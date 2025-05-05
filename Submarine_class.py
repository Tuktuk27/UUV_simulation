import numpy as np
import casadi as ca

class UnderwaterVehicle:
    """
    4-DOF underwater vehicle (submarine) simulation class.
    State:
      eta = [x, y, z, psi]  (inertial position x,y,z and yaw psi)
      nu  = [u, v, w, r]    (body-fixed velocities: surge u, sway v, heave w, yaw rate r)
    Dynamics (body-frame):
      M*dot(nu) + C(nu)*nu + D(nu)*nu + g(eta) = tau + w
    Kinematics (inertial):
      dot(x,y,z,psi) = J(psi) * [u, v, w, r]^T
    """
    def __init__(self, 
                 M=None, C_func=None, D_lin=None, D_quad=None,
                 I_z=None, weight=None, buoyancy=None,
                 initial_state=None, dt=0.1):
        """
        Initialize the vehicle with optional custom parameters.
        M        : 4x4 inertia matrix (diagonal by default)
        C_func   : function C(nu) to compute Coriolis matrix (or None to use default)
        D_lin    : array of linear damping coefficients [Xu, Yv, Zw, Nr]
        D_quad   : array of quadratic damping coefficients [Xu, Yv, Zw, Nr]
        I_z      : yaw moment of inertia (overrides last diag entry of M)
        weight   : weight (N), for hydrostatic force
        buoyancy : buoyant force (N)
        initial_state : tuple (eta0, nu0) to set initial state
        dt       : default timestep for integration
        """
        # Inertia matrix M (mass and inertia). Diagonal by default.
        # Units: M = diag([m, m, m, I_z])
        if M is None:
            m = 1000.0  # mass in kg (example)
            I_z = 500.0 if I_z is None else I_z
            self.M = np.diag([m, m, m, I_z])
        else:
            self.M = np.array(M, dtype=float)
        
        # Pre-compute inverse of M
        self.M_inv = np.linalg.inv(self.M)
        
        # Default Coriolis: zero or simple skew-symmetric.
        if C_func is None:
            # Example: simple constant Coriolis (could be improved)
            self.C_func = lambda nu: np.zeros((4,4))
        else:
            self.C_func = C_func
        
        # Damping: linear + quadratic (diagonal)
        if D_lin is None:
            # Example linear damping coefficients
            D_lin = np.array([50, 50, 50, 10], dtype=float)  # [Xu, Yv, Zw, Nr]
        if D_quad is None:
            D_quad = np.array([100, 100, 100, 20], dtype=float)  # [Xu|u|, Yv|v|, Zw|w|, Nr|r|]
        self.D_lin = np.array(D_lin, dtype=float)
        self.D_quad = np.array(D_quad, dtype=float)
        
        # Hydrostatic forces (gravity vs buoyancy)
        # Positive weight acts downward, buoyancy upward.
        if weight is None:
            weight = self.M[0,0] * 9.81  # assume weight = mass * g
        if buoyancy is None:
            buoyancy = weight  # neutrally buoyant by default
        # g_vec = [Fz (gravity-buoyancy), rest zeros]; no yaw restoring in basic model
        self.weight = weight
        self.buoyancy = buoyancy
        
        # Initial state
        if initial_state is None:
            # eta0 = [0, 0, 0, 0], nu0 = [0, 0, 0, 0]
            self.eta = np.zeros(4)
            self.nu = np.zeros(4)
        else:
            eta0, nu0 = initial_state
            self.eta = np.array(eta0, dtype=float)
            self.nu  = np.array(nu0, dtype=float)
        self.initial_eta = self.eta.copy()
        self.initial_nu = self.nu.copy()
        
        self.dt = dt

        # Build CasADi dynamics immediately
        self._build_casadi_dynamics()

    def __str__(self):
        x, y, z, psi = self.eta
        u, v, w, r = self.nu
        return (f"Pose: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={np.degrees(psi):.1f}°\n"
                f"Vel:  u={u:.2f}, v={v:.2f}, w={w:.2f}, r={r:.2f}")
    
    def _build_casadi_dynamics(self):
        # 1) symbolic states and inputs
        x = ca.SX.sym('x', 8)   # [x,y,z,psi, u,v,w,r]
        u = ca.SX.sym('u', 4)   # [Fx,Fy,Fz,Mz]

        # 2) split into eta and nu symbols
        eta = x[0:4]            # CasADi slice, keeps it symbolic
        nu  = x[4:8]

        # 3) hydrostatic term (fixed)
        fgz = float(self.buoyancy - self.weight)
        g_vec = ca.vertcat(0, 0, fgz, 0)

        # 4) build Coriolis (zero) and damping symbolically
        #    C = zero(4,4)
        D_lin = ca.diag(ca.vertcat(*[float(d) for d in self.D_lin]))
        # D_quad * |nu|
        abs_nu = ca.fabs(nu)
        D_quad = ca.diag(ca.vertcat(*[float(d)*abs_nu[i]
                                      for i,d in enumerate(self.D_quad)]))
        D = D_lin + D_quad

        # 5) kinematics: eta_dot = J(psi) * nu
        psi = eta[3]
        c = ca.cos(psi); s = ca.sin(psi)
        J = ca.vertcat(
            ca.horzcat( c, -s, 0, 0),
            ca.horzcat( s,  c, 0, 0),
            ca.horzcat( 0,  0, 1, 0),
            ca.horzcat( 0,  0, 0, 1)
        )
        eta_dot = J @ nu

        # 6) dynamics: M*nu_dot = tau - (C + D)*nu - g
        #    here C is zero, so just D
        tau = u
        nu_dot = self.M_inv @ (tau - D @ nu - g_vec)

        # 7) stack RHS
        x_dot = ca.vertcat(eta_dot, nu_dot)

        # 8) CasADi Function for continuous dynamics
        f_cont = ca.Function('f_cont', [x, u], [x_dot])

        # 9) RK4 discretization
        k1 = f_cont(x,         u)
        k2 = f_cont(x + 0.5*self.dt*k1, u)
        k3 = f_cont(x + 0.5*self.dt*k2, u)
        k4 = f_cont(x +     self.dt*k3, u)
        x_next = x + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # 10) expose discrete dynamics
        self.discrete_dynamics = ca.Function(
            'discrete_dynamics', [x, u], [x_next],
            ['x','u'], ['x_next']
        )

    def rotation_matrix(self, psi):
        """
        Rotation matrix J(psi) that maps body-frame velocities [u,v,w,r]
        to inertial-frame rates [x_dot, y_dot, z_dot, psi_dot].
        Here orientation is yaw-only, so we rotate the (u,v) components.
        """
        c, s = np.cos(psi), np.sin(psi)
        J = np.array([[c, -s, 0, 0],
                      [s,  c, 0, 0],
                      [0,  0, 1, 0],
                      [0,  0, 0, 1]])
        return J

    def damping_matrix(self, nu):
        """
        Compute the damping matrix D(nu) = D_lin + D_quad * |v|.
        Here v = [u,v,w,r].
        """
        abs_nu = np.abs(nu)
        D_lin_mat = np.diag(self.D_lin)
        D_quad_mat = np.diag(self.D_quad * abs_nu)
        return D_lin_mat + D_quad_mat

    def hydrostatic_force(self, eta):
        """
        Hydrostatic restoring force vector g(eta) due to gravity and buoyancy.
        We assume zero moments in yaw for simplicity. Weight acts in -z.
        """
        # For vertical (z): net force = buoyancy - weight
        fgz = self.buoyancy - self.weight
        # No force in x,y, and no yaw moment (simplified)
        return np.array([0.0, 0.0, fgz, 0.0])

    def state_derivatives(self, eta, nu, tau, w):
        """
        Given state (eta, nu) and inputs (tau: control, w: disturbance),
        compute time derivatives (eta_dot, nu_dot).
        """
        # Kinematics: inertial rates = J(psi) * body velocities
        psi = eta[3]
        J = self.rotation_matrix(psi)
        eta_dot = J @ nu
        
        # Dynamics: M * nu_dot = tau + w - (C + D)*nu - g
        C = self.C_func(nu)  # Coriolis matrix (4x4)
        D = self.damping_matrix(nu)
        g = self.hydrostatic_force(eta)
        
        nu_dot = self.M_inv @ (tau + w - (C + D) @ nu - g)
        return eta_dot, nu_dot

    def step(self, tau, w=None):
        """
        Advance the simulation by one timestep (self.dt) using Euler integration.
        tau : control input vector [Fx, Fy, Fz, Mz] in body frame
        w   : external disturbance vector (same shape as tau), defaults to zero.
        """
        if w is None:
            w = np.zeros(4)
        # Current state
        eta = self.eta
        nu = self.nu
        
        # Compute derivatives at current state
        eta_dot, nu_dot = self.state_derivatives(eta, nu, tau, w)
        
        # Simple Euler integration
        self.eta += eta_dot * self.dt
        self.nu  += nu_dot  * self.dt

    def get_state(self):
        """Return current state as (eta, nu)."""
        return self.eta.copy(), self.nu.copy()

    def reset(self):
        """Reset state to initial conditions."""
        self.eta = self.initial_eta.copy()
        self.nu  = self.initial_nu.copy()



# Example usage:
sub = UnderwaterVehicle()
print(sub)
tau = np.array([100, 0, 0, 10])  # some forward force and yaw moment
for _ in range(100):
    sub.step(tau)
print("After motion:", sub)
