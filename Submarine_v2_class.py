import numpy as np
import casadi as ca
from typing import Callable, Optional, Tuple, Union

class UnderwaterVehicle_v2:
    """
    4-DOF underwater vehicle (submarine) simulation class.

    STATE:
      η = [x, y, z, ψ]    (inertial position x,y,z and yaw ψ)
      ν = [u, v, w, r]    (body-fixed velocities: surge u, sway v, heave w, yaw rate r)

    DYNAMICS (body-frame):
      M * ν̇ + C(ν)·ν + D(ν)·ν + g(η) = τ_control + τ_disturbance

    KINEMATICS (inertial):
      η̇ = J(ψ) · [u, v, w, r]^T

    USAGE:
      - Call `step(τ_control, τ_disturbance)` once per time-step to advance fidelity.
      - Use `observe()` to obtain a noisy measurement of (η, ν).
      - Use `get_state()` to obtain the (un-noisy) internal state.
      - `reset()` returns the vehicle to its initial condition.
    """

    def __init__(
        self,
        M: Optional[np.ndarray] = None,
        C_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        D_lin: Union[np.ndarray, Tuple[float, float, float, float]] = (50.0, 50.0, 50.0, 10.0),
        D_quad: Union[np.ndarray, Tuple[float, float, float, float]] = (100.0, 100.0, 100.0, 20.0),
        mass: float = 12.0,
        I_z: float = 0.12,
        initial_state: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]] = ((0, 0, 0, 0), (0, 0, 0, 0)),
        dt: float = 0.1,
        max_thrust: float = 90.0,
        max_torque: float = 30.0,
        pos_noise_std: float = 0.1,
        vel_noise_std: float = 0.05,
        z_ref: float = 0.0,
        rho_water: float = 1025.0,
        volume: Optional[float] = None,
        beta: float = 4.5e-6,
        compressibility_depth: float = 2e4,
        p_atm: float = 101325.0,
        g_gravity: float = 9.81,
    ):
        """
        Initialize the vehicle with optional custom parameters.

        Parameters:
        -----------
        M : (4×4) added-mass + rigid-body inertia matrix. If None, a diagonal matrix
            with added-mass fractions will be created automatically:
              [m + 0.15·m, m + 0.15·m, m + 0.15·m, I_z + 0.10·m]
        C_func : function C(ν) → (4×4) Coriolis matrix. If None, a simple skew-symmetric
            form based on the first diagonal of M will be used.
        D_lin : length-4 array of linear damping coefficients [X_u, Y_v, Z_w, N_r].
            If None, defaults to [50, 50, 50, 10].
        D_quad : length-4 array of quadratic damping coefficients [X_{u|u|}, Y_{v|v|}, Z_{w|w|}, N_{r|r|}].
            If None, defaults to [100, 100, 100, 20].
        mass : vehicle mass in kg. If None and M is provided, taken from M[0,0].
            Otherwise defaults to 37 kg.
        I_z : yaw moment of inertia in kg·m². Overrides the last diagonal entry of M if provided.
        initial_state : ((x0, y0, z0, ψ0), (u0, v0, w0, r0)). Defaults to all zeros.
        dt : simulation time-step (s).
        max_thrust : maximum force (N) in surge/sway/heave directions.
        max_torque : maximum yaw moment (N·m).
        pos_noise_std : standard deviation (m) for additive noise on η during observation.
        vel_noise_std : standard deviation (m/s) for additive noise on ν during observation.
        z_ref : neutral-buoyancy reference depth (m).
        rho_water : nominal water density (kg/m³).
        volume : vehicle displaced volume (m³). If None, set to mass / rho_water.
        beta : hull compressibility coefficient [1/m].
        compressibility_depth : depth constant for ρ(z) variation (m).
        p_atm : atmospheric pressure at surface (Pa).
        g_gravity : gravitational acceleration (m/s²).
        """
        # Environmental constants
        self.dt = dt
        self.g_gravity = g_gravity
        self.rho_water = rho_water
        self.z_ref = z_ref
        self.beta = beta
        self.compressibility_depth = compressibility_depth
        self.p_atm = p_atm

        # Sensor noise
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std

        # Actuator limits
        self.max_thrust = max_thrust
        self.max_torque = max_torque

        # Mass properties
        self.mass = mass
        if M is None:
            ma_lin = 0.15 * self.mass
            ma_yaw = 0.10 * self.mass
            M_rigid = np.diag([self.mass] * 3 + [I_z])
            M_added = np.diag([ma_lin] * 3 + [ma_yaw])
            self.M = M_rigid + M_added
        else:
            self.M = np.array(M, dtype=float)
        assert self.M.shape == (4,4) and np.all(np.linalg.eigvals(self.M) > 0), "Inertia M must be positive definite 4×4"
        self.M_inv = np.linalg.inv(self.M)

        # Coriolis
        self.C_func = C_func if C_func is not None else self._default_coriolis

        # Damping
        self.D_lin = np.array(D_lin, dtype=float).reshape(4,)
        self.D_quad = np.array(D_quad, dtype=float).reshape(4,)

        # Buoyancy
        self.volume = volume if volume is not None else self.mass / rho_water
        self.weight = self.mass * self.g_gravity

        # State initialization
        eta0, nu0 = initial_state
        self.eta = np.array(eta0, dtype=float)
        self.nu = np.array(nu0, dtype=float)
        self.initial_eta = self.eta.copy()
        self.initial_nu = self.nu.copy()

        # Build dynamics
        self._build_casadi_dynamics()

    def __str__(self) -> str:
        x, y, z, psi = self.eta
        u, v, w, r = self.nu
        return (
            f"Pose:   x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={np.degrees(psi):.1f}°\n"
            f"Vel:    u={u:.2f}, v={v:.2f}, w={w:.2f}, r={r:.2f}"
        )

    def _default_coriolis(self, ν: np.ndarray) -> np.ndarray:
        """
        Simple skew-symmetric Coriolis matrix based on first diagonal of M (mass).
        C(ν) ∈ ℝ⁴×⁴ such that C·ν captures added-mass Coriolis.
        """
        m = float(self.M[0, 0])
        u, v, w, r = ν
        return np.array([
            [  0.0,    0.0,    0.0, -m * v],
            [  0.0,    0.0,    0.0,  m * u],
            [  0.0,    0.0,    0.0,    0.0],
            [ m * v, -m * u,    0.0,    0.0]
        ])

    def _build_casadi_dynamics(self) -> None:
        """
        Build a CasADi function for one RK4 step of:
          x = [η; ν] ∈ ℝ⁸,  u = [τ_control + τ_disturbance] ∈ ℝ⁴,
          x_dot( x, u ) = [ η̇; ν̇ ],
        Then discretize via RK4 with step = self.dt.
        """
        x_sym = ca.SX.sym("x", 8)  # [x, y, z, ψ, u, v, w, r]
        u_sym = ca.SX.sym("u", 4)  # [F_x, F_y, F_z, M_z]

        η_sym = x_sym[0:4]     # [x, y, z, ψ]
        ν_sym = x_sym[4:8]     # [u, v, w, r]

        # Hydrostatic force: depends on η_sym (z portion)
        f_z_sym = self._casadi_hydrostatic_force(η_sym)
        g_vec = ca.vertcat(0, 0, f_z_sym, 0)  # only vertical component

        # Damping: D_lin + D_quad * |ν|
        D_lin_mat = ca.diag(ca.vertcat(*[float(d) for d in self.D_lin]))
        abs_nu = ca.fabs(ν_sym)
        D_quad_mat = ca.diag(ca.vertcat(*[float(d) * abs_nu[i] for i, d in enumerate(self.D_quad)]))
        D_mat = D_lin_mat + D_quad_mat

        # Kinematics: η̇ = J(ψ) * ν
        ψ = η_sym[3]
        c = ca.cos(ψ)
        s = ca.sin(ψ)
        J = ca.vertcat(
            ca.horzcat(c, -s, 0, 0),
            ca.horzcat(s,  c, 0, 0),
            ca.horzcat(0,  0, 1, 0),
            ca.horzcat(0,  0, 0, 1),
        )
        η_dot = J @ ν_sym

        # Dynamics: M * ν̇ = u_sym - D·ν - g_vec  (Coriolis neglected here, since it’s small
        # or you can add C(ν)*ν if you want a symbolic C; for now we assume C = 0 or negligible.)
        # If you really need C(ν), you’d need to build a CasADi expression for it.
        ν_dot = ca.mtimes(ca.inv(self.M), (u_sym - D_mat @ ν_sym + g_vec))

        x_dot = ca.vertcat(η_dot, ν_dot)
        f_cont = ca.Function("f_cont", [x_sym, u_sym], [x_dot])

        # RK4 discretization
        k1 = f_cont(x_sym,     u_sym)
        k2 = f_cont(x_sym + 0.5 * self.dt * k1, u_sym)
        k3 = f_cont(x_sym + 0.5 * self.dt * k2, u_sym)
        k4 = f_cont(x_sym +     self.dt * k3, u_sym)
        x_next = x_sym + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self._discrete_dynamics = ca.Function(
            "discrete_dynamics",
            [x_sym, u_sym],
            [x_next],
            ["x", "u"],
            ["x_next"]
        )

    def _casadi_hydrostatic_force(self, η: ca.SX) -> ca.SX:
        """
        CasADi expression for depth-dependent hydrostatic force f_z(η).
        η = [x, y, z, ψ], so η[2] = z.
        Returns f_z symbolically.
        """
        z = η[2]
        delta_z = z - self.z_ref
        # Prevent over-compression:
        V = self.volume * (1 - self.beta * delta_z)
        V = ca.fmax(V, 0.95 * self.volume)
        effective_rho = self.rho_water * (1 + delta_z / self.compressibility_depth)
        buoyancy = effective_rho * self.g_gravity * V
        f_z = self.weight - buoyancy
        return f_z

    def rotation_matrix(self, ψ: float) -> np.ndarray:
        """
        Rotation J(ψ) ∈ ℝ⁴×⁴ mapping body-frame velocities [u,v,w,r] → inertial rates [ẋ,ẏ,ż,ψ̇].
        Only yaw rotation is considered.
        """
        c = np.cos(ψ)
        s = np.sin(ψ)
        return np.array([
            [ c, -s, 0, 0],
            [ s,  c, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1],
        ])

    def damping_matrix(self, ν: np.ndarray) -> np.ndarray:
        """
        Compute D(ν) = diag(D_lin) + diag(D_quad * |ν|).
        ν = [u, v, w, r].
        """
        abs_nu = np.abs(ν)
        D_lin_mat = np.diag(self.D_lin)
        D_quad_mat = np.diag(self.D_quad * abs_nu)
        return D_lin_mat + D_quad_mat

    def hydrostatic_force(self, η: np.ndarray) -> float:
        """
        Compute scalar vertical hydrostatic force f_z given η = [x, y, z, ψ].
        Positive downwards (NED convention), so f_z = weight - buoyancy.
        """
        z = η[2]
        delta_z = z - self.z_ref
        V = self.volume * (1 - self.beta * delta_z)
        V = max(V, 0.95 * self.volume)
        effective_rho = self.rho_water * (1 + delta_z / self.compressibility_depth)
        buoyancy = effective_rho * self.g_gravity * V
        return self.weight - buoyancy

    def step(
        self,
        τ_control: np.ndarray,
        τ_disturbance: Optional[np.ndarray] = None
    ) -> None:
        """
        Advance one time-step (self.dt) using RK4 integration.

        Parameters:
        -----------
        τ_control : length-4 array [F_x, F_y, F_z, M_z], control forces/torques in body frame.
        τ_disturbance : optional length-4 array of external forces/torques.
                        If None, zeros are assumed.
        """
        if τ_disturbance is None:
            τ_disturbance = np.zeros(4)

        # Saturate control inputs
        Fx, Fy, Fz, Mz = τ_control
        Fx = np.clip(Fx, -self.max_thrust,  self.max_thrust)
        Fy = np.clip(Fy, -self.max_thrust,  self.max_thrust)
        Fz = np.clip(Fz, -self.max_thrust,  self.max_thrust)
        Mz = np.clip(Mz, -self.max_torque,  self.max_torque)
        τ_sat = np.array([Fx, Fy, Fz, Mz])

        # Total input to dynamics
        u_in = τ_sat + τ_disturbance

        # Pack current state into CasADi vector
        x0 = ca.vertcat(
            self.eta[0], self.eta[1], self.eta[2], self.eta[3],
            self.nu[0], self.nu[1], self.nu[2], self.nu[3]
        )

        # One RK4 step
        x_next = self._discrete_dynamics(x0, u_in).full().flatten()
        self.eta = x_next[0:4]
        self.nu  = x_next[4:8]

        # Optionally wrap yaw into [-π, π]
        # self.eta[3] = (self.eta[3] + np.pi) % (2 * np.pi) - np.pi

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the true internal state:
          (η, ν) as NumPy arrays of shape (4,) each.
        """
        return self.eta.copy(), self.nu.copy()

    def observe(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a noisy measurement of (η, ν):
          - adds Gaussian noise to position/angle (std = pos_noise_std),
          - adds Gaussian noise to body velocities (std = vel_noise_std).
        """
        noisy_eta = self.eta + np.random.normal(0.0, self.pos_noise_std, size=4)
        noisy_nu  = self.nu + np.random.normal(0.0, self.vel_noise_std, size=4)
        return noisy_eta, noisy_nu

    def reset(self) -> None:
        """
        Reset internal state (η, ν) back to the initial conditions provided at construction.
        """
        self.eta = self.initial_eta.copy()
        self.nu  = self.initial_nu.copy()

    def control_cmds(Fx,Fy,Fz,Mz):
        d = 0.16    # m, half‐arm offset from center
        # order: thrusters [FL, FR, BL, BR, V1, V2]
        B = np.array([
        # Fx  Fy  Fz  Mz   -> thruster forces
        [+.707, +.707, 0,  0],   # FL  (45° forward-right)
        [+.707, -.707, 0,  0],   # FR  (45° forward-left)
        [+.707, -.707, 0,  0],   # BL  (45° backward-left)
        [+.707, +.707, 0,  0],   # BR  (45° backward-right)
        [   0,     0,  1,  0],   # V1  vertical
        [   0,     0,  1,  0],   # V2  vertical
        ])
        # And yaw moment Mz is realized by opposite‐signed thrusts on FL vs FR (and/or BL vs BR):
        # you’d augment B’s Mz column accordingly, e.g. ±d lever arm.
        thruster_cmds = np.linalg.pinv(B) @ np.array([Fx,Fy,Fz,Mz])

        return thruster_cmds


