import logging
from typing import Optional
import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are

logger = logging.getLogger(__name__)

class OneStepTVLQRController:
    def __init__(self, system, nx, nu, dt=0.1, Fmax=200.0, Q=None, R=None):
        # ... initialization ...
        
        # Discrete dynamics and Jacobians
        x = ca.MX.sym('x', nx)
        u = ca.MX.sym('u', nu)
        x_next = system._discrete_dynamics(x, u)

        self.system = system
        self.nx, self.nu = nx, nu
        self.dt, self.Fmax = dt, Fmax
        self.last_u_ff = np.zeros(nu)

        default_Q = np.diag([500,500,500,100,50,50] + [10]*(nx-6))
        self.Q = Q if Q is not None else default_Q
        self.R = R if R is not None else np.eye(nu)*0.01
        
        # CORRECT: Discrete Jacobians
        self.A_fun = ca.Function('A_discrete', [x,u], [ca.jacobian(x_next, x)])
        self.B_fun = ca.Function('B_discrete', [x,u], [ca.jacobian(x_next, u)])
        self.dyn_fun = ca.Function('dyn', [x,u], [x_next])

    def _compute_feedforward(self, x_ref, x_ref_next):
        u_ff = self.last_u_ff.copy()  # Warm start
        prev_residual_norm = np.inf
        
        for _ in range(5):
            # 1. Compute prediction and residual
            x_pred = self.dyn_fun(x_ref, u_ff).full().flatten()
            residual = x_ref_next - x_pred
            residual_norm = np.linalg.norm(residual)
            
            # Break if residual increases (divergence)
            if residual_norm > prev_residual_norm * 1.1:
                break
            prev_residual_norm = residual_norm
            
            # 2. Regularized Jacobian
            B = self.B_fun(x_ref, u_ff).full()
            B_reg = B + 1e-5 * np.eye(*B.shape)
            
            # 3. Regularized least-squares solution
            du = np.linalg.lstsq(
                B_reg, 
                residual,
                rcond=1e-6  # Explicit regularization parameter
            )[0]
            
            # 4. Update with momentum
            u_ff += 0.7 * du  # Fixed sign (ADDITIVE update)
            u_ff = np.clip(u_ff, -self.Fmax, self.Fmax)
            
            # 5. Convergence check
            if np.linalg.norm(du) < 1e-4 or residual_norm < 1e-5:
                break
                
        self.last_u_ff = u_ff
        return u_ff

    def solve(self, x_current, ref, k_time = None): #k_time used to debug and see when the true ARE is solvable compared to good controlling results
        """
        x_current: shape (nx,)
        ref:       shape (nx, 2)   [x_ref[k], x_ref[k+1]]
        """
        # unpack references
        x_ref      = ref[:,0]
        x_ref_next = ref[:,1] if ref.shape[1]>1 else ref[:,0]

        # 1. Iterative feedforward (discrete)
        u_ff = self._compute_feedforward(x_ref, x_ref_next)

        # 2. Discrete linearization
        A_d = self.A_fun(x_current, u_ff).full()
        B_d = self.B_fun(x_current, u_ff).full()

        # 3. Solve DISCRETE Riccati
        try:
            P = solve_discrete_are(A_d, B_d, self.Q, self.R)
            K = np.linalg.inv(self.R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        except:
            # Fallback to one-step LQR
            K = np.linalg.solve(B_d.T @ self.Q @ B_d + self.R, 
                                B_d.T @ self.Q @ A_d)
            print("LQR DARE solving failed. ")

        # 4. Compute control
        e = x_current - x_ref  # State deviation
        u = u_ff - K @ e  # u_ff minus feedback
        u = np.clip(u, -self.Fmax, self.Fmax)
        
        return u, e.T @ self.Q @ e + u.T @ self.R @ u