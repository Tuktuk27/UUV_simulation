import logging
from typing import Optional
import numpy as np
import casadi as ca
from scipy.linalg import norm
from scipy.linalg import solve_discrete_are

logger = logging.getLogger(__name__)

class OneStepTVLQIController:
    """
    Numerically stable LQI controller for submarine dynamics
    - One-step finite horizon LQR instead of DARE
    - Comprehensive numerical safeguards
    - Gain fallback mechanism
    """
    def __init__(
        self,
        system,
        nx: int,
        nu: int,
        dt: float = 0.1,
        Fmax: float = 200.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Qi: Optional[np.ndarray] = None,
    ):
        self.system = system
        self.nx, self.nu = nx, nu
        self.dt, self.Fmax = dt, Fmax

        # Regularized weighting matrices
        default_Q = np.diag([500,500,500,100,50,50] + [10]*(nx-6))
        self.Q = self._regularize(Q, default_Q, 1e-4)
        self.R = self._regularize(R, np.eye(nu)*0.01, 1e-3)
        self.Qi = self._regularize(Qi, np.eye(nx)*0.5, 1e-4)
        
        self.xi = np.zeros(nx)
        self.last_u_ff = np.zeros(nu)
        self.last_Kx = np.zeros((nu, nx))
        self.last_Ki = np.zeros((nu, nx))
        self.last_A_aug = np.eye(2*nx)
        self.last_B_aug = np.zeros((2*nx, nu))
        self.fallback_count = 0
        
        # Compile efficient Jacobians
        self._setup_jacobians()

    def _regularize(self, matrix, default, epsilon):
        """Ensure well-conditioned matrices"""
        if matrix is None:
            matrix = default
        return matrix + epsilon * np.eye(matrix.shape[0])
    
    def _setup_jacobians(self):
        """Create Jacobian functions with proper input dependencies"""
        x = ca.MX.sym('x', self.nx)
        u = ca.MX.sym('u', self.nu)
        
        # Discrete dynamics
        x_next = self.system._discrete_dynamics(x, u)
        
        # Discrete Jacobians
        self.A_fun = ca.Function('A_jac', [x, u], [ca.jacobian(x_next, x)])
        self.B_fun = ca.Function('B_jac', [x, u], [ca.jacobian(x_next, u)])
        
        # Feedforward function
        x_ref_next_sym = ca.MX.sym('x_ref_next', self.nx)
        residual = x_next - x_ref_next_sym
        self.res_fun = ca.Function('residual', 
                                   [x, u, x_ref_next_sym], 
                                   [residual])
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

    def _compute_gains(self, A_aug, B_aug, Q_aug, R_aug, k_time):
        """Safe gain computation with multiple fallbacks"""
        # 5. Compute *true* infinite‐horizon LQI gains
        try:
            P_aug = solve_discrete_are(A_aug, B_aug, Q_aug, R_aug)
            K_aug = np.linalg.inv(R_aug + B_aug.T @ P_aug @ B_aug) \
                    @ (B_aug.T @ P_aug @ A_aug)
            used_fallback = False
            # print(k_time)
        except Exception as err:
            try:
                P = Q_aug.copy()
                for _ in range(100):
                    S = R_aug + B_aug.T @ P @ B_aug
                    K_pi = np.linalg.solve(S, B_aug.T @ P @ A_aug)
                    P_next = A_aug.T @ P @ (A_aug - B_aug @ K_pi) + Q_aug
                    if np.linalg.norm(P_next - P) < 1e-8:
                        break
                    P = P_next
                # then extract gains
                K_aug = np.linalg.solve(R_aug + B_aug.T @ P @ B_aug,
                                        B_aug.T @ P @ A_aug)
            except:
                logger.warning(f"LQI DARE failed: {err}; falling back to one‐step LQR")
                # your existing one‐step fallback:
                S = B_aug.T @ Q_aug @ B_aug + R_aug
                T = B_aug.T @ Q_aug @ A_aug
                K_aug = np.linalg.solve(S, T)
                used_fallback = True

        # Save successful matrices for potential fallback
        self.last_A_aug = A_aug
        self.last_B_aug = B_aug
        self.fallback_count = 0
        
        return K_aug, False


    def solve(self, x_current: np.ndarray, ref: np.ndarray, k_time = None): #k_time used to debug and see when the true ARE is solvable compared to good controlling results
        x_ref = ref[:, 0]
        x_ref_next = ref[:, 1] if ref.shape[1] > 1 else x_ref.copy()

        # 1. Robust feedforward calculation
        u_ff = self._compute_feedforward(x_ref, x_ref_next)
        
        # 2. Discrete linearization with regularization
        try:
            A_d = self.A_fun(x_current, u_ff).full()
            B_d = self.B_fun(x_current, u_ff).full()
            
        except:
            logger.error("Jacobian evaluation failed, using last matrices")
            A_d = self.last_A_aug[:self.nx, :self.nx]
            B_d = self.last_B_aug[:self.nx, :]

        # 3. Build augmented system
        A_aug = np.block([
            [A_d,                        np.zeros((self.nx, self.nx))],
            [-self.dt * np.eye(self.nx),   np.eye(self.nx)]
        ])
        B_aug = np.vstack([B_d, np.zeros((self.nx, self.nu))])
        
        # 4. Regularized augmented cost
        Q_reg = self.Q + 1e-6 * np.eye(self.nx)
        Qi_reg = self.Qi + 1e-6 * np.eye(self.nx)
        R_reg = self.R + 1e-6 * np.eye(self.nu)
        Q_aug = np.block([
            [Q_reg, np.zeros((self.nx, self.nx))],
            [np.zeros((self.nx, self.nx)), Qi_reg]
        ])
        
        # 5. Compute gains with fallback
        K_aug, used_fallback = self._compute_gains(A_aug, B_aug, Q_aug, R_reg, k_time)
        
        # 6. Extract gains
        Kx = K_aug[:, :self.nx]
        Ki = K_aug[:, self.nx:]
        
        # Save gains for potential fallback
        self.last_Kx = Kx
        self.last_Ki = Ki
        
        # 7. Error calculation
        e = x_current - x_ref
        
        # 8. Adaptive integral control
        u_unclipped = u_ff - Kx @ e - Ki @ self.xi
        saturated = np.any(np.abs(u_unclipped) > self.Fmax)
        
        if not saturated:
            # Standard integration
            self.xi += e * self.dt
        else:
            # Conditional reset to prevent windup
            self.xi *= 0.8  # Leaky integrator
        
        # 9. Apply control limits
        u = np.clip(u_unclipped, -self.Fmax, self.Fmax)
        
        # 10. Cost calculation
        cost = e.T @ self.Q @ e + self.xi.T @ self.Qi @ self.xi + u.T @ self.R @ u
        
        # Log controller status
        status = "FALLBACK" if used_fallback else "NORMAL"
        logger.info(f"Control status: {status} | Saturated: {saturated} | Cost: {cost:.2f}")
        
        return u, float(cost)