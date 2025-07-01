"""
Module: hybrid_tvlqi_submarine.py

Receding‑Horizon TV‑LQI Controller for submarine trajectory tracking.
Computes feed‑forward and LQI gains over each horizon window using helper
methods for clarity and maintainability.

Interface: `solve(current_state, reference_window) -> (u0, cost)`
"""
import logging
from typing import Optional, Tuple, List
import numpy as np
import casadi as ca

logger = logging.getLogger(__name__)

class HorizonTVLQIController:
    """
    Receding‑Horizon Time‑Varying Linear‑Quadratic Integrator (TV‑LQI).

    - Computes feed‑forward and LQI gains over the provided reference window.
    - Uses physics‑based feed‑forward specific to submarine dynamics.
    - Matches `MPCController.solve(state, horizon_window)` interface.
    """

    def __init__(
        self,
        system,
        nx: int,
        nu: int,
        N=20,
        dt: float = 0.1,
        Fmax: float = 200.0,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        Qi: Optional[np.ndarray] = None,
    ):
        # dynamics and dimensions
        self.system = system
        self.nx = nx
        self.nu = nu
        self.N = N                 # prediction horizon
        self.dt = dt
        self.Fmax = Fmax

        # cost weights
        default_Q = np.diag([500, 500, 500, 100, 50, 50] + [10] * (nx - 6))
        self.Q  = Q  if Q  is not None else default_Q
        self.R  = R  if R  is not None else np.diag([0.1]*nu)
        self.Qi = Qi if Qi is not None else np.eye(nx) * 0.5

        # persistent integral state
        self.integral_error = np.zeros(self.nx)

        # build CasADi Jacobian once
        self._build_casadi_functions()

    def _build_casadi_functions(self):
        x = ca.MX.sym('x', self.nx)
        u = ca.MX.sym('u', self.nu)
        x_next = self.system._discrete_dynamics(x, u)
        f_cont = (x_next - x) / self.dt
        self._A_fun = ca.Function('A_jac', [x, u], [ca.jacobian(f_cont, x)])
        self._B_fun = ca.Function('B_jac', [x, u], [ca.jacobian(f_cont, u)])

    def _compute_feedforward_and_linearization(
        self,
        traj: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Compute feed-forward sequence and linearization matrices A_seq, B_seq
        over the horizon defined by traj (shape nx x (H+1)).
        """

        # pad ref to N+1
        H = traj.shape[1] - 1

        if H < self.N:
            last = traj[:, -1:]
            pad  = np.tile(last, (1, self.N - H))
            traj = np.hstack([traj, pad])
            H = self.N
        
        ff_seq = np.zeros((self.nu, H))
        A_seq: List[np.ndarray] = []
        B_seq: List[np.ndarray] = []

        for k in range(H):
            xk = traj[:,k]
            xk1 = traj[:,k+1]
            # feed-forward via pseudo-inverse
            x_pred0 = self.system._discrete_dynamics(xk, np.zeros(self.nu)).full().flatten()
            delta = xk1 - x_pred0
            Bk = self._B_fun(xk, np.zeros(self.nu)).full()
            u_ff = np.linalg.pinv(Bk) @ delta
            ff_seq[:,k] = u_ff
            # linearizations
            A_seq.append(self._A_fun(xk, u_ff).full())
            B_seq.append(Bk)

        return ff_seq, A_seq, B_seq

    def _compute_lqi_gains(
        self,
        A_seq: List[np.ndarray],
        B_seq: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward Riccati recursion to compute TV-LQI gain sequences Ks, Ki.
        """
        H = len(A_seq)
        Qa = np.block([
            [self.Q,               np.zeros((self.nx, self.nx))],
            [np.zeros((self.nx, self.nx)), self.Qi]
        ])
        Ra = self.R
        P = Qa.copy()
        Ks: List[np.ndarray] = [None]*H
        Ki: List[np.ndarray] = [None]*H

        for k in reversed(range(H)):
            Ak = A_seq[k]
            Bk = B_seq[k]
            # corrected augmented dynamics
            Aaug = np.block([
                [Ak,                           np.zeros((self.nx, self.nx))],
                [-self.dt * np.eye(self.nx),   np.eye(self.nx)]
            ])
            Baug = np.vstack([Bk, np.zeros((self.nx, self.nu))])
            S = Ra + Baug.T @ P @ Baug
            Kaug = np.linalg.lstsq(S, Baug.T @ P @ Aaug, rcond=None)[0]
            P = Qa + Aaug.T @ P @ Aaug - Aaug.T @ P @ Baug @ Kaug
            Ks[k] = Kaug[:, :self.nx]
            Ki[k] = Kaug[:, self.nx:]

        return Ks, Ki

    def solve(self, current_state: np.ndarray, ref_window: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve for the first control input over a receding horizon window:
        - ref_window: shape (nx, H+1)
        """
        # Prepare horizon data
        ff_seq, A_seq, B_seq = self._compute_feedforward_and_linearization(ref_window)
        Ks, Ki = self._compute_lqi_gains(A_seq, B_seq)

        # Compute error and update integral
        e = current_state - ref_window[:,0]
        self.integral_error += e * self.dt

        # First-step control
        u_ff  = ff_seq[:,0]
        u_fb  = Ks[0] @ e
        u_int = Ki[0] @ self.integral_error
        u0 = u_ff - (u_fb + u_int)
        u0 = np.clip(u0, -self.Fmax, self.Fmax)

        # Cost with cross-term
        cost = float(
            e.T @ self.Q @ e +
            self.integral_error.T @ self.Qi @ self.integral_error +
            u0.T @ self.R @ u0 +
            2 * e.T @ self.Qi @ self.integral_error
        )

        return u0, cost
