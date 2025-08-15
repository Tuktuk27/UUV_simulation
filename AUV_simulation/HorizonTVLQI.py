import numpy as np
from TVLQI_v2 import OneStepTVLQIController
import logging
from typing import Optional
import casadi as ca
from scipy.linalg import norm
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

logger = logging.getLogger(__name__)

class HorizonTVLQIController(OneStepTVLQIController):
    def __init__(self, *args, horizon: int = 10, Q_term: Optional[np.ndarray]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = horizon
        # Terminal cost: if not given, reuse Q_aug
        if Q_term is None:
            self.Q_term = np.block([
                [self.Q + 1e-6*np.eye(self.nx),            np.zeros((self.nx,self.nx))],
                [np.zeros((self.nx,self.nx)), self.Qi + 1e-6*np.eye(self.nx)]
            ])
        else:
            self.Q_term = Q_term

    def solve(self, x_current, ref, k_time=None):
        # 1) compute feedforward for step 0
        u_ff0 = self._compute_feedforward(ref[:,0], ref[:,1] if ref.shape[1]>1 else ref[:,0])

        # 2) build nominal rollout and linearizations
        X_nom = np.zeros((self.N+1,self.nx))
        U_nom = np.zeros((self.N,  self.nu))
        A_list = []
        B_list = []
        X_nom[0] = x_current
        U_nom[0] = u_ff0
        for k in range(self.N):
            # roll out nominal
            xk = X_nom[k]; uk = U_nom[k]
            X_nom[k+1] = self.dyn_fun(xk, uk).full().flatten()
            # linearize at (xk,uk)
            Ak = self.A_fun(xk, uk).full()
            Bk = self.B_fun(xk, uk).full()
            # build augmented
            A_augk = np.block([
                [Ak,                     np.zeros((self.nx,self.nx))],
                [-self.dt*np.eye(self.nx), np.eye(self.nx)]
            ])
            B_augk = np.vstack([Bk, np.zeros((self.nx,self.nu))])
            A_list.append(A_augk)
            B_list.append(B_augk)
            # feedforward guess for next u (simple horizon‑0 warm start)
            if k+1 < self.N:
                U_nom[k+1] = uk  

        # 3) build costs
        Q_aug = np.block([
            [self.Q+1e-6*np.eye(self.nx),           np.zeros((self.nx,self.nx))],
            [np.zeros((self.nx,self.nx)), self.Qi+1e-6*np.eye(self.nx)]
        ])
        R_aug = self.R + 1e-6*np.eye(self.nu)
        P = [None]*(self.N+1); K = [None]*self.N
        P[self.N] = self.Q_term
        # 4) backward pass over k=N‑1…0
        for k in reversed(range(self.N)):
            A_k = A_list[k]; B_k = B_list[k]
            S = R_aug + B_k.T @ P[k+1] @ B_k
            invS = np.linalg.inv(S)
            K[k] = invS @ (B_k.T @ P[k+1] @ A_k)
            P[k] = (A_k.T @ P[k+1] @ A_k + Q_aug
                    - A_k.T @ P[k+1] @ B_k @ invS @ B_k.T @ P[k+1] @ A_k)

        # 5) form error‑augmented state
        e0 = x_current - ref[:,0]
        x_aug = np.concatenate([ e0, self.xi ])

        # 6) apply first gain
        K0 = K[0]
        u_unclipped = u_ff0 - K0 @ x_aug
        u = np.clip(u_unclipped, -self.Fmax, self.Fmax)

        # 7) integrator, cost, logging (same as before)
        saturated = np.any(np.abs(u_unclipped) > self.Fmax)
        if not saturated:
            self.xi += e0 * self.dt
        else:
            self.xi *= 0.8
        cost = e0.T@self.Q@e0 + self.xi.T@self.Qi@self.xi + u.T@self.R@u
        logger.info(f"Horizon LQI | N={self.N} | Saturated: {saturated} | Cost: {cost:.2f}")

        # 8) save for fallback
        self.last_Kx = K0[:,:self.nx]
        self.last_Ki = K0[:,self.nx:]
        return u, float(cost)
