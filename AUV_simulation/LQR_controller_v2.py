import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are
from typing import Callable, Optional, Tuple, Union

class LQRController:
    """
    Simple LQR controller for tracking using a constant gain K.
    Interface matches MPCController.solve:
        u0, cost = solve(current_state, reference_window)
    """
    def __init__(self, system, nx: int, nu: int, N=20, dt=0.1,
                 Q: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None):
        # store dims
        self.system = system
        self.nx = nx
        self.nu = nu
        self.N = N
        self.dt = dt
        # default cost matrices
        self.Q = Q if Q is not None else np.diag([500,100,100,500] + [10]*(nx-4))
        self.R = R if R is not None else np.diag([0.01]*nu)
        # linearize around equilibrium x=0, u=0 once
        x0 = np.zeros(nx)
        u0 = np.zeros(nu)
        # use CasADi to get continuous A, B
        x_sym = ca.SX.sym('x', nx)
        u_sym = ca.SX.sym('u', nu)
        f_cont = system._discrete_dynamics
        # define f_dot = (x_next - x)/dt
        x_next = f_cont(x_sym, u_sym)
        f_dot = (x_next - x_sym) / dt
        A_sym = ca.jacobian(f_dot, x_sym)
        B_sym = ca.jacobian(f_dot, u_sym)
        A = np.array(ca.Function('A', [x_sym,u_sym],[A_sym])(x0, u0)).reshape(nx,nx)
        B = np.array(ca.Function('B', [x_sym,u_sym],[B_sym])(x0, u0)).reshape(nx,nu)
        # discrete linear model
        self.Ad = np.eye(nx) + A*dt
        self.Bd = B*dt
        # solve discrete ARE
        P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        # compute constant gain
        self.K = np.linalg.inv(self.Bd.T @ P @ self.Bd + self.R) @ (self.Bd.T @ P @ self.Ad)

    def solve(self, current_state: np.ndarray, reference: np.ndarray):
        """
        current_state: (nx,)
        reference:   (nx, N+1) reference trajectory over horizon
        Returns tuple (u0, cost)
            u0: (nu,) first control input
            cost: scalar instantaneous cost
        """
        # tracking error at first step
        x_ref = reference[:,0]
        dx = current_state - x_ref
        # LQR control law (no feedforward)
        u0 = -self.K @ dx
        # instantaneous cost
        cost = float(dx.T @ self.Q @ dx + u0.T @ self.R @ u0)
        return u0, cost
