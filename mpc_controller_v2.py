import casadi as ca
import numpy as np

class MPCController:
    """
    Generic MPC controller for a discrete-time system with state x and input u.
    The provided 'system' must implement a CasADi-compatible discrete dynamics function:
      x_next = system.discrete_dynamics(x, u)

    - horizon N
    - timestep dt
    - state dimension nx, input dimension nu
    """
    def __init__(self, system, nx: int, nu: int, N=20, dt=0.1):
        # system: any object with attribute 'discrete_dynamics(x,u) -> x_next'
        self.system = system
        self.N = N                 # prediction horizon
        self.dt = dt
        self.nx = nx                # dimension of state
        self.nu = nu                # dimension of input
        self._build_optimizer()

    def _build_optimizer(self):
        opti = ca.Opti()

        # decision vars
        X = opti.variable(self.nx, self.N+1)
        U = opti.variable(self.nu, self.N)

        # parameters
        x0  = opti.parameter(self.nx)
        ref = opti.parameter(self.nx, self.N+1)

        # initial state constraint
        opti.subject_to(X[:,0] == x0)

        # cost
        Q = ca.diag([500.0,500.0,500.0,100.0] + [10.0]*(self.nx-4))
        R = ca.diag([0.01]*self.nu)

        cost = 0
        for k in range(self.N):
            e = X[:,k] - ref[:,k]
            cost += e.T @ Q @ e + U[:,k].T @ R @ U[:,k]
        # terminal cost
        eN = X[:,self.N] - ref[:,self.N]
        cost += eN.T @ Q @ eN

        opti.minimize(cost)
        self.cost = cost

        # dynamics
        for k in range(self.N):
            x_next = self.system.discrete_dynamics(X[:,k], U[:,k])
            opti.subject_to(X[:,k+1] == x_next)

        # input bounds
        Fmax = 200.0
        opti.subject_to(opti.bounded(-Fmax, U, Fmax))

        # solver
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 800,
            'ipopt.tol': 1e-6,
            'ipopt.linear_solver': 'mumps',
        }
        opti.solver('ipopt', opts)

        # store
        self.opti = opti
        self.X, self.U = X, U
        self.x0, self.ref = x0, ref

    def solve(self, current_state: np.ndarray, reference: np.ndarray):
        # pad ref to N+1
        M = reference.shape[1]
        if M < self.N+1:
            last = reference[:, -1:]
            pad  = np.tile(last, (1, self.N+1 - M))
            reference = np.hstack([reference, pad])

        # set parameters
        self.opti.set_value(self.x0, current_state)
        self.opti.set_value(self.ref, reference)

        # warm start
        self.opti.set_initial(self.X, 
            np.tile(current_state.reshape(-1,1), (1, self.N+1)))
        self.opti.set_initial(self.U, np.zeros((self.nu, self.N)))

        # solve
        sol = self.opti.solve()
        print("MPC status:", sol.stats()['return_status'])
        print("Final cost:", sol.value(self.cost))

        u_opt = sol.value(self.U)[:,0]
        return u_opt
