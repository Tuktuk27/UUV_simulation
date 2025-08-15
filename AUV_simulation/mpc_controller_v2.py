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
    def __init__(self, system, nx: int, nu: int, N=20, dt=0.1, Fmax = 200):
        # system: any object with attribute 'discrete_dynamics(x,u) -> x_next'
        self.system = system
        self.N = N                 # prediction horizon
        self.dt = dt
        self.nx = nx                # dimension of state
        self.nu = nu                # dimension of input
        self.Fmax = Fmax
        self._prev_U = np.zeros((self.nu, self.N))
        self._prev_X = np.tile(np.zeros(self.nx).reshape(-1,1),
                       (1, self.N+1))

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
        # system state (position) Q = diag([ Q_x, Q_y, Q_z, Q_ψ, Q_u, Q_v, Q_w, Q_r ])
        # Q = ca.diag([100.0,100.0,100.0,500.0] + [10.0]*(self.nx-4))
        Q = ca.diag([ 500, 500, 500, 100,  50, 50, 10, 10 ])

        # control effort
        R = ca.diag([0.001]*self.nu)

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
            x_next = self.system._discrete_dynamics(X[:,k], U[:,k])
            opti.subject_to(X[:,k+1] == x_next)

        # input bounds
        opti.subject_to(opti.bounded(-self.Fmax, U, self.Fmax))

        # solver
        opts = {
            'ipopt.print_level': 0,
            'print_time': False,      # disables IPOPT timing output
            'ipopt.sb': 'yes',        # suppress IPOPT banner (optional)
            'ipopt.max_iter': 800,
            'ipopt.tol': 1e-5,
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
        # shift last_U forward by one, and repeat the last column
        U_init = np.hstack([self._prev_U[:,1:], self._prev_U[:,-1:]])
        X_init = np.hstack([self._prev_X[:,1:],   self._prev_X[:,-1:]])
        self.opti.set_initial(self.U, U_init)
        self.opti.set_initial(self.X, X_init)

        # solve
        # sol = self.opti.solve()
        try:
            sol = self.opti.solve()
        except RuntimeError as e:
            status = self.opti.return_status()
            print("MPC return status:", status)
            # extract the last trial values:
            X_cur = self.opti.debug.value(self.X)
            U_cur = self.opti.debug.value(self.U)
            print("Last X[ :,0 ]:", X_cur[:,0])
            print("Last U[ :,0 ]:", U_cur[:,0])
            raise
        # print("MPC status:", sol.stats()['return_status'])
        # print("Final cost:", sol.value(self.cost))

        # 3) Extract *full* solution
        U_opt = sol.value(self.U)    # shape (nu, N)
        X_opt = sol.value(self.X)    # shape (nx, N+1)

        # 4) Store for *next* time
        self._prev_U = U_opt
        self._prev_X = X_opt

        # 5) Return only the first control
        u0 = U_opt[:, 0]
        cost_final = float(sol.value(self.cost))
        return u0, cost_final
