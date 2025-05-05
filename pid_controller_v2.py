import numpy as np


class PositionPID3D:
    def __init__(self, Kp, Ki, Kd, dt, limits=None):
        """
        PID in x/y/z (inertial frame).
        
        Kp, Ki, Kd : arrays of length 3 (gains for x, y, z)
        dt         : timestep
        limits     : tuple (min_forces, max_forces), each length-3 or scalar
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.dt = dt

        self.int_e = np.zeros(3)
        self.prev_e = np.zeros(3)

        if limits is not None:
            lo, hi = limits
            self.min_forces = np.array(lo)
            self.max_forces = np.array(hi)
        else:
            self.min_forces = None
            self.max_forces = None

    def compute(self, current_pos, current_yaw, current_vel, 
                        ref_pos, ref_vel):
            # 1) Position error in inertial frame
            e_pos = ref_pos - current_pos
            self.int_e += e_pos * self.dt
            d_e    = (e_pos - self.prev_e) / self.dt
            self.prev_e = e_pos.copy()

            # 2) Compute inertial‐frame velocity command
            vel_cmd_inertial = (
                self.Kp * e_pos
                + self.Ki * self.int_e
                + self.Kd * d_e
                + ref_vel           # feed-forward in inertial
            )

            # 3) Rotate that command into the body frame
            R_inv = self.rotation_matrix(current_yaw).T
            vel_cmd_body = R_inv @ vel_cmd_inertial

            # 4) Velocity error in body frame
            e_vel = vel_cmd_body - current_vel

            # 5) Force command (body frame)
            F = self.Kp * e_vel

            # 6) Enforce force limits, if any
            if self.min_forces is not None:
                F = np.clip(F, self.min_forces, self.max_forces)

            return F
    
    def rotation_matrix(self, psi: float) -> np.ndarray:
        """
        Build a 3×3 yaw‐only rotation matrix.
        R maps body→inertial: [u_body; v_body; w_body] → [x_dot; y_dot; z_dot].
        R.T maps inertial→body.
        """
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [ c, -s, 0],
            [ s,  c, 0],
            [ 0,  0, 1],
        ])


class YawPID:
    def __init__(self, Kp, Ki, Kd, dt, limits=None):
        """
        PID in yaw.
        
        Kp, Ki, Kd : scalars
        dt         : timestep
        limits     : (min_moment, max_moment)
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.int_e = 0.0
        self.prev_e = 0.0

        if limits is not None:
            self.min_m, self.max_m = limits
        else:
            self.min_m = None
            self.max_m = None

    def wrap(self, err):
        """Shortest‐path yaw error into [−π,π)."""
        return ((err + np.pi) % (2*np.pi)) - np.pi

    def compute(self, current_yaw, current_r, ref_yaw, ref_r=0.0):
        # 1) Error
        e = self.wrap(ref_yaw - current_yaw)
        self.int_e += e * self.dt
        d_e = (e - self.prev_e) / self.dt
        self.prev_e = e

        # 2) PID
        M = self.Kp*e + self.Ki*self.int_e + self.Kd*d_e
        M += ref_r  # yaw‐rate feed-forward

        # 3) Limits
        if self.min_m is not None:
            M = float(np.clip(M, self.min_m, self.max_m))
        return M
