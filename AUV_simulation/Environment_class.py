import numpy as np
from typing import Callable, Optional, Tuple

class Environment:
    """
    A modular “environment/disturbance” generator for UnderwaterVehicle_v2.

    It can produce:
      1. A steady or time-varying current (in the inertial frame), converted to body-frame drag.
      2. A depth-dependent wave force (vertical).
      3. Optional random disturbance jitter.
      4. A yaw moment induced by relative velocity in surge/sway.

    USAGE:
      env = Environment(
          current_profile=lambda t: np.array([0.2, 0.0, 0.0]),
          drag_coeffs=(10.0, 10.0, 5.0),
          current_moment_coeff=2.0,
          wave_amplitude=0.1,
          wave_frequency=0.5,
          wave_decay_rate=0.05,
          noise_intensity=0.01,
          seed=123
      )
      ...
      τ_dist = env.get_disturbance(η, ν, t)
      sub.step(τ_control, τ_dist)
    """

    def __init__(
        self,
        current_profile: Optional[Callable[[float], np.ndarray]] = None,
        drag_coeffs: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        current_moment_coeff: float = 0.0,
        wave_amplitude: float = 0.0,
        wave_frequency: float = 1.0,
        wave_decay_rate: float = 0.05,
        noise_intensity: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Parameters:
        -----------
        current_profile : Callable[[t], np.ndarray]
            Function returning inertial current velocity [v_x, v_y, v_z] at time t.
            If None, defaults to zero current.
        drag_coeffs : (C_x, C_y, C_z)
            Quadratic drag coefficients for body-frame relative velocity [u, v, w].
            Force_i = C_i * v_rel_i * |v_rel_i|.
        current_moment_coeff : float
            Coefficient mapping lateral relative flow in surge/sway to a yaw moment:
            Mz_current = coeff * v_rel_x * v_rel_y.
        wave_amplitude : float
            Amplitude of the sinusoidal wave force (N).
        wave_frequency : float
            Angular frequency (rad/s) of the sinusoidal wave disturbance.
        wave_decay_rate : float
            Exponential decay rate (1/m) of wave amplitude with depth.
        noise_intensity : float
            Standard deviation (N or N·m) of additive Gaussian noise on all disturbance channels.
        seed : Optional[int]
            Seed for the internal random number generator (for reproducibility).
        """
        # Random number generator for stochastic disturbances
        self.rng = np.random.default_rng(seed)

        # Current profile (inertial)
        if current_profile is None:
            self.current_profile = self.zero_current_profile
        else:
            self.current_profile = current_profile

        # Quadratic drag coefficients (body frame) for [surge, sway, heave]
        self.drag_coeffs = tuple(float(c) for c in drag_coeffs)
        self.current_moment_coeff = float(current_moment_coeff)

        # Wave parameters
        self.wave_amplitude = float(wave_amplitude)
        self.wave_frequency = float(wave_frequency)
        self.wave_decay_rate = float(wave_decay_rate)

        # Noise intensity for all four disturbance channels [Fx, Fy, Fz, Mz]
        self.noise_intensity = float(noise_intensity)

    def get_disturbance(
        self,
        η: np.ndarray,
        ν: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Compute a 4-vector of disturbance forces/torques [F_x, F_y, F_z, M_z] at time t.

        Parameters:
        -----------
        η : np.ndarray, shape=(4,)
            Inertial pose [x, y, z, ψ].  η[2] = depth (positive down).
        ν : np.ndarray, shape=(4,)
            Body-fixed velocities [u, v, w, r].
        t : float
            Current simulation time (s).

        Returns:
        --------
        τ_disturbance : np.ndarray, shape=(4,)
            Body-frame disturbance forces/torques:
            [F_x_drag + F_x_current, F_y_drag + F_y_current,
             F_z_wave + F_z_drag, M_z_current] + noise.
        """
        # Unpack state
        z = float(η[2])    # depth (NED: positive down)
        ψ = float(η[3])    # yaw
        u, v, w, _ = ν     # body-frame velocities

        # ----- 1) Wave-induced vertical force -----
        # Fz_wave = A * sin(ω t) * exp(-k * depth)
        Fz_wave = self.wave_amplitude * np.sin(self.wave_frequency * t) * np.exp(-self.wave_decay_rate * z)

        # ----- 2) Current-induced relative velocity in body frame -----
        v_current_inertial = self.current_profile(t).reshape(3,)
        c, s = np.cos(ψ), np.sin(ψ)
        R_inertial_to_body = np.array([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1]
        ])
        v_current_body = R_inertial_to_body @ v_current_inertial

        # Relative velocity (body frame) between water and vehicle:
        v_rel = v_current_body - np.array([u, v, w])

        # ----- 3) Quadratic drag in body axes -----
        Cx, Cy, Cz = self.drag_coeffs
        Fx_drag = Cx * v_rel[0] * abs(v_rel[0])
        Fy_drag = Cy * v_rel[1] * abs(v_rel[1])
        Fz_drag = Cz * v_rel[2] * abs(v_rel[2])

        # ----- 4) Yaw moment from lateral relative flow -----
        # Mz_current = coeff * v_rel_x * v_rel_y
        Mz_current = self.current_moment_coeff * v_rel[0] * v_rel[1]

        # ----- 5) Aggregate forces (no constant “Fx_current” term—drag is computed above) -----
        Fx_total = Fx_drag
        Fy_total = Fy_drag
        Fz_total = Fz_wave + Fz_drag
        Mz_total = Mz_current

        τ_disturbance = np.array([Fx_total, Fy_total, Fz_total, Mz_total], dtype=float)

        # ----- 6) Add random disturbance noise if specified -----
        if self.noise_intensity > 0.0:
            noise_vec = self.rng.normal(scale=self.noise_intensity, size=4)
            τ_disturbance += noise_vec

        return τ_disturbance
    
    def zero_current_profile(self, t):
        return np.zeros(3)

    def reset(self) -> None:
        """
        Reset any internal state. Currently, RNG is stateless beyond its initial seed,
        so nothing to reset here. If you add timers or internal wave phases later,
        do so in this method.
        """
        pass
