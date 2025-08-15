# test_velocity_decay.py
import numpy as np
from simulator_3d import UnderwaterPointMass3D
from utile import plot_velocity_decay

def simulate_velocity_decay():
    # Initialize with damping
    sim = UnderwaterPointMass3D(damping_linear=0.5, damping_quad=0.1)
    sim.state[3:] = [3.0, 2.0, 1.0]  # Initial velocity
    
    dt = 0.1
    time_steps = []
    velocities = []
    
    for t in np.arange(0, 5.0, dt):
        sim.step([0.0, 0.0, 0.0])  # No control input
        velocities.append(sim.state[3:].copy())
        time_steps.append(t)
    
    return np.array(time_steps), np.array(velocities)

if __name__ == "__main__":
    time, velocities = simulate_velocity_decay()
    plot_velocity_decay(time, velocities)