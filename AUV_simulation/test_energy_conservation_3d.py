# test_energy_conservation_3d.py
import numpy as np
from simulator_3d import UnderwaterPointMass3D

def test_energy_conservation():
    # Case 1: No damping → energy conserved
    sim = UnderwaterPointMass3D(damping_linear=0.0, damping_quad=0.0)
    sim.state[3:] = [2.0, 1.5, 0.8]  # Initial velocity
    initial_ke = 0.5 * sim.mass * np.sum(sim.state[3:]**2)
    
    for _ in range(10):
        sim.step([0.0, 0.0, 0.0])
        current_ke = 0.5 * sim.mass * np.sum(sim.state[3:]**2)
        np.testing.assert_almost_equal(current_ke, initial_ke, decimal=6,
                                       err_msg="Energy not conserved when damping=0!")

    # Case 2: With damping → energy decreases
    sim = UnderwaterPointMass3D(damping_linear=0.5, damping_quad=0.1)
    sim.state[3:] = [2.0, 1.5, 0.8]
    initial_ke = 0.5 * sim.mass * np.sum(sim.state[3:]**2)
    
    for _ in range(10):
        sim.step([0.0, 0.0, 0.0])
        current_ke = 0.5 * sim.mass * np.sum(sim.state[3:]**2)
        assert current_ke < initial_ke, "Energy should decrease with damping!"
    
if __name__ == "__main__":
    test_energy_conservation()
    print("All energy tests passed!")