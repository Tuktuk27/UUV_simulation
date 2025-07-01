import numpy as np
from pid_controller_v2 import PositionPID3D, YawPID
from mpc_controller_v2 import MPCController
from TVLQI_v2 import OneStepTVLQIController
from TVLQR_v2 import OneStepTVLQRController
from HorizonTVLQI import HorizonTVLQIController
from utils import plot_comparison, plot_velocity_comparison, plot_comparison_dynamic_3d, helix_pattern, lawnmower_pattern
import minsnap_trajectories as ms
from Submarine_v2_class import UnderwaterVehicle_v2
from Environment_class import Environment
import time


def simulate_uncontrolled(initial_state, env, t, dt=0.1, steps=100, noise = False):
    """Simulate UUV with initial velocity and no control."""
    if noise:
        uuv = UnderwaterVehicle_v2(initial_state=initial_state, dt=dt)
    else:
        uuv = UnderwaterVehicle_v2(
                initial_state=initial_state,
                dt=dt,
                pos_noise_std=0.0,
                vel_noise_std=0.0
            )
    # initial_state: (eta0, nu0)
    history = []
    # record initial
    eta, nu = uuv.get_state()
    history.append(np.hstack((eta, nu)))
    for k in range(steps):
        # zero control: tau = [0,0,0,0]
        tau_disturbance = env.get_disturbance(eta, nu, t[k])
        uuv.step(τ_control=np.zeros(4), τ_disturbance= tau_disturbance)
        eta, nu = uuv.get_state()
        history.append(np.hstack((eta, nu)))
    return np.array(history)


def generate_snap_min(t, pattern, step_size, speed):
    """
    Given:
      - t        : (M,) array of times at which you want to sample the trajectory
      - pattern  : (N×3) array of [x, y, z] waypoints (with N ≪ M)
      - step_size: spacing between each waypoint time = pattern[k] occurs at time k·step_size

    Returns:
      - position : (M×3) array of [x, y, z] along the _smoothed_ trajectory
      - velocity : (M×3) array of [v_x, v_y, v_z]
      - psi      : (M,)   array of yaw angles = atan2(v_y, v_x)
      - psi_vel  : (M,)   array of yaw rates
      - ref_full : (M×8)  array of [ x, y, z, psi, v_x, v_y, v_z, psi_dot ]
    """
    # 1) Build waypoints at times k·step_size
    refs = []
    current_time = 0.0
    for k in range(pattern.shape[0]):
        x_k, y_k, z_k = pattern[k]
        refs.append(
            ms.Waypoint(time=current_time,
                        position=np.array([x_k, y_k, z_k]))
        )
        current_time += np.floor(step_size/speed).astype(int)

    print(f"{refs = }")

    # 2) Fit a degree-8 minimum-snap polynomial through those N waypoints
    polys = ms.generate_trajectory(
        refs,
        degree=8,
        idx_minimized_orders=(3, 4),
        num_continuous_orders=3,
        algorithm="closed-form",
    )

    # 3) Instead of using polys.time_reference, use the caller’s dense “t” to sample
    #    (t should go from 0 to final_time = (N−1)·step_size, in many small steps).
    t_grid = t.copy()

    # 4) Evaluate up to third derivatives at each of those M time points
    #    pva[0] = (M×3) positions, pva[1] = (M×3) velocities, pva[2] = (M×3) accelerations
    pva = ms.compute_trajectory_derivatives(polys, t_grid, order=3)
    position = pva[0]   # shape (M, 3)
    velocity = pva[1]   # shape (M, 3)
    accel    = pva[2]   # shape (M, 3)

    # 5) Compute yaw ψ = atan2(vy, vx) and yaw-rate ψ̇ = (v_x·a_y – v_y·a_x)/(v_x²+v_y²)
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    ax = accel[:, 0]
    ay = accel[:, 1]

    psi = np.unwrap(np.arctan2(vy, vx))            # shape (M,)
    psi[0] = psi[1]
    
    denom = vx**2 + vy**2
    eps = 1e-8
    psi_vel = (vx * ay - vy * ax) / (denom + eps)  # shape (M,)

    # 6) Stack into (M×8): [x, y, z, ψ, v_x, v_y, v_z, ψ̇]
    psi_column     = psi.reshape(-1, 1)        # (M×1)
    psi_vel_column = psi_vel.reshape(-1, 1)    # (M×1)

    ref_full = np.hstack([
        position,          # (M×3)  → columns 0,1,2
        psi_column,        # (M×1)  → column 3
        velocity,          # (M×3)  → columns 4,5,6
        psi_vel_column     # (M×1)  → column 7
    ])

    return position, velocity, psi_column, psi_vel_column, ref_full




def pattern(steps, t, dt, sim_length, step_size, pat_num=0, speed =1, **kwargs):
    """
    pat_num=0 → helix_pattern
    pat_num=1 → lawnmower_pattern
    Any extra pattern parameters go in **kwargs (e.g. helix_sharpness, width).
    """
    if pat_num == 0:
        pattern =  helix_pattern(steps = steps, sim_length = sim_length, step_size=step_size, speed = speed, **kwargs)
    elif pat_num == 1:
        pattern = lawnmower_pattern(steps= steps, sim_length = sim_length, step_size=step_size, speed = speed, **kwargs)
    else:
        raise ValueError("Unknown pat_num: use 0 for helix, 1 for lawnmower")
    
    return generate_snap_min(t= t, pattern=pattern, step_size=step_size, speed=speed )
    
import numpy as np

def compute_tracking_cost(states, ref_full, dt=0.1, Q=None, R=None):
    """
    Returns:
      J_total : scalar total cost
      loss_t  : array of per-step costs, shape (T,)
    """
    T = min(len(states), len(ref_full))
    Q = np.eye(8) if Q is None else Q
    R = np.zeros((4,4)) if R is None else R

    loss_t = np.zeros(T)
    for k in range(T):
        xk   = states[k]     # actual state
        xref = ref_full[k]   # reference state

        # control effort (u,v,w,r) via finite diff
        if k == 0:
            uk = np.zeros(R.shape[0])
        else:
            uk = (states[k,4:] - states[k-1,4:]) / dt

        e = xk - xref
        loss_t[k] = e.T @ Q @ e + uk.T @ R @ uk

    J_total = loss_t.sum()
    return J_total, loss_t


def simulate_controller(controller_type, init_state, ref_full_8, t, dt, steps, 
                        noise, env, prediction_horizon=None, pos_pid=None, yaw_pid=None, controller= None):
    """
    Unified simulation function for all controller types
    """
    # Initialize vehicle
    if noise:
        uuv = UnderwaterVehicle_v2(initial_state=init_state, dt=dt)
    else:
        uuv = UnderwaterVehicle_v2(
            initial_state=init_state,
            dt=dt,
            pos_noise_std=0.0,
            vel_noise_std=0.0
        )
    
    states = []
    control_log = []
    
    
    for k in range(steps):
        k_time = k * dt
        # Get current state
        eta, nu = uuv.get_state()
        states.append(np.hstack((eta, nu)))
        
        # Compute control based on controller type
        if controller_type == "PID":
            # Extract current reference
            ref_k = ref_full_8[k]
            px, py, pz = ref_k[0:3]
            ux_ref, vy_ref, wz_ref = ref_k[4:7]
            ψ_ref = ref_k[3]
            r_ψ_ref = ref_k[7]
            
            # Compute control forces
            F = pos_pid.compute(
                current_pos=np.array([eta[0], eta[1], eta[2]]),
                current_yaw=eta[3],
                current_vel=np.array([nu[0], nu[1], nu[2]]),
                ref_pos=np.array([px, py, pz]),
                ref_vel=np.array([ux_ref, vy_ref, wz_ref])
            )
            
            # Compute yaw moment
            Mz = yaw_pid.compute(
                current_yaw=eta[3],
                current_r=nu[3],
                ref_yaw=ψ_ref,
                ref_r=r_ψ_ref
            )
            tau_control = np.array([F[0], F[1], F[2], Mz])

        elif controller_type in {"TVLQI", "TVLQR"}:
            # Build reference window
            state_8 = np.hstack((eta, nu))
            ref_k = ref_full_8[k:k+2].T
            # Call the controller's solve method
            u_opt, _ = controller.solve(state_8, ref_k, k_time)
            tau_control = u_opt[0:4]
            
        elif controller_type in {"MPC", "TVLQI_horizon"}:
            # Build reference window
            end_index = min(k + prediction_horizon + 1, ref_full_8.shape[0])
            window = ref_full_8[k:end_index].T
            state_8 = np.hstack((eta, nu))
            
            # Call the controller's solve method
            u_opt, _ = controller.solve(state_8, window)
            tau_control = u_opt[0:4]
            
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        
        # Get environmental disturbance
        tau_disturbance = env.get_disturbance(eta, nu, t[k])
        
        # Apply control
        uuv.step(tau_control, tau_disturbance)
        control_log.append(tau_control)
    
    return np.array(states), np.array(control_log)

def test_comparison_dynamic(
    dt: float = 0.1,
    sim_length: float = 10,
    step_size: float = 1,
    pat_num: int = 1,
    noise: bool = False,
    speed: float = 1,
    prediction_horizon: int = 20,
):
    # Setup (unchanged)
    steps = int(np.floor(sim_length / dt))
    t = np.linspace(0, sim_length, steps)
    ref_pos, ref_vel, psi_ref, r_ref, ref_full_8 = pattern(
        steps=steps, t=t, dt=dt, sim_length=sim_length, 
        step_size=step_size, pat_num=pat_num, speed=speed
    )
    
    # Initial conditions
    eta0 = [float(ref_pos[0, i]) for i in range(3)] + [float(psi_ref[0, 0])]
    nu0 = [float(ref_vel[0, i]) for i in range(3)] + [float(r_ref[0, 0])]
    init = (eta0, nu0)
    env = Environment()
    
    # Controller setup
    max_effort = 500.0
    results = {}
    controllers = []  # Initialize controllers list

    # Add PID controller
    controllers.append(
        {
            'name': 'PID',
            'type': 'PID',
            # Pass both PID controllers directly
            'controller': {  
                'pos_pid': PositionPID3D(
                    Kp=[20, 20, 15], Ki=[0.1, 0.1, 0.05], Kd=[10, 10, 5],
                    Kff=0.5, dt=dt, limits=([-max_effort]*3, [max_effort]*3)
                ),
                'yaw_pid': YawPID(
                    Kp=5.0, Ki=0.1, Kd=1.0, Kff=0.8, dt=dt, 
                    limits=(-max_effort, max_effort)
                )
            }
        })
    
    # Add MPC controller
    controllers.append(
        {
            'name': 'MPC',
            'type': 'MPC',
            # Pass the controller instance and prediction horizon
            'controller': {
                'controller': MPCController(
                    system=UnderwaterVehicle_v2(initial_state=init, dt=dt),
                    nx=8, nu=4, N=prediction_horizon, 
                    dt=dt, Fmax=max_effort
                ),
                'prediction_horizon': prediction_horizon
            }
        })
    
    # Add TVLQI controller    
    controllers.append({
        'name': 'TVLQI',
        'type': 'TVLQI',
        # Pass the controller instance and prediction horizon
        'controller': {
            'controller': OneStepTVLQIController(
                system=UnderwaterVehicle_v2(initial_state=init, dt=dt),
                nx=8,
                nu=4,
                dt=dt,
                Fmax=max_effort,
                Q=np.diag([500, 500, 500, 100, 50, 50, 10, 10]),
                R=np.diag([0.001, 0.001, 0.001, 0.001]),
                Qi=np.eye(8) * 0.5,
            ),
            'prediction_horizon': prediction_horizon
        }
    })
    
    controllers.append({
            'name': 'TVLQR',
            'type': 'TVLQR',
            # Pass the controller instance and prediction horizon
            'controller': {
                'controller': OneStepTVLQRController(
                    system=UnderwaterVehicle_v2(initial_state=init, dt=dt),
                    nx=8,
                    nu=4,
                    dt=dt,
                    Fmax=max_effort,
                    Q=np.diag([500, 500, 500, 100, 50, 50, 10, 10]),
                    R=np.diag([0.001, 0.001, 0.001, 0.001]),
                ),
                'prediction_horizon': prediction_horizon
            }
        })
    
    # controllers.append({
    #     'name': 'TVLQI_horizon',
    #     'type': 'TVLQI_horizon',
    #     # Pass the controller instance and prediction horizon
    #     'controller': {
    #         'controller': HorizonTVLQIController(
    #             system=UnderwaterVehicle_v2(initial_state=init, dt=dt),
    #             nx=8,
    #             nu=4,
    #             N=prediction_horizon,
    #             dt=dt,
    #             Fmax=max_effort,
    #             Q=np.diag([500, 500, 500, 100, 50, 50, 10, 10]),
    #             R=np.diag([0.1, 0.1, 0.1, 0.01]),
    #             Qi=np.eye(8) * 0.5,
    #         ),
    #         'prediction_horizon': prediction_horizon
    #     }
    # })
    
    # Simulate all controllers in a loop
    for ctrl in controllers:
        print(f"Starting simulation for {ctrl['name']} controller...")
        start_time = time.time()
        
        sim_result = simulate_controller(
            controller_type=ctrl['type'],
            init_state=init,
            ref_full_8=ref_full_8,
            t=t,
            dt=dt,
            steps=steps,
            noise=noise,
            env=env,
            **ctrl['controller']
        )
        
        elapsed = time.time() - start_time
        print(f"  {ctrl['name']} simulation completed in {elapsed:.2f} seconds")
        print(f"  Average per-step time: {elapsed/steps*1000:.2f} ms")
        results[ctrl['name']] = sim_result
    
    # Add uncontrolled simulation
    results['Unctrl'] = (
        simulate_uncontrolled(init, t=t, dt=dt, steps=steps, noise=noise, env=env), 
        np.zeros((steps, 4))
    )
    
    # Process results
    T_ref = ref_full_8.shape[0]
    costs = {}
    for name, (states, _) in results.items():
        states = states[:T_ref]
        cost, loss = compute_tracking_cost(states, ref_full_8[:T_ref], dt=dt)
        costs[name] = cost
        print(f"{name} tracking cost: {cost}")
    
    # Plot results
    plot_comparison_dynamic_3d(
        ref_pos, ref_vel, psi_ref, r_ref,
        [results[name][0][:T_ref] for name in results],
        list(results.keys()),
        [compute_tracking_cost(results[name][0][:T_ref], ref_full_8[:T_ref], dt=dt)[1] 
         for name in results],
        dt=dt,
        controller_efforts=[results[name][1][:T_ref] for name in results if name != 'Unctrl']
    )


if __name__ == "__main__":
    # Example call:
    test_comparison_dynamic(
        dt=0.05,
        sim_length=50,
        step_size=2,
        pat_num=1,
        speed=0.5,
        noise=False,
        prediction_horizon=40,
    )