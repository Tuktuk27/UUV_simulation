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
import os
from rand_path import generate_minimum_snap_trajectory, generate_random_waypoints, generate_trajectory_dataset, generate_small_auv_configs
import pickle
import h5py
from tqdm import tqdm, trange


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




def pattern(steps, t, dt, sim_length, step_size, pat_num=0, speed=1, **kwargs):
    """
    Unified trajectory pattern generator
    pat_num: 0=helix, 1=lawnmower, 2=random
    """
    if pat_num == 0:
        return helix_pattern(steps, sim_length, step_size, t, dt, **kwargs)
    elif pat_num == 1:
        pattern = lawnmower_pattern(steps, sim_length, step_size, speed, **kwargs)
        return generate_snap_min(t, pattern, step_size, speed)
    elif pat_num == 2:
        # Random trajectory parameters
        avg_segment_time = 4.0  # Average time between waypoints (seconds)
        fixed_waypoint = int(sim_length / avg_segment_time) + 1

        num_waypoints = kwargs.get('num_waypoints', np.random.randint(fixed_waypoint*0.8, fixed_waypoint*1.2))

        # Calculate adaptive number of waypoints
        
        print(f"{num_waypoints = }")
        
        bounds = kwargs.get('bounds', (-50, 50, -50, 50, -30, -2))
        
        # Generate waypoints
        waypoints = generate_random_waypoints(
            num_waypoints=num_waypoints,
            bounds=bounds
        )
        
        # Generate trajectory
        traj = generate_minimum_snap_trajectory(
            waypoints=waypoints,
            t_total=sim_length,
            dt=dt
        )
        
        return (
            traj['position'],
            traj['velocity'],
            traj['yaw'].reshape(-1, 1),
            traj['yaw_rate'].reshape(-1, 1),
            traj['ref_full']
        )
    else:
        raise ValueError("Unknown pat_num: 0=helix, 1=lawnmower, 2=random")
    
import numpy as np

def compute_tracking_cost(states, control_log, ref_full, dt=0.1, Q=None, R=None):
    """
    Returns:
      J_total : scalar total cost
      loss_t  : array of per-step costs, shape (T,)
    """
    T = min(len(states), len(ref_full), len(control_log))
    Q = np.eye(8) if Q is None else Q
    R = np.zeros((4,4)) if R is None else R

    loss_t = np.zeros(T)
    for k in range(T):
        xk   = states[k]
        xref = ref_full[k]
        uk   = control_log[k]    # <— direct use of your logged control

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
                    # Q=np.diag([1.00000000e+05, 4.48521793e+04, 2.86140177e+04, 4.83188227e+04, 1.00000000e-03, 1.84970854e+03, 4.40753655e+04, 2.95797234e+04]),
                    # R=np.diag([1.00000000e-03, 1.00000000e-03, 7.30923312e+02, 2.46127869e+02]),
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
    #             dt=dt,
    #             Fmax=max_effort,
    #             Q=np.diag([500, 500, 500, 100, 50, 50, 10, 10]),
    #             R=np.diag([0.1, 0.1, 0.1, 0.01]),
    #             Qi=np.eye(8) * 0.5,
    #             horizon = prediction_horizon,
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
    for name, (states, control_logs) in results.items():
        cost, loss = compute_tracking_cost(states[:T_ref], control_logs[:T_ref],ref_full_8[:T_ref], dt=dt)
        costs[name] = cost
        print(f"{name} tracking cost: {cost}")
    
    # Plot results
    plot_comparison_dynamic_3d(
        ref_pos, ref_vel, psi_ref, r_ref,
        [results[name][0][:T_ref] for name in results],
        list(results.keys()),
        [compute_tracking_cost(results[name][0][:T_ref], results[name][1][:T_ref], ref_full_8[:T_ref], dt=dt)[1] 
         for name in results],
        dt=dt,
        controller_efforts=[results[name][1][:T_ref] for name in results if name != 'Unctrl']
    )

def test_comparison_dynamic_v2(
        dt: float = 0.1,
        sim_length: float = 30,  # Increased for more variety
        step_size: float = 2,
        pat_num: int = 2,        # Default to random pattern
        noise: bool = False,
        speed: float = 1,
        prediction_horizon: int = 20,
        Fmax = 140,
        Upmax = 100,
        Tmax = 30,
        **kwargs  # Accept pattern parameters
    ):
    # Setup (unchanged)
    steps = int(np.floor(sim_length / dt))
    t = np.linspace(0, sim_length, steps)
        
    ref_pos, ref_vel, psi_ref, r_ref, ref_full_8 = pattern(
        steps=steps, t=t, dt=dt, sim_length=sim_length, 
        step_size=step_size, pat_num=pat_num, speed=speed, **kwargs
    )
        

    # Initial conditions
    eta0 = [float(ref_pos[0, i]) for i in range(3)] + [float(psi_ref[0, 0])]
    nu0 = [float(ref_vel[0, i]) for i in range(3)] + [float(r_ref[0, 0])]
    init = (eta0, nu0)
    env = Environment()

    # Controller setup
    results = {}
    controllers = []  # Initialize controllers list

    # # Add MPC controller
    # controllers.append(
    #     {
    #         'name': 'MPC',
    #         'type': 'MPC',
    #         # Pass the controller instance and prediction horizon
    #         'controller': {
    #             'controller': MPCController(
    #                 system=UnderwaterVehicle_v2(initial_state=init, dt=dt),
    #                 nx=8, nu=4, N=prediction_horizon, 
    #                 dt=dt, Fmax=max_effort
    #             ),
    #             'prediction_horizon': prediction_horizon
    #         }
    #     })
    

    # Add PID controller
    controllers.append(
        {
            'name': 'PID',
            'type': 'PID',
            # Pass both PID controllers directly
            'controller': {  
                'pos_pid': PositionPID3D(
                    Kp=[20, 20, 15], Ki=[0.1, 0.1, 0.05], Kd=[10, 10, 5],
                    Kff=0.5, dt=dt, limits=([-Fmax]*3, [Fmax]*3)
                ),
                'yaw_pid': YawPID(
                    Kp=5.0, Ki=0.1, Kd=1.0, Kff=0.8, dt=dt, 
                    limits=(-Tmax, Tmax)
                )
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
                    Fmax=Fmax,
                    Upmax=Upmax,
                    Tmax=Tmax,
                    Q=np.diag([500, 500, 500, 100, 50, 50, 10, 10]),
                    R=np.diag([0.001, 0.001, 0.001, 0.001]),
                    # Q=np.diag([1.00000000e+05, 4.48521793e+04, 2.86140177e+04, 4.83188227e+04, 1.00000000e-03, 1.84970854e+03, 4.40753655e+04, 2.95797234e+04]),
                    # R=np.diag([1.00000000e-03, 1.00000000e-03, 7.30923312e+02, 2.46127869e+02]),
                ),
                'prediction_horizon': prediction_horizon
            }
        })


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
    for name, (states, control_logs) in results.items():
        cost, loss = compute_tracking_cost(states[:T_ref], control_logs[:T_ref],ref_full_8[:T_ref], dt=dt)
        costs[name] = cost
        print(f"{name} tracking cost: {cost}")

    # Plot results
    plot_comparison_dynamic_3d(
        ref_pos, ref_vel, psi_ref, r_ref,
        [results[name][0][:T_ref] for name in results],
        list(results.keys()),
        [compute_tracking_cost(results[name][0][:T_ref], results[name][1][:T_ref], ref_full_8[:T_ref], dt=dt)[1] 
            for name in results],
        dt=dt,
        controller_efforts=[results[name][1][:T_ref] for name in results if name != 'Unctrl']
    )




def generate_full_dataset(
    controller_type = "TVLQR",
    subs_file="small_auv_configs.pkl",
    dataset_dir="trajectory_data",
    output_file="dataset.h5",
    dt=0.05,
    noise=False,
    window_size=50,
    include_padded=True,
    buffer_size=4096
):
    # Load submarine configurations
    with open(subs_file, 'rb') as f:
        subs = pickle.load(f)

    traj_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npz')])

    # Pre-calculate total samples depending on include_padded
    total_samples = 0
    for filename in traj_files:
        data = np.load(os.path.join(dataset_dir, filename))
        T = len(data['ref_full'])
        if include_padded:
            total_samples += max(0, T) * len(subs)     # one sample per timestep per sub
        else:
            total_samples += max(0, T - window_size + 1) * len(subs)  # only full windows

    print(f"number of total samples : {total_samples = }")
    # Create HDF5 and datasets
    with h5py.File(output_file, 'w') as hf:
        hf_ref     = hf.create_dataset('ref_windows', (total_samples, window_size, 8), dtype='f4', compression='gzip')
        hf_case     = hf.create_dataset('case_number', (total_samples,), dtype='i4', compression='gzip')
        hf_control = hf.create_dataset('controls', (total_samples, 4), dtype='f4', compression='gzip')
        # store valid length (how many real future points in the window) -> useful for masking
        hf_valid_len = hf.create_dataset('valid_length', (total_samples,), dtype='i2', compression='gzip')
        hf_sub_params = hf.create_dataset('sub_params', (total_samples, 23), dtype= 'f4', compression = 'gzip')

        n_subs = len(subs)
        # create per-sub datasets (note correct shapes)
        hf.create_dataset('per_sub_M', (n_subs, 4), dtype='f4', compression='gzip')        # diagonal only
        hf.create_dataset('per_sub_D_lin', (n_subs, 4), dtype='f4', compression='gzip')
        hf.create_dataset('per_sub_D_quad', (n_subs, 4), dtype='f4', compression='gzip')
        hf.create_dataset('per_sub_static_scalars', (n_subs, 11), dtype='f4', compression='gzip')

        hf_sub_id = hf.create_dataset('sub_id', (total_samples,), dtype='f4', compression='gzip')

        # metadata
        hf.attrs['window_size'] = window_size
        hf.attrs['dt'] = dt
        hf.attrs['submarine_params'] = list(next(iter(subs.values())).keys())

        # Buffers in RAM
        B = buffer_size
        buf_ref    = np.zeros((B, window_size, 8), dtype=np.float32)
        buf_ctrl   = np.zeros((B, 4), dtype=np.float32)
        buf_valid  = np.zeros((B,), dtype=np.int16)
        buf_pos = 0
        written = 0  # how many samples already written to hf
        case_number = 0
        sub_idx = 0

        for auv_name, config in tqdm(subs.items(), desc="Number of AUV iterated through"):
            print(f"Starting simulation for {auv_name} ...")
            start_time = time.time()
            
            scalar_features = np.array([
                config['L'], config['D'],config['volume_coeff'],config['m'],config['cd'],config['cf'],
                config['V'], config['Fmax'], config['Upmax'], config['Tmax'], config['payload_mass'],
            ], dtype='f4')

            # inside loop, build static_vector once per submarine:
            M_flat = np.diag(config['M']).astype(np.float32)
            Dlin = np.asarray(config['D_lin'], dtype='f4')   # 4
            Dquad = np.asarray(config['D_quad'], dtype='f4') # 4

            static_vector = np.concatenate([M_flat, Dlin, Dquad, scalar_features])  # length 23

            # hf['sub_M'][sub_idx] = config['M'].ravel().astype('f4')       # if choose to keep all inertia matrix length 16
            hf['per_sub_M'][sub_idx] = M_flat
            hf['per_sub_D_lin'][sub_idx] = Dlin
            hf['per_sub_D_quad'][sub_idx] = Dquad
            hf['per_sub_static_scalars'][sub_idx] = scalar_features


            for filename in traj_files:
                data = np.load(os.path.join(dataset_dir, filename))
                ref_full = data['ref_full']   # shape (T, 8)
                T = len(ref_full)
                if T == 0:
                    continue

                # init vehicle at first reference
                uuv = UnderwaterVehicle_v2(
                    M=config['M'],
                    D_lin=config['D_lin'],
                    D_quad=config['D_quad'],
                    volume=config['V'],
                    max_thrust = config['Fmax'],
                    max_upforce = config['Upmax'],
                    max_torque = config['Tmax'],
                    initial_state=(ref_full[0, :4], ref_full[0, 4:]),
                    dt=dt,
                    pos_noise_std=0.1 if noise else 0.0,
                    vel_noise_std=0.05 if noise else 0.0
                )

                uuv_controller = UnderwaterVehicle_v2(
                    M=config['M'],
                    D_lin=config['D_lin'],
                    D_quad=config['D_quad'],
                    volume=config['V'],
                    max_thrust = config['Fmax'],
                    max_upforce = config['Upmax'],
                    max_torque = config['Tmax'],
                    initial_state=(ref_full[0, :4], ref_full[0, 4:]),
                    dt=dt,
                    pos_noise_std=0.1 if noise else 0.0,
                    vel_noise_std=0.05 if noise else 0.0
                )

                if controller_type == "TVLQR":
                    controller = OneStepTVLQRController(
                        system=uuv_controller,
                        nx=8,
                        nu=4,
                        dt=dt,
                        Fmax = config['Fmax'],
                        Upmax = config['Upmax'],
                        Tmax = config['Tmax'],
                        Q=np.diag([500, 500, 500, 100, 50, 50, 10, 10]),
                        R=np.diag([0.001, 0.001, 0.001, 0.001]),
                        # Q=np.diag([1.00000000e+05, 4.48521793e+04, 2.86140177e+04, 4.83188227e+04, 1.00000000e-03, 1.84970854e+03, 4.40753655e+04, 2.95797234e+04]),
                        # R=np.diag([1.00000000e-03, 1.00000000e-03, 7.30923312e+02, 2.46127869e+02]),
                    )

                for k in range(T):
                    # current state
                    eta, nu = uuv.get_state()

                    # construct window and compute valid_length
                    if k + window_size <= T:
                        ref_window = ref_full[k:k+window_size]
                        valid_len = window_size
                    else:
                        # pad with last point
                        n_real = max(0, T - k)
                        pad_count = window_size - n_real
                        ref_window = np.vstack([ref_full[k:], np.tile(ref_full[-1], (pad_count, 1))])
                        valid_len = n_real  # how many are real (could be 0 near empty traj)

                    # compute control
                    u_opt, _ = controller.solve(np.hstack((eta, nu)), ref_window.T, k * dt)
                    tau_control = u_opt[:4]

                    # step sim
                    uuv.step(tau_control, np.zeros(4))

                    # decide whether to save this sample
                    save_sample = include_padded or (valid_len == window_size)
                    if save_sample:
                        
                        # write into buffers
                        buf_ref[buf_pos]    = ref_window.astype(np.float32)
                        buf_ctrl[buf_pos]   = tau_control.astype(np.float32)
                        buf_valid[buf_pos]  = int(valid_len)

                        buf_pos += 1

                        # flush buffer if full
                        if buf_pos == B:
                            start = written
                            end = start + buf_pos
                            print(f"buf number: {buf_pos = }")
                            print(f"number of iterations equal {k = }")
                            print(f"{buf_ref.shape = }")
                            print(f"{hf_ref[start:end].shape = }")
                            print(f"{case_number = }")
                            hf_ref[start:end] = buf_ref[:buf_pos]
                            hf_control[start:end] = buf_ctrl[:buf_pos]
                            hf_valid_len[start:end] = buf_valid[:buf_pos]
                            hf_case[start:end] = case_number
                            hf_sub_params[start:end] = static_vector
                            hf_sub_id[start:end] = sub_idx
                            
                            written = end
                            buf_pos = 0

                # flush remaining
                if buf_pos > 0:
                    start = written
                    end = start + buf_pos
                    print(f"buf number: {buf_pos = }")
                    print(f"number of iterations equal {k = }")
                    print(f"{buf_ref.shape = }")
                    print(f"{hf_ref[start:end].shape = }")
                    print(f"{case_number = }")
                    hf_ref[start:end]    = buf_ref[:buf_pos]
                    hf_control[start:end] = buf_ctrl[:buf_pos]
                    hf_valid_len[start:end] = buf_valid[:buf_pos]
                    hf_case[start:end] = case_number
                    hf_sub_params[start:end] = static_vector
                    hf_sub_id[start:end] = sub_idx
                    written = end
                    buf_pos = 0
            
                print(f"{case_number = }")
                case_number += 1
            sub_idx += 1

            elapsed = time.time() - start_time
            print(f"  {auv_name} went through all trajectories {elapsed:.2f} seconds")

        if written != total_samples:
            raise RuntimeError(f"Written samples {written} != expected {total_samples}")
        print(f"Done. Wrote {written} samples to {output_file}")


# def generate_full_dataset(
#     controller,
#     subs_file="submarine_configs_physical.npz",
#     dataset_dir="reference_dataset",
#     output_file="imitation_dataset.h5",
#     dt=0.05,
#     noise=False,
#     window_size=50
# ):
#     # Load submarine configurations
#     with open(subs_file, 'rb') as f:
#         subs = pickle.load(f)
    
#     traj_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.npz')])
    
#     # Pre-calculate total samples
#     total_samples = 0
#     for filename in traj_files:
#         data = np.load(os.path.join(dataset_dir, filename))
#         T = len(data['ref_full'])
#         total_samples += max(0, T - window_size) * len(subs)
    
#     with h5py.File(output_file, 'w') as hf:
#         # Create datasets (simplified structure)
#         hf_static = hf.create_dataset('static_features', (total_samples, 7), 
#                                      dtype='f4', compression='gzip')
#         hf_ref = hf.create_dataset('ref_windows', (total_samples, window_size, 8), 
#                                   dtype='f4', compression='gzip')
#         hf_control = hf.create_dataset('controls', (total_samples, 4), 
#                                       dtype='f4', compression='gzip')
        
#         # Store metadata
#         hf.attrs['window_size'] = window_size
#         hf.attrs['dt'] = dt
#         hf.attrs['submarine_params'] = list(subs[next(iter(subs))].keys())
        
#         idx = 0
#         progress_bar = tqdm(total=total_samples, desc="Generating dataset")
        
#         for auv_name, config in subs.items():
#             # Extract static features
#             static_features = np.array([
#                 config['L'], config['D'], config['V'],
#                 config['alpha'], config['c_rot'],
#                 config['cf'], config['cd']
#             ], dtype=np.float32)
            
#             for filename in traj_files:
#                 data = np.load(os.path.join(dataset_dir, filename))
#                 ref_full = data['ref_full']
#                 T = len(ref_full)
                
#                 if T < window_size:
#                     continue
                
#                 # Initialize submarine
#                 uuv = UnderwaterVehicle_v2(
#                     M=config['M'],
#                     D_lin=config['D_lin'],
#                     D_quad=config['D_quad'],
#                     volume=config['V'],
#                     initial_state=(ref_full[0, :4], ref_full[0, 4:]),
#                     dt=dt,
#                     pos_noise_std=0.1 if noise else 0.0,
#                     vel_noise_std=0.05 if noise else 0.0
#                 )
                
#                 controls = []
                
#                 # Run simulation
#                 for k in range(T):
#                     # Get current state (not stored separately)
#                     eta, nu = uuv.get_state()
                    
#                     # Get reference window (includes current state as first point)
#                     if k + window_size <= T:
#                         ref_window = ref_full[k:k+window_size]
#                     else:
#                         # Pad with last state if near end
#                         ref_window = np.vstack([
#                             ref_full[k:],
#                             np.tile(ref_full[-1], (window_size - (T - k), 1))
#                         ])
                    
#                     # Get control action
#                     u_opt, _ = controller.solve(
#                         np.hstack((eta, nu)), 
#                         ref_window.T, 
#                         k*dt
#                     )
#                     tau_control = u_opt[:4]
                    
#                     # Apply control
#                     uuv.step(tau_control, np.zeros(4))
#                     controls.append(tau_control)
                
#                 controls = np.array(controls)
                
#                 # Create samples
#                 for k in range(0, T - window_size):
#                     hf_static[idx] = static_features
#                     hf_ref[idx] = ref_full[k:k+window_size].astype(np.float32)
#                     hf_control[idx] = controls[k].astype(np.float32)
#                     idx += 1
#                     progress_bar.update(1)
        
#         progress_bar.close()
#         print(f"Generated {idx} samples in {output_file}")

def check_dataset(dt = 0.05):
    with h5py.File('dataset.h5', 'r') as hf:
        print(f"{hf.keys() = }")                 # ['static_features', 'ref_windows', 'controls']
        print(f"{hf['sub_params'][0] = }")  # first static feature row
        print(f"{hf.attrs['window_size'] = }")   # metadata

        print(f"{hf['controls'].shape = }")
        print(f"{hf['ref_windows'].shape = }")
        print(f"{hf['case_number'].shape = }")



        # Process results
        T_ref = hf['ref_windows'].shape[0]
        ref_full_8 = []
        control_logs = []
        results = []
        case_numbers = None

        auv_features = hf['sub_params'][0]

        # inside loop, build static_vector once per submarine:
        M = auv_features[0:4]
        Dlin = auv_features[4:8]
        Dquad = auv_features[8:12]

        scalar_features = ['L' ,'D',
        'volume_coeficient',
        'm' ,
        'cd',
        'cf',
        'V' ,
        'Fmax',
        'Upmax' ,
        'Tmax',
        'payload_mass']

        L = auv_features[12]
        D = auv_features[13]
        volume_coeficient = auv_features[14]
        m = auv_features[15]
        cd = auv_features[16]
        cf = auv_features[17]
        V = auv_features[18]
        Fmax = auv_features[19]
        Upmax = auv_features[20]
        Tmax = auv_features[21]
        print(f"{M = }")
        print(f"{Dlin = }")
        print(f"{Dquad = }")

        for i, feat in enumerate(scalar_features):
            print(f"{feat} : {auv_features[i+12]}")

        for i, data in enumerate(hf['ref_windows']):
            if case_numbers is None or case_numbers == hf['case_number'][i+1]:
                ref_full_8.append(data[0])
                control_logs.append(hf['controls'][i])
                case_numbers = hf['case_number'][i]
            else:
                print("STOP ")
                break
        
        
        ref_full_8 = np.array(ref_full_8)
        control_logs = np.array(control_logs)
        print(f"{ref_full_8.shape = }")
        print(f"{control_logs.shape = }")

        init = (ref_full_8[0,:4], ref_full_8[0,4:])

        print(f"{init = }")

        auv = UnderwaterVehicle_v2(initial_state=init, 
                                    M=np.diag(auv_features[0:4]),
                                    D_lin=auv_features[4:8],
                                    D_quad=auv_features[8:12],
                                    volume=auv_features[18],
                                    max_thrust = auv_features[19],
                                    max_upforce = auv_features[20],
                                    max_torque = auv_features[21],
                                    dt=dt)
        env = Environment()

        
        for control in control_logs:
            # Get current state
            eta, nu = auv.get_state()
            results.append(np.hstack((eta, nu)))

            tau_disturbance = env.get_disturbance(eta, nu, 0)

            # Apply control
            auv.step(control, tau_disturbance)

        results = np.array(results)
        print(f"{results.shape = }")
        print(f"{ref_full_8[:,0:4].shape = }")
        print(f"{ref_full_8[:,4:8].shape = }")
        print(f"{ref_full_8[:,4].shape = }")

        # Plot results
        plot_comparison_dynamic_3d(
            ref_full_8[:,0:3], ref_full_8[:,4:7], ref_full_8[:,3], ref_full_8[:,7],
            [results],
            list(["LQR"]),
            dt=dt,
            controller_efforts=[control_logs]
        )


if __name__ == "__main__":
    # Example call:
    speed = 1
    sim_length = 100/speed
    dt = 0.05
    Fmax = 140
    Upmax = 100
    Tmax = 30
    prediction_horizon = 50
    step_size = 2
    pat_num = 2
    noise = False
    controller_type = "TVLQR"
    print(f"{speed = } and {sim_length = }")
    # test_comparison_dynamic_v2(
    #     dt=dt,
    #     sim_length=sim_length,
    #     step_size=step_size,
    #     pat_num=pat_num,
    #     speed=speed,
    #     noise=noise,
    #     prediction_horizon=prediction_horizon,
    #     Fmax = Fmax,
    #     Upmax = Upmax,
    #     Tmax = Tmax,
        
    # )
    generate_small_auv_configs(n_auvs=2)

    generate_trajectory_dataset(num_trajectories=2, dt=dt)
    
    generate_full_dataset(controller_type = controller_type)
    check_dataset()
    