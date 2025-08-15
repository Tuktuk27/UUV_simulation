import numpy as np
import minsnap_trajectories as ms
from tqdm import trange
import os
import pickle  # For more flexible data storage

def generate_minimum_snap_trajectory(
    waypoints, 
    t_total, 
    dt=0.05,
    degree=8,
    minimize_orders=(3, 4),
    continuity_orders=3
):
    """
    Generates minimum-snap trajectory through arbitrary waypoints
    """
    # Create waypoint timing (velocity-based timing)
    distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumulative_dist = np.cumsum(distances)
    waypoint_times = [0]
    waypoint_times.extend(list(cumulative_dist / cumulative_dist[-1] * t_total))
    
    # Build waypoint objects
    refs = []
    for i, (pos, t_wp) in enumerate(zip(waypoints, waypoint_times)):
        refs.append(ms.Waypoint(time=t_wp, position=pos))
    
    # Generate trajectory
    polys = ms.generate_trajectory(
        refs,
        degree=degree,
        idx_minimized_orders=minimize_orders,
        num_continuous_orders=continuity_orders,
        algorithm="closed-form",
    )
    
    # Sample trajectory
    t_grid = np.arange(0, t_total, dt)
    pva = ms.compute_trajectory_derivatives(polys, t_grid, order=3)
    
    # Calculate yaw and yaw rate
    vx, vy = pva[1][:, 0], pva[1][:, 1]
    ax, ay = pva[2][:, 0], pva[2][:, 1]
    
    psi = np.unwrap(np.arctan2(vy, vx))

    psi[0]= psi[1]


    denom = vx**2 + vy**2 + 1e-8
    psi_vel = (vx * ay - vy * ax) / denom
    
    # Create full reference trajectory
    ref_full = np.hstack([
        pva[0],                          # position (x,y,z)
        psi.reshape(-1, 1),               # yaw
        pva[1],                           # velocity (vx,vy,vz)
        psi_vel.reshape(-1, 1)            # yaw rate
    ])
    
    return {
        'position': pva[0],
        'velocity': pva[1],
        'acceleration': pva[2],
        'yaw': psi,
        'yaw_rate': psi_vel,
        'ref_full': ref_full,
        'waypoints': waypoints,
        'timestamps': t_grid
    }

import numpy as np

def generate_random_waypoints(
    num_waypoints=15,
    bounds=(-100, 100, -100, 100, -50, 0),
    min_dist=1.5,
    max_dist=4.0,
    max_heading_change=np.pi/9,  # 30 degrees max turn angle
    bias_factor=0.5,
):
    """
    Generates feasible 3D waypoints with yaw alignment and constrained turns
    """
    # Unpack bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Start at origin
    waypoints = [np.zeros(3)]

    waypoints[0][2] = -1
    
    # Generate first waypoint with simple direction
    valid = False
    attempts = 0
    while not valid and attempts < 100:
        direction = np.random.uniform(-1, 1, 3)
        direction[2] *= 0.3  # Reduce vertical movement
        norm = np.linalg.norm(direction)
        if norm < 1e-5:
            continue
            
        direction /= norm
        step_length = np.random.uniform(min_dist, max_dist)
        candidate = waypoints[0] + direction * step_length
        
        # Check bounds
        in_bounds = (x_min <= candidate[0] <= x_max and
                     y_min <= candidate[1] <= y_max and
                     z_min <= candidate[2] <= z_max)
        
        if in_bounds:
            waypoints.append(candidate)
            valid = True
            prev_direction = waypoints[-1] - waypoints[-2]
            new_heading = np.arctan2(prev_direction[1], prev_direction[0])
            heading_change = new_heading
            
        attempts += 1
    
    # Generate subsequent waypoints with heading constraints
    for i in range(num_waypoints - 2):
        valid = False
        attempts = 0
        # prev_direction = waypoints[-1] - waypoints[-2]
        # prev_heading = np.arctan2(prev_direction[1], prev_direction[0])
        prev_direction = direction
        prev_heading = new_heading
        
        while not valid and attempts < 100:
            if num_waypoints%10 ==0:
                heading_change = np.random.uniform(-max_heading_change, max_heading_change)
                new_heading = prev_heading + heading_change
            # Constrain new heading to max_heading_change
            else:
                mu    = np.sign(heading_change) * max_heading_change * bias_factor
                sigma = max_heading_change / 2
                raw   = np.random.normal(loc=mu, scale=sigma)
                heading_change = np.clip(raw, -max_heading_change, +max_heading_change)

            new_heading = prev_heading + heading_change

            # Create new direction with constrained vertical movement
            vertical_angle = np.random.uniform(-0.3, 0.3)  # ±17 degrees
            direction = np.array([
                np.cos(vertical_angle) * np.cos(new_heading),
                np.cos(vertical_angle) * np.sin(new_heading),
                np.sin(vertical_angle)
            ])
            
            # Generate candidate point
            step_length = np.random.uniform(min_dist, max_dist)
            candidate = waypoints[-1] + direction * step_length
            
            # Check bounds and collision
            in_bounds = (x_min <= candidate[0] <= x_max and
                         y_min <= candidate[1] <= y_max and
                         z_min <= candidate[2] <= z_max)
            
            # collision = any(np.linalg.norm(candidate - wp) < min_dist*0.7 
            #              for wp in waypoints)
            
            if in_bounds:# and not collision:
                waypoints.append(candidate)
                valid = True
                
            attempts += 1
        
        # if not valid:
        #     # Fallback with simple direction
        #     print(f"Fallback, iteration number: {i = }")
        #     direction = np.random.uniform(-1, 1, 3)
        #     direction[2] *= 0.3
        #     if np.linalg.norm(direction) > 1e-5:
        #         direction /= np.linalg.norm(direction)
        #         step_length = np.random.uniform(min_dist, max_dist)
        #         candidate = waypoints[-1] + direction * step_length
        #         waypoints.append(np.clip(candidate, 
        #                                 [x_min, y_min, z_min], 
        #                                 [x_max, y_max, z_max]))

        # Check initial alignment

            
    return np.array(waypoints)

def is_trajectory_feasible(traj, max_vel=2.0, max_accel=1.0, max_yaw_rate=0.5):
    # Check velocity limits
    vel_magnitudes = np.linalg.norm(traj['velocity'], axis=1)
    if np.any(vel_magnitudes > max_vel):
        print(f"\n\n\n\nvelocity problem: {vel_magnitudes = }\n\n\n\n")
        return False
        
    # Check acceleration limits
    accel_magnitudes = np.linalg.norm(traj['acceleration'], axis=1)
    if np.any(accel_magnitudes > max_accel):
        print(f"\n\n\n\nacceleration problem: {accel_magnitudes = }\n\n\n\n")
        return False
        
    # Check yaw rate limits
    abs_rates = np.abs(traj['yaw_rate'])
    max_val   = abs_rates.max()
    max_idx   = abs_rates.argmax()
    if max_val > max_yaw_rate:
        print(
            f"\n\n\n\nYaw problem:\n"
            f"  max yaw_rate = {max_val:.4f} rad/s\n"
            f"  at index      = {max_idx}\n"
            f"before -1: {traj['yaw_rate'][max_idx-1]} and after+1: {traj['yaw_rate'][max_idx+1]}\n"
            f"before -2: {traj['yaw_rate'][max_idx-2]} and after+2: {traj['yaw_rate'][max_idx+2]}\n\n\n"
        )
        return False
        
    return True

def generate_trajectory_dataset(
    num_trajectories=1000,
    dt=0.05,
    min_duration=100,
    max_duration=150,
    output_dir="trajectory_data",
    avg_segment_time = 5.0,  # Average time between waypoints (seconds)
):
    """
    Generates a dataset of random trajectories
    """
    os.makedirs(output_dir, exist_ok=True)

    num_saved = 0
    
    for _ in trange(num_trajectories, desc="Trajectories generations iteration"):
        # Random parameters
        duration = np.random.uniform(min_duration, max_duration)
                # Random trajectory parameters
        fixed_waypoint_num = int(duration / avg_segment_time) + 1
        num_waypoints = fixed_waypoint_num - np.random.randint(- fixed_waypoint_num*0.2, fixed_waypoint_num*0.2)
        # bounds = (
        #     np.random.uniform(-40, 0),   # x_min
        #     np.random.uniform(5, 20),     # x_max
        #     np.random.uniform(-20, -5),   # y_min
        #     np.random.uniform(5, 20),     # y_max
        #     np.random.uniform(-25, -5),   # z_min
        #     np.random.uniform(-3, -0.5)   # z_max
        # )
        
        # Generate waypoints
        waypoints = generate_random_waypoints(
            num_waypoints=num_waypoints,
            # bounds=bounds
        )
        
        # Generate trajectory
        traj = generate_minimum_snap_trajectory(
            waypoints=waypoints,
            t_total=duration,
            dt=dt
        )

        # Verify feasibility
        if not is_trajectory_feasible(traj):
            print("Warning: Generated trajectory may be infeasible")
        else:
            # Save trajectory
            np.savez(
                os.path.join(output_dir, f"trajectory_{num_saved:04d}.npz"),
                position=traj['position'],
                velocity=traj['velocity'],
                acceleration=traj['acceleration'],
                yaw=traj['yaw'],
                yaw_rate=traj['yaw_rate'],
                ref_full=traj['ref_full'],
                waypoints=waypoints,
                timestamps=traj['timestamps'],
                duration=duration,
                # bounds=bounds
            )

            num_saved += 1
    print(f"Generated {num_saved} of feasible trajectories")

import numpy as np

def generate_small_auv_configs(n_auvs=100, output_path="small_auv_configs.pkl",
                               
                               a_fwd_range=(1, 8),     # m/s^2
                               a_vert_range=(2, 6),   # m/s^2
                               alpha_range=(1, 6),       # rad/s^2
                               rho = 1025
                               ):
    """
    Generates physically-plausible submarine configurations with validated hydrodynamics
    using robust dictionary-per-submarine storage
    """
    submarines = {}
    
    for sub_id in range(n_auvs):
        config = {}
        
        # 1. Geometric parameters
        config['L'] = np.random.uniform(0.4, 0.9)      # Length [m]
        config['D'] = np.random.uniform(0.2, 0.4)     # Width [m]
        config['volume_coeff'] = np.random.uniform(0.45, 0.65)
        
        # 2. Volume and mass
        config['V'] = config['volume_coeff'] * config['L'] * config['D']**2
        config['payload_mass'] = 0 #np.random.uniform(0.1, 5.0)
        config['m'] = rho * config['V'] + config['payload_mass']

        displaced_mass = rho * config['V']
        buoyancy_offset = config['m'] - displaced_mass  # positive => negatively buoyant (kg)
        print(f"{buoyancy_offset = }")
        
        # 3. Hydrodynamics
        config['U_cruise'] = np.random.uniform(0.5, 2.5)
        Re = config['U_cruise'] * config['L'] / 1e-6

        # Skin friction coeficient
        config['cf'] = 0.0045 if Re < 1e5 else 0.075/(np.log10(Re)-2)**1.8
        
        fineness = config['L'] / config['D']
        config['cd_base'] = 0.12 + 0.03*(config['D']/config['L'])
        config['cd'] = config['cd_base'] + 0.15*(1-np.exp(-0.3*abs(fineness-5.0)))
        
        # 4. Added mass (empirical for small AUVs)
        config['M'] = np.array([
            [0.1*config['m'], 0, 0, 0],
            [0, 0.2*config['m'], 0, 0],
            [0, 0, 0.8*config['m'], 0],
            [0, 0, 0, 0.05*config['m']*config['L']**2]
        ])
        
        # 5. Damping
        frontal_area = config['D']**2
        config['D_lin'] = np.array([
            0.1 * config['m'],
            0.15 * config['m'],
            0.25 * config['m'],
            0.01 * config['m'] * config['L']**2
        ])
        
        config['D_quad'] = np.array([
            0.5 * 1025 * frontal_area * config['cd'],
            0.7 * 0.5 * 1025 * frontal_area * config['cd'],
            1.2 * 0.5 * 1025 * frontal_area * config['cd'],
            0.3 * 0.5 * 1025 * config['L']**3 * config['cd']
        ])
        

        # 6. Thrust characteristics
        # Potential way to cmpute thrusts and torque 
        # config['max_thrust'] = np.random.uniform(5, 30)  # [N]
        # config['thrust_ratio'] = np.array([
        #     np.random.uniform(0.8, 1.0),  # Surge efficiency
        #     np.random.uniform(0.6, 0.8),  # Sway efficiency
        #     np.random.uniform(0.7, 0.9),  # Heave efficiency
        #     np.random.uniform(0.4, 0.6)   # Yaw efficiency
        # ])

        # ---- 6. Thrust & torque computed consistently with mass/inertia ----
        # sample target max linear accelerations (m/s^2)
        a_fwd_max = np.random.uniform(*a_fwd_range)   # forward/back accel
        a_vert_max = np.random.uniform(*a_vert_range) # vertical accel (heave)
        
        # approximate rigid-body yaw moment of inertia (about body z)
        I_rigid_z = config['m'] * (config['L']**2 + config['D']**2) / 12.0
        
        # added (hydrodynamic) rotational inertia from M matrix (you stored it at M[3,3])
        I_added_z = config['M'][3,3]
        I_tot_z = I_rigid_z + I_added_z
        
        # sample target angular accel (rad/s^2)
        alpha_max = np.random.uniform(*alpha_range)

        # compute max forces/torque
        config['Fmax'] = float(config['m'] * a_fwd_max)   # [N], forward/backward capability
        config['Upmax'] = float(config['m'] * a_vert_max) # [N], upward/downward capability
        config['Tmax'] = float(I_tot_z * alpha_max)       # [N*m], yaw torque capability
        
        # Validate
        valid = True
        if config['V'] < 0.01: valid = False
        if not np.all(np.linalg.eigvals(config['M'] > 0)): valid = False
        if not np.all(config['D_lin'] > 0): valid = False
        
        if valid:
            submarines[f"auv_{sub_id}"] = config
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(submarines, f)

    n_valid = len(submarines)
    print(f"Generated {n_valid} valid configurations ({n_valid/n_auvs*100:.1f}%)")
    print(f"Saved to {output_path}")
    
    return submarines


# def generate_small_auv_configs(
#     n_auvs=100,
#     a_fwd_range=(1, 8),     # m/s^2
#     a_vert_range=(2, 6),    # m/s^2
#     alpha_range=(1, 6),     # rad/s^2
#     rho=1025,
#     nu=1e-6,
#     g =9.81,                                # kinematic viscosity [m^2/s]
#     payload_range=(0.1, 5.0),
#     structure_mass_range=(1.0, 6.0),
#     volume_coeff_range=(0.45, 0.65),
#     enforce_buoyancy_offset_kg=0.5,         # soft enforce |m - rho*V| <= this (kg)
#     vertical_safety_margin=1.1,             # Upmax must be >= needed_N * margin
#     output_path="small_auv_configs.pkl"
# ):
#     """
#     Generates physically-plausible submarine configurations (improved & validated).
#     """
#     submarines = {}

#     for sub_id in range(n_auvs):
#         config = {}

#         # 1. Geometry
#         config['L'] = np.random.uniform(0.4, 0.9)
#         config['D'] = np.random.uniform(0.2, 0.4)
#         config['volume_coeff'] = np.random.uniform(*volume_coeff_range)

#         # 2. Volume
#         config['V'] = config['volume_coeff'] * config['L'] * config['D']**2

#         # 2b. Masses: explicit structure + payload (fixes original bug)
#         config['payload_mass'] = np.random.uniform(*payload_range)
#         config['structure_mass'] = np.random.uniform(*structure_mass_range)
#         config['m'] = float(config['structure_mass'] + config['payload_mass'])

#         # displaced mass (mass of displaced water)
#         displaced_mass = rho * config['V']
#         buoyancy_offset = config['m'] - displaced_mass  # positive => negatively buoyant (kg)

#         # Soft enforce small buoyancy offset by nudging structure mass
#         if abs(buoyancy_offset) > enforce_buoyancy_offset_kg:
#             target_m = displaced_mass + np.sign(buoyancy_offset) * enforce_buoyancy_offset_kg
#             config['structure_mass'] = max(0.1, target_m - config['payload_mass'])
#             config['m'] = float(config['structure_mass'] + config['payload_mass'])
#             buoyancy_offset = config['m'] - displaced_mass

#         config['displaced_mass'] = float(displaced_mass)
#         config['buoyancy_offset_kg'] = float(buoyancy_offset)

#         # 3. Hydrodynamics
#         config['U_cruise'] = np.random.uniform(0.5, 2.5)
#         Re = config['U_cruise'] * config['L'] / nu
#         config['Re'] = float(Re)
#         # ITTC friction formula (standard) for typical Re; fallback to small constant
#         config['cf'] = 0.075 / (np.log10(Re) - 2.0)**2 if Re > 1e4 else 0.0045

#         fineness = config['L'] / config['D']
#         config['cd_base'] = 0.12 + 0.03 * (config['D'] / config['L'])
#         config['cd'] = float(config['cd_base'] + 0.15 * (1 - np.exp(-0.3 * abs(fineness - 5.0))))

#         # 4. Added mass (diagonal empirical) — these are *added* masses
#         M_added = np.array([
#             0.1 * config['m'],                 # surge added mass
#             0.15 * config['m'],                # sway added mass
#             0.1 * displaced_mass,              # heave added mass (scaled with displaced mass)
#             0.05 * config['m'] * config['L']**2  # added yaw inertia (empirical)
#         ], dtype=float)
#         config['M_added'] = M_added

#         # 5. Rigid-body inertia (diag translational + approx yaw inertia)
#         I_z = 0.25 * config['m'] * config['L']**2
#         M_rigid = np.array([config['m'], config['m'], config['m'], I_z], dtype=float)
#         config['M_rigid'] = M_rigid

#         # 6. Total inertia (rigid + added) used by the sim. Assign to config['M']
#         M_total = np.diag(M_rigid) + np.diag(M_added)
#         config['M'] = M_total  # keep compatibility with older code expecting config['M']
#         config['M_total'] = M_total

#         # 7. Damping (use correct frontal & lateral areas)
#         frontal_area = np.pi * (config['D'] / 2.0)**2
#         lateral_area = config['L'] * config['D']
#         # quadratic base for frontal area:
#         Dq_base = 0.5 * rho * frontal_area * config['cd']

#         # Assign per-axis quadratic damping (surge, sway, heave, yaw)
#         config['D_quad'] = np.array([
#             1.0 * Dq_base,                            # surge (frontal)
#             0.7 * 0.5 * rho * lateral_area * config['cd'],   # sway (lateral)
#             1.2 * Dq_base,                            # heave (frontal scaled)
#             0.02 * 0.5 * rho * (config['L']**3) * config['cd'] # yaw (small tuned)
#         ], dtype=float)

#         # 8. Linear damping (small empirical values)
#         config['D_lin'] = np.array([
#             0.05 * config['m'],
#             0.08 * config['m'],
#             0.15 * config['m'],
#             0.01 * config['m'] * config['L']**2
#         ], dtype=float)

#         # 9. Thruster & torque sizing: sample accelerations, but ensure Upmax sufficient
#         a_fwd_max = np.random.uniform(*a_fwd_range)
#         # compute needed N to counter buoyancy offset
#         needed_N = max(0.0, buoyancy_offset * g)  # N upward required to hover (if positive)
#         # minimal vertical accel required so that m*a_vert >= needed_N
#         min_a_vert_required = needed_N / max(1e-9, config['m'])

#         # choose a_vert ensuring it meets the requirement + safety margin
#         a_vert_lower = max(a_vert_range[0], min_a_vert_required * vertical_safety_margin)
#         a_vert = np.random.uniform(a_vert_lower, a_vert_range[1])

#         alpha_max = np.random.uniform(*alpha_range)

#         config['a_fwd'] = float(a_fwd_max)
#         config['a_vert'] = float(a_vert)
#         config['alpha'] = float(alpha_max)

#         config['Fmax'] = float(config['m'] * a_fwd_max)
#         config['Upmax'] = float(config['m'] * a_vert)
#         config['Tmax'] = float(I_z * alpha_max)

#         # 10. Validation checks (positive-definite inertia, positive damping, reasonable volume)
#         valid = True
#         if config['V'] < 1e-3: valid = False
#         # eigenvalues of M_total must be > 0 (positive-definite diagonal here)
#         if not np.all(np.linalg.eigvals(config['M']) > 0): valid = False
#         if not np.all(config['D_lin'] > 0): valid = False
#         # thruster must be able to overcome buoyancy offset within margin
#         if config['Upmax'] < needed_N * vertical_safety_margin:
#             # If this happens despite our selection, mark invalid (or you can choose to bump a_vert)
#             valid = False

#         if valid:
#             submarines[f"auv_{sub_id}"] = config

#     # Save
#     with open(output_path, 'wb') as f:
#         pickle.dump(submarines, f)

#     n_valid = len(submarines)
#     print(f"Generated {n_valid} valid configurations ({n_valid/n_auvs*100:.1f}%)")
#     print(f"Saved to {output_path}")

#     return submarines

