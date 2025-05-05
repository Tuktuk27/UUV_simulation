# utils.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _draw_submarine(ax, x, y, z, yaw, size=0.2, color='k'):
    """
    Draw a small rectangular hull + nose arrow at (x,y,z) with heading yaw.
    """
    # Define corners in the body frame
    corners = np.array([
        [-size, -size/2, 0],
        [ size, -size/2, 0],
        [ size,  size/2, 0],
        [-size,  size/2, 0],
        [-size, -size/2, 0],
    ]).T  # shape (3,5)

    # Rotation matrix about Z
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[ c,-s,0],
                  [ s, c,0],
                  [ 0, 0,1]])
    pts = R @ corners
    ax.plot(pts[0]+x, pts[1]+y, pts[2]+z, color=color, linewidth=1.5)

    # Nose arrow
    nose = np.array([size*1.5, 0, 0])
    tip = R @ nose + np.array([x, y, z])
    ax.quiver(x, y, z,
              tip[0]-x, tip[1]-y, tip[2]-z,
              length=1.0, normalize=True, color=color,
              arrow_length_ratio=0.3)
    
def plot_comparison(states_pid, states_uncontrolled, ref_pos, draw_every=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot paths
    ax.plot(states_uncontrolled[:,0], states_uncontrolled[:,1], states_uncontrolled[:,2],
            'r--', label='Uncontrolled')
    ax.plot(states_pid[:,0], states_pid[:,1], states_pid[:,2],
            'g-', label='PID Controlled')
    ax.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2],
            'b:', label='Reference')

    # Draw submarine body at intervals
    for i in range(0, len(states_pid), draw_every):
        x, y, z, yaw = states_pid[i,0], states_pid[i,1], states_pid[i,2], states_pid[i,3]
        _draw_submarine(ax, x, y, z, yaw, size=0.1, color='g')

    for i in range(0, len(states_uncontrolled), draw_every):
        x, y, z, yaw = states_uncontrolled[i,0], states_uncontrolled[i,1], states_uncontrolled[i,2], states_uncontrolled[i,3]
        _draw_submarine(ax, x, y, z, yaw, size=0.1, color='r')

    # Formatting
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Trajectory & Vehicle Pose")
    ax.legend()
    plt.show()


def plot_velocity_comparison(states_pid, states_uncontrolled, ref_vel):
    """
    Time-series of surge, sway, heave velocities:
    - ref_vel: array (T,3) of desired [u,v,w]
    """
    T = states_pid.shape[0]
    t = np.linspace(0, T-1, T) * (t.max()/ (T-1))  # or pass actual time vector if available
    u_pid = states_pid[:,4]; v_pid = states_pid[:,5]; w_pid = states_pid[:,6]
    u_un = states_uncontrolled[:,4]; v_un = states_uncontrolled[:,5]; w_un = states_uncontrolled[:,6]

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(t, u_un, 'r--', label='u uncontrolled')
    plt.plot(t, u_pid,'g-',  label='u PID'       )
    plt.plot(t, ref_vel[:,0],'b:', label='u ref')
    plt.ylabel('u (m/s)')
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(t, v_un,'r--', label='v uncontrolled')
    plt.plot(t, v_pid,'g-',  label='v PID'       )
    plt.plot(t, ref_vel[:,1],'b:', label='v ref')
    plt.ylabel('v (m/s)')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(t, w_un,'r--', label='w uncontrolled')
    plt.plot(t, w_pid,'g-',  label='w PID'       )
    plt.plot(t, ref_vel[:,2],'b:', label='w ref')
    plt.ylabel('w (m/s)')
    plt.xlabel('Step')
    plt.legend()
    plt.suptitle("Velocity Comparison")

def plot_comparison_dynamic_3d(ref_pos, ref_vel, controller_states, controller_labels,
                               draw_every=10, dt=1.0, psi_ref=None):
    """
    4x4 Grid:
      - Rows 0–1: 3D trajectory + submarine pose icons
      - Row  2:   X, Y, Z, and Yaw vs time (4 subplots)
      - Row  3:   Velocity magnitude vs time
    """
    T_ref = ref_pos.shape[0]
    time  = np.arange(T_ref) * dt

    # Align trajectories to T_ref
    aligned = []
    for states in controller_states:
        if states.shape[0] == T_ref + 1:
            aligned.append(states[:-1])
        elif states.shape[0] == T_ref - 1:
            aligned.append(np.vstack([states, states[-1]]))
        else:
            aligned.append(states)
    controller_states = aligned

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs  = fig.add_gridspec(4, 4)

    # --- Top: 3D Trajectories ---
    ax3d = fig.add_subplot(gs[:2, :], projection='3d')
    colors = ['g','m','r','c','y']
    ax3d.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2],
              'k--', label='Reference', linewidth=2)
    for i, (states, label) in enumerate(zip(controller_states, controller_labels)):
        ax3d.plot(states[:,0], states[:,1], states[:,2],
                  color=colors[i], label=label, linewidth=1.5)
        for k in range(0, T_ref, draw_every):
            x,y,z,yaw = states[k,0], states[k,1], states[k,2], states[k,3]
            _draw_submarine(ax3d, x, y, z, yaw, size=0.2, color=colors[i])
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D Trajectories & Vehicle Pose"); ax3d.legend()

    # Equalize aspect
    all_pts = np.vstack([ref_pos, *[s[:,:3] for s in controller_states]])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    R = (maxs - mins).max()/2.0; mid = (maxs + mins)/2.0
    ax3d.set_xlim(mid[0]-R, mid[0]+R); ax3d.set_ylim(mid[1]-R, mid[1]+R); ax3d.set_zlim(mid[2]-R, mid[2]+R)

    # --- Middle row: X, Y, Z, Yaw ---
    labels = ['X Position', 'Y Position', 'Z Position', 'Yaw (ψ)']
    refs = [ref_pos[:,0], ref_pos[:,1], ref_pos[:,2], psi_ref]
    for idx in range(4):
        ax = fig.add_subplot(gs[2, idx])
        ax.plot(time, refs[idx], 'k--', label=f'Ref {labels[idx]}')
        for states, label in zip(controller_states, controller_labels):
            data = states[:, idx] if idx<3 else states[:,3]
            ax.plot(time, data, label=f'{label} {labels[idx]}')
        ax.set_ylabel(labels[idx])
        if idx==0: ax.legend()
        ax.grid(True)
        if idx==3:
            ax.set_xlabel("Time (s)")
    # --- Bottom: Velocity magnitude ---
    axv = fig.add_subplot(gs[3, :])
    v_ref = np.linalg.norm(ref_vel, axis=1)
 
    axv.plot(time, v_ref, 'k--', linewidth=2, label='Ref Velocity')
    for states, label in zip(controller_states, controller_labels):
        v = np.linalg.norm(states[:,4:7], axis=1)
        axv.plot(time, v, label=f'{label} Velocity')
    axv.set_xlabel("Time (s)"); axv.set_ylabel("Velocity Magnitude (m/s)")
    axv.set_title("Velocity Tracking"); axv.grid(True); axv.legend()

    plt.suptitle("Dynamic Comparison: Trajectory, Pose & Velocity", fontsize=18)
    plt.show()

def plot_comparison_dynamic(ref_pos, ref_vel, controller_states, controller_labels):
    """
    Plot a dynamic comparison of trajectories and velocities.
    
    Parameters:
      ref_pos           : array of shape (T, 3) for the reference positions.
      ref_vel           : array of shape (T, 3) for the reference velocities.
      controller_states : a list of arrays, each with shape (T' , 6) representing
                          the state trajectories for a controller. The state should contain
                          [x, y, z, vx, vy, vz] in each row.
      controller_labels : a list of strings representing the labels for each controller.
    """
    # Use the length of the reference as the baseline time dimension.
    T_ref = ref_pos.shape[0]
    time = np.arange(T_ref)
    
    # Adjust each state trajectory if its length does not match the reference.
    adjusted_states = []
    for states in controller_states:
        if states.shape[0] == T_ref + 1:
            # Trim the final extra sample
            adjusted_states.append(states[:-1])
        elif states.shape[0] == T_ref - 1:
            # Append the last sample (not typical, but handling this case)
            adjusted_states.append(np.vstack([states, states[-1]]))
        else:
            adjusted_states.append(states)
    
    # Create a figure with a gridspec layout:
    # Top: 3D trajectory (spans top 2 rows)
    # Middle: X, Y, Z positions vs time (3 subplots in one row)
    # Bottom: Velocity magnitude vs time (one subplot spanning entire width)
    fig = plt.figure(constrained_layout=True, figsize=(18, 14))
    gs = fig.add_gridspec(4, 3)
    
    # Top: 3D Trajectory Plot
    ax3d = fig.add_subplot(gs[:2, :], projection='3d')
    # Plot reference 3D trajectory:
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
              'g--', label='Reference', linewidth=2)
    # Plot each controller trajectory in 3D:
    for states, label in zip(adjusted_states, controller_labels):
        ax3d.plot(states[:, 0], states[:, 1], states[:, 2],
                  label=label, linewidth=1.5)
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D Trajectory Comparison")
    ax3d.legend()

    # Middle row: X, Y, Z position evolution
    ax_x = fig.add_subplot(gs[2, 0])
    ax_x.plot(time, ref_pos[:, 0], 'g--', label='Ref X')
    for states, label in zip(adjusted_states, controller_labels):
        ax_x.plot(time, states[:, 0], label=label + " X")
    ax_x.set_ylabel("X Position")
    ax_x.legend()
    ax_x.grid(True)
    
    ax_y = fig.add_subplot(gs[2, 1])
    ax_y.plot(time, ref_pos[:, 1], 'g--', label='Ref Y')
    for states, label in zip(adjusted_states, controller_labels):
        ax_y.plot(time, states[:, 1], label=label + " Y")
    ax_y.set_ylabel("Y Position")
    ax_y.legend()
    ax_y.grid(True)
    
    ax_z = fig.add_subplot(gs[2, 2])
    ax_z.plot(time, ref_pos[:, 2], 'g--', label='Ref Z')
    for states, label in zip(adjusted_states, controller_labels):
        ax_z.plot(time, states[:, 2], label=label + " Z")
    ax_z.set_xlabel("Time Steps")
    ax_z.set_ylabel("Z Position")
    ax_z.legend()
    ax_z.grid(True)
    
    # Bottom row: Velocity magnitude comparison
    ax_vel = fig.add_subplot(gs[3, :])
    # Compute the velocity magnitude for the reference
    v_ref = np.linalg.norm(ref_vel, axis=1)
    ax_vel.plot(time, v_ref, 'g--', label='Ref Velocity', linewidth=2)
    # Plot velocity magnitudes for each controller
    for states, label in zip(adjusted_states, controller_labels):
        v = np.linalg.norm(states[:, 3:6], axis=1)
        ax_vel.plot(time, v, label=label + " Velocity", linewidth=1.5)
    ax_vel.set_xlabel("Time Steps")
    ax_vel.set_ylabel("Velocity Magnitude (m/s)")
    ax_vel.legend()
    ax_vel.grid(True)

    plt.suptitle("Trajectory and Velocity Comparison", fontsize=16)
    plt.show()
