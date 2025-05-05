import numpy as np
from Submarine_class import UnderwaterVehicle
from pid_controller_v2 import PositionPID3D, YawPID
from mpc_controller_v2 import MPCController
from utils import plot_comparison, plot_velocity_comparison, plot_comparison_dynamic_3d

def simulate_uncontrolled(initial_state, dt=0.1, steps=100):
    """Simulate UUV with initial velocity and no control."""
    uuv = UnderwaterVehicle(initial_state=initial_state, dt=dt)
    # initial_state: (eta0, nu0)
    history = []
    # record initial
    eta, nu = uuv.get_state()
    history.append(np.hstack((eta, nu)))
    for _ in range(steps):
        # zero control: tau = [0,0,0,0]
        uuv.step(tau=np.zeros(4))
        eta, nu = uuv.get_state()
        history.append(np.hstack((eta, nu)))
    return np.array(history)


def test_comparison(dt=0.1):
    # Reference helix trajectory
    t = np.linspace(0, 100, 1000)
    ref_pos = np.zeros((len(t), 3))
    ref_vel = np.zeros((len(t), 3))
    ref_pos[:, 0] = 2 * np.cos(0.5 * t) - 2
    ref_pos[:, 1] = 2 * np.sin(0.5 * t)
    ref_pos[:, 2] = 0.1 * t
    ref_vel[:, 0] = -0.5 * np.sin(0.5 * t)
    ref_vel[:, 1] =  0.5 * np.cos(0.5 * t)
    ref_vel[:, 2] = 0.1
    psi_ref   = np.arctan2(ref_vel[:,1], ref_vel[:,0])
    r_ref     = np.gradient(psi_ref, dt)   # or analytic derivative

    # PID-Controlled Simulation
    init_state = ([0.0, 0.0, 0.0, np.pi/4], [0.0, 0.0, 0.0, 0.0])
    uuv_pid = UnderwaterVehicle(initial_state=init_state, dt=0.1)
    # 4×4 diagonal gains (tune these)
    Kp = np.diag([50, 50, 30, 10])
    Ki = np.diag([ 1,  1,  0.5, 1 ])
    Kd = np.diag([20, 20, 10, 5 ])

    pid = PIDController4DOF(Kp, Ki, Kd, dt=0.1)

    for k in range(len(t)):
        eta, nu = uuv_pid.get_state()         # [x,y,z,psi], [u,v,w,r]
        ref = np.hstack((ref_pos[k], psi_ref[k], ref_vel[k], r_ref[k]))
        forces, moment = pid.compute_control(eta, nu, ref)
        tau = np.array([*forces, moment])
        uuv_pid.step(tau=tau)
    states_pid = np.array(states_pid)

    # Uncontrolled Simulation
    # initial forward velocity 1 m/s
    init_state2 = ([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    states_unctrl = simulate_uncontrolled(init_state2, )

    # Plots
    plot_comparison(
        states_pid=states_pid,
        states_uncontrolled=states_unctrl,
        ref_pos=ref_pos
    )
    plot_velocity_comparison(states_pid, states_unctrl, ref_vel)


def test_comparison_dynamic(dt=0.1, helix_sharpness = 0.1, sim_length = 10, step_size = 10):
    # Helix reference
    steps = sim_length * step_size
    t = np.linspace(0, sim_length, steps)
    ref_pos = np.stack([2*np.cos(helix_sharpness*t)-2,
                        2*np.sin(helix_sharpness*t),
                        0.1*t], axis=1)
    ref_vel = np.stack([-helix_sharpness*np.sin(helix_sharpness*t),
                        helix_sharpness*np.cos(helix_sharpness*t),
                        0.1*np.ones_like(t)], axis=1)

    psi_wrapped = np.arctan2(ref_vel[:,1], ref_vel[:,0])

    # Create a continuous, unwrapped heading
    psi_ref = np.unwrap(psi_wrapped)
    r_ref     = np.gradient(psi_ref, dt)   # or analytic derivative

    ref_full_8 = np.hstack((ref_pos, 
                        psi_ref.reshape(-1,1),
                        ref_vel, 
                        r_ref.reshape(-1,1)))
    
    # at k=0
    eta0 = [ 
        ref_pos[0,0],       # x0
        ref_pos[0,1],       # y0
        ref_pos[0,2],       # z0
        psi_ref[0]          # yaw0  
    ]
    nu0 = [
        ref_vel[0,0],       # u0 (surge)
        ref_vel[0,1],       # v0 (sway)
        ref_vel[0,2],       # w0 (heave)
        r_ref[0]           # r0 (yaw rate)
    ]

    # PID
    init = (eta0, nu0)
    uuv_pid = UnderwaterVehicle(initial_state=init, dt=0.1)

    states_pid = []
    pos_pid = PositionPID3D(
    Kp=[20, 20, 15],
    Ki=[0.1,0.1,0.05],
    Kd=[10,10,5],
    dt=dt,
    limits=([-200,-200,-200],[200,200,200])
    )
    yaw_pid = YawPID(
        Kp=5.0, Ki=0.1, Kd=1.0,
        dt=dt,
        limits=(-50.0, 50.0)
    )

    # --- simulation loop ---
    states = []
    for k in range(len(t)):
        eta, nu = uuv_pid.get_state()

        states_pid.append(np.hstack((eta, nu)))
        x,y,z, yaw = eta
        u, v, w, r = nu

        # desired references
        px, py, pz   = ref_pos[k]
        ux_ref, vy_ref, wz_ref = ref_vel[k]
        ψ_ref = psi_ref[k]
        r_ψ_ref = r_ref[k]

        # translate‐pid
        F = pos_pid.compute(
            current_pos=np.array([x,y,z]),
            current_yaw = yaw,
            current_vel=np.array([u,v,w]),
            # current_yaw_r=r,
            ref_pos=np.array([px,py,pz]),
            # ψ_ref = ψ_ref,
            ref_vel=np.array([ux_ref,vy_ref,wz_ref]),
            # r_ψ_ref = r_ψ_ref,
        )

        # yaw‐pid
        Mz = yaw_pid.compute(
            current_yaw=yaw,
            current_r=r,
            ref_yaw=ψ_ref,
            ref_r=r_ψ_ref
        )

        # combine & step
        tau = np.array([F[0], F[1], F[2], Mz])
        uuv_pid.step(tau=tau)

        states.append(np.hstack((eta, nu)))
    states = np.array(states)
    states_pid = np.array(states_pid)

    states_unctrl = simulate_uncontrolled(init, dt=dt, steps=steps)

    # MPC
    uuv_mpc = UnderwaterVehicle(initial_state=init, dt=0.1)
    mpc = MPCController(
    system=uuv_mpc,
    nx=8,         # [x,y,z,psi,u,v,w,r]
    nu=4,         # [Fx,Fy,Fz,Mz]
    N=20,
    dt=0.1
    )
    states_mpc = []
    for k in range(len(t)):
        eta, nu = uuv_mpc.get_state()
        states_mpc.append(np.hstack((eta, nu)))

        # end index for this window
        end = min(k + mpc.N + 1, len(ref_full_8))

        # slice and transpose → shape (8, window_len)
        window = ref_full_8[k:end].T  

        # solve takes an 8-vector state and an 8×(≤N+1) reference
        u = mpc.solve(np.hstack((eta, nu)), window)  
        tau = np.array([*u[:3], u[3]])  # first 3 are forces, 4th is Mz

        uuv_mpc.step(tau=tau)
    states_mpc = np.array(states_mpc)

    # find minimum length
    T_ref = ref_vel.shape[0]          # 100
    states_pid  = states_pid[:T_ref]
    states_mpc  = states_mpc[:T_ref]
    states_unctrl = states_unctrl[:T_ref]

    # Dynamic comparison
    controller_states = [states_pid, states_mpc, states_unctrl]
    controller_labels = ['PID Controlled', 'MPC Controlled', 'Uncontrolled']
    plot_comparison_dynamic_3d(ref_pos, ref_vel, controller_states, controller_labels, psi_ref=psi_ref)

if __name__ == "__main__":
    # test_comparison()
    test_comparison_dynamic(dt=0.1, sim_length=100, helix_sharpness=0.1, step_size=10)
