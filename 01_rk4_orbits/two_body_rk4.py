import numpy as np
import matplotlib.pyplot as plt

MU_EARTH = 398600.4418

def two_body_ode(t, state, mu=MU_EARTH):
    r_vec = state[:3]
    v_vec = state[3:]
    r_norm = np.linalg.norm(r_vec)
    a_vec = -(mu / (r_norm ** 3)) * r_vec
    return np.concatenate((v_vec, a_vec))

def euler_step(f, t, y, dt):
    return y + dt * f(t, y)

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def compute_specific_energy(states, mu=MU_EARTH):
    r_norms = np.linalg.norm(states[:, :3], axis=1)
    v_norms = np.linalg.norm(states[:, 3:], axis=1)
    return 0.5 * (v_norms ** 2) - (mu / r_norms)

if __name__ == '__main__':
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, 8.2, 3.5])
    state0 = np.concatenate((r0, v0))
    dt = 20.0
    t_steps = np.arange(0.0, 35000.0, dt)
    n_steps = len(t_steps)
    traj_euler = np.zeros((n_steps, 6))
    traj_rk4 = np.zeros((n_steps, 6))
    traj_euler[0] = state0
    traj_rk4[0] = state0
    for i in range(1, n_steps):
        t_prev = t_steps[i - 1]
        traj_euler[i] = euler_step(two_body_ode, t_prev, traj_euler[i - 1], dt)
        traj_rk4[i] = rk4_step(two_body_ode, t_prev, traj_rk4[i - 1], dt)
    E0 = compute_specific_energy(state0.reshape(1, 6))[0]
    drift_euler = np.abs((compute_specific_energy(traj_euler) - E0) / E0)
    drift_rk4 = np.abs((compute_specific_energy(traj_rk4) - E0) / E0)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(traj_euler[:, 0], traj_euler[:, 1], traj_euler[:, 2], 'r--', label='Forward Euler (Unstable)', alpha=0.7)
    ax1.plot(traj_rk4[:, 0], traj_rk4[:, 1], traj_rk4[:, 2], 'b-', linewidth=2, label='Custom RK4 (Stable Orbit)')
    ax1.scatter([0], [0], [0], color='green', s=120, label='Earth Center')
    ax1.set_title('3D Two-Body Orbital Trajectory')
    ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)'); ax1.set_zlabel('Z (km)')
    ax1.legend()
    ax2 = fig.add_subplot(122)
    ax2.semilogy(t_steps / 3600.0, drift_euler, 'r--', label='Euler Energy Error')
    ax2.semilogy(t_steps / 3600.0, drift_rk4 + 1e-16, 'b-', label='RK4 Energy Error')
    ax2.set_title('Relative Specific Energy Error |dE / E0|')
    ax2.set_xlabel('Simulation Time (Hours)'); ax2.set_ylabel('Relative Error (Log Scale)')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('01_rk4_orbits/orbit_comparison.png', dpi=200)
    print('Saved 01_rk4_orbits/orbit_comparison.png successfully!')
