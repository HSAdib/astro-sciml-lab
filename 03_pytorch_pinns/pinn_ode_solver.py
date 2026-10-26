import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training PINN on device: {device}')

# 1. Exact Analytical Solution for Verification (Underdamped Oscillator)
def exact_solution(t, zeta=0.15, omega0=4.0):
    omega_d = omega0 * np.sqrt(1.0 - zeta**2)
    return np.exp(-zeta * omega0 * t) * (np.cos(omega_d * t) + (zeta * omega0 / omega_d) * np.sin(omega_d * t))

# 2. Physics-Informed Neural Network Architecture
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, t):
        return self.net(t)

if __name__ == '__main__':
    torch.manual_seed(42)
    zeta, omega0 = 0.15, 4.0
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Collocation points in domain t in [0, 3] where ODE residual is enforced
    t_physics = torch.linspace(0.0, 3.0, 300, device=device).view(-1, 1).requires_grad_(True)
    # Boundary/Initial condition at t = 0: u(0) = 1, du/dt(0) = 0
    t_ic = torch.tensor([[0.0]], device=device, requires_grad=True)

    for epoch in range(1, 4001):
        optimizer.zero_grad()

        # Initial Condition Loss
        u_ic = model(t_ic)
        du_dt_ic = torch.autograd.grad(u_ic, t_ic, torch.ones_like(u_ic), create_graph=True)[0]
        loss_ic = (u_ic - 1.0)**2 + (du_dt_ic - 0.0)**2

        # Physics ODE Residual Loss: u_tt + 2*zeta*omega0*u_t + omega0^2 * u = 0
        u = model(t_physics)
        du_dt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        d2u_dt2 = torch.autograd.grad(du_dt, t_physics, torch.ones_like(du_dt), create_graph=True)[0]
        ode_residual = d2u_dt2 + 2.0 * zeta * omega0 * du_dt + (omega0 ** 2) * u
        loss_physics = torch.mean(ode_residual ** 2)

        loss = 10.0 * loss_ic + loss_physics
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}/4000 | Total Loss: {loss.item():.6e} | ODE Residual: {loss_physics.item():.6e}')

    # Evaluate and Plot
    t_test = np.linspace(0.0, 3.0, 400)
    u_exact = exact_solution(t_test, zeta, omega0)
    with torch.no_grad():
        u_pred = model(torch.tensor(t_test, dtype=torch.float32, device=device).view(-1, 1)).cpu().numpy().flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(t_test, u_exact, 'k-', linewidth=2.5, label='Exact Analytical Solution')
    plt.plot(t_test, u_pred, 'r--', linewidth=2, label='PINN Prediction (Learned from ODE Residual)')
    plt.scatter([0.0], [1.0], color='blue', s=80, zorder=5, label='Single Initial Condition Point (t=0)')
    plt.title('Physics-Informed Neural Network (PINN) Solving a 2nd-Order ODE')
    plt.xlabel('Time (t)'); plt.ylabel('State u(t)')
    plt.grid(True, ls='--', alpha=0.5); plt.legend()
    plt.tight_layout()
    plt.savefig('03_pytorch_pinns/pinn_solution.png', dpi=200)
    print('Saved 03_pytorch_pinns/pinn_solution.png successfully!')
