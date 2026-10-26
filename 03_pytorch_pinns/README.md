# Module 3: Physics-Informed Neural Networks (PINNs) & SciML

This module implements a **Physics-Informed Neural Network (PINN)** in PyTorch that solves a second-order ordinary differential equation without labeled trajectory data inside the domain, relying solely on the initial condition at $t=0$ and automatic differentiation (`torch.autograd`).

## 1. Governing Differential Equation

$$\frac{d^2 u}{dt^2} + 2\zeta\omega_0\frac{du}{dt} + \omega_0^2 u = 0, \quad u(0)=1, \quad \dot{u}(0)=0$$

## 2. Physics-Informed Loss Function

$$\mathcal{L}_{\text{total}} = w_{\text{IC}} \left( |u_\theta(0) - 1|^2 + |\dot{u}_\theta(0)|^2 \right) + \frac{1}{N}\sum_{i=1}^{N} \left| \ddot{u}_\theta(t_i) + 2\zeta\omega_0\dot{u}_\theta(t_i) + \omega_0^2 u_\theta(t_i) \right|^2$$

## 3. Results

![PINN vs Exact ODE Solution](pinn_solution.png)

```bash
python pinn_ode_solver.py
```
