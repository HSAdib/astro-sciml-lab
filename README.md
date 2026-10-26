# Astro-SciML Lab: Orbital Mechanics & Scientific Machine Learning

An Applied Mathematics research and simulation repository bridging **Differential Equations**, **Numerical Analysis**, and **Scientific Machine Learning (SciML)** for astrodynamics, implemented using pure NumPy and SciPy.

## Repository Architecture

| Module | Directory | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **01** | [`01_rk4_orbits`](./01_rk4_orbits) | 3D Two-Body & Three-Body ODE integrators (Euler, RK4, Adaptive RK45) built from scratch | Active |
| **02** | [`02_ntrs_notes`](./02_ntrs_notes) | Technical literature reviews from NASA Technical Reports Server (NTRS) & GMAT architecture | Active |
| **03** | [`03_pytorch_pinns`](./03_pytorch_pinns) | Physics-Informed Collocation & Neural Network architectures built in pure NumPy | Active |

## Quick Start

```bash
# Run Module 1: 3D Two-Body Keplerian Orbit & Energy Conservation Benchmark
python 01_rk4_orbits/two_body_rk4.py

# Run Module 3: Pure NumPy SciML ODE Collocation Solver
python 03_pytorch_pinns/pinn_ode_solver.py
```
