# Astro-SciML Lab: Orbital Mechanics & Physics-Informed Machine Learning

An Applied Mathematics research and simulation repository bridging **Differential Equations**, **Numerical Analysis**, and **Scientific Machine Learning (SciML)** for astrodynamics.

## Repository Architecture

| Module | Directory | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **01** | [`01_rk4_orbits`](./01_rk4_orbits) | 3D Two-Body & Three-Body ODE integrators (Euler, RK4, Adaptive RK45) built from scratch | Active |
| **02** | [`02_ntrs_notes`](./02_ntrs_notes) | Technical literature reviews from NASA Technical Reports Server (NTRS) & GMAT architecture | Active |
| **03** | [`03_pytorch_pinns`](./03_pytorch_pinns) | Physics-Informed Neural Networks (PINNs) solving differential equations via `torch.autograd` | In Progress |

## Quick Start

```bash
# Run Module 1: 3D Two-Body Keplerian Orbit & Energy Conservation Benchmark
python 01_rk4_orbits/two_body_rk4.py
```
