# Module 2: NASA Technical Reports Server (NTRS) Reading Log

This directory tracks weekly literature reviews from [NASA NTRS](https://ntrs.nasa.gov) and [NASA GitHub](https://github.com/nasa), connecting undergraduate Applied Mathematics coursework (ODEs, Vector Analysis, Linear Algebra, Numerical Methods) to production mission design.

## The 15-30-45 Minute Audit Protocol
1. **15 Mins (Problem Scope):** Abstract, Introduction, Conclusion — identify the mission constraint.
2. **30 Mins (Mathematical Model):** Extract the state vector, governing differential equations, and coordinate frames.
3. **45 Mins (Numerical Implementation):** Document the solver architecture, step-size control, and error bounds.

---

## Log 01: Numerical Integration & Energy Drift in Orbital Propagation
* **Topic Search Query:** `"Runge-Kutta orbital propagation"` / `"Symplectic integrators astrodynamics"`
* **Reference NASA Tool:** [NASA General Mission Analysis Tool (GMAT)](https://github.com/nasa/GMAT)

### 1. Mission Problem Statement
Long-duration satellite missions require propagating the 6D state vector $\mathbf{Y} = [\mathbf{r}, \mathbf{v}]^T$ over thousands of orbits. Standard explicit integrators accumulate local truncation error $\mathcal{O}(\Delta t^{p+1})$, causing secular drift in total mechanical energy $\mathcal{E}$.

### 2. Mathematical Connection to Syllabus
* **Fixed-Step RK4:** Local truncation error is $\mathcal{O}(\Delta t^5)$ and global error is $\mathcal{O}(\Delta t^4)$. While vastly superior to Forward Euler $\mathcal{O}(\Delta t)$, RK4 is not strictly symplectic—over millions of steps, phase-space volume is not exactly conserved.
* **Adaptive Runge-Kutta-Fehlberg (RK45 / Prince-Dormand RK89):** Used in NASA GMAT to dynamically shrink $\Delta t$ near perigee (where $\|\mathbf{a}\| \propto r^{-2}$ changes rapidly) and expand $\Delta t$ near apogee.

### 3. Actionable Takeaway for `01_rk4_orbits`
Upgrade the fixed-step `rk4_step` solver in Week 3 to an embedded adaptive **RK45** step controller using local error estimation $\epsilon = \|\mathbf{Y}_{5} - \mathbf{Y}_{4}\|$.
