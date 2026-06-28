
# Solving 2D Navier–Stokes Equations Using Physics-Informed Neural Networks (PINNs)

A deep learning-based approach to solving the two-dimensional incompressible Navier–Stokes equations using Physics-Informed Neural Networks (PINNs). Instead of relying on traditional numerical solvers such as Finite Difference or Finite Element Methods, this project embeds the governing physical laws directly into the neural network loss function to predict velocity and pressure fields.

---

## Overview

Physics-Informed Neural Networks (PINNs) combine deep learning with physical constraints by incorporating partial differential equations (PDEs) into the training process. This project demonstrates the application of PINNs to solve the steady-state 2D incompressible Navier–Stokes equations for the lid-driven cavity flow problem.

The neural network learns the solution by minimizing:

- Continuity equation residual
- Momentum equation residuals
- Boundary condition losses

without requiring labeled simulation data.

---

## Features

- Physics-informed neural network implemented in PyTorch
- Solves 2D incompressible Navier–Stokes equations
- Automatic differentiation for PDE residual computation
- Implements lid-driven cavity flow boundary conditions
- Fully connected neural network with Tanh activation
- GPU support using CUDA (if available)
- Visualization of velocity and pressure fields

---

## Governing Equations

### Continuity Equation

∂u/∂x + ∂v/∂y = 0

### Momentum Equations

u ∂u/∂x + v ∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)

u ∂v/∂x + v ∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)

where

- u = x-velocity
- v = y-velocity
- p = pressure
- ν = kinematic viscosity

---

## Methodology

1. Generate collocation points inside the computational domain.
2. Apply boundary conditions on all cavity walls.
3. Construct a fully connected neural network.
4. Use automatic differentiation to compute PDE derivatives.
5. Minimize the combined physics and boundary losses.
6. Predict velocity and pressure fields across the domain.

---

## Project Structure

```
Navier-Stokes-PINN/
│
├── data/
│
├── models/
│   └── pinn.py
│
├── training/
│   └── train.py
│
├── utils/
│   ├── boundary_conditions.py
│   ├── losses.py
│   └── visualization.py
│
├── results/
│
├── requirements.txt
│
└── README.md
```

---

## Neural Network Architecture

| Layer | Configuration |
|--------|---------------|
| Input | (x, y) |
| Hidden Layers | Fully Connected |
| Activation | Tanh |
| Output | (u, v, p) |

---

## Loss Function

The total loss is computed as

```
Loss =
Physics Loss
+ Boundary Loss
```

where

- Physics Loss enforces the Navier–Stokes equations
- Boundary Loss enforces velocity boundary conditions

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Automatic Differentiation (Autograd)

---

## Results

The model successfully learns:

- Velocity field (u)
- Velocity field (v)
- Pressure distribution

while satisfying the governing Navier–Stokes equations without supervised training data.

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Navier-Stokes-PINN.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python train.py
```

---

## Future Improvements

- Time-dependent Navier–Stokes equations
- Higher Reynolds number flows
- Adaptive collocation sampling
- Turbulence modeling
- 3D flow simulations
- Comparison with CFD solvers (OpenFOAM/ANSYS)

---

## Applications

- Computational Fluid Dynamics (CFD)
- Aerospace Engineering
- Heat Transfer
- Fluid Mechanics
- Scientific Machine Learning
- Physics-Informed Deep Learning

---

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
- PyTorch Documentation
