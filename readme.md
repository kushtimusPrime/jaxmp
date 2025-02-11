# jaxmp

JAX-based robot library, focusing on modularity and ease of use.

We formulate goals and tasks as a nonlinear least squares problem, and use [jaxls](https://github.com/brentyi/jaxls) to solve it in a sparse manner.

Includes:
- Differentiable forward robot kinematics model, given a URDF from [`yourdfpy`](https://github.com/clemense/yourdfpy/tree/main) as input.
  - Supports a wide range of robots, through [robot-descriptions](https://github.com/robot-descriptions/robot_descriptions.py).
  - Automatic robot collision geometry generation (e.g., with capsules).
- Differentiable collision bodies with numpy broadcasting logic, using a thin wrapper around [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html). 
- Common cost factors (e.g., EE pose, self/world-collision, manipulability).

Supports:
- Arbitrary costs, as long as autodiff Jacobians are feasible.

  
---

## Installation
```
pip install git+https://github.com/kushtimusprime/jaxmp.git
```

To run examples, install with `pip install -e .[examples]`.
