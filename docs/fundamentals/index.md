# Fundamentals

Before you start building reservoir models and running simulations, it helps to understand the core concepts that BORES is built on. This section provides background knowledge for users who may be new to reservoir simulation, new to Python-based scientific computing, or both.

If you are a **petroleum engineer**, you will find explanations of how BORES translates familiar reservoir engineering concepts into Python code - how grids are stored as NumPy arrays, how the simulation loop works as a Python generator, and how the IMPES method is implemented under the hood.

If you are a **Python developer**, you will find approachable explanations of the petroleum engineering fundamentals - what reservoir simulation actually models, why we care about fluid properties at different pressures, and what the black-oil model assumes about the physics of underground flow.

Either way, spending 20 minutes here will save you hours of confusion later.

## What You Will Learn

| Page | Description |
|------|-------------|
| [What is Reservoir Simulation?](reservoir-simulation.md) | The big picture - why we simulate reservoirs, what physics are involved, and where BORES fits in the landscape of simulation tools. |
| [The Black-Oil Model](black-oil-model.md) | The fluid model at the heart of BORES - three phases, solution gas, formation volume factors, and PVT relationships. |
| [Grid Systems](grid-systems.md) | How BORES divides a reservoir into discrete cells, stores properties on those cells, and handles boundaries with ghost-cell padding. |
| [How BORES Runs a Simulation](simulation-workflow.md) | A step-by-step walkthrough of the IMPES method - from pressure solve to saturation update to convergence checking. |

!!! tip
    You do not need to read these pages in order, but if you are brand new to reservoir simulation, starting with [What is Reservoir Simulation?](reservoir-simulation.md) and working through sequentially will give you the smoothest experience.
