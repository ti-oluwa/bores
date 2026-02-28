# Tutorials

Step-by-step guides that walk you through complete reservoir simulation workflows, from simple depletion to advanced miscible flooding.

---

## Prerequisites

Before starting these tutorials, make sure you have completed the following:

1. **[Installation](../getting-started/installation.md)** - BORES is installed and working in your environment.
2. **[Quickstart](../getting-started/quickstart.md)** - You have run the quickstart example successfully.
3. **[Core Concepts](../getting-started/concepts.md)** - You understand the simulation pipeline, immutable models, and factory functions.

You should also be comfortable with Python and NumPy basics. Each tutorial builds on the previous one, so we recommend following them in order.

---

## Tutorial Roadmap

The tutorials progress from fundamental skills to advanced recovery techniques. Each one introduces new concepts while reinforcing what you have already learned.

<div class="grid cards" markdown>

-   **Your First Simulation**

    ---

    Build a simple depletion model with a single production well. Learn the complete simulation workflow from grid construction to result visualization.

    [:octicons-arrow-right-24: Tutorial 1](01-first-simulation.md)

-   **Building Reservoir Models**

    ---

    Construct realistic heterogeneous models with layered properties, anisotropic permeability, structural dip, and 3D visualization.

    [:octicons-arrow-right-24: Tutorial 2](02-building-models.md)

-   **Waterflood Simulation**

    ---

    Set up a complete waterflood with injection and production wells. Track water breakthrough, water cut, and oil recovery factor.

    [:octicons-arrow-right-24: Tutorial 3](03-waterflood.md)

-   **Gas Injection**

    ---

    Simulate immiscible gas injection and observe gravity override, gas mobility effects, and compare recovery against waterflooding.

    [:octicons-arrow-right-24: Tutorial 4](04-gas-injection.md)

-   **Miscible Gas Flooding**

    ---

    Model miscible displacement with the Todd-Longstaff method. Configure CO2 injection with custom fluid properties and analyze mixing behavior.

    [:octicons-arrow-right-24: Tutorial 5](05-miscible-flooding.md)

</div>

---

## Suggested Reading Order

| Order | Tutorial | What You Will Learn | Time |
|-------|----------|---------------------|------|
| 1 | [Your First Simulation](01-first-simulation.md) | Full simulation workflow, depletion drive, basic visualization | 20 min |
| 2 | [Building Reservoir Models](02-building-models.md) | Heterogeneity, anisotropy, structural dip, 3D visualization | 25 min |
| 3 | [Waterflood Simulation](03-waterflood.md) | Injection wells, water breakthrough, recovery analysis | 30 min |
| 4 | [Gas Injection](04-gas-injection.md) | Gas injectors, gravity effects, gas-oil displacement | 25 min |
| 5 | [Miscible Gas Flooding](05-miscible-flooding.md) | Todd-Longstaff model, CO2 properties, miscible displacement | 30 min |

!!! tip "Learning Approach"

    Each tutorial is self-contained with complete, runnable code. However, the explanations assume familiarity with earlier tutorials. If you jump ahead and encounter an unfamiliar concept, check the earlier tutorials or the [Core Concepts](../getting-started/concepts.md) page.
