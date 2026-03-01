# Best Practices

## Overview

Building a reservoir simulation model involves dozens of decisions, from grid resolution to solver tolerances to timestep control. Each choice affects accuracy, performance, and stability in ways that are not always obvious. This section distills practical guidance from numerical simulation experience into actionable recommendations that help you make informed decisions for your specific problem.

The advice here is organized by topic. Each page addresses a specific area of model construction or simulation execution, provides rules of thumb for common situations, and explains the reasoning behind each recommendation so you can adapt the guidance to unusual cases. The pages are self-contained, so you can read them in any order. That said, if you are setting up your first model, reading them in the order listed below will give you a natural progression from spatial discretization through solver configuration to validation.

These are not rigid rules. Every reservoir is different, and what works well for a homogeneous sandstone waterflood may not be appropriate for a fractured carbonate under miscible gas injection. Use these recommendations as starting points, and always validate your specific model against analytical solutions or benchmark cases when possible. The [Validation](validation.md) page covers how to do exactly that.

---

## Pages

- [Grid Design](grid-design.md) - Cell sizing, aspect ratios, vertical resolution, and when to refine
- [Timestep Selection](timestep-selection.md) - Initial timestep, adaptive control, CFL limits, and stability
- [Solver Selection](solver-selection.md) - Choosing solvers and preconditioners for different problems
- [Error Handling](errors.md) - Understanding and resolving common simulation errors
- [Performance](performance.md) - Precision settings, grid size, preconditioner caching, and memory
- [Validation](validation.md) - Material balance checks, analytical comparisons, and benchmarking
