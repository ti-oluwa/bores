# User Guide

The User Guide provides detailed documentation for every major subsystem in BORES. Each page covers one topic in depth, with working code examples, parameter references, and practical guidance.

## Sections

| Section | Description |
|---|---|
| [Grids](grids.md) | Grid construction utilities: uniform, layered, depth, structural dip, and saturation grids |
| [Rock Properties](rock-properties.md) | Porosity, permeability (isotropic and anisotropic), and rock compressibility |
| [Fluid Properties](fluid-properties.md) | PVT correlations, oil/gas/water properties, and the reservoir model factory |
| [Relative Permeability](relative-permeability.md) | Brooks-Corey model, Corey exponents, wettability, and three-phase mixing rules |
| [Capillary Pressure](capillary-pressure.md) | Brooks-Corey, Van Genuchten, Leverett J-function, and tabular capillary pressure |
| **Wells** | |
| [Well Basics](wells/basics.md) | Production and injection wells, perforations, well index, skin factor |
| [Well Controls](wells/controls.md) | Rate controls, BHP controls, PrimaryPhaseRateControl, MultiPhaseRateControl |
| [Well Fluids](wells/fluids.md) | Produced fluids, injected fluids, CO2 property overrides |
| [Well Schedules](wells/schedules.md) | Time-dependent well events, predicates, and actions |
| [Well Patterns](wells/patterns.md) | Common well placement patterns for waterfloods and EOR |
| **Simulation** | |
| [Running Simulations](simulation/running.md) | The bores.run() function, iterating over results, early stopping |
| [Time Stepping](simulation/timestepping.md) | Timer class, adaptive control, CFL conditions, ramp-up |
| [Solvers](simulation/solvers.md) | Pressure solvers, preconditioners, CachedPreconditionerFactory |
| [Streaming and Storage](simulation/streaming.md) | StateStream, storage backends, replay |
| [Post-Processing](simulation/post-processing.md) | Analysis utilities, recovery factors, production profiles |
| **Advanced** | |
| [Boundary Conditions](advanced/boundary-conditions.md) | No-flow, constant pressure, periodic, Carter-Tracy aquifer |
| [Faults and Fractures](advanced/faults-fractures.md) | Transmissibility modifiers, fault definitions |
| [PVT Tables](advanced/pvt-tables.md) | Tabular PVT as alternative to correlations |
| [Serialization](advanced/serialization.md) | Saving/loading models, custom type registration |
| [Miscible Flooding](advanced/miscible-flooding.md) | Todd-Longstaff model, MMP, mixing parameter |
| [Performance](advanced/performance.md) | Numba JIT, precision control, grid coarsening |
| [Events System](advanced/events.md) | Custom event predicates and actions |
| [Configuration](config.md) | Complete Config class reference |
| [Constants](constants.md) | Physical constants and the Constants class |

!!! tip "Where to Start"

    If you completed the [Tutorials](../tutorials/), start with the sections most relevant to your current project. The sections on wells, relative permeability, and simulation controls are the most commonly referenced.
