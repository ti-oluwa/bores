# Advanced Topics

Specialized features for advanced reservoir simulation scenarios.

---

## Overview

These guides cover advanced BORES features for specific use cases.

<div class="grid cards" markdown>

-   :material-water-outline:{ .lg .middle } **Boundary Conditions**

    ---

    Aquifer support, constant pressure, no-flow, and periodic boundaries

    [:octicons-arrow-right-24: Read Guide](boundary-conditions.md)

-   :material-vector-line:{ .lg .middle } **Faults & Fractures**

    ---

    Modeling sealing faults, conductive fractures, and transmissibility barriers

    [:octicons-arrow-right-24: Read Guide](faults-fractures.md)

-   :material-table:{ .lg .middle } **PVT Tables**

    ---

    Building and using PVT lookup tables for performance

    [:octicons-arrow-right-24: Read Guide](pvt-tables.md)

-   :material-speedometer:{ .lg .middle } **Performance Optimization**

    ---

    Tips and techniques for faster simulations and lower memory usage

    [:octicons-arrow-right-24: Read Guide](performance-optimization.md)

-   :material-content-save:{ .lg .middle } **Storage & Serialization**

    ---

    Saving large simulations with HDF5, Zarr, checkpointing, and streaming

    [:octicons-arrow-right-24: Read Guide](storage-serialization.md)

</div>

---

## When to Use These Features

### Boundary Conditions

Use when:
- Modeling aquifer support (Carter-Tracy)
- Need constant pressure boundaries
- Testing periodic domains
- Matching field behavior with edge/bottom water drive

### Faults & Fractures

Use when:
- Reservoir has known faults
- Modeling fracture corridors
- Need to reduce transmissibility across barriers
- Studying fault seal capacity

### PVT Tables

Use when:
- Simulations > 100 timesteps
- Large grids (> 10K cells)
- Need to incorporate lab PVT data
- Want faster execution (interpolation vs correlations)

### Performance Optimization

Use when:
- Simulations are too slow
- Running out of memory
- Need to run many scenarios
- Optimizing for production use

### Storage & Serialization

Use when:
- Simulation results > 1 GB
- Need to resume interrupted runs
- Want to post-process later
- Streaming results to disk

---

## Quick Links

- [User Guide](../guides/index.md) - Core features
- [Examples](../examples/index.md) - Complete working code
- [Best Practices](../best-practices/index.md) - Tips and tricks
