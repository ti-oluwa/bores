# User Guide

Comprehensive reference for all BORES features.

---

## Overview

The user guide provides detailed documentation for each major component of BORES. Use this as a reference when building your simulations.

---

## Core Components

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } **Building Models**

    ---

    Learn to construct reservoir models with grids, properties, and structure

    [:octicons-arrow-right-24: Read Guide](building-models.md)

-   :material-water:{ .lg .middle } **Rock & Fluid Properties**

    ---

    Configure PVT properties, rel perm, capillary pressure, and correlations

    [:octicons-arrow-right-24: Read Guide](rock-fluid-properties.md)

-   :material-gas-cylinder:{ .lg .middle } **Wells & Controls**

    ---

    Add injection and production wells with various control strategies

    [:octicons-arrow-right-24: Read Guide](wells-and-controls.md)

-   :material-play-circle:{ .lg .middle } **Running Simulations**

    ---

    Configure solvers, timesteps, and execute simulations

    [:octicons-arrow-right-24: Read Guide](running-simulations.md)

-   :material-chart-line:{ .lg .middle } **Analyzing Results**

    ---

    Extract production data, calculate recovery factors, and track fronts

    [:octicons-arrow-right-24: Read Guide](analyzing-results.md)

-   :material-chart-box:{ .lg .middle } **Visualization**

    ---

    Create 1D plots, 2D maps, and interactive 3D visualizations

    [:octicons-arrow-right-24: Read Guide](visualization.md)

</div>

---

## Quick Reference

### Common Tasks

| Task | Guide Section |
|------|---------------|
| Create heterogeneous permeability | [Building Models → Grid Builders](building-models.md#grid-builders) |
| Initialize saturations with contacts | [Building Models → Saturation Init](building-models.md#saturation-initialization) |
| Use PVT correlations | [Rock & Fluid → PVT Correlations](rock-fluid-properties.md#pvt-correlations) |
| Build PVT tables | [Rock & Fluid → PVT Tables](rock-fluid-properties.md#pvt-tables) |
| Add production well | [Wells → Production Wells](wells-and-controls.md#production-wells) |
| Add water injector | [Wells → Injection Wells](wells-and-controls.md#injection-wells) |
| Schedule well changes | [Wells → Scheduling](wells-and-controls.md#well-scheduling) |
| Choose solver | [Running → Solvers](running-simulations.md#pressure-solvers) |
| Stream large results | [Running → State Streams](running-simulations.md#state-streams) |
| Calculate recovery factor | [Analysis → Recovery Factors](analyzing-results.md#recovery-factors) |
| Plot pressure maps | [Visualization → 2D Maps](visualization.md#2d-maps) |
| Render 3D saturation | [Visualization → 3D Volume](visualization.md#3d-volume-rendering) |

---

## Advanced Topics

For specialized features:

- **Boundary Conditions (see advanced/boundary-conditions.md)** - Aquifer support, periodic boundaries
- **[Faults & Fractures](../advanced/faults-fractures.md)** - Transmissibility modifications
- **[Performance Optimization](../advanced/performance-optimization.md)** - Speed and memory tips
- **[Storage & Serialization](../advanced/storage-serialization.md)** - HDF5, Zarr, checkpointing

---

## How to Use This Guide

This guide is organized by workflow:

1. **Start with [Building Models](building-models.md)** - Learn grid construction
2. **Then [Rock & Fluid Properties](rock-fluid-properties.md)** - Configure physics
3. **Add [Wells & Controls](wells-and-controls.md)** - Define injection/production
4. **Configure [Running Simulations](running-simulations.md)** - Set up solvers
5. **Use [Analyzing Results](analyzing-results.md)** - Extract data
6. **Visualize with [Visualization](visualization.md)** - Create plots

!!! tip "Search"
    Use the search bar (top right) to quickly find specific topics like "relative permeability" or "well control".

---

## Need Help?

- **Tutorials**: For step-by-step learning, see [Tutorials](../tutorials/index.md)
- **Examples**: For complete working code, see [Examples](../examples/index.md)
- **API Reference**: For function signatures, see [API Reference](../reference/api.md)

---

**Ready to dive in?** Start with [Building Models →](building-models.md)
