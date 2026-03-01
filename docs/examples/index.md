# Examples

## Overview

This section provides complete, runnable simulation examples that demonstrate BORES in realistic workflows. Each example includes the full model setup, simulation execution, results analysis, and visualization, with explanations of the physical reasoning behind each step.

The examples are designed to be self-contained. You can copy any example into a Python script and run it directly. All examples use the BORES public API with correct imports, and produce output that you can inspect, modify, and build upon for your own work.

If you are new to BORES, start with the [Tutorials](../tutorials/index.md) section, which introduces concepts progressively. The examples here assume you are comfortable with the basics (model construction, well placement, running simulations) and focus on complete end-to-end workflows that illustrate how all the pieces fit together.

---

## Running Examples

All examples use standard BORES imports and produce results using the simulation API. To run any example, make sure you have BORES installed:

```bash
uv add bores-framework
```

Then copy the code into a Python script or Jupyter notebook and execute it. Most examples complete in under a minute on a modern laptop. Examples that produce visualizations use the BORES visualization API (`bores.make_series_plot`, `bores.visualization.plotly2d`, `bores.visualization.plotly3d`), which requires Plotly to be installed (included as a BORES dependency).

---

## Available Examples

The `benchmarks/` directory in the BORES repository contains complete simulation setups, including the SPE benchmark cases:

| Example | Description | Physics |
| --- | --- | --- |
| SPE 1 | Gas reservoir depletion | Single-phase gas, compressibility, PVT |

Additional examples are available in the `scenarios/` directory as marimo notebooks that you can run interactively.

---

## Example Structure

Each complete example follows a consistent structure that mirrors a real simulation workflow:

1. **Model initialization**: Define the grid, rock properties, and fluid properties using `bores.reservoir_model()`.
2. **Well configuration**: Place wells, assign controls, and optionally define schedules using `bores.injection_well()`, `bores.production_well()`, and `bores.wells_()`.
3. **Simulation setup**: Create a `Config` with timer, wells, solver, and other parameters.
4. **Execution**: Run the simulation and collect results using `bores.run()` or `StateStream`.
5. **Analysis**: Compute recovery factors, plot production profiles, and visualize spatial distributions.

---

## Building Your Own Examples

When building your own simulation cases, refer to the following documentation:

- [Building Reservoir Models](../tutorials/02-building-models.md) for detailed model construction guidance
- [User Guide](../user-guide/index.md) for comprehensive coverage of every component
- [Best Practices](../best-practices/index.md) for grid design, timestep, and solver recommendations
- [API Reference](../api-reference/index.md) for function signatures and parameter details

If you create an interesting simulation case that demonstrates BORES capabilities, consider contributing it to the project.
