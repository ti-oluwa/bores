# Visualization Module

BORES visualization tools for 1D time series, 2D maps, and 3D volume rendering.

## Modules

- **[plotly1d](plotly1d.md)** - Time series plots, production curves, decline analysis
- **[plotly2d](plotly2d.md)** - Heatmaps, contour plots, layer visualizations
- **[plotly3d](plotly3d.md)** - Volume rendering, isosurfaces, 3D reservoir visualization

## Quick Examples

### 1D Time Series

```python
import bores

# Get production history from analyst
oil_rates = []
time_days = []
for step, rate in analyst.oil_production_history(interval=10):
    state = analyst.get_state(step)
    time_days.append(state.time / 86400)
    oil_rates.append(rate)

# Create time series plot
viz = bores.plotly1d.DataVisualizer()
fig = viz.make_plot(
    data={"Oil Production": oil_rates},
    x_data=time_days,
    x_label="Time (days)",
    y_label="Rate (STB/day)",
    title="Oil Production History",
)
fig.show()
```

### 2D Heatmaps

```python
# Extract pressure map from a specific timestep
final_state = analyst.get_state(-1)
pressure_grid = final_state.model.fluid_properties.pressure_grid
layer_index = 3  # Middle layer

# Create 2D heatmap
viz = bores.plotly2d.DataVisualizer()
fig = viz.make_plot(
    data=pressure_grid[:, :, layer_index],
    plot_type="heatmap",
    title=f"Pressure Distribution (Layer {layer_index})",
    x_label="X Index",
    y_label="Y Index",
)
fig.show()
```

### 3D Volume Rendering

```python
viz = bores.plotly3d.DataVisualizer()
fig = viz.make_plot(
    state,
    property="pressure",
    plot_type="volume",
    opacity=0.7,
)
fig.show()
```

## See Also

- [Visualization Guide](../guides/visualization.md) - Comprehensive usage guide
- [Analyzing Results](../guides/analyzing-results.md) - Computing metrics for visualization
