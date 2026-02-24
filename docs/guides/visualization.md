# Visualization

BORES provides comprehensive 1D, 2D, and 3D visualization capabilities using Plotly for creating interactive, publication-quality visualizations of reservoir simulation results. All visualization tools are designed to work seamlessly with simulation outputs and analysis data.

---

## Overview

The visualization module consists of three main components:

- **1D Visualization** (`plotly1d`) - Time series plots, production curves, decline analysis, and sensitivity analysis
- **2D Visualization** (`plotly2d`) - Heatmaps, contour plots, and layer visualizations
- **3D Visualization** (`plotly3d`) - Volume rendering, isosurfaces, and interactive 3D reservoir visualization

All visualizations support interactive features including zoom, pan, hover tooltips, and export to various formats (HTML, PNG, JPEG, SVG, PDF).

---

## Color Schemes

BORES uses perceptually uniform, colorblind-friendly color schemes by default to ensure accessibility and scientific accuracy.

### Available Color Schemes

The [`ColorScheme`](../reference/api.md#colorscheme) enum provides the following options:

#### Colorblind-Friendly Schemes (Recommended)

- `VIRIDIS` - Blue → green → yellow (perceptually uniform, most recommended)
- `CIVIDIS` - Optimized specifically for colorblind accessibility
- `PLASMA` - Purple → pink → yellow
- `INFERNO` - Black → purple → yellow → white
- `MAGMA` - Black → purple → pink → yellow

#### Diverging Schemes

- `BALANCE` - Balanced diverging with good accessibility
- `RdYlBu` - Red-yellow-blue diverging
- `RdBu` - Red-blue diverging (caution: not ideal for red-green colorblindness)
- `SPECTRAL` - Spectral diverging (not colorblind-friendly)

#### Other Schemes

- `TURBO` - High-contrast rainbow (not colorblind-friendly, use sparingly)
- `EARTH` - Earth tones for geological visualization

```python
import bores

# Access color scheme
color_scheme = bores.ColorScheme.VIRIDIS
# or as string
color_scheme = "viridis"
```

!!! tip "Accessibility"
    For maximum accessibility and scientific rigor, prefer `VIRIDIS`, `CIVIDIS`, `PLASMA`, `INFERNO`, or `MAGMA`.

---

## 1D Visualization (Time Series and Line Plots)

The primary function for creating 1D plots is [`bores.make_series_plot()`](../reference/api.md#make_series_plot), which creates line plots for time series data, production histories, and other 1D data.

### Data Format

The `make_series_plot()` function accepts data in the following formats:

- **Dictionary of series**: `{"Series Name": np.array([[x1, y1], [x2, y2], ...])}`
- **Single series**: `np.array([[x1, y1], [x2, y2], ...])`  (shape: n×2)
- **List of series**: `[series1_array, series2_array, ...]`

Each series must be a 2D NumPy array where each row contains an `(x, y)` pair.

### Basic Usage

```python
import bores
import numpy as np
from pathlib import Path

# Load simulation states
store = bores.ZarrStore(Path("./results/simulation.zarr"))
stream = bores.StateStream(store=store, auto_replay=True)
states = list(stream.replay())

# Create analyst
analyst = bores.ModelAnalyst(states)

# Get production history data
oil_production = analyst.oil_production_history(interval=1, cumulative=False, from_step=1)

# Convert to numpy array format
oil_data = np.array(list(oil_production))  # shape: (n_steps, 2) with columns [timestep, rate]

# Create plot
fig = bores.make_series_plot(
    data={"Oil Production": oil_data},
    title="Oil Production History",
    x_label="Time Step",
    y_label="Oil Rate (STB/day)",
    marker_sizes=6,
    show_markers=True,
    width=800,
    height=600,
)
fig.show()
```

### Multiple Series

```python
# Collect multiple production histories
oil_data = np.array(list(analyst.oil_production_history(interval=1, from_step=1)))
water_data = np.array(list(analyst.water_production_history(interval=1, from_step=1)))
gas_data = np.array(list(analyst.free_gas_production_history(interval=1, from_step=1)))

# Plot all series together
fig = bores.make_series_plot(
    data={
        "Oil Production": oil_data,
        "Water Production": water_data,
        "Gas Production": gas_data,
    },
    title="Production History - All Phases",
    x_label="Time Step",
    y_label="Production Rate (STB/day for Oil/Water, SCF/day for Gas)",
    marker_sizes=6,
    show_markers=True,
    line_colors=["brown", "blue", "green"],
    line_widths=[2.5, 2.0, 2.0],
    width=900,
    height=600,
)
fig.show()
```

### Customization Options

The `make_series_plot()` function supports extensive customization:

```python
fig = bores.make_series_plot(
    data={"Pressure": pressure_data},
    title="Reservoir Pressure Decline",
    x_label="Time Step",
    y_label="Average Pressure (psia)",

    # Line styling
    line_colors="darkblue",  # Single color or list of colors
    line_widths=3.0,  # Single width or list of widths
    line_dashes="solid",  # "solid", "dot", "dash", "dashdot", or list

    # Marker styling
    show_markers=True,
    marker_sizes=8,  # Single size or list of sizes
    marker_symbols="circle",  # "circle", "square", "diamond", etc.

    # Axis scaling
    log_x=False,  # Logarithmic x-axis
    log_y=False,  # Logarithmic y-axis

    # Dimensions
    width=1000,
    height=700,

    # Fill area under curve
    fill="tonexty",  # Fill to next y-value
    fill_colors="rgba(100, 149, 237, 0.2)",  # Semi-transparent fill
)
fig.show()
```

### Saving Plots

```python
# Save as interactive HTML
fig.write_html("production_history.html")

# Save as static image
fig.write_image("production_history.png", width=1200, height=800, scale=2)
fig.write_image("production_history.pdf", width=1200, height=800)
fig.write_image("production_history.svg", width=1200, height=800)
```

### Merging Multiple Plots

Use [`bores.merge_plots()`](../reference/api.md#merge_plots) to combine multiple figures into a single subplot layout:

```python
# Create individual plots
pressure_fig = bores.make_series_plot(
    data={"Avg. Pressure": np.array(pressure_history)},
    title="Pressure Analysis",
    x_label="Time Step",
    y_label="Pressure (psia)",
    marker_sizes=6,
    width=720,
    height=460,
)

saturation_fig = bores.make_series_plot(
    data={
        "Oil Saturation": np.array(oil_sat_history),
        "Water Saturation": np.array(water_sat_history),
        "Gas Saturation": np.array(gas_sat_history),
    },
    title="Saturation Analysis",
    x_label="Time Step",
    y_label="Saturation (fraction)",
    marker_sizes=6,
    width=720,
    height=460,
)

# Merge into single figure with 2 columns
combined_fig = bores.merge_plots(
    pressure_fig,
    saturation_fig,
    cols=2,
    title="Reservoir Analysis",
    height=500,
    width=1400,
)
combined_fig.show()
```

The `merge_plots()` function automatically handles subplot layout, spacing, and legend positioning.

---

## 2D Visualization (Maps and Heatmaps)

The 2D visualization module provides tools for creating heatmaps, contour plots, and layer visualizations. While BORES provides 2D renderers, most users will work directly with Plotly for custom 2D visualizations.

### Basic Heatmap Example

```python
import bores

# Get final state
final_state = states[-1]

# Extract oil saturation for a specific layer (k=0 is top layer)
oil_sat_layer = final_state.model.fluid_properties.oil_saturation_grid[:, :, 0]

# Create heatmap using BORES 2D visualizer
viz = bores.plotly2d.DataVisualizer()
fig = viz.make_plot(
    data=oil_sat_layer,
    plot_type="heatmap",
    title="Oil Saturation - Top Layer",
    x_label="J (cell index)",
    y_label="I (cell index)",
    colorbar_title="Oil Saturation",
)
fig.show()
```

### Multiple Layers with Subplots

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Extract saturation data for three layers
layers_to_plot = [0, 1, 2]
oil_sat_grid = final_state.model.fluid_properties.oil_saturation_grid

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"Layer {k+1}" for k in layers_to_plot],
)

for idx, k in enumerate(layers_to_plot, start=1):
    fig.add_trace(
        go.Heatmap(
            z=oil_sat_grid[:, :, k],
            colorscale="viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="So") if idx == 3 else None,  # Only show colorbar on last plot
        ),
        row=1,
        col=idx,
    )

fig.update_layout(
    title="Oil Saturation by Layer",
    height=400,
    width=1400,
)
fig.show()
```

### Adding Well Locations

```python
import bores
import plotly.graph_objects as go

# Create pressure heatmap using BORES 2D visualizer
pressure_layer = final_state.model.fluid_properties.pressure_grid[:, :, 0]

viz = bores.plotly2d.DataVisualizer()
fig = viz.make_plot(
    data=pressure_layer,
    plot_type="heatmap",
    title="Pressure Map with Wells - Top Layer",
    x_label="J (cell index)",
    y_label="I (cell index)",
    colorbar_title="Pressure (psia)",
)

# Add well markers
wells = final_state.wells

# Production wells (green circles)
for well in wells.producers:
    # Get first perforation location
    i, j, k = well.perforating_intervals[0][0]  # First interval, start coordinate
    fig.add_trace(
        go.Scatter(
            x=[j],
            y=[i],
            mode="markers+text",
            marker=dict(
                size=15,
                color="green",
                symbol="circle",
                line=dict(width=2, color="white"),
            ),
            text=[well.name],
            textposition="top center",
            name=well.name,
            showlegend=True,
        )
    )

# Injection wells (red triangles)
for well in wells.injectors:
    i, j, k = well.perforating_intervals[0][0]
    fig.add_trace(
        go.Scatter(
            x=[j],
            y=[i],
            mode="markers+text",
            marker=dict(
                size=15,
                color="red",
                symbol="triangle-up",
                line=dict(width=2, color="white"),
            ),
            text=[well.name],
            textposition="top center",
            name=well.name,
            showlegend=True,
        )
    )

# Optionally customize figure size
fig.update_layout(height=700, width=800)
fig.show()
```

---

## 3D Visualization (Volume Rendering and Isosurfaces)

The 3D visualization module provides the [`DataVisualizer`](../reference/api.md#datavisualizer) class for creating interactive 3D visualizations of reservoir properties.

### Getting Started with 3D Visualization

```python
import bores

# Load simulation states
store = bores.ZarrStore("./results/simulation.zarr")
stream = bores.StateStream(store=store, auto_replay=True)
states = list(stream.replay())

# Get a specific timestep
state = states[100]

# Create visualizer
viz = bores.plotly3d.DataVisualizer()

# Create 3D plot
fig = viz.make_plot(
    state,
    property="pressure",
    plot_type="volume",
    title="Pressure Distribution at Timestep 100",
    width=1200,
    height=960,
    opacity=0.7,
)
fig.show()
```

### Available Plot Types

The `plot_type` parameter in `make_plot()` accepts:

- `"volume"` - Volume rendering showing continuous data distribution
- `"isosurface"` - Isosurface plots showing surfaces at specific data values
- `"scatter_3d"` - 3D scatter plots for point data
- `"cell_blocks"` - Individual reservoir cells rendered as 3D blocks

### Available Properties

Common property names for visualization (see [`PropertyRegistry`](../reference/api.md#propertyregistry) for full list):

- `"pressure"` - Reservoir pressure
- `"oil-saturation"` - Oil saturation
- `"water-saturation"` - Water saturation
- `"gas-saturation"` - Gas saturation
- `"porosity"` - Rock porosity
- `"permeability-x"`, `"permeability-y"`, `"permeability-z"` - Directional permeabilities
- `"oil-viscosity"` - Oil viscosity
- `"gas-viscosity"` - Gas viscosity
- `"oil-density"` - Oil density
- `"gas-compressibility-factor"` - Gas Z-factor
- `"oil-formation-volume-factor"` - Oil FVF (Bo)
- `"gas-formation-volume-factor"` - Gas FVF (Bg)
- `"oil-effective-viscosity"` - Todd-Longstaff effective oil viscosity (miscible)
- `"oil-effective-density"` - Todd-Longstaff effective oil density (miscible)
- `"solvent-concentration"` - Solvent concentration in oil phase (miscible)

### Isosurface Visualization

Isosurfaces are useful for visualizing specific value thresholds, such as the oil-water contact or gas front:

```python
# Visualize water front at 50% saturation
fig = viz.make_plot(
    state,
    property="water-saturation",
    plot_type="isosurface",
    title="Water Front (Sw = 0.5)",
    isomin=0.5,
    isomax=0.5,
    opacity=0.7,
    width=1200,
    height=960,
)
fig.show()
```

### Data Slicing

Extract and visualize specific regions using slice parameters:

```python
# Visualize top 10 layers only
fig = viz.make_plot(
    state,
    property="oil-saturation",
    plot_type="volume",
    title="Oil Saturation - Top 10 Layers",
    z_slice=(0, 10),  # Slice layers 0-9
    opacity=0.8,
)
fig.show()

# Visualize corner section
fig = viz.make_plot(
    state,
    property="pressure",
    plot_type="isosurface",
    x_slice=(0, 20),  # First 20 cells in X
    y_slice=(0, 20),  # First 20 cells in Y
    z_slice=(0, 5),   # Top 5 layers
    title="Pressure - Corner Section",
)
fig.show()
```

### Well Visualization

Display wells in 3D plots:

```python
fig = viz.make_plot(
    state,
    property="pressure",
    plot_type="isosurface",
    title="Pressure with Wells",
    show_wells=True,
    show_perforations=True,
    show_surface_marker=True,
    injection_color="#ff4444",
    production_color="#44dd44",
    wellbore_width=15.0,
    marker_size=12,
    opacity=0.7,
)
fig.show()
```

### Adding Labels

Use the [`Labels`](../reference/api.md#labels) class to add text annotations to 3D plots:

```python
# Create labels for wells
wells = state.wells
injector_locations, producer_locations = wells.locations
injector_names, producer_names = wells.names

well_positions = [*injector_locations, *producer_locations]
well_names = [*injector_names, *producer_names]

# Create labels object
labels = bores.plotly3d.Labels()
labels.add_well_labels(well_positions, well_names)

# Use in visualization
fig = viz.make_plot(
    state,
    property="oil-saturation",
    plot_type="volume",
    title="Oil Saturation with Well Labels",
    labels=labels,
    show_wells=True,
    opacity=0.6,
)
fig.show()
```

### Customizing 3D Plots

The `make_plot()` method accepts additional keyword arguments for customization:

```python
fig = viz.make_plot(
    state,
    property="oil-effective-viscosity",
    plot_type="isosurface",
    title="Oil Effective Viscosity Profile",

    # Isosurface parameters
    isomin=0.5,  # Minimum isosurface value
    isomax=2.0,  # Maximum isosurface value

    # Visual styling
    opacity=0.75,
    cmin=0.0,  # Minimum colorbar value
    cmax=3.0,  # Maximum colorbar value

    # Dimensions
    width=1260,
    height=960,

    # Scene aspect ratio
    aspect_mode="data",  # Use actual data aspect ratios
    z_scale=2.0,  # Scale Z axis by factor of 2

    # Wells
    show_wells=True,
    show_surface_marker=True,
    show_perforations=True,
    marker_size=12,
)
fig.show()
```

### Complete 3D Visualization Example

Here's a comprehensive example showing multiple timesteps:

```python
import bores

# Load states
store = bores.ZarrStore("./results/simulation.zarr")
stream = bores.StateStream(store=store, auto_replay=True)
states = list(stream.replay())

# Create visualizer
viz = bores.plotly3d.DataVisualizer()

# Prepare well labels
wells = states[0].wells
injector_locations, producer_locations = wells.locations
injector_names, producer_names = wells.names
well_positions = [*injector_locations, *producer_locations]
well_names = [*injector_names, *producer_names]

labels = bores.plotly3d.Labels()
labels.add_well_labels(well_positions, well_names)

# Shared visualization parameters
shared_kwargs = dict(
    plot_type="isosurface",
    width=1260,
    height=960,
    opacity=0.7,
    labels=labels,
    aspect_mode="data",
    z_scale=1.5,
    marker_size=12,
    show_wells=True,
    show_surface_marker=True,
    show_perforations=True,
)

# Visualize multiple timesteps
property_name = "oil-saturation"
timesteps = [0, 50, 100, 150]
figures = []

for timestep in timesteps:
    fig = viz.make_plot(
        states[timestep],
        property=property_name,
        title=f"Oil Saturation at Timestep {timestep}",
        **shared_kwargs,
    )
    figures.append(fig)

# Merge into single visualization
combined_fig = bores.merge_plots(*figures, cols=2, height=1000, width=2000)
combined_fig.show()

# Save
combined_fig.write_html("oil_saturation_evolution.html")
```

---

## Configuration

### Image Export Configuration

Set global image export settings:

```python
# Configure export resolution
bores.image_config(scale=3, width=1920, height=1080)

# Now all figure.write_image() calls use these settings by default
```

### Visualization Environment Variables

The visualization module respects several environment variables for performance tuning:

- `BORES_MAX_VOLUME_CELLS` - Maximum cells for volume rendering (default: 1,000,000)
- `BORES_RECOMMENDED_VOLUME_CELLS` - Target cells after auto-coarsening (default: 512,000)
- `BORES_MAX_ISOSURFACE_CELLS` - Maximum cells for isosurface rendering (default: 2,000,000)
- `BORES_COLORBAR_THICKNESS` - Colorbar thickness in pixels (default: 15)
- `BORES_MARKER_LINE_WIDTH` - Marker outline width (default: 1)

Set these before importing BORES:

```bash
export BORES_MAX_VOLUME_CELLS=2000000
export BORES_COLORBAR_THICKNESS=20
```

---

## Best Practices

### Performance Optimization

1. **Downsample large grids** - For grids with > 1M cells, use slicing to reduce data before 3D visualization
2. **Use appropriate plot types** - Isosurfaces are faster than volume rendering for large datasets
3. **Limit animation frames** - Sample every 5th or 10th timestep rather than all timesteps
4. **Export format selection** - Use HTML for interactive exploration, PNG/PDF for publications

### Visual Design

1. **Use colorblind-friendly schemes** - Always use `VIRIDIS`, `CIVIDIS`, `PLASMA`, `INFERNO`, or `MAGMA`
2. **Include units** - Always label axes with units in parentheses (e.g., "Pressure (psia)")
3. **Descriptive titles** - Include timestep, property name, and context in titles
4. **Consistent colorbars** - Use `cmin` and `cmax` to maintain consistent scales across timesteps
5. **Appropriate opacity** - Use 0.6-0.8 opacity for volume rendering to see internal features

### Layout and Composition

1. **Aspect ratios** - Use `aspect_mode="data"` for geologically accurate 3D views
2. **Z-axis scaling** - Adjust `z_scale` parameter to avoid vertical exaggeration
3. **Legend placement** - Use `merge_plots()` for automatic legend management in multi-panel figures
4. **Resolution** - Export at 2-3x scale for high-DPI displays and publications

---

## See Also

- [Analyzing Results](analyzing-results.md) - Computing metrics for visualization
- [States, Streams, and Stores](states-streams-stores.md) - Loading simulation data
- [Examples](../examples/primary-depletion.md) - Complete working examples with visualizations
