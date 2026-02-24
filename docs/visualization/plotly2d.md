# plotly2d - 2D Visualization Module

The `plotly2d` module provides tools for creating 2D visualizations including heatmaps, contour plots, filled contours, scatter plots, and line plots for reservoir layer analysis and spatial property visualization. All visualizations are built on Plotly for interactivity.

---

## Overview

While the plotly2d module provides renderers for structured 2D visualization, most users will work directly with Plotly's native functions for flexibility. The module includes:

- [`HeatmapRenderer`](#heatmaprenderer) - Heat

maps for spatial data visualization
- [`ContourRenderer`](#contourrenderer) - Contour and filled contour plots
- [`ScatterRenderer`](#scatterrenderer) - 2D scatter plots for sparse data
- [`PlotConfig`](#plotconfig) - Configuration for appearance customization

---

## Common Use Cases

### Extracting 2D Layer Data

Before visualizing, you need to extract a 2D slice from your 3D reservoir model:

```python
import bores
from pathlib import Path

# Load simulation states
store = bores.ZarrStore(Path("./results/simulation.zarr"))
stream = bores.StateStream(store=store, auto_replay=True)
states = list(stream.replay())

# Get final state
final_state = states[-1]

# Extract a 2D layer (k=0 is the top layer)
oil_sat_layer = final_state.model.fluid_properties.oil_saturation_grid[:, :, 0]
pressure_layer = final_state.model.fluid_properties.pressure_grid[:, :, 1]  # Second layer

print(oil_sat_layer.shape)  # (nx, ny) - 2D array
```

---

## Using Plotly Directly (Recommended)

For maximum flexibility, use Plotly's native functions directly. This is the approach used in the BORES analysis scenarios.

### Basic Heatmap

```python
import plotly.graph_objects as go

# Extract layer data
final_state = states[-1]
oil_sat_layer = final_state.model.fluid_properties.oil_saturation_grid[:, :, 0]

# Create heatmap
fig = go.Figure(
    data=go.Heatmap(
        z=oil_sat_layer,
        colorscale="viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Oil Saturation (fraction)"),
    )
)

fig.update_layout(
    title="Oil Saturation - Top Layer (k=0)",
    xaxis_title="J (cell index)",
    yaxis_title="I (cell index)",
    width=800,
    height=700,
)
fig.show()
```

### Pressure Heatmap with Custom Colorscale

```python
# Extract pressure from layer 2
pressure_layer = final_state.model.fluid_properties.pressure_grid[:, :, 1]

fig = go.Figure(
    data=go.Heatmap(
        z=pressure_layer,
        colorscale=bores.ColorScheme.PLASMA.value,
        colorbar=dict(
            title="Pressure (psia)",
            tickformat=".0f",
        ),
    )
)

fig.update_layout(
    title="Reservoir Pressure - Layer 2",
    xaxis_title="J (cell index)",
    yaxis_title="I (cell index)",
    width=900,
    height=750,
)
fig.show()
```

### Multiple Layer Subplots

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Extract saturation for multiple layers
layers_to_plot = [0, 1, 2]
oil_sat_grid = final_state.model.fluid_properties.oil_saturation_grid

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"Layer {k+1}" for k in layers_to_plot],
    horizontal_spacing=0.08,
)

for idx, k in enumerate(layers_to_plot, start=1):
    fig.add_trace(
        go.Heatmap(
            z=oil_sat_grid[:, :, k],
            colorscale="viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title="Oil Saturation",
                x=1.02 if idx == 3 else None,  # Colorbar on rightmost plot only
            ) if idx == 3 else None,
            showscale=(idx == 3),  # Only show colorbar on last plot
        ),
        row=1,
        col=idx,
    )

fig.update_layout(
    title="Oil Saturation Distribution by Layer",
    height=500,
    width=1600,
)

# Update all subplot axes
for i in range(1, 4):
    fig.update_xaxes(title_text="J (cell index)", row=1, col=i)
    fig.update_yaxes(title_text="I (cell index)", row=1, col=i)

fig.show()
```

### Adding Well Locations to Maps

Wells can be overlaid on heatmaps to show their spatial distribution:

```python
import plotly.graph_objects as go

# Create base pressure heatmap
pressure_layer = final_state.model.fluid_properties.pressure_grid[:, :, 0]

fig = go.Figure(
    data=go.Heatmap(
        z=pressure_layer,
        colorscale="viridis",
        colorbar=dict(title="Pressure (psia)"),
    )
)

# Get wells from state
wells = final_state.wells

# Add production well markers (green circles)
for well in wells.producers:
    # Get first perforation location - structure is ((i,j,k), (i,j,k))
    # Each perforation interval is a tuple of (start_coords, end_coords)
    start_coords, end_coords = well.perforating_intervals[0]
    i, j, k = start_coords  # Use start coordinate

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
            textfont=dict(color="white", size=10),
            name=well.name,
            showlegend=True,
        )
    )

# Add injection well markers (red triangles)
for well in wells.injectors:
    start_coords, end_coords = well.perforating_intervals[0]
    i, j, k = start_coords

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
            textfont=dict(color="white", size=10),
            name=well.name,
            showlegend=True,
        )
    )

fig.update_layout(
    title="Pressure Map with Wells - Top Layer",
    xaxis_title="J (cell index)",
    yaxis_title="I (cell index)",
    height=750,
    width=850,
)
fig.show()
```

### Contour Plots

Contour plots show lines of constant property values:

```python
# Extract pressure data
pressure_layer = final_state.model.fluid_properties.pressure_grid[:, :, 0]

fig = go.Figure(
    data=go.Contour(
        z=pressure_layer,
        colorscale="RdYlBu_r",  # Red (high) to Blue (low)
        colorbar=dict(title="Pressure (psia)"),
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color="white"),
        ),
        line=dict(width=1.5),
    )
)

fig.update_layout(
    title="Pressure Contours - Top Layer",
    xaxis_title="J (cell index)",
    yaxis_title="I (cell index)",
    width=900,
    height=750,
)
fig.show()
```

### Filled Contour Plots

```python
# Filled contours with smooth transitions
saturation_layer = final_state.model.fluid_properties.water_saturation_grid[:, :, 0]

fig = go.Figure(
    data=go.Contour(
        z=saturation_layer,
        colorscale="Blues",
        colorbar=dict(title="Water Saturation (fraction)"),
        contours_coloring="fill",  # Fill between contour lines
        contours=dict(
            start=0.0,
            end=1.0,
            size=0.1,  # Contour interval
            showlabels=True,
        ),
        line=dict(width=0.5, color="black"),
    )
)

fig.update_layout(
    title="Water Saturation - Filled Contours",
    xaxis_title="J (cell index)",
    yaxis_title="I (cell index)",
    width=900,
    height=750,
)
fig.show()
```

### Comparing Properties Side-by-Side

```python
from plotly.subplots import make_subplots

# Get properties from final state
oil_sat = final_state.model.fluid_properties.oil_saturation_grid[:, :, 0]
water_sat = final_state.model.fluid_properties.water_saturation_grid[:, :, 0]
gas_sat = final_state.model.fluid_properties.gas_saturation_grid[:, :, 0]

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Oil Saturation", "Water Saturation", "Gas Saturation"],
    horizontal_spacing=0.08,
)

# Add heatmaps
for idx, (sat, name) in enumerate([(oil_sat, "Oil"), (water_sat, "Water"), (gas_sat, "Gas")], start=1):
    fig.add_trace(
        go.Heatmap(
            z=sat,
            colorscale="viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(
                title=f"{name} Sat.",
                x=1.02 if idx == 3 else None,
            ) if idx == 3 else None,
            showscale=(idx == 3),
        ),
        row=1,
        col=idx,
    )

fig.update_layout(
    title=f"Phase Saturations - Top Layer (Timestep {final_state.step})",
    height=500,
    width=1800,
)

for i in range(1, 4):
    fig.update_xaxes(title_text="J (cell index)", row=1, col=i)
    fig.update_yaxes(title_text="I (cell index)", row=1, col=i)

fig.show()
```

### Merging 2D Maps with bores.merge_plots()

You can also use `bores.merge_plots()` to combine heatmaps:

```python
# Create individual figures
oil_fig = go.Figure(
    data=go.Heatmap(
        z=final_state.model.fluid_properties.oil_saturation_grid[:, :, 0],
        colorscale="viridis",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Oil Sat."),
    )
)
oil_fig.update_layout(title="Oil Saturation", xaxis_title="J", yaxis_title="I")

pressure_fig = go.Figure(
    data=go.Heatmap(
        z=final_state.model.fluid_properties.pressure_grid[:, :, 0],
        colorscale="plasma",
        colorbar=dict(title="Pressure (psia)"),
    )
)
pressure_fig.update_layout(title="Pressure", xaxis_title="J", yaxis_title="I")

# Merge using bores.merge_plots()
combined_fig = bores.merge_plots(
    oil_fig,
    pressure_fig,
    cols=2,
    title="Reservoir Properties - Top Layer",
    height=600,
    width=1600,
)
combined_fig.show()
```

---

## Renderers (Advanced)

For users who need structured 2D rendering with configuration management, the plotly2d module provides renderer classes.

### HeatmapRenderer

Renderer for creating heatmaps with automatic metadata handling.

#### Example

```python
from bores.visualization.plotly2d import HeatmapRenderer, PlotConfig
from bores.visualization.base import PropertyMeta
import plotly.graph_objects as go

# Create configuration
config = PlotConfig(
    width=900,
    height=750,
    show_colorbar=True,
    color_scheme="viridis",
)

# Create renderer
renderer = HeatmapRenderer(config)

# Create empty figure
fig = go.Figure()

# Create property metadata
oil_sat_meta = PropertyMeta(
    name="oil_saturation",
    display_name="Oil Saturation",
    unit="fraction",
    color_scheme="viridis",
    log_scale=False,
    cmin=0.0,
    cmax=1.0,
)

# Extract 2D data
oil_sat_layer = final_state.model.fluid_properties.oil_saturation_grid[:, :, 0]

# Render
fig = renderer.render(
    figure=fig,
    data=oil_sat_layer,
    metadata=oil_sat_meta,
    x_label="J (cell index)",
    y_label="I (cell index)",
)
fig.show()
```

### ContourRenderer

Renderer for contour and filled contour plots.

#### Example

```python
from bores.visualization.plotly2d import ContourRenderer, PlotConfig
from bores.visualization.base import PropertyMeta
import plotly.graph_objects as go

config = PlotConfig(
    width=900,
    height=750,
    contour_levels=15,
    contour_line_width=1.5,
)

# Create filled contour renderer
renderer = ContourRenderer(config, filled=True)

# Metadata
pressure_meta = PropertyMeta(
    name="pressure",
    display_name="Pressure",
    unit="psia",
    color_scheme="plasma",
    log_scale=False,
)

# Extract data
pressure_layer = final_state.model.fluid_properties.pressure_grid[:, :, 0]

# Render
fig = go.Figure()
fig = renderer.render(
    figure=fig,
    data=pressure_layer,
    metadata=pressure_meta,
    x_label="J (cell index)",
    y_label="I (cell index)",
    contour_levels=20,
)
fig.show()
```

---

## PlotConfig

Configuration class for customizing 2D plot appearance.

### Definition

```python
@attrs.frozen
class PlotConfig:
    # Dimensions
    width: int = 800
    height: int = 600

    # Display
    show_colorbar: bool = True
    title: Optional[str] = None

    # Styling
    color_scheme: str = "viridis"
    opacity: float = 0.8

    # Grid and axes
    show_grid: bool = True
    grid_color: str = "lightgray"
    xaxis_grid_color: Optional[str] = None
    yaxis_grid_color: Optional[str] = None
    axis_line_color: str = "black"
    axis_line_width: float = 1.0

    # Text and labels
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 16
    axis_title_font_size: int = 14

    # Colorbar
    colorbar_thickness: int = 20
    colorbar_len: float = 0.8

    # Contour-specific
    contour_line_width: float = 1.5
    contour_levels: int = 20

    # Scatter-specific
    scatter_marker_size: int = 6
    line_width: float = 2.0

    # Margins
    margin_left: int = 80
    margin_right: int = 80
    margin_top: int = 80
    margin_bottom: int = 80

    # Background
    plot_bgcolor: str = "#f8f9fa"
    paper_bgcolor: str = "#ffffff"
```

### Usage

```python
from bores.visualization.plotly2d import PlotConfig, HeatmapRenderer

config = PlotConfig(
    width=1000,
    height=850,
    color_scheme="plasma",
    show_colorbar=True,
    colorbar_thickness=25,
    font_family="Times New Roman, serif",
    font_size=14,
)

renderer = HeatmapRenderer(config)
```

---

## Saving and Exporting

All Plotly figures can be saved in various formats:

```python
# Create figure (any method)
fig = go.Figure(data=go.Heatmap(...))

# Save as interactive HTML
fig.write_html("saturation_map.html")

# Save as static images (requires kaleido)
fig.write_image("saturation_map.png", width=1200, height=900, scale=2)
fig.write_image("saturation_map.pdf", width=1200, height=900)
fig.write_image("saturation_map.svg", width=1200, height=900)
fig.write_image("saturation_map.jpeg", width=1200, height=900, scale=2)

# Display
fig.show()
```

---

## Best Practices

### Layer Selection

Choose representative layers that show key features:

- **Top layer (k=0)**: Surface effects, gas cap behavior
- **Middle layers**: Primary reservoir zone
- **Bottom layer (k=nz-1)**: Water contact, bottom aquifer

### Colorscale Selection

- **Sequential data** (saturation, pressure): Use `viridis`, `plasma`, `inferno`
- **Diverging data** (pressure change): Use `RdYlBu`, `balance`
- **Water properties**: Use `Blues`, `RdBu` (reversed)
- **Oil properties**: Use `YlOrBr`, `oranges`
- **Gas properties**: Use `Greens`, `viridis`

### Resolution

- Use full resolution for publication figures
- Downsample large grids (> 200×200) for quick visualization
- Set appropriate `zmin` and `zmax` for consistent color mapping across timesteps

### Aspect Ratio

Maintain realistic aspect ratios:

```python
fig.update_layout(
    yaxis_scaleanchor="x",  # Link y-axis scale to x-axis
    yaxis_scaleratio=1,     # 1:1 aspect ratio
)
```

---

## See Also

- [Visualization Overview](../guides/visualization.md) - Overview of all visualization modules
- [plotly1d](plotly1d.md) - 1D visualization (time series, line plots)
- [plotly3d](plotly3d.md) - 3D visualization (volume rendering, isosurfaces)
- [Analyzing Results](../guides/analyzing-results.md) - Computing metrics for visualization
