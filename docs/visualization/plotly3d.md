# plotly3d - 3D Visualization Module

The `plotly3d` module provides comprehensive tools for creating interactive 3D visualizations of reservoir simulation data including volume rendering, isosurface plots, scatter plots, and cell block visualizations. This is the primary module for visualizing full 3D reservoir models.

---

## Overview

The module centers around the [`DataVisualizer`](#datavisualizer) class which provides a high-level interface for 3D visualization. Key components include:

- [`DataVisualizer`](#datavisualizer) - Main visualization class with `make_plot()` method
- [`Labels`](#labels) - Text annotation system for 3D plots
- [`PlotType`](#plottype) - Enumeration of available 3D plot types
- [`PlotConfig`](#plotconfig) - Configuration for appearance and behavior
- [`WellKwargs`](#wellkwargs) - Well visualization customization options

---

## Getting Started

### Basic Usage

```python
import bores
from pathlib import Path

# Load simulation states
store = bores.ZarrStore(Path("./results/simulation.zarr"))
stream = bores.StateStream(store=store, auto_replay=True)
states = list(stream.replay())

# Get a specific state
state = states[100]

# Create visualizer
viz = bores.plotly3d.DataVisualizer()

# Create 3D visualization
fig = viz.make_plot(
    state,
    property="pressure",
    plot_type="volume",
    title="Reservoir Pressure Distribution",
    width=1200,
    height=960,
    opacity=0.7,
)
fig.show()
```

---

## DataVisualizer

The `DataVisualizer` class is the main entry point for creating 3D visualizations.

### Class Definition

```python
class DataVisualizer:
    def __init__(
        self,
        config: Optional[PlotConfig] = None,
        registry: Optional[PropertyRegistry] = None,
    )
```

#### Parameters

- **config** (`Optional[PlotConfig]`) - Optional configuration (uses defaults if None)
- **registry** (`Optional[PropertyRegistry]`) - Optional property registry (uses default if None)

### make_plot()

The primary method for creating 3D visualizations.

#### Full Signature

```python
def make_plot(
    self,
    source: Union[
        ReservoirModel[ThreeDimensions],
        ModelState[ThreeDimensions],
        ThreeDimensionalGrid,
    ],
    property: Optional[str] = None,
    plot_type: Optional[Union[PlotType, str]] = None,
    figure: Optional[go.Figure] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    x_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
    y_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
    z_slice: Optional[Union[int, slice, Tuple[int, int]]] = None,
    labels: Optional[Labels] = None,
    show_wells: bool = False,
    **kwargs: Any,
) -> go.Figure
```

#### Parameters

**Source and Property**

- **source** - Either a `ModelState`, `ReservoirModel`, or raw `ThreeDimensionalGrid` numpy array
- **property** (`Optional[str]`) - Property name from [`PropertyRegistry`](#available-properties) (required for ModelState/ReservoirModel)

**Plot Configuration**

- **plot_type** (`Optional[Union[PlotType, str]]`) - Type of 3D plot (see [PlotType](#plottype))
- **figure** (`Optional[go.Figure]`) - Existing figure to add to (creates new if None)
- **title** (`Optional[str]`) - Custom plot title
- **width**, **height** (`Optional[int]`) - Figure dimensions in pixels

**Data Slicing**

- **x_slice**, **y_slice**, **z_slice** - Slice specifications:
  - `int`: Single index (e.g., `5` extracts cells `[5:6, :, :]`)
  - `Tuple[int, int]`: Range (e.g., `(10, 20)` extracts cells `[10:20, :, :]`)
  - `slice`: Full slice object (e.g., `slice(10, 20, 2)` for every 2nd cell)
  - `None`: Use full dimension

**Annotations and Wells**

- **labels** (`Optional[Labels]`) - Label collection for text annotations
- **show_wells** (`bool`) - Whether to visualize wells (only works with `ModelState`)

**Additional Parameters**

- **kwargs** - Plot-type-specific parameters and well visualization options (see [WellKwargs](#wellkwargs))

#### Returns

A Plotly `Figure` object.

---

## Available Properties

Properties for the `property` parameter (from [`PropertyRegistry`](../guides/visualization.md#property-registry)):

**Fluid Properties**

- `"pressure"` - Reservoir pressure (psia)
- `"oil-saturation"` - Oil saturation (fraction)
- `"water-saturation"` - Water saturation (fraction)
- `"gas-saturation"` - Gas saturation (fraction)

**PVT Properties**

- `"oil-viscosity"` - Oil viscosity (cP)
- `"gas-viscosity"` - Gas viscosity (cP)
- `"water-viscosity"` - Water viscosity (cP)
- `"oil-density"` - Oil density (lbm/ft³)
- `"gas-density"` - Gas density (lbm/ft³)
- `"water-density"` - Water density (lbm/ft³)
- `"oil-formation-volume-factor"` - Oil FVF, Bo (rb/stb)
- `"gas-formation-volume-factor"` - Gas FVF, Bg (rb/scf)
- `"gas-compressibility-factor"` - Gas Z-factor (dimensionless)
- `"solution-gas-oil-ratio"` - Solution GOR, Rs (scf/stb)

**Todd-Longstaff Miscible Properties**

- `"oil-effective-viscosity"` - Effective oil viscosity with dissolved solvent (cP)
- `"oil-effective-density"` - Effective oil density with dissolved solvent (lbm/ft³)
- `"solvent-concentration"` - Solvent concentration in oil phase (fraction)

**Rock Properties**

- `"porosity"` - Rock porosity (fraction)
- `"permeability-x"` - X-direction permeability (mD)
- `"permeability-y"` - Y-direction permeability (mD)
- `"permeability-z"` - Z-direction permeability (mD)

---

## PlotType

Enumeration of available 3D plot types.

### Values

- **`PlotType.VOLUME`** (or `"volume"`) - Volume rendering showing continuous 3D data distribution
- **`PlotType.ISOSURFACE`** (or `"isosurface"`) - Isosurface plots showing surfaces at specific data values
- **`PlotType.SCATTER_3D`** (or `"scatter_3d"`) - 3D scatter plots for point data visualization
- **`PlotType.CELL_BLOCKS`** (or `"cell_blocks"`) - Individual reservoir cells rendered as 3D blocks

### Usage

```python
# Using enum
viz.make_plot(state, "pressure", plot_type=bores.plotly3d.PlotType.VOLUME)

# Using string (automatically converted)
viz.make_plot(state, "pressure", plot_type="volume")
viz.make_plot(state, "oil-saturation", plot_type="isosurface")
```

---

## Examples

### Basic Volume Rendering

```python
import bores

# Load states
store = bores.ZarrStore("./results/simulation.zarr")
stream = bores.StateStream(store=store, auto_replay=True)
states = list(stream.replay())

# Create visualizer
viz = bores.plotly3d.DataVisualizer()

# Visualize pressure
fig = viz.make_plot(
    states[100],
    property="pressure",
    plot_type="volume",
    title="Pressure Distribution at Timestep 100",
    width=1200,
    height=960,
    opacity=0.7,
)
fig.show()
```

### Isosurface Visualization

Isosurfaces are ideal for visualizing specific value thresholds:

```python
# Visualize water front at 50% saturation
fig = viz.make_plot(
    states[-1],
    property="water-saturation",
    plot_type="isosurface",
    title="Water Front (Sw = 0.5)",
    isomin=0.5,  # Minimum isosurface value
    isomax=0.5,  # Maximum isosurface value
    opacity=0.8,
    cmin=0.0,  # Colorbar minimum
    cmax=1.0,  # Colorbar maximum
    width=1260,
    height=960,
)
fig.show()
```

### Data Slicing

Extract and visualize specific regions:

```python
# Visualize top 10 layers only
fig = viz.make_plot(
    states[50],
    property="oil-saturation",
    plot_type="volume",
    title="Oil Saturation - Top 10 Layers",
    z_slice=(0, 10),  # Layers 0-9
    opacity=0.75,
)
fig.show()

# Visualize corner section
fig = viz.make_plot(
    states[50],
    property="pressure",
    plot_type="isosurface",
    title="Pressure - Corner Section",
    x_slice=(0, 20),  # First 20 cells in X
    y_slice=(0, 20),  # First 20 cells in Y
    z_slice=(0, 5),   # Top 5 layers
)
fig.show()

# Use slice object for every other cell
fig = viz.make_plot(
    states[50],
    property="porosity",
    plot_type="cell_blocks",
    x_slice=slice(None, None, 2),  # Every 2nd cell in X
    y_slice=slice(None, None, 2),  # Every 2nd cell in Y
)
fig.show()
```

### Well Visualization

Display wells with customizable styling:

```python
# Basic well visualization
fig = viz.make_plot(
    states[100],
    property="pressure",
    plot_type="isosurface",
    title="Pressure with Wells",
    show_wells=True,
    show_perforations=True,
    show_surface_marker=True,
    opacity=0.7,
)
fig.show()

# Custom well colors and styling
fig = viz.make_plot(
    states[100],
    property="oil-saturation",
    plot_type="volume",
    title="Oil Saturation with Custom Well Styling",
    show_wells=True,
    injection_color="#ff4444",     # Red for injectors
    production_color="#44dd44",    # Green for producers
    shut_in_color="#888888",       # Gray for shut-in wells
    wellbore_width=15.0,           # Wellbore line width
    surface_marker_size=2.5,       # Surface marker size
    marker_size=12,                # Perforation marker size
    show_wellbore=True,
    show_surface_marker=True,
    show_perforations=True,
    opacity=0.6,
)
fig.show()
```

### Todd-Longstaff Miscible Flooding Visualization

```python
# Visualize solvent concentration
fig = viz.make_plot(
    states[500],
    property="solvent-concentration",
    plot_type="isosurface",
    title="Solvent Concentration Profile",
    isomin=0.1,
    isomax=0.9,
    opacity=0.7,
    width=1260,
    height=960,
)
fig.show()

# Effective oil viscosity
fig = viz.make_plot(
    states[500],
    property="oil-effective-viscosity",
    plot_type="volume",
    title="Oil Effective Viscosity with Dissolved Solvent",
    opacity=0.65,
    cmin=0.5,
    cmax=3.0,
)
fig.show()
```

### Adding Labels

Use the `Labels` class to add text annotations:

```python
# Create labels for wells
wells = states[0].wells
injector_locations, producer_locations = wells.locations
injector_names, producer_names = wells.names

well_positions = [*injector_locations, *producer_locations]
well_names = [*injector_names, *producer_names]

# Create labels object
labels = bores.plotly3d.Labels()
labels.add_well_labels(well_positions, well_names)

# Use in visualization
fig = viz.make_plot(
    states[100],
    property="pressure",
    plot_type="isosurface",
    title="Pressure with Well Labels",
    labels=labels,
    show_wells=True,
    opacity=0.7,
)
fig.show()
```

### Multiple Timesteps Comparison

```python
# Visualize evolution over time
timesteps = [0, 50, 100, 150]
figures = []

# Shared visualization parameters
shared_kwargs = dict(
    plot_type="isosurface",
    width=630,
    height=480,
    opacity=0.7,
    aspect_mode="data",
    z_scale=1.5,
    marker_size=10,
    show_wells=True,
    show_surface_marker=True,
    show_perforations=True,
)

for timestep in timesteps:
    fig = viz.make_plot(
        states[timestep],
        property="oil-saturation",
        title=f"Oil Saturation - Timestep {timestep}",
        **shared_kwargs,
    )
    figures.append(fig)

# Merge into single visualization
combined_fig = bores.merge_plots(
    *figures,
    cols=2,
    title="Oil Saturation Evolution",
    height=1000,
    width=1400,
)
combined_fig.show()
```

### Visualizing Custom Properties

You can visualize raw 3D numpy arrays directly:

```python
import numpy as np

# Create custom property (e.g., mobility)
state = states[-1]
krو = state.relative_permeabilities.kro
mu_o = state.model.fluid_properties.oil_effective_viscosity_grid
mobility = kro / mu_o  # Shape: (nx, ny, nz)

# Visualize
fig = viz.make_plot(
    mobility,  # Raw 3D numpy array
    property="oil-mobility",  # Optional: uses metadata if registered
    plot_type="volume",
    title="Oil Mobility Distribution",
    opacity=0.65,
)
fig.show()
```

---

## Labels

The `Labels` class provides a system for adding text annotations to 3D plots.

### Class Definition

```python
class Labels:
    def __init__(self, labels: Optional[Iterable[Label]] = None)
```

### Methods

#### add_well_labels()

Add labels for well locations.

```python
def add_well_labels(
    self,
    well_positions: List[Tuple[int, int, int]],
    well_names: Optional[List[str]] = None,
    template: str = "Well - '{name}' ({x_index}, {y_index}, {z_index}): {formatted_value} ({unit})",
    **label_kwargs,
) -> None
```

**Parameters**

- **well_positions** - List of `(i, j, k)` well coordinates
- **well_names** - Optional well names (defaults to Well_1, Well_2, etc.)
- **template** - Text template for label formatting
- **label_kwargs** - Additional `Label` constructor arguments

**Example**

```python
# Get well positions from state
wells = states[0].wells
injector_locations, producer_locations = wells.locations
injector_names, producer_names = wells.names

# Combine all wells
well_positions = [*injector_locations, *producer_locations]
well_names = [*injector_names, *producer_names]

# Create labels
labels = bores.plotly3d.Labels()
labels.add_well_labels(well_positions, well_names)

# Use in plot
fig = viz.make_plot(
    states[100],
    property="pressure",
    labels=labels,
    show_wells=True,
)
```

#### add_grid_labels()

Add labels at regular grid intervals.

```python
def add_grid_labels(
    self,
    data_shape: Tuple[int, int, int],
    spacing: Tuple[int, int, int] = (10, 10, 5),
    template: str = "({x_index}, {y_index}, {z_index})",
    **label_kwargs,
) -> None
```

**Example**

```python
labels = bores.plotly3d.Labels()
labels.add_grid_labels(
    data_shape=(50, 50, 20),
    spacing=(15, 15, 5),  # Label every 15th cell in X/Y, 5th in Z
)
```

#### add_boundary_labels()

Add labels at grid corners.

```python
def add_boundary_labels(
    self,
    data_shape: Tuple[int, int, int],
    template: str = "Boundary ({x_index}, {y_index}, {z_index})",
    **label_kwargs,
) -> None
```

---

## WellKwargs

Well visualization customization options (passed as `**kwargs` to `make_plot()`):

- **show_wellbore** (`bool`) - Show wellbore trajectory as colored tubes (default: `True`)
- **show_surface_marker** (`bool`) - Show arrows at surface location (default: `True`)
- **show_perforations** (`bool`) - Highlight perforated intervals (default: `False`)
- **injection_color** (`str`) - Color for injection wells (default: `"#ff4444"` - red)
- **production_color** (`str`) - Color for production wells (default: `"#44dd44"` - green)
- **shut_in_color** (`str`) - Color for shut-in wells (default: `"#888888"` - gray)
- **wellbore_width** (`float`) - Wellbore line width in pixels (default: `15.0`)
- **surface_marker_size** (`float`) - Surface marker size scaling (default: `2.0`)
- **marker_size** (`float`) - Perforation marker size (default: varies by plot type)

### Example

```python
fig = viz.make_plot(
    state,
    property="pressure",
    show_wells=True,
    injection_color="#ff6b6b",      # Custom red
    production_color="#51cf66",     # Custom green
    wellbore_width=20.0,            # Thicker wellbores
    surface_marker_size=3.0,        # Larger surface markers
    show_wellbore=True,
    show_surface_marker=True,
    show_perforations=True,
)
```

---

## PlotConfig

Configuration class for customizing 3D plot appearance and behavior.

### Key Fields

```python
@attrs.frozen
class PlotConfig:
    # Dimensions
    width: int = 1200
    height: int = 960

    # Plot type
    plot_type: PlotType = PlotType.VOLUME

    # Styling
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    opacity: float = 0.85
    show_colorbar: bool = True
    show_axes: bool = True

    # Camera
    camera_position: CameraPosition = DEFAULT_CAMERA_POSITION

    # Title
    title: str = ""

    # Cell blocks
    show_cell_outlines: bool = False
    cell_outline_color: str = "#404040"
    cell_outline_width: float = 0.5
```

### Usage

```python
from bores.visualization.plotly3d import PlotConfig, PlotType
from bores.visualization.base import ColorScheme

config = PlotConfig(
    width=1400,
    height=1100,
    plot_type=PlotType.ISOSURFACE,
    color_scheme=ColorScheme.PLASMA,
    opacity=0.75,
    show_colorbar=True,
)

viz = bores.plotly3d.DataVisualizer(config)
```

---

## Performance Considerations

### Automatic Coarsening

For large grids (> 1M cells), the visualizer automatically coarsens data to maintain performance:

```python
# Environment variables for performance tuning
import os

os.environ["BORES_MAX_VOLUME_CELLS"] = "2000000"  # Max cells before coarsening
os.environ["BORES_RECOMMENDED_VOLUME_CELLS"] = "512000"  # Target after coarsening
os.environ["BORES_MAX_ISOSURFACE_CELLS"] = "2000000"  # Max for isosurfaces

# Must be set BEFORE importing bores
import bores
```

### Manual Downsampling

Use slicing for manual control:

```python
# Visualize every 2nd cell
fig = viz.make_plot(
    state,
    property="pressure",
    x_slice=slice(None, None, 2),
    y_slice=slice(None, None, 2),
    z_slice=slice(None, None, 2),
)
```

---

## Saving and Exporting

```python
# Create visualization
fig = viz.make_plot(state, "pressure", plot_type="volume")

# Save as interactive HTML
fig.write_html("pressure_3d.html")

# Save as static images (requires kaleido)
fig.write_image("pressure_3d.png", width=1600, height=1200, scale=2)
fig.write_image("pressure_3d.pdf", width=1600, height=1200)
fig.write_image("pressure_3d.svg", width=1600, height=1200)
```

---

## Best Practices

### Opacity Settings

- **Volume rendering**: 0.6-0.8 for seeing internal features
- **Isosurfaces**: 0.7-0.9 for solid appearance with some transparency
- **Multiple objects**: Lower opacity (0.4-0.6) when showing wells + data

### Z-axis Scaling

Adjust vertical exaggeration with `z_scale`:

```python
fig = viz.make_plot(
    state,
    property="pressure",
    z_scale=2.0,  # 2x vertical exaggeration
    aspect_mode="data",
)
```

### Colorbar Ranges

Set consistent ranges for time series comparison:

```python
# Define global min/max
pressure_min = 1000.0
pressure_max = 5000.0

for state in states[::50]:
    fig = viz.make_plot(
        state,
        property="pressure",
        cmin=pressure_min,
        cmax=pressure_max,
    )
```

---

## See Also

- [Visualization Overview](../guides/visualization.md) - Overview of all visualization modules
- [plotly1d](plotly1d.md) - 1D visualization (time series, line plots)
- [plotly2d](plotly2d.md) - 2D visualization (heatmaps, contours)
- [Analyzing Results](../guides/analyzing-results.md) - Computing metrics for visualization
