# plotly1d - 1D Visualization Module

The `plotly1d` module provides comprehensive tools for creating 1D visualizations including time series plots, production curves, bar charts, tornado plots for sensitivity analysis, and scatter plots with trendlines. All visualizations are built on Plotly for interactivity and publication-quality output.

---

## Overview

The module includes four main renderer classes and a convenience function:

- [`LineRenderer`](#linerenderer) - Line plots for time series and continuous data
- [`BarRenderer`](#barrenderer) - Vertical bar charts for categorical or discrete data
- [`TornadoRenderer`](#tornadorenderer) - Tornado plots for sensitivity analysis
- [`ScatterRenderer`](#scatterrenderer) - Scatter plots with optional trendlines
- [`make_series_plot()`](#make_series_plot) - Convenience function for quick line plots

---

## Data Format

All renderers accept data in the `SeriesData` format, which can be:

1. **Dictionary of named series**: `{"Series Name": np.array([[x1, y1], [x2, y2], ...])}`
2. **Single series array**: `np.array([[x1, y1], [x2, y2], ...])` (shape: n×2)
3. **List of series**: `[series1_array, series2_array, ...]`

Each series must be a 2D NumPy array where each row contains an `(x, y)` pair.

### Example Data Preparation

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

# Get production history (returns generator of (timestep, value) tuples)
oil_production = analyst.oil_production_history(interval=1, cumulative=False, from_step=1)

# Convert to numpy array format required by plotly1d
# Each row is [timestep, production_rate]
oil_data = np.array(list(oil_production))  # shape: (n_steps, 2)

print(oil_data.shape)  # (150, 2) for 150 timesteps
print(oil_data[0])     # [0, 1234.5] - timestep 0, rate 1234.5 STB/day
```

---

## make_series_plot()

Convenience function for quickly creating line plots without instantiating a renderer.

### Function Signature

```python
def make_series_plot(
    data: SeriesData,
    title: str = "Series Plot",
    x_label: str = "X",
    y_label: str = "Y",
    **kwargs: Any,
) -> go.Figure
```

### Parameters

- **data** - Series data (see [Data Format](#data-format))
- **title** - Plot title
- **x_label** - X-axis label
- **y_label** - Y-axis label
- **kwargs** - Additional parameters passed to [`LineRenderer.render()`](#linerendererrender)

### Returns

A Plotly `Figure` object that can be displayed with `.show()` or saved with `.write_html()` / `.write_image()`.

### Basic Usage

```python
import bores
import numpy as np

# Single series
pressure_data = np.array([[0, 3500], [10, 3200], [20, 2900], [30, 2600]])

fig = bores.make_series_plot(
    data={"Avg. Pressure": pressure_data},
    title="Reservoir Pressure Decline",
    x_label="Time Step",
    y_label="Pressure (psia)",
)
fig.show()
```

### Multiple Series

```python
# Multiple series with custom styling
oil_data = np.array(list(analyst.oil_production_history(interval=2, from_step=1)))
water_data = np.array(list(analyst.water_production_history(interval=2, from_step=1)))

fig = bores.make_series_plot(
    data={
        "Oil Production": oil_data,
        "Water Production": water_data,
    },
    title="Production History",
    x_label="Time Step",
    y_label="Production Rate (STB/day)",
    line_colors=["brown", "blue"],
    line_widths=[2.5, 2.0],
    marker_sizes=6,
    show_markers=True,
    width=900,
    height=600,
)
fig.show()
```

### Customization Examples

```python
# Logarithmic y-axis
fig = bores.make_series_plot(
    data={"Gas Rate": gas_data},
    title="Gas Production (Log Scale)",
    x_label="Time Step",
    y_label="Gas Rate (SCF/day)",
    log_y=True,
    line_colors="green",
    line_widths=3.0,
)

# Fill area under curve
fig = bores.make_series_plot(
    data={"Cumulative Oil": cumulative_oil_data},
    title="Cumulative Oil Production",
    x_label="Time Step",
    y_label="Cumulative Production (STB)",
    fill_area=True,
    fill_colors="rgba(139, 69, 19, 0.3)",  # Semi-transparent brown
    line_colors="brown",
)

# Markers only (no lines)
fig = bores.make_series_plot(
    data={"Recovery Factor": rf_data},
    title="Recovery Factor vs Time",
    x_label="Time Step",
    y_label="Recovery Factor (fraction)",
    mode="markers",
    marker_sizes=10,
    marker_symbols="diamond",
    marker_colors="darkgreen",
)
```

---

## LineRenderer

Renderer for creating line plots with extensive customization options for time series analysis, production curves, and trend visualization.

### Class Definition

```python
class LineRenderer(BaseRenderer):
    def __init__(self, config: Optional[PlotConfig] = None)
```

### LineRenderer.render()

Main rendering method for creating line plots.

#### Full Signature

```python
def render(
    self,
    data: SeriesData,
    x_label: str = "X",
    y_label: str = "Y",
    series_names: Optional[Sequence[str]] = None,
    line_colors: Optional[Union[str, Sequence[str]]] = None,
    line_widths: Optional[Union[float, Sequence[float]]] = None,
    line_styles: Optional[Union[str, Sequence[str]]] = None,
    show_markers: bool = True,
    marker_sizes: Optional[Union[int, Sequence[int]]] = None,
    marker_symbols: Optional[Union[str, Sequence[str]]] = None,
    marker_colors: Optional[Union[str, Sequence[str]]] = None,
    mode: Literal["lines", "markers", "lines+markers"] = "lines+markers",
    fill_area: Optional[Union[bool, Sequence[bool]]] = None,
    fill_colors: Optional[Union[str, Sequence[str]]] = None,
    fill_opacity: float = 0.3,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    log_x: bool = False,
    log_y: bool = False,
    hover_template: Optional[str] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs: Any,
) -> go.Figure
```

#### Parameters

**Data and Labels**

- **data** (`SeriesData`) - Input data (see [Data Format](#data-format))
- **x_label** (`str`) - X-axis label (default: "X")
- **y_label** (`str`) - Y-axis label (default: "Y")
- **series_names** (`Optional[Sequence[str]]`) - Custom series names for legend

**Line Styling**

- **line_colors** (`Optional[Union[str, Sequence[str]]]`) - Line color(s) (CSS colors, hex, or rgb)
- **line_widths** (`Optional[Union[float, Sequence[float]]]`) - Line width(s) in pixels
- **line_styles** (`Optional[Union[str, Sequence[str]]]`) - Line style(s): `"solid"`, `"dash"`, `"dot"`, `"dashdot"`

**Marker Styling**

- **show_markers** (`bool`) - Whether to show markers on data points (default: `True`)
- **marker_sizes** (`Optional[Union[int, Sequence[int]]]`) - Marker size(s) in pixels
- **marker_symbols** (`Optional[Union[str, Sequence[str]]]`) - Marker symbol(s): `"circle"`, `"square"`, `"diamond"`, `"triangle-up"`, etc.
- **marker_colors** (`Optional[Union[str, Sequence[str]]]`) - Marker color(s) (defaults to line colors)

**Plot Mode**

- **mode** (`Literal["lines", "markers", "lines+markers"]`) - Display mode (default: `"lines+markers"`)

**Fill Options**

- **fill_area** (`Optional[Union[bool, Sequence[bool]]]`) - Whether to fill area under curve(s)
- **fill_colors** (`Optional[Union[str, Sequence[str]]]`) - Fill color(s) (defaults to line colors)
- **fill_opacity** (`float`) - Opacity for filled areas (0.0-1.0, default: 0.3)

**Axes Configuration**

- **x_range** (`Optional[Tuple[float, float]]`) - X-axis range as `(min, max)`
- **y_range** (`Optional[Tuple[float, float]]`) - Y-axis range as `(min, max)`
- **log_x** (`bool`) - Use logarithmic x-axis (default: `False`)
- **log_y** (`bool`) - Use logarithmic y-axis (default: `False`)

**Layout**

- **hover_template** (`Optional[str]`) - Custom Plotly hover template
- **title** (`Optional[str]`) - Plot title
- **width** (`Optional[int]`) - Figure width in pixels (overrides config)
- **height** (`Optional[int]`) - Figure height in pixels (overrides config)
- **kwargs** - Additional parameters passed to `go.Scatter()`

#### Returns

A Plotly `Figure` object.

### Examples

#### Basic Line Plot

```python
from bores.visualization.plotly1d import LineRenderer
import numpy as np

# Create renderer with custom configuration
from bores.visualization.plotly1d import PlotConfig

config = PlotConfig(
    width=1000,
    height=700,
    line_width=2.5,
    marker_size=8,
    show_grid=True,
    color_palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
)

renderer = LineRenderer(config)

# Prepare data
saturation_history = []
for state in states[::5]:  # Every 5th timestep
    timestep = state.step
    avg_oil_sat = np.mean(state.model.fluid_properties.oil_saturation_grid)
    saturation_history.append((timestep, avg_oil_sat))

sat_data = np.array(saturation_history)

# Render
fig = renderer.render(
    data={"Oil Saturation": sat_data},
    x_label="Time Step",
    y_label="Average Oil Saturation (fraction)",
    title="Oil Saturation Decline",
    line_colors="brown",
    line_widths=3.0,
    marker_sizes=8,
    show_markers=True,
)
fig.show()
```

#### Multiple Series with Custom Styling

```python
# Collect multiple saturation histories
oil_sat_history = []
water_sat_history = []
gas_sat_history = []

for state in states[::2]:
    timestep = state.step
    avg_oil_sat = np.mean(state.model.fluid_properties.oil_saturation_grid)
    avg_water_sat = np.mean(state.model.fluid_properties.water_saturation_grid)
    avg_gas_sat = np.mean(state.model.fluid_properties.gas_saturation_grid)

    oil_sat_history.append((timestep, avg_oil_sat))
    water_sat_history.append((timestep, avg_water_sat))
    gas_sat_history.append((timestep, avg_gas_sat))

# Convert to arrays
oil_data = np.array(oil_sat_history)
water_data = np.array(water_sat_history)
gas_data = np.array(gas_sat_history)

# Render with custom styling for each series
renderer = LineRenderer()
fig = renderer.render(
    data={
        "Oil Saturation": oil_data,
        "Water Saturation": water_data,
        "Gas Saturation": gas_data,
    },
    x_label="Time Step",
    y_label="Saturation (fraction)",
    title="Phase Saturations Evolution",
    line_colors=["brown", "blue", "green"],
    line_widths=[2.5, 2.0, 2.0],
    line_styles=["solid", "dash", "dot"],
    marker_sizes=[8, 6, 6],
    marker_symbols=["circle", "square", "diamond"],
    show_markers=True,
    y_range=(0.0, 1.0),
    width=1000,
    height=600,
)
fig.show()
```

#### Logarithmic Scale

```python
# Pressure decline on log scale
pressure_history = []
for state in states:
    timestep = state.step
    avg_pressure = np.mean(state.model.fluid_properties.pressure_grid)
    pressure_history.append((timestep, avg_pressure))

pressure_data = np.array(pressure_history)

renderer = LineRenderer()
fig = renderer.render(
    data={"Reservoir Pressure": pressure_data},
    x_label="Time Step",
    y_label="Pressure (psia, log scale)",
    title="Pressure Decline (Logarithmic)",
    log_y=True,
    line_colors="darkblue",
    line_widths=3.0,
    marker_sizes=0,  # No markers
    show_markers=False,
)
fig.show()
```

#### Fill Area Under Curve

```python
# Cumulative production with filled area
cumulative_oil = analyst.oil_production_history(interval=1, cumulative=True, from_step=1)
cum_oil_data = np.array(list(cumulative_oil))

renderer = LineRenderer()
fig = renderer.render(
    data={"Cumulative Oil Production": cum_oil_data},
    x_label="Time Step",
    y_label="Cumulative Production (STB)",
    title="Cumulative Oil Production",
    fill_area=True,
    fill_colors="rgba(139, 69, 19, 0.2)",  # Semi-transparent brown
    line_colors="brown",
    line_widths=2.5,
    marker_sizes=6,
)
fig.show()
```

---

## BarRenderer

Renderer for creating vertical bar charts with support for grouped, stacked, and overlaid configurations.

### Class Definition

```python
class BarRenderer(BaseRenderer):
    def __init__(self, config: Optional[PlotConfig] = None)
```

### BarRenderer.render()

#### Full Signature

```python
def render(
    self,
    data: SeriesData,
    x_label: str = "Category",
    y_label: str = "Value",
    series_names: Optional[Sequence[str]] = None,
    bar_colors: Optional[Union[str, Sequence[str]]] = None,
    bar_mode: Literal["group", "stack", "overlay", "relative"] = "group",
    bar_width: Optional[float] = None,
    show_values: bool = False,
    value_format: str = ".4f",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    log_y: bool = False,
    hover_template: Optional[str] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> go.Figure
```

#### Parameters

- **data** (`SeriesData`) - Input data (x-values are category indices, y-values are bar heights)
- **x_label** (`str`) - X-axis label (default: "Category")
- **y_label** (`str`) - Y-axis label (default: "Value")
- **series_names** (`Optional[Sequence[str]]`) - Custom series names
- **bar_colors** (`Optional[Union[str, Sequence[str]]]`) - Bar color(s)
- **bar_mode** (`Literal["group", "stack", "overlay", "relative"]`) - How to display multiple series (default: "group")
- **bar_width** (`Optional[float]`) - Bar width (None = auto)
- **show_values** (`bool`) - Whether to show values on bars (default: `False`)
- **value_format** (`str`) - Format string for values on bars (default: ".4f")
- **x_range**, **y_range**, **log_y**, **hover_template**, **title**, **width**, **height** - Same as LineRenderer
- **categories** (`Optional[Sequence[str]]`) - Custom labels for x-axis tick marks (e.g., `["Jan", "Feb", "Mar"]`)
- **kwargs** - Additional parameters passed to `go.Bar()`

#### Returns

A Plotly `Figure` object.

### Examples

#### Basic Bar Chart

```python
from bores.visualization.plotly1d import BarRenderer
import numpy as np

# Production by well
well_production = np.array([
    [0, 1500],  # Well 1
    [1, 2300],  # Well 2
    [2, 1800],  # Well 3
    [3, 2100],  # Well 4
])

renderer = BarRenderer()
fig = renderer.render(
    data={"Oil Production": well_production},
    categories=["PROD-1", "PROD-2", "PROD-3", "PROD-4"],
    x_label="Well Name",
    y_label="Production Rate (STB/day)",
    title="Production by Well",
    bar_colors="green",
    show_values=True,
    value_format=".1f",
)
fig.show()
```

#### Grouped Bars

```python
# Quarterly production comparison
quarters = [0, 1, 2, 3]

oil_quarterly = np.array([[0, 45000], [1, 52000], [2, 48000], [3, 51000]])
gas_quarterly = np.array([[0, 120000], [1, 135000], [2, 128000], [3, 140000]])

renderer = BarRenderer()
fig = renderer.render(
    data={
        "Oil": oil_quarterly,
        "Gas": gas_quarterly,
    },
    categories=["Q1", "Q2", "Q3", "Q4"],
    x_label="Quarter",
    y_label="Production (STB for Oil, SCF for Gas)",
    title="Quarterly Production - 2025",
    bar_mode="group",
    bar_colors=["brown", "green"],
    show_values=True,
    value_format=".0f",
)
fig.show()
```

#### Stacked Bars

```python
# Production composition
renderer = BarRenderer()
fig = renderer.render(
    data={
        "Primary Recovery": np.array([[0, 1200], [1, 1100], [2, 1000]]),
        "Secondary Recovery": np.array([[0, 300], [1, 400], [2, 350]]),
        "Tertiary Recovery": np.array([[0, 0], [1, 50], [2, 150]]),
    },
    categories=["Year 1", "Year 2", "Year 3"],
    x_label="Year",
    y_label="Production (STB/day)",
    title="Production by Recovery Method",
    bar_mode="stack",
    bar_colors=["#8c564b", "#e377c2", "#7f7f7f"],
    width=800,
    height=600,
)
fig.show()
```

---

## TornadoRenderer

Renderer for creating tornado plots used in sensitivity analysis to visualize the impact of different variables on a base case outcome.

### Class Definition

```python
class TornadoRenderer(BaseRenderer):
    def __init__(self, config: Optional[PlotConfig] = None)
```

### TornadoRenderer.render()

#### Full Signature

```python
def render(
    self,
    data: Union[
        TwoDimensionalGrid,  # Shape (n, 3): [low, base, high]
        Mapping[str, Tuple[float, float, float]],  # {var: (low, base, high)}
        SeriesData,
    ],
    x_label: str = "Impact",
    y_label: str = "Variable",
    series_names: Optional[Sequence[str]] = None,
    positive_color: str = "#2ca02c",
    negative_color: str = "#d62728",
    base_value: Optional[float] = None,
    show_values: bool = True,
    value_format: str = ".2f",
    sort_by_impact: bool = True,
    hover_template: Optional[str] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    **kwargs: Any,
) -> go.Figure
```

#### Parameters

- **data** - Either an `(n, 3)` array with columns `[low, base, high]` or dictionary `{variable_name: (low, base, high)}`
- **x_label** (`str`) - X-axis label (default: "Impact")
- **y_label** (`str`) - Y-axis label (default: "Variable")
- **series_names** (`Optional[Sequence[str]]`) - Variable names (if data is array)
- **positive_color** (`str`) - Color for positive impacts (default: green)
- **negative_color** (`str`) - Color for negative impacts (default: red)
- **base_value** (`Optional[float]`) - Base case value (if None, computed from data)
- **show_values** (`bool`) - Whether to show impact values on bars (default: `True`)
- **value_format** (`str`) - Format string for values (default: ".2f")
- **sort_by_impact** (`bool`) - Sort variables by total impact magnitude (default: `True`)
- **hover_template**, **title**, **width**, **height** - Same as other renderers
- **kwargs** - Additional parameters passed to `go.Bar()`

#### Returns

A Plotly `Figure` object.

### Example

```python
from bores.visualization.plotly1d import TornadoRenderer

# Sensitivity analysis results
# Each row: [low case NPV, base case NPV, high case NPV]
sensitivity_data = {
    "Oil Price": (15.2, 25.0, 35.8),        # $M NPV
    "Permeability": (20.5, 25.0, 28.5),
    "Porosity": (22.0, 25.0, 27.5),
    "Well Count": (18.0, 25.0, 31.0),
    "OPEX": (27.0, 25.0, 23.5),
}

renderer = TornadoRenderer()
fig = renderer.render(
    data=sensitivity_data,
    x_label="NPV Impact ($M)",
    y_label="Parameter",
    title="Sensitivity Analysis - NPV",
    base_value=25.0,
    positive_color="#2ca02c",
    negative_color="#d62728",
    show_values=True,
    value_format=".1f",
    sort_by_impact=True,
    width=900,
    height=600,
)
fig.show()
```

---

## ScatterRenderer

Renderer for creating scatter plots with optional trendlines for correlation analysis.

### Class Definition

```python
class ScatterRenderer(BaseRenderer):
    def __init__(self, config: Optional[PlotConfig] = None)
```

Note: For detailed API documentation of `ScatterRenderer.render()`, refer to the source code at `src/bores/visualization/plotly1d.py`.

### Example

```python
from bores.visualization.plotly1d import ScatterRenderer
import numpy as np

# Correlation between permeability and production
perm_values = []
prod_values = []

for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
    perm = model.rock_properties.permeability_grid[i, j, k, 0]  # X-direction
    # Get average production in this cell (simplified example)
    prod = calculate_cell_production(i, j, k)
    perm_values.append(perm)
    prod_values.append(prod)

# Create scatter data
scatter_data = np.column_stack([perm_values, prod_values])

renderer = ScatterRenderer()
fig = renderer.render(
    data={"Perm vs Production": scatter_data},
    x_label="Permeability (mD)",
    y_label="Production Rate (STB/day)",
    title="Permeability-Production Correlation",
    marker_sizes=8,
    marker_colors="blue",
    log_x=True,  # Permeability often log-distributed
)
fig.show()
```

---

## Configuration

### PlotConfig

Configuration class for customizing plot appearance and behavior.

```python
@attrs.frozen
class PlotConfig:
    # Dimensions
    width: int = 800
    height: int = 600

    # Display
    title: Optional[str] = None
    show_legend: bool = True
    legend_position: Literal["top", "bottom", "left", "right"] = "right"

    # Styling
    color_palette: Sequence[str] = ("#1f77b4", "#ff7f0e", "#2ca02c", ...)
    opacity: float = 0.8

    # Grid and axes
    show_grid: bool = True
    grid_color: str = "lightgray"
    xaxis_grid_color: Optional[str] = None
    yaxis_grid_color: Optional[str] = None
    axis_line_color: str = "black"
    axis_line_width: float = 1.0

    # Text
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    title_font_size: int = 16
    axis_title_font_size: int = 14

    # Plot defaults
    line_width: float = 2.0
    marker_size: int = 8
    bar_width: Optional[float] = None

    # Margins
    margin_left: int = 80
    margin_right: int = 80
    margin_top: int = 80
    margin_bottom: int = 80

    # Background
    background_color: str = "white"
    plot_background_color: str = "white"

    # Interactive
    show_hover: bool = True
    hover_mode: Literal["x", "y", "closest", "x unified", "y unified"] = "x unified"
```

### Usage

```python
from bores.visualization.plotly1d import PlotConfig, LineRenderer

# Create custom configuration
config = PlotConfig(
    width=1200,
    height=800,
    color_palette=["#8B4513", "#1E90FF", "#32CD32"],  # Brown, Blue, Green
    line_width=3.0,
    marker_size=10,
    show_grid=True,
    grid_color="lightgray",
    font_family="Times New Roman, serif",
    font_size=14,
    background_color="#f5f5f5",
)

# Use with renderer
renderer = LineRenderer(config)
fig = renderer.render(data=my_data, ...)
```

---

## Saving and Exporting

All renderers return Plotly `Figure` objects which can be saved in various formats:

```python
# Render plot
fig = bores.make_series_plot(data=oil_data, title="Oil Production")

# Save as interactive HTML
fig.write_html("oil_production.html")

# Save as static images (requires kaleido package)
fig.write_image("oil_production.png", width=1200, height=800, scale=2)
fig.write_image("oil_production.pdf", width=1200, height=800)
fig.write_image("oil_production.svg", width=1200, height=800)
fig.write_image("oil_production.jpeg", width=1200, height=800, scale=2)

# Display in browser or notebook
fig.show()
```

---

## See Also

- [Visualization Guide](../guides/visualization.md) - Overview of all visualization modules
- [plotly2d](plotly2d.md) - 2D visualization (heatmaps, contours)
- [plotly3d](plotly3d.md) - 3D visualization (volume rendering, isosurfaces)
- [Analyzing Results](../guides/analyzing-results.md) - Computing metrics for visualization
