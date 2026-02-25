# Reference

Quick reference and lookup for BORES framework.

---

## Documentation Sections

<div class="grid cards" markdown>

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete API documentation with all functions, classes, and parameters

    [:octicons-arrow-right-24: Browse API](api.md)

-   :material-book-alphabet:{ .lg .middle } **Glossary**

    ---

    Definitions of reservoir engineering and BORES-specific terms

    [:octicons-arrow-right-24: View Glossary](glossary.md)

</div>

---

## Quick Links

### Core Modules

| Module | Description |
|--------|-------------|
| `bores.factories` | Model and well factory functions |
| `bores.models` | Data models (ReservoirModel, FluidProperties, etc.) |
| `bores.config` | Configuration classes (Config, Timer) |
| `bores.simulate` | Simulation execution (Run) |
| `bores.wells` | Well classes and controls |
| `bores.grids` | Grid builders and utilities |
| `bores.correlations` | PVT correlations |
| `bores.tables` | PVT and rock-fluid tables |
| `bores.analyses` | Post-simulation analysis |
| `bores.visualization` | Plotting and visualization |

### Common Functions

```python
# Factories
bores.reservoir_model(...)      # Build reservoir model
bores.production_well(...)      # Create production well
bores.injection_well(...)       # Create injection well
bores.wells_(...)               # Combine wells

# Grid Builders
bores.uniform_grid(...)         # Uniform property grid
bores.layered_grid(...)         # Layered property grid
bores.depth_grid(...)           # Depth from thickness
bores.build_saturation_grids(...)  # Initialize saturations

# Utilities
bores.array(...)                # Array with configured dtype
bores.Time(...)                 # Time conversion
bores.use_32bit_precision()     # Set precision
bores.get_dtype()               # Get current dtype

# Storage
model.to_file(path)             # Save model
bores.ReservoirModel.from_file(path)  # Load model
```

---

## Units Reference

All quantities in **Oilfield Units**:

| Quantity | Unit | Symbol |
|----------|------|--------|
| Length | feet | ft |
| Pressure | pounds per square inch | psi |
| Temperature | Fahrenheit | °F |
| Oil rate | stock tank barrels per day | STB/day |
| Gas rate | standard cubic feet per day | SCF/day |
| Water rate | barrels per day | bbl/day |
| Permeability | millidarcy | mD |
| Viscosity | centipoise | cP |
| Density | pounds per cubic foot | lbm/ft³ |
| Time (internal) | seconds | s |
| Porosity | fraction | - |
| Saturation | fraction | - |

### Conversion Factors

```python
# Time
1 day = 86400 seconds
1 year = 365.25 days = 31557600 seconds

# Pressure
1 atm = 14.7 psi
1 bar = 14.5038 psi

# Volume
1 bbl = 42 gallons = 5.615 ft³

# Common usage
bores.Time(days=365)  # → 31536000.0 seconds
bores.c.DAYS_PER_YEAR  # → 365.25
```

---

## Enum Reference

### FluidPhase

```python
bores.FluidPhase.OIL
bores.FluidPhase.WATER
bores.FluidPhase.GAS
```

### Orientation

```python
bores.Orientation.X  # Layer along x-axis
bores.Orientation.Y  # Layer along y-axis
bores.Orientation.Z  # Layer along z-axis (most common)
```

### Wettability / Wettability

```python
bores.Wettability.WATER_WET  # For rel perm
bores.Wettability.WATER_WET      # For cap pressure
bores.Wettability.OIL_WET
bores.Wettability.OIL_WET
```

---

## Configuration Defaults

### Timer Defaults

```python
bores.Timer(
    initial_step_size=bores.Time(hours=1),
    max_step_size=bores.Time(days=10),
    min_step_size=bores.Time(seconds=1),
    simulation_time=bores.Time(days=365),
    max_cfl_number=0.95,
    ramp_up_factor=1.25,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
    max_rejects=50,
)
```

### Config Defaults

```python
bores.Config(
    scheme="impes",
    output_frequency=1,
    log_interval=10,
    pressure_solver="bicgstab",
    pressure_preconditioner="ilu",
    pressure_convergence_tolerance=1e-6,
    saturation_convergence_tolerance=1e-4,
    max_iterations=200,
    use_pseudo_pressure=True,
    disable_capillary_effects=False,
    capillary_strength_factor=1.0,
)
```

---

## Getting Help

- **Search**: Use search bar (top right) to find specific terms
- **Examples**: See [Examples](../examples/index.md) for working code
- **API Docs**: See [API Reference](api.md) for detailed signatures
- **GitHub**: [Report issues](https://github.com/ti-oluwa/bores/issues)
