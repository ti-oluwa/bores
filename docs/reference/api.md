# API Reference

Complete API documentation for the BORES framework.

---

## Overview

This page documents all public APIs in BORES using the `bores.*` notation. For detailed examples, see the [User Guide](../guides/index.md) and [Tutorials](../tutorials/index.md).

!!! info "Notation"
    All APIs are accessed through the `bores` namespace:
    ```python
    import bores

    # Use bores.* notation
    model = bores.reservoir_model(...)
    well = bores.production_well(...)
    ```

---

## Core Factories

### `bores.reservoir_model()`

Build a reservoir model from grids and properties.

```python
model = bores.reservoir_model(
    grid_shape: tuple[int, int, int],
    cell_dimension: tuple[float, float],
    thickness_grid: NDArray,
    pressure_grid: NDArray,
    oil_bubble_point_pressure_grid: NDArray,
    absolute_permeability: RockPermeability,
    porosity_grid: NDArray,
    temperature_grid: NDArray,
    rock_compressibility: float,
    oil_saturation_grid: NDArray,
    water_saturation_grid: NDArray,
    gas_saturation_grid: NDArray,
    oil_viscosity_grid: NDArray,
    oil_specific_gravity_grid: NDArray,
    oil_compressibility_grid: NDArray,
    gas_gravity_grid: NDArray,
    residual_oil_saturation_water_grid: NDArray,
    residual_oil_saturation_gas_grid: NDArray,
    irreducible_water_saturation_grid: NDArray,
    connate_water_saturation_grid: NDArray,
    residual_gas_saturation_grid: NDArray,
    reservoir_gas: str = "methane",
    dip_angle: float = 0.0,
    dip_azimuth: float = 0.0,
    net_to_gross_ratio_grid: Optional[NDArray] = None,
    pvt_tables: Optional[PVTTables] = None,
) -> ReservoirModel[ThreeDimensions]
```

**Parameters:**

- `grid_shape`: Grid dimensions (nx, ny, nz)
- `cell_dimension`: Horizontal cell size (dx, dy) in feet
- `thickness_grid`: Cell thickness in feet, shape (nx, ny, nz)
- `pressure_grid`: Initial pressure in psi
- `oil_bubble_point_pressure_grid`: Bubble point pressure in psi
- `absolute_permeability`: RockPermeability object with x, y, z permeabilities in mD
- `porosity_grid`: Porosity fraction (0-1)
- `temperature_grid`: Temperature in °F
- `rock_compressibility`: Rock compressibility in 1/psi
- `oil_saturation_grid`: Initial oil saturation (0-1)
- `water_saturation_grid`: Initial water saturation (0-1)
- `gas_saturation_grid`: Initial gas saturation (0-1)
- `oil_viscosity_grid`: Oil viscosity in cP
- `oil_specific_gravity_grid`: Oil specific gravity (relative to water)
- `oil_compressibility_grid`: Oil compressibility in 1/psi
- `gas_gravity_grid`: Gas specific gravity (relative to air)
- `residual_oil_saturation_water_grid`: Sor to water
- `residual_oil_saturation_gas_grid`: Sor to gas
- `irreducible_water_saturation_grid`: Swi
- `connate_water_saturation_grid`: Swc
- `residual_gas_saturation_grid`: Sgr
- `reservoir_gas`: Gas type (e.g., "methane", "co2", "n2") - must be supported by CoolProp
- `dip_angle`: Structural dip angle in degrees (0-90)
- `dip_azimuth`: Dip direction in degrees (0-360, clockwise from North)
- `net_to_gross_ratio_grid`: Net-to-gross ratio (0-1), optional
- `pvt_tables`: PVTTables object for property lookups, optional

**Returns:** `ReservoirModel[ThreeDimensions]`

**Example:**

```python
model = bores.reservoir_model(
    grid_shape=(20, 20, 10),
    cell_dimension=(100.0, 100.0),
    thickness_grid=thickness,
    pressure_grid=pressure,
    oil_bubble_point_pressure_grid=bubble_point,
    absolute_permeability=perm,
    porosity_grid=porosity,
    temperature_grid=temperature,
    rock_compressibility=4.5e-6,
    oil_saturation_grid=oil_sat,
    water_saturation_grid=water_sat,
    gas_saturation_grid=gas_sat,
    oil_viscosity_grid=oil_visc,
    oil_specific_gravity_grid=oil_sg,
    oil_compressibility_grid=oil_comp,
    gas_gravity_grid=gas_gravity,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    irreducible_water_saturation_grid=swi,
    connate_water_saturation_grid=swc,
    residual_gas_saturation_grid=sgr,
    reservoir_gas="methane",
)
```

---

### `bores.production_well()`

Create a production well.

```python
well = bores.production_well(
    well_name: str,
    perforating_intervals: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    radius: float,
    control: WellControl,
    produced_fluids: Optional[tuple[ProducedFluid, ...]] = None,
    skin_factor: float = 0.0,
    is_active: bool = True,
) -> ProductionWell[ThreeDimensions]
```

**Parameters:**

- `well_name`: Unique well identifier
- `perforating_intervals`: List of perforation ranges, each as `((x1, y1, z1), (x2, y2, z2))`
- `radius`: Wellbore radius in feet
- `control`: Well control object (e.g., `AdaptiveBHPRateControl`)
- `produced_fluids`: Tuple of `ProducedFluid` objects, optional
- `skin_factor`: Dimensionless skin (positive = damage, negative = stimulation)
- `is_active`: Whether well is initially active

**Returns:** `ProductionWell[ThreeDimensions]`

**Example:**

```python
producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((10, 10, 2), (10, 10, 5))],
    radius=0.354,  # 8.5" hole
    control=bores.AdaptiveBHPRateControl(
        target_rate=-200,
        target_phase="oil",
        bhp_limit=800,
    ),
    skin_factor=2.5,
    is_active=True,
)
```

---

### `bores.injection_well()`

Create an injection well.

```python
well = bores.injection_well(
    well_name: str,
    perforating_intervals: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    radius: float,
    control: WellControl,
    injected_fluid: InjectedFluid,
    skin_factor: float = 0.0,
    is_active: bool = True,
) -> InjectionWell[ThreeDimensions]
```

**Parameters:**

- `well_name`: Unique well identifier
- `perforating_intervals`: List of perforation ranges
- `radius`: Wellbore radius in feet
- `control`: Well control (e.g., `ConstantRateControl`)
- `injected_fluid`: `InjectedFluid` object specifying what to inject
- `skin_factor`: Dimensionless skin
- `is_active`: Whether well is initially active

**Returns:** `InjectionWell[ThreeDimensions]`

**Example:**

```python
injector = bores.injection_well(
    well_name="I-1",
    perforating_intervals=[((5, 5, 2), (5, 5, 5))],
    radius=0.354,
    control=bores.ConstantRateControl(target_rate=500),  # STB/day
    injected_fluid=bores.InjectedFluid(
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.05,
        molecular_weight=18.015,
    ),
    skin_factor=0.0,
    is_active=True,
)
```

---

### `bores.wells_()`

Combine injectors and producers into a Wells object.

```python
wells = bores.wells_(
    injectors: Optional[list[InjectionWell]] = None,
    producers: Optional[list[ProductionWell]] = None,
) -> Wells[ThreeDimensions]
```

**Example:**

```python
wells = bores.wells_(
    injectors=[injector1, injector2],
    producers=[producer1, producer2, producer3],
)
```

---

## Grid Builders

### `bores.uniform_grid()`

Create a grid with uniform values.

```python
grid = bores.uniform_grid(
    grid_shape: tuple[int, int, int],
    value: float,
) -> NDArray
```

**Example:**

```python
porosity = bores.uniform_grid(grid_shape=(20, 20, 10), value=0.20)
```

---

### `bores.layered_grid()`

Create a grid with values varying by layer.

```python
grid = bores.layered_grid(
    grid_shape: tuple[int, int, int],
    layer_values: NDArray,
    orientation: Orientation,
) -> NDArray
```

**Parameters:**

- `grid_shape`: Grid dimensions (nx, ny, nz)
- `layer_values`: Array of values, one per layer. Length must match grid_shape dimension along orientation
- `orientation`: `bores.Orientation.X`, `.Y`, or `.Z`

**Example:**

```python
porosity = bores.layered_grid(
    grid_shape=(20, 20, 10),
    layer_values=bores.array([0.15, 0.18, 0.20, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10]),
    orientation=bores.Orientation.Z,
)
```

---

### `bores.depth_grid()`

Calculate depth grid from thickness.

```python
depth = bores.depth_grid(
    thickness_grid: NDArray,
) -> NDArray
```

**Returns:** Depth at cell centers (cumulative sum of thickness)

---

### `bores.build_saturation_grids()`

Initialize saturations with fluid contacts and transition zones.

```python
water_sat, oil_sat, gas_sat = bores.build_saturation_grids(
    depth_grid: NDArray,
    gas_oil_contact: float,
    oil_water_contact: float,
    connate_water_saturation_grid: NDArray,
    residual_oil_saturation_water_grid: NDArray,
    residual_oil_saturation_gas_grid: NDArray,
    residual_gas_saturation_grid: NDArray,
    porosity_grid: NDArray,
    use_transition_zones: bool = True,
    gas_oil_transition_thickness: float = 5.0,
    oil_water_transition_thickness: float = 10.0,
    transition_curvature_exponent: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray]
```

**Parameters:**

- `depth_grid`: Cell depths in feet
- `gas_oil_contact`: GOC depth below reservoir top (feet)
- `oil_water_contact`: OWC depth below reservoir top (feet)
- `connate_water_saturation_grid`: Swc grid
- `residual_oil_saturation_water_grid`: Sor to water
- `residual_oil_saturation_gas_grid`: Sor to gas
- `residual_gas_saturation_grid`: Sgr
- `porosity_grid`: Porosity
- `use_transition_zones`: Enable smooth transitions
- `gas_oil_transition_thickness`: GOC transition zone thickness (feet)
- `oil_water_transition_thickness`: OWC transition zone thickness (feet)
- `transition_curvature_exponent`: Controls transition shape

**Returns:** `(water_sat_grid, oil_sat_grid, gas_sat_grid)`

---

## Well Controls

### `bores.ConstantRateControl`

Fixed rate control.

```python
control = bores.ConstantRateControl(
    target_rate: float,  # STB/day (negative for production)
)
```

---

### `bores.AdaptiveBHPRateControl`

Rate control with BHP limit.

```python
control = bores.AdaptiveBHPRateControl(
    target_rate: float,
    target_phase: str,  # "oil", "water", or "gas"
    bhp_limit: float,  # psi
    clamp: Optional[ProductionClamp] = None,
)
```

**Example:**

```python
control = bores.AdaptiveBHPRateControl(
    target_rate=-300,  # STB/day
    target_phase="oil",
    bhp_limit=800,  # psi minimum
)
```

---

### `bores.MultiPhaseRateControl`

Independent controls for each phase.

```python
control = bores.MultiPhaseRateControl(
    oil_control: Optional[WellControl] = None,
    water_control: Optional[WellControl] = None,
    gas_control: Optional[WellControl] = None,
)
```

---

## Configuration

### `bores.Config`

Simulation configuration.

```python
config = bores.Config(
    timer: Timer,
    rock_fluid_tables: RockFluidTables,
    wells: Optional[Wells] = None,
    well_schedules: Optional[WellSchedules] = None,
    boundary_conditions: Optional[BoundaryConditions] = None,
    pvt_tables: Optional[PVTTables] = None,
    scheme: str = "impes",
    output_frequency: int = 1,
    log_interval: int = 10,
    pressure_solver: str = "bicgstab",
    pressure_preconditioner: Optional[str] = "ilu",
    pressure_convergence_tolerance: float = 1e-6,
    saturation_convergence_tolerance: float = 1e-4,
    max_iterations: int = 200,
    use_pseudo_pressure: bool = True,
    disable_capillary_effects: bool = False,
    capillary_strength_factor: float = 1.0,
    miscibility_model: str = "immiscible",
    max_gas_saturation_change: float = 0.9,
)
```

**Key Parameters:**

- `timer`: Timer object
- `rock_fluid_tables`: Rel perm and cap pressure models
- `scheme`: `"impes"` or `"explicit"`
- `pressure_solver`: `"bicgstab"`, `"gmres"`, `"lgmres"`, etc.
- `pressure_preconditioner`: `"ilu"`, `"amg"`, `"diagonal"`, `"cpr"`, or `None`
- `miscibility_model`: `"immiscible"` or `"todd_longstaff"`

---

### `bores.Timer`

Adaptive timestep controller.

```python
timer = bores.Timer(
    initial_step_size: float,
    max_step_size: float,
    min_step_size: float,
    simulation_time: float,
    max_cfl_number: float = 0.95,
    ramp_up_factor: float = 1.25,
    backoff_factor: float = 0.5,
    aggressive_backoff_factor: float = 0.25,
    max_rejects: int = 50,
)
```

**Use with `bores.Time()`:**

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=4),
    max_step_size=bores.Time(days=5),
    min_step_size=bores.Time(minutes=10),
    simulation_time=bores.Time(days=365 * 5),
)
```

---

## Simulation Execution

### `bores.Run`

Execute simulation.

```python
run = bores.Run(
    model: ReservoirModel,
    config: Config,
)

# Iterate through timesteps
for state in run():
    # Process state
    pass
```

**State attributes:**

```python
state.step          # Timestep index
state.time          # Simulation time (seconds)
state.step_size     # Timestep size (seconds)
state.model         # Updated ReservoirModel
state.wells         # Wells configuration
state.injection     # Injection rate grids
state.production    # Production rate grids
```

---

## PVT Tables

### `bores.build_pvt_table_data()`

Build PVT table data from correlations.

```python
pvt_data = bores.build_pvt_table_data(
    pressures: NDArray,
    temperatures: NDArray,
    salinities: NDArray,
    oil_specific_gravity: float,
    gas_gravity: float,
    reservoir_gas: str = "methane",
) -> PVTTableData
```

---

### `bores.PVTTables`

PVT lookup tables.

```python
pvt_tables = bores.PVTTables(
    data: PVTTableData,
    interpolation_method: str = "linear",  # or "cubic"
)
```

---

## Rock-Fluid Models

### Relative Permeability

```python
# Brooks-Corey
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation: float,
    residual_oil_saturation_gas: float,
    residual_oil_saturation_water: float,
    residual_gas_saturation: float,
    wettability: WettabilityType,
    water_exponent: float = 2.0,
    oil_exponent: float = 2.0,
    gas_exponent: float = 2.0,
    mixing_rule: Callable = bores.eclipse_rule,
)
```

### Capillary Pressure

```python
# Brooks-Corey
cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet: float,
    oil_water_pore_size_distribution_index_water_wet: float,
    gas_oil_entry_pressure: float,
    gas_oil_pore_size_distribution_index: float,
    wettability: Wettability,
)
```

### Rock-Fluid Tables

```python
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table: RelativePermeabilityModel,
    capillary_pressure_table: CapillaryPressureModel,
)
```

---

## Utilities

### `bores.Time()`

Convert time to seconds.

```python
seconds = bores.Time(
    weeks: float = 0,
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
    seconds: float = 0,
) -> float
```

**Example:**

```python
five_years = bores.Time(days=365 * 5)
one_day = bores.Time(hours=24)
mixed = bores.Time(days=7, hours=12, minutes=30)
```

---

### `bores.array()`

Create array with configured dtype.

```python
arr = bores.array(data) -> NDArray
```

Uses `bores.get_dtype()` to determine float32 or float64.

---

### Precision Control

```python
bores.use_32bit_precision()  # Set to float32 (default)
bores.use_64bit_precision()  # Set to float64
dtype = bores.get_dtype()    # Get current dtype
```

---

## Enums and Constants

### `bores.FluidPhase`

```python
bores.FluidPhase.OIL
bores.FluidPhase.WATER
bores.FluidPhase.GAS
```

### `bores.Orientation`

```python
bores.Orientation.X
bores.Orientation.Y
bores.Orientation.Z
```

### `bores.WettabilityType` / `bores.Wettability`

```python
bores.WettabilityType.WATER_WET
bores.WettabilityType.OIL_WET
bores.Wettability.WATER_WET
bores.Wettability.OIL_WET
```

### Constants

```python
bores.c.DAYS_PER_YEAR  # 365.25
```

---

## Storage

### Save/Load

```python
# Save
model.to_file("model.h5")
config.to_file("config.yaml")
run.to_file("run.h5")

# Load
model = bores.ReservoirModel.from_file("model.h5")
config = bores.Config.from_file("config.yaml")
run = bores.Run.from_files(
    model_path="model.h5",
    config_path="config.yaml",
    pvt_table_path="pvt.h5",
)
```

---

## More Information

For detailed usage examples:

- [Tutorials](../tutorials/index.md) - Step-by-step learning
- [User Guide](../guides/index.md) - Comprehensive documentation
- [Examples](../examples/index.md) - Complete working code

For conceptual understanding:

- [Core Concepts](../getting-started/core-concepts.md)
- [Glossary](glossary.md)
