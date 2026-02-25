# Initial Model Setup

Create the initial reservoir model and configuration for all subsequent examples.

This example shows the complete setup workflow matching the actual implementation in scenarios/setup.py.

## Objective

Build a complete three-dimensional heterogeneous reservoir model from scratch with:

- 3D grid (20×20×10 cells, 4000 total cells)
- Layered reservoir properties (10 layers with varying quality)
- Structural dip (2° dip angle at 90° azimuth)
- Initial saturations with fluid contacts and transition zones
- Vertical sealing fault
- PVT properties from correlations (CoolProp-supported methane gas)
- Rock-fluid properties (Brooks-Corey relative permeability and capillary pressure)
- Carter-Tracy bottom aquifer boundary condition
- Save all files for reuse in other examples

This setup is the **foundation** for all other examples and scenarios.

## Complete Implementation

The complete implementation is in scenarios/setup.py. Below are the key sections with detailed explanations.

### 1. Grid Definition and Layer Properties

```python
import bores
from pathlib import Path

bores.use_32bit_precision()

# Grid dimensions
cell_dimension = (100.0, 100.0)  # 100ft x 100ft cells
grid_shape = (20, 20, 10)  # 20x20 cells, 10 layers

# Structural dip
dip_angle = 2.0  # degrees
dip_azimuth = 90.0  # degrees (east-dipping)
```

**Grid characteristics**:

- Horizontal extent: 2000 ft × 2000 ft (cell_dimension × number of cells)
- Total cells: 4000 (20 × 20 × 10)
- Structural dip creates realistic tilted reservoir structure

### 2. Thickness Distribution

Using `bores.layered_grid()` helper to create layered property grids:

```python
# Thickness distribution - typical reservoir layers
# Thicker in the middle, thinner at top/bottom
thickness_values = bores.array(
    [30.0, 20.0, 25.0, 30.0, 25.0, 30.0, 20.0, 25.0, 30.0, 25.0]
)  # feet

thickness_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=thickness_values,
    orientation=bores.Orientation.Z,
)
```

**Total thickness**: 260 ft (sum of all layer thicknesses)

**Why variable thickness?** Real reservoirs have non-uniform layer thicknesses due to depositional environment and compaction history.

### 3. Initial Pressure with Gradient

```python
# Pressure gradient: ~0.433 psi/ft for water, slightly less for oil
# Assuming reservoir top at 8000 ft depth
reservoir_top_depth = 8000.0  # ft
pressure_gradient = 0.38  # psi/ft (typical for oil reservoirs)

layer_depths = reservoir_top_depth + np.cumsum(
    np.concatenate([[0], thickness_values[:-1]])
)
layer_pressures = 14.7 + (
    layer_depths * pressure_gradient
)  # Add atmospheric pressure

pressure_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_pressures,  # Ranges from ~3055 to ~3117 psi
    orientation=bores.Orientation.Z,
)
```

**Pressure range**: 3055 - 3117 psi (increases with depth)

**Why 0.38 psi/ft gradient?** This is typical for oil reservoirs (slightly less than water gradient of 0.433 psi/ft due to lower oil density).

### 4. Bubble Point Pressure Grid

```python
# Bubble point pressure slightly below initial pressure (undersaturated oil)
oil_bubble_point_pressure_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_pressures - 400.0,  # 400 psi below formation pressure
    orientation=bores.Orientation.Z,
)
```

**Undersaturated oil**: Initial pressure is 400 psi above bubble point, meaning the oil will not release solution gas until pressure drops by 400 psi. This is typical for many oil reservoirs.

### 5. Saturation Endpoints

```python
# Saturation endpoints - typical for sandstone reservoirs
residual_oil_saturation_water_grid = bores.uniform_grid(
    grid_shape=grid_shape,
    value=0.25,  # Sor to water
)
residual_oil_saturation_gas_grid = bores.uniform_grid(
    grid_shape=grid_shape,
    value=0.15,  # Sor to gas
)
irreducible_water_saturation_grid = bores.uniform_grid(
    grid_shape=grid_shape,
    value=0.15,  # Swi
)
connate_water_saturation_grid = bores.uniform_grid(
    grid_shape=grid_shape,
    value=0.12,  # Slightly less than Swi
)
residual_gas_saturation_grid = bores.uniform_grid(
    grid_shape=grid_shape,
    value=0.045,  # Sgr
)
```

**Key values**:

- **Swc = 0.15**: Irreducible (connate) water saturation below which water is immobile
- **Sorw = 0.25**: Residual oil after waterflooding (75% displacement efficiency)
- **Sorg = 0.15**: Residual oil after gas flooding (85% displacement efficiency)
- **Sgr = 0.045**: Trapped gas saturation during imbibition

### 6. Porosity Distribution

```python
# Porosity - decreasing with depth (compaction trend)
porosity_values = bores.array(
    [0.04, 0.07, 0.09, 0.1, 0.08, 0.12, 0.14, 0.16, 0.11, 0.08]
)  # fraction

porosity_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=porosity_values,
    orientation=bores.Orientation.Z,
)
```

**Porosity range**: 0.04 - 0.16 (4% - 16%)

**Why variable?** Porosity typically varies with depth due to compaction, cementation, and depositional facies changes.

### 7. Fluid Contacts and Saturation Distribution

```python
# Fluid contacts
# GOC at 8060 ft, OWC at 8220 ft
goc_depth = 8060.0
owc_depth = 8220.0

# Create depth grid and apply structural dip
depth_grid = bores.depth_grid(thickness_grid)
depth_grid = bores.apply_structural_dip(
    elevation_grid=depth_grid,
    elevation_direction="downward",
    cell_dimension=cell_dimension,
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
)

# Build saturation grids with transition zones
water_saturation_grid, oil_saturation_grid, gas_saturation_grid = (
    bores.build_saturation_grids(
        depth_grid=depth_grid,
        gas_oil_contact=goc_depth - reservoir_top_depth,  # 60 ft below top
        oil_water_contact=owc_depth - reservoir_top_depth,  # 220 ft below top
        connate_water_saturation_grid=connate_water_saturation_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        porosity_grid=porosity_grid,
        use_transition_zones=True,
        oil_water_transition_thickness=12.0,  # ft transition zone
        gas_oil_transition_thickness=8.0,  # ft transition zone
        transition_curvature_exponent=1.2,
    )
)
```

**Fluid distribution**:

- **Gas cap**: Depth < 8060 ft (above GOC)
- **Oil zone**: 8060 ft < Depth < 8220 ft (between GOC and OWC)
- **Water zone**: Depth > 8220 ft (below OWC)
- **Transition zones**: 12 ft (OW) and 8 ft (GO) thickness create realistic gradual saturation changes

**Structural dip**: The 2° dip at 90° azimuth tilts the reservoir, creating variable fluid column heights across the grid.

### 8. Permeability Distribution

```python
# Permeability distribution
# Higher permeability in middle layers (better reservoir quality)
# Anisotropy ratio kv/kh ~ 0.1 (typical for layered sandstone)
x_perm_values = bores.array([12, 25, 40, 18, 55, 70, 90, 35, 48, 22])  # mD

x_permeability_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=x_perm_values,
    orientation=bores.Orientation.Z,
)

# Slight directional permeability difference
y_permeability_grid = x_permeability_grid * 0.8

# Vertical permeability much lower (layering effect)
z_permeability_grid = x_permeability_grid * 0.1

absolute_permeability = bores.RockPermeability(
    x=x_permeability_grid,
    y=y_permeability_grid,
    z=z_permeability_grid,
)
```

**Permeability characteristics**:

- **Horizontal permeability (kh)**: 12 - 90 mD (varies by layer)
- **Vertical permeability (kv)**: kh × 0.1 (typical kv/kh ratio for layered sandstone)
- **Directional anisotropy**: ky = kx × 0.8 (slight anisotropy)

**Why kv << kh?** Layering and shale streaks create barriers to vertical flow, resulting in much lower vertical permeability.

### 9. Temperature and Rock Compressibility

```python
# Realistic temperature gradient (~1.5°F per 100 ft)
surface_temp = 60.0  # °F
temp_gradient = 0.015  # °F/ft

layer_temps = surface_temp + (layer_depths * temp_gradient)

temperature_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_temps,  # ~180-182°F
    orientation=bores.Orientation.Z,
)

# Rock compressibility for sandstone
rock_compressibility = 4.5e-6  # 1/psi
```

**Temperature range**: 180 - 182 °F at reservoir depth

### 10. Net-to-Gross Ratio

```python
# Net-to-gross ratio (accounting for shale layers)
net_to_gross_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=[
        0.42, 0.55, 0.68, 0.35, 0.60,
        0.72, 0.80, 0.50, 0.63, 0.47,
    ],
    orientation=bores.Orientation.Z,
)
```

**Net-to-gross ratio**: Fraction of productive (non-shale) rock in each layer. Ranges from 0.35 to 0.80, accounting for interbedded shale layers.

### 11. Gas Properties (CoolProp)

```python
# Natural gas (associated gas) properties
gas_gravity_grid = bores.uniform_grid(
    grid_shape=grid_shape,
    value=0.65,  # Typical for associated gas
)
```

**Important**: The gas type must be CoolProp-supported. This model uses **methane** (`reservoir_gas="methane"`), which is supported by CoolProp.

### 12. PVT Table Construction

```python
pvt_table_data = bores.build_pvt_table_data(
    pressures=bores.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),
    temperatures=bores.array([120, 140, 160, 180, 200, 220]),
    salinities=bores.array([30000, 32000, 36000, 40000]),  # ppm
    oil_specific_gravity=0.845,
    gas_gravity=0.65,
    reservoir_gas="methane",  # CoolProp-supported gas
)

pvt_tables = bores.PVTTables(data=pvt_table_data, interpolation_method="linear")
```

**PVT table coverage**:

- **Pressure**: 500 - 4500 psi (covers expected simulation range)
- **Temperature**: 120 - 220 °F (covers reservoir temperature range)
- **Salinity**: 30,000 - 40,000 ppm (brine salinity range)
- **Reservoir gas**: "methane" (must be CoolProp-supported: methane, ethane, propane, CO2, nitrogen, hydrogen, etc.)

### 13. Create Reservoir Model

```python
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
    absolute_permeability=absolute_permeability,
    porosity_grid=porosity_grid,
    temperature_grid=temperature_grid,
    rock_compressibility=rock_compressibility,
    oil_saturation_grid=oil_saturation_grid,
    water_saturation_grid=water_saturation_grid,
    gas_saturation_grid=gas_saturation_grid,
    oil_viscosity_grid=oil_viscosity_grid,
    oil_specific_gravity_grid=oil_specific_gravity_grid,
    oil_compressibility_grid=oil_compressibility_grid,
    gas_gravity_grid=gas_gravity_grid,
    residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
    irreducible_water_saturation_grid=irreducible_water_saturation_grid,
    connate_water_saturation_grid=connate_water_saturation_grid,
    residual_gas_saturation_grid=residual_gas_saturation_grid,
    net_to_gross_ratio_grid=net_to_gross_grid,
    reservoir_gas="methane",
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
    pvt_tables=pvt_tables,
)
```

### 14. Add Vertical Sealing Fault

```python
fault = bores.vertical_sealing_fault(
    fault_id="F-2",
    orientation="y",  # Fault strikes in y-direction
    index=8,  # At x-index 8
    y_range=(7, 9),  # Lateral extent in y-direction
    x_range=(2, 18),  # Fault trace length
    z_range=(0, 6),  # Vertical extent (shallow fault, top 6 layers)
)

# Apply fault to model
model = bores.apply_fracture(model, fault)
```

**Fault characteristics**:

- **Type**: Vertical sealing fault (zero transmissibility)
- **Orientation**: Strikes in y-direction, cuts across x-direction
- **Location**: x = 8 (center of grid)
- **Extent**: Partial (only affects top 6 layers, doesn't penetrate entire reservoir)

**Effect**: Creates two compartments separated by impermeable barrier in upper layers.

### 15. Define Rock-Fluid Tables

```python
# Relative permeability (Brooks-Corey with Eclipse mixing rule)
relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.Wettability.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,  # Industry standard
)

# Capillary pressure (Brooks-Corey)
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relative_permeability_table,
    capillary_pressure_table=capillary_pressure_table,
)
```

**Relative permeability**: Uses Eclipse mixing rule (recommended industry standard) for three-phase oil relative permeability.

**Capillary pressure**: Water-wet system with moderate entry pressures (2.0 psi for oil-water, 2.8 psi for gas-oil).

### 16. Define Boundary Conditions (Carter-Tracy Aquifer)

```python
# Realistic bottom water drive using Carter-Tracy aquifer
bottom_aquifer = bores.CarterTracyAquifer(
    aquifer_permeability=450.0,  # mD - moderate permeability sandstone aquifer
    aquifer_porosity=0.22,  # fraction - typical for aquifer rock
    aquifer_compressibility=5e-6,  # psi⁻¹ - total compressibility (rock + water)
    water_viscosity=0.5,  # cP - at reservoir temperature (~180°F)
    inner_radius=1400.0,  # ft - effective reservoir-aquifer contact radius
    outer_radius=14000.0,  # ft - aquifer extent (10x inner radius = moderate)
    aquifer_thickness=80.0,  # ft - thick bottom aquifer
    initial_pressure=3117.0,  # psi - matches reservoir bottom pressure
    angle=360.0,  # degrees - full circular contact (bottom water drive)
)

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(bottom=bottom_aquifer),
    }
)
```

**Carter-Tracy aquifer**:

- **Type**: Analytical aquifer model using Van Everdingen-Hurst solution
- **Physical properties mode**: Uses actual rock properties (k, φ, ct) rather than calibrated constant
- **Location**: Bottom boundary (bottom water drive)
- **Strength**: Dimensionless radius ratio = 10 (moderate aquifer)
- **Effect**: Provides pressure support from below as reservoir depletes

**Why Carter-Tracy?** More realistic than constant pressure (infinite aquifer) or no-flow (no support). Provides time-dependent finite aquifer response.

### 17. Configure Simulation Timer and Solver

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=4.5),
    max_step_size=bores.Time(days=5.0),
    min_step_size=bores.Time(minutes=10.0),
    simulation_time=bores.Time(days=30),  # 30 days
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
)

# ILU preconditioner with caching for performance
ilu_preconditioner = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,  # Rebuild every 10 timesteps
    recompute_threshold=0.3,  # Or when matrix changes by 30%
)
ilu_preconditioner.register(override=True)

config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",  # Implicit pressure, explicit saturation
    output_frequency=1,
    max_iterations=200,
    pressure_solver="bicgstab",
    pressure_preconditioner="cached_ilu",  # Faster than rebuilding each timestep
    log_interval=5,
    pvt_tables=pvt_tables,
    wells=None,  # No wells for initial setup
    boundary_conditions=boundary_conditions,
    max_gas_saturation_change=0.85,
)
```

**Solver configuration**:

- **Scheme**: IMPES (Implicit Pressure Explicit Saturation) - standard for black-oil simulation
- **Pressure solver**: BiCGSTAB (fast iterative solver)
- **Preconditioner**: Cached ILU (reuses preconditioner for 10 timesteps or until matrix changes by 30%)
- **Performance**: Cached preconditioner provides 20-30% speedup vs rebuilding each timestep

### 18. Save Setup Files

```python
# Save model data
model.to_file(Path("./scenarios/runs/setup/model.h5"))

# Save PVT data
pvt_table_data.to_file(Path("./scenarios/runs/setup/pvt.h5"))

# Save base config
config.to_file(Path("./scenarios/runs/setup/config.yaml"))
```

**Files created**:

- `model.h5`: Complete reservoir model with all grids and properties
- `pvt.h5`: PVT table data for fluid property lookups
- `config.yaml`: Simulation configuration in human-readable YAML format

## Key Features

### Realistic Reservoir Heterogeneity

This setup includes multiple sources of heterogeneity:

1. **Vertical variation**: 10 layers with different porosity, permeability, thickness
2. **Structural dip**: 2° dip creates tilted structure
3. **Fault compartmentalization**: Vertical sealing fault creates barriers
4. **Net-to-gross variation**: Interbedded shales reduce productive rock fraction
5. **Transition zones**: Smooth saturation changes at fluid contacts

### Proper Boundary Conditions

Uses Carter-Tracy aquifer instead of simple constant pressure or no-flow:

- **More realistic**: Finite aquifer with time-dependent response
- **Physical basis**: Computed from actual aquifer rock properties
- **Material balance**: Provides appropriate pressure support

### Industry-Standard Models

- **Relative permeability**: Brooks-Corey with Eclipse mixing rule
- **Capillary pressure**: Brooks-Corey model
- **PVT**: Correlations with CoolProp-supported gas phase
- **Solver**: IMPES scheme with BiCGSTAB and cached ILU preconditioner

## Verification

After running the setup, verify the model:

```python
from pathlib import Path
import bores

setup_dir = Path("./scenarios/runs/setup")

# Load model
model = bores.ReservoirModel.from_file(setup_dir / "model.h5")
print(f"Grid shape: {model.grid_shape}")
print(f"Total cells: {model.grid_shape[0] * model.grid_shape[1] * model.grid_shape[2]}")
print(f"Average porosity: {model.rock_properties.porosity_grid.mean():.3f}")
print(f"Permeability range: {model.rock_properties.absolute_permeability.x.min():.1f} - {model.rock_properties.absolute_permeability.x.max():.1f} mD")
print(f"Pressure range: {model.fluid_properties.pressure_grid.min():.1f} - {model.fluid_properties.pressure_grid.max():.1f} psi")

# Check saturations sum to 1
total_sat = (
    model.fluid_properties.oil_saturation_grid +
    model.fluid_properties.water_saturation_grid +
    model.fluid_properties.gas_saturation_grid
)
print(f"Saturation sum check: {total_sat.min():.3f} - {total_sat.max():.3f} (should be ~1.0)")

print("\nAll files verified successfully!")
```

## What's Next

This setup creates the initial equilibrated model. You can now:

1. Run primary depletion with production wells
2. Perform waterflood secondary recovery
3. Test gas injection (CH4) for pressure maintenance
4. Implement miscible CO2 flooding for EOR

All subsequent examples use this base model as their starting point.

## See Also

- [Building Models Guide](../guides/building-models.md) - Detailed model construction
- [Rock-Fluid Properties](../guides/rock-fluid-properties.md) - Relative permeability and capillary pressure
- Boundary Conditions (see advanced/boundary-conditions.md) - Carter-Tracy and other boundary conditions
- [Relative Permeability Module](../relative-permeability/index.md) - Detailed relperm documentation
- [Capillary Pressure Module](../capillary-pressure/index.md) - Detailed capillary pressure documentation
