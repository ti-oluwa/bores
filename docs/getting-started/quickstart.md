# Quick Start

Let's build and run your first reservoir simulation in **10 minutes**!

---

## What We'll Build

A simple 3D reservoir model with:

- Single production well
- Natural depletion (primary recovery)
- Heterogeneous properties
- 30-day simulation

!!! tip "Copy and Run"
    Copy the complete code at the end into a Python file and run it. We'll explain each step below.

---

## Step 1: Import and Setup

```python
import bores
import numpy as np

# Use 32-bit precision (default - saves memory)
bores.use_32bit_precision()
```

**What's happening?**

- Import BORES framework
- Set precision to 32-bit float (half the memory of 64-bit, still accurate)

---

## Step 2: Define Grid Dimensions

```python
# Grid size: 20x20x10 cells
grid_shape = (20, 20, 10)  # (nx, ny, nz)

# Cell size: 100ft x 100ft horizontal
cell_dimension = (100.0, 100.0)  # (dx, dy) in feet
```

!!! info "Grid Anatomy"
    - **grid_shape**: Number of cells in each direction (x, y, z)
    - **cell_dimension**: Horizontal cell size (dz comes from thickness)
    - Total cells: 20 × 20 × 10 = 4,000 cells

---

## Step 3: Build Property Grids

### Thickness (varies by layer)

```python
# 10 layers with different thicknesses
thickness_values = bores.array([30, 20, 25, 30, 25, 30, 20, 25, 30, 25])  # ft

thickness_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=thickness_values,
    orientation=bores.Orientation.Z,  # Stack along z-axis
)
```

### Porosity (varies by layer)

```python
porosity_values = bores.array([0.04, 0.07, 0.09, 0.10, 0.08,
                                0.12, 0.14, 0.16, 0.11, 0.08])

porosity_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=porosity_values,
    orientation=bores.Orientation.Z,
)
```

### Pressure (increases with depth)

```python
reservoir_top_depth = 8000.0  # ft
pressure_gradient = 0.38  # psi/ft

layer_depths = reservoir_top_depth + np.cumsum(np.concatenate([[0], thickness_values[:-1]]))
layer_pressures = 14.7 + (layer_depths * pressure_gradient)

pressure_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_pressures,
    orientation=bores.Orientation.Z,
)
```

!!! tip "Pressure Gradient"
    - Water: ~0.433 psi/ft
    - Oil: ~0.35-0.40 psi/ft (lighter)
    - We use 0.38 psi/ft for an oil reservoir

### Permeability (anisotropic)

```python
# Horizontal permeability varies by layer (mD)
x_perm_values = bores.array([12, 25, 40, 18, 55, 70, 90, 35, 48, 22])

x_permeability_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=x_perm_values,
    orientation=bores.Orientation.Z,
)

# Y-direction: 80% of X-direction
y_permeability_grid = x_permeability_grid * 0.8

# Vertical: 10% of horizontal (typical for layered rocks)
z_permeability_grid = x_permeability_grid * 0.1

absolute_permeability = bores.RockPermeability(
    x=x_permeability_grid,
    y=y_permeability_grid,
    z=z_permeability_grid,
)
```

### Uniform Properties

```python
# Temperature increases with depth
surface_temp = 60.0  # °F
temp_gradient = 0.015  # °F/ft
layer_temperatures = surface_temp + (layer_depths * temp_gradient)
temperature_grid = bores.layered_grid(grid_shape, layer_temperatures, bores.Orientation.Z)

# Simple uniform properties
oil_viscosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.5)  # cP
oil_compressibility_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.2e-5)  # 1/psi
oil_specific_gravity_grid = bores.uniform_gridgrid_shape=(grid_shape, value=0.845)  # ~36° API
gas_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.65)  # relative to air
```

---

## Step 4: Initialize Saturations

```python
# Saturation endpoints
connate_water_sat_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.12)
residual_oil_sat_water_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.25)
residual_oil_sat_gas_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.15)
residual_gas_sat_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.045)
irreducible_water_sat_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.15)

# Fluid contacts
goc_depth = 8060.0  # ft
owc_depth = 8220.0  # ft

depth_grid = bores.depth_grid(thickness_grid)

water_sat_grid, oil_sat_grid, gas_sat_grid = bores.build_saturation_grids(
    depth_grid=depth_grid,
    gas_oil_contact=goc_depth - reservoir_top_depth,
    oil_water_contact=owc_depth - reservoir_top_depth,
    connate_water_saturation_grid=connate_water_sat_grid,
    residual_oil_saturation_water_grid=residual_oil_sat_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_sat_gas_grid,
    residual_gas_saturation_grid=residual_gas_sat_grid,
    porosity_grid=porosity_grid,
    use_transition_zones=True,
    oil_water_transition_thickness=12.0,  # ft
    gas_oil_transition_thickness=8.0,  # ft
)
```

!!! info "Fluid Contacts"
    - **GOC** (8060 ft): Above this = gas cap, below = oil
    - **OWC** (8220 ft): Above this = oil, below = water
    - **Transition zones**: Smooth gradients (more realistic)

---

## Step 5: Build the Model

```python
# Bubble point: 400 psi below reservoir pressure (undersaturated oil)
bubble_point_grid = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=layer_pressures - 400.0,
    orientation=bores.Orientation.Z,
)

model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    oil_bubble_point_pressure_grid=bubble_point_grid,
    absolute_permeability=absolute_permeability,
    porosity_grid=porosity_grid,
    temperature_grid=temperature_grid,
    rock_compressibility=4.5e-6,  # 1/psi
    oil_saturation_grid=oil_sat_grid,
    water_saturation_grid=water_sat_grid,
    gas_saturation_grid=gas_sat_grid,
    oil_viscosity_grid=oil_viscosity_grid,
    oil_specific_gravity_grid=oil_specific_gravity_grid,
    oil_compressibility_grid=oil_compressibility_grid,
    gas_gravity_grid=gas_gravity_grid,
    residual_oil_saturation_water_grid=residual_oil_sat_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_sat_gas_grid,
    irreducible_water_saturation_grid=irreducible_water_sat_grid,
    connate_water_saturation_grid=connate_water_sat_grid,
    residual_gas_saturation_grid=residual_gas_sat_grid,
    reservoir_gas="methane",  # CoolProp-supported: methane, ethane, propane, CO2, nitrogen, hydrogen
)
```

!!! info "CoolProp Supported Gases"
    The `reservoir_gas` parameter accepts CoolProp fluid names:

    - **Hydrocarbons**: "methane", "ethane", "propane", "n-butane", "isobutane"
    - **Injection gases**: "CO2" (carbon dioxide), "nitrogen", "hydrogen"
    - **Custom**: Any fluid name supported by CoolProp library

    These are used for accurate gas property calculations at reservoir conditions.

!!! success "Model Built!"
    You now have a complete 3D reservoir model with 4,000 cells, heterogeneous properties, and realistic fluid distribution.

---

## Step 6: Add a Production Well

```python
production_clamp = bores.ProductionClamp() # Clamps rate to prevent backflow/injection in the production well

# Well rate control mechanism
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-150, # STB/day (negative = production)
        target_phase="oil",
        bhp_limit=1200, # psi (minimum bottomhole pressure)
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-500, # SCF/day (negative = production)
        target_phase="gas",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-10, # STB/day (negative = production)
        target_phase="water",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((10, 10, 3), (10, 10, 6))],  # Just one perforation interval at the enter of reservoir, at layers 3-6
    radius=0.354,  # ft (8.5" hole)
    control=control,
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.845,
            molecular_weight=180.0,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.05,
            molecular_weight=18.015,
        ),
    ),
    skin_factor=2.5,  # Well damage
    is_active=True,
)

wells = bores.wells_(injectors=None, producers=[producer])
```

!!! tip "Well Location"
    `(10, 10, 3)` to `(10, 10, 6)` means:

      - X-index: 10 (middle of 20 cells)
      - Y-index: 10 (middle of 20 cells)
      - Z-index: 3 to 6 (perforated in 4 layers)

---

## Step 7: Configure Rock-Fluid Properties

```python
# Relative permeability
relperm_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.Wettability.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Capillary pressure
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,  # psi
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,  # psi
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relperm_table,
    capillary_pressure_table=capillary_pressure_table,
)
```

---

## Step 8: Setup Simulation Config

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=4),
    max_step_size=bores.Time(days=2),
    min_step_size=bores.Time(minutes=10),
    simulation_time=bores.Time(days=30),  # 30-day simulation
    max_cfl_number=0.9,
    ramp_up_factor=1.2,  # Increase timestep by 20% on success
    backoff_factor=0.5,  # Halve timestep on failure
)

config = bores.Config(
    timer=timer,
    wells=wells,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",  # Implicit pressure, explicit saturation
    output_frequency=1,
    log_interval=5,
    pressure_solver="bicgstab",
    pressure_preconditioner="ilu",
    max_iterations=200,
)
```

---

## Step 9: Run Simulation

```python
run = bores.Run(model=model, config=config)

print("Starting simulation...")
for state in run():
    fluid_props = state.model.fluid_properties
    avg_pressure = fluid_props.pressure_grid.mean()
    print(f"Day {state.time / 86400:.1f}: Avg Pressure = {avg_pressure:.1f} psi")

print("Simulation complete!")
```

Expected output:

```txt
Starting simulation...
Day 0.2: Avg Pressure = 3085.4 psi
Day 0.4: Avg Pressure = 3082.1 psi
Day 0.7: Avg Pressure = 3078.3 psi
...
Day 30.0: Avg Pressure = 2912.7 psi
Simulation complete!
```

---

## Complete Code

Here's the full working example:

```python
import bores
import numpy as np

bores.use_32bit_precision()

# Grid definition
grid_shape = (20, 20, 10)
cell_dimension = (100.0, 100.0)

# Thickness
thickness_values = bores.array([30, 20, 25, 30, 25, 30, 20, 25, 30, 25])
thickness_grid = bores.layered_grid(
    grid_shape=grid_shape, 
    layer_values=thickness_values, 
    orientation=bores.Orientation.Z
)

# Porosity
porosity_values = bores.array([0.04, 0.07, 0.09, 0.10, 0.08, 0.12, 0.14, 0.16, 0.11, 0.08])
porosity_grid = bores.layered_grid(
    grid_shape=grid_shape, 
    layer_values=porosity_values, 
    orientation=bores.Orientation.Z
)

# Pressure
reservoir_top_depth = 8000.0
pressure_gradient = 0.38
layer_depths = reservoir_top_depth + np.cumsum(np.concatenate([[0], thickness_values[:-1]]))
layer_pressures = 14.7 + (layer_depths * pressure_gradient)
pressure_grid = bores.layered_grid(
    grid_shape=grid_shape, 
    layer_values=layer_pressures, 
    orientation=bores.Orientation.Z
)

# Permeability
x_perm_values = bores.array([12, 25, 40, 18, 55, 70, 90, 35, 48, 22])
x_permeability_grid = bores.layered_grid(
    grid_shape=grid_shape, 
    layer_values=x_perm_values, 
    orientation=bores.Orientation.Z
)
y_permeability_grid = x_permeability_grid * 0.8
z_permeability_grid = x_permeability_grid * 0.1
absolute_permeability = bores.RockPermeability(x=x_permeability_grid, y=y_permeability_grid, z=z_permeability_grid)

# Temperature
surface_temp = 60.0
temp_gradient = 0.015
layer_temperatures = surface_temp + (layer_depths * temp_gradient)
temperature_grid = bores.layered_grid(
    grid_shape=grid_shape, 
    layer_values=layer_temperatures, 
    orientation=bores.Orientation.Z
)

# Uniform properties
oil_viscosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.5)
oil_compressibility_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.2e-5)
oil_specific_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.845)
gas_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.65)

# Saturation endpoints
connate_water_sat_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.12)
residual_oil_sat_water_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.25)
residual_oil_sat_gas_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.15)
residual_gas_sat_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.045)
irreducible_water_sat_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.15)

# Initialize saturations
goc_depth = 8060.0
owc_depth = 8220.0
depth_grid = bores.depth_grid(thickness_grid)
water_sat_grid, oil_sat_grid, gas_sat_grid = bores.build_saturation_grids(
    depth_grid=depth_grid,
    gas_oil_contact=goc_depth - reservoir_top_depth,
    oil_water_contact=owc_depth - reservoir_top_depth,
    connate_water_saturation_grid=connate_water_sat_grid,
    residual_oil_saturation_water_grid=residual_oil_sat_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_sat_gas_grid,
    residual_gas_saturation_grid=residual_gas_sat_grid,
    porosity_grid=porosity_grid,
    use_transition_zones=True,
    oil_water_transition_thickness=12.0,
    gas_oil_transition_thickness=8.0,
)

# Bubble point
bubble_point_grid = bores.layered_grid(
    grid_shape=grid_shape, 
    layer_values=layer_pressures - 400.0, 
    orientation=bores.Orientation.Z
)

# Build model
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    oil_bubble_point_pressure_grid=bubble_point_grid,
    absolute_permeability=absolute_permeability,
    porosity_grid=porosity_grid,
    temperature_grid=temperature_grid,
    rock_compressibility=4.5e-6,
    oil_saturation_grid=oil_sat_grid,
    water_saturation_grid=water_sat_grid,
    gas_saturation_grid=gas_sat_grid,
    oil_viscosity_grid=oil_viscosity_grid,
    oil_specific_gravity_grid=oil_specific_gravity_grid,
    oil_compressibility_grid=oil_compressibility_grid,
    gas_gravity_grid=gas_gravity_grid,
    residual_oil_saturation_water_grid=residual_oil_sat_water_grid,
    residual_oil_saturation_gas_grid=residual_oil_sat_gas_grid,
    irreducible_water_saturation_grid=irreducible_water_sat_grid,
    connate_water_saturation_grid=connate_water_sat_grid,
    residual_gas_saturation_grid=residual_gas_sat_grid,
    reservoir_gas="methane",
)

# Production well
production_clamp = bores.ProductionClamp() # Clamps rate to prevent backflow/injection in the production well
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-150, # STB/day (negative = production)
        target_phase="oil",
        bhp_limit=1200, # psi (minimum bottomhole pressure)
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-500, # SCF/day (negative = production)
        target_phase="gas",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-10, # STB/day (negative = production)
        target_phase="water",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
)
producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((10, 10, 3), (10, 10, 6))],
    radius=0.354,
    control=control,
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.845,
            molecular_weight=180.0,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.05,
            molecular_weight=18.015,
        ),
    ),
    skin_factor=2.5,
    is_active=True,
)
wells = bores.wells_(injectors=None, producers=[producer])

# Rock-fluid properties
relperm_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.Wettability.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relperm_table,
    capillary_pressure_table=capillary_pressure_table,
)

# Simulation config
timer = bores.Timer(
    initial_step_size=bores.Time(hours=4),
    max_step_size=bores.Time(days=2),
    min_step_size=bores.Time(minutes=10),
    simulation_time=bores.Time(days=30),
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
)
config = bores.Config(
    timer=timer,
    wells=wells,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",
    output_frequency=1,
    log_interval=5,
    pressure_solver="bicgstab",
    pressure_preconditioner="ilu",
    max_iterations=200,
)

# Run simulation
run = bores.Run(model=model, config=config)
print("Starting simulation...")
for state in run():
    fluid_props = state.model.fluid_properties
    avg_pressure = fluid_props.pressure_grid.mean()
    print(f"Day {state.time / 86400:.1f}: Avg Pressure = {avg_pressure:.1f} psi")
print("Simulation complete!")
```

---

## What's Next?

**Congratulations!** You have successfully run your first BORES simulation.

Now explore:

- **[Core Concepts](core-concepts.md)** - Understand how BORES works
- **[Tutorials](../tutorials/index.md)** - Build more complex simulations
- **[Examples](../examples/index.md)** - Complete working examples

!!! tip "Save Your Work"
    You can save models and results:
    ```python
    model.save("my_model.h5")
    ```
    Learn more in [Storage & Serialization](../advanced/storage-serialization.md)
