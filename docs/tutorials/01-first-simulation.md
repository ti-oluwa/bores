# Tutorial 1: Your First Simulation

Build a simple reservoir model and run a primary depletion simulation in 15 minutes.

**Time**: 15 minutes | **Level**: Beginner

---

## What You'll Build

A 1D reservoir model with the following characteristics:

- Geometry: 50 cells arranged in a vertical 1D column (grid shape 1×1×50)
- Production: Single vertical production well perforated in the middle section
- Fluid system: Undersaturated oil (initial pressure above bubble point)
- Recovery mechanism: Primary depletion over 30 days
- Numerical scheme: IMPES (Implicit Pressure, Explicit Saturation)

---

## Objectives

By the end of this tutorial, you will know how to:

- Construct a 1D reservoir model using the `reservoir_model()` factory function
- Define rock and fluid properties as numpy grids
- Configure a production well with adaptive rate/BHP control
- Set up relative permeability and capillary pressure models
- Configure simulation timing, solvers, and convergence parameters
- Execute a simulation and monitor results during runtime

---

## Step 1: Setup

Create a new Python file `first_simulation.py`:

```python
import bores

# Use 32-bit precision (default)
bores.use_32bit_precision()

# 1D grid: 1x1x50 cells
grid_shape = (1, 1, 50)
cell_dimension = (100.0, 100.0)  # ft (not used in 1D, but required)
```

!!! info "1D Simulation"
    A 1D model has shape `(1, 1, nz)` - single column of cells. Great for testing and learning!

---

## Step 2: Define Properties

```python
# Uniform thickness
thickness = bores.uniform_grid(grid_shape, value=20.0)  # ft

# Pressure increases with depth
pressure_values = bores.array([3000 + i * 2 for i in range(50)])  # psi
pressure = bores.layered_grid(grid_shape, pressure_values, bores.Orientation.Z)

# Uniform properties
porosity = bores.uniform_grid(grid_shape, 0.20)
permeability_val = bores.uniform_grid(grid_shape, 100.0)  # mD
temperature = bores.uniform_grid(grid_shape, 180.0)  # °F
oil_viscosity = bores.uniform_grid(grid_shape, 1.5)  # cP
oil_compressibility = bores.uniform_grid(grid_shape, 1.2e-5)  # 1/psi
oil_sg = bores.uniform_grid(grid_shape, 0.85)
gas_gravity = bores.uniform_grid(grid_shape, 0.65)

# Permeability structure
perm = bores.RockPermeability(
    x=permeability_val,
    y=permeability_val,
    z=permeability_val * 0.1,  # Lower vertical perm
)

# Bubble point below initial pressure (undersaturated)
bubble_point = bores.layered_grid(
    grid_shape,
    [3000 + i * 2 - 300 for i in range(50)],  # 300 psi below pressure
    bores.Orientation.Z,
)
```

---

## Step 3: Initialize Saturations

```python
# Saturation endpoints
swc = bores.uniform_grid(grid_shape, 0.12)
sorw = bores.uniform_grid(grid_shape, 0.25)
sorg = bores.uniform_grid(grid_shape, 0.15)
sgr = bores.uniform_grid(grid_shape, 0.05)
swi = bores.uniform_grid(grid_shape, 0.15)

# Initial saturations: oil zone only (no gas cap, no water)
oil_sat = bores.uniform_grid(grid_shape, 0.80)  # 80% oil
water_sat = bores.uniform_grid(grid_shape, 0.15)  # 15% connate water
gas_sat = bores.uniform_grid(grid_shape, 0.05)  # 5% initial gas
```

!!! tip "Saturation Sum"
    Always ensure: `So + Sw + Sg = 1.0`

---

## Step 4: Build the Model

```python
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
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
    oil_viscosity_grid=oil_viscosity,
    oil_specific_gravity_grid=oil_sg,
    oil_compressibility_grid=oil_compressibility,
    gas_gravity_grid=gas_gravity,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    irreducible_water_saturation_grid=swi,
    connate_water_saturation_grid=swc,
    residual_gas_saturation_grid=sgr,
    reservoir_gas="methane",
)

print(f"Model built: {grid_shape} cells")
```

---

## Step 5: Add Production Well

```python
# Define produced fluids
produced_fluids = (
    bores.ProducedFluid(
        name="Oil",
        phase=bores.FluidPhase.OIL,
        specific_gravity=0.85,
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
)

# Create production well
producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((0, 0, 20), (0, 0, 30))],  # Middle section
    radius=0.354,  # ft
    control=bores.AdaptiveBHPRateControl(
        target_rate=-150,  # STB/day (negative for production)
        target_phase="oil",
        bhp_limit=1500,  # psi minimum
    ),
    produced_fluids=produced_fluids,
    skin_factor=0.0,  # No damage
    is_active=True,
)

wells = bores.wells_(injectors=None, producers=[producer])
print("Well added")
```

---

## Step 6: Configure Rock-Fluid Properties

```python
# Simple Brooks-Corey relative permeability
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.05,
    wettability=bores.Wettability.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=1.5,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=rel_perm,
    capillary_pressure_table=cap_pressure,
)
```

---

## Step 7: Setup Simulation

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=2),
    max_step_size=bores.Time(days=1),
    min_step_size=bores.Time(minutes=5),
    simulation_time=bores.Time(days=30),  # 30 days
    max_cfl_number=0.9,
    ramp_up_factor=1.15,
    backoff_factor=0.5,
)

config = bores.Config(
    timer=timer,
    wells=wells,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",
    output_frequency=1,
    log_interval=10,
    pressure_solver="bicgstab",
    pressure_preconditioner="ilu",
    max_iterations=150,
)

print("Config ready")
```

---

## Step 8: Run Simulation

```python
run = bores.Run(model=model, config=config)

print("\nStarting simulation...")
print(f"{'Day':>6} {'Pressure (psi)':>15} {'Oil Sat':>10}")
print("-" * 35)

for state in run():
    day = state.time / 86400
    avg_pressure = state.model.fluid_properties.pressure_grid.mean()
    avg_oil_sat = state.model.fluid_properties.oil_saturation_grid.mean()

    if state.step % 10 == 0:  # Print every 10 steps
        print(f"{day:6.1f} {avg_pressure:15.1f} {avg_oil_sat:10.3f}")

print("\nSimulation complete!")
```

---

## Expected Output

```
Model built: (1, 1, 50) cells
Well added
Config ready

Starting simulation...
   Day  Pressure (psi)   Oil Sat
-----------------------------------
   0.1          3048.3      0.800
   0.5          3041.2      0.798
   1.2          3032.8      0.795
   2.5          3021.4      0.791
   5.0          3005.2      0.785
  10.0          2975.1      0.773
  20.0          2930.8      0.756
  30.0          2895.4      0.742

Simulation complete!
```

!!! success "What's Happening?"
    - Pressure declines as oil is produced
    - Oil saturation decreases (more gas evolves)
    - Production rate adapts to maintain BHP limit

---

## Complete Code

```python
import bores

bores.use_32bit_precision()

# Grid
grid_shape = (1, 1, 50)
cell_dimension = (100.0, 100.0)

# Properties
thickness = bores.uniform_grid(grid_shape, 20.0)
pressure_values = bores.array([3000 + i * 2 for i in range(50)])
pressure = bores.layered_grid(grid_shape, pressure_values, bores.Orientation.Z)
porosity = bores.uniform_grid(grid_shape, 0.20)
permeability_val = bores.uniform_grid(grid_shape, 100.0)
temperature = bores.uniform_grid(grid_shape, 180.0)
oil_viscosity = bores.uniform_grid(grid_shape, 1.5)
oil_compressibility = bores.uniform_grid(grid_shape, 1.2e-5)
oil_sg = bores.uniform_grid(grid_shape, 0.85)
gas_gravity = bores.uniform_grid(grid_shape, 0.65)

perm = bores.RockPermeability(x=permeability_val, y=permeability_val, z=permeability_val * 0.1)
bubble_point = bores.layered_grid(grid_shape, [3000 + i * 2 - 300 for i in range(50)], bores.Orientation.Z)

# Saturations
swc = bores.uniform_grid(grid_shape, 0.12)
sorw = bores.uniform_grid(grid_shape, 0.25)
sorg = bores.uniform_grid(grid_shape, 0.15)
sgr = bores.uniform_grid(grid_shape, 0.05)
swi = bores.uniform_grid(grid_shape, 0.15)
oil_sat = bores.uniform_grid(grid_shape, 0.80)
water_sat = bores.uniform_grid(grid_shape, 0.15)
gas_sat = bores.uniform_grid(grid_shape, 0.05)

# Model
model = bores.reservoir_model(
    grid_shape=grid_shape, cell_dimension=cell_dimension, thickness_grid=thickness,
    pressure_grid=pressure, oil_bubble_point_pressure_grid=bubble_point,
    absolute_permeability=perm, porosity_grid=porosity, temperature_grid=temperature,
    rock_compressibility=4.5e-6, oil_saturation_grid=oil_sat, water_saturation_grid=water_sat,
    gas_saturation_grid=gas_sat, oil_viscosity_grid=oil_viscosity,
    oil_specific_gravity_grid=oil_sg, oil_compressibility_grid=oil_compressibility,
    gas_gravity_grid=gas_gravity, residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg, irreducible_water_saturation_grid=swi,
    connate_water_saturation_grid=swc, residual_gas_saturation_grid=sgr,
    reservoir_gas="methane",
)

# Well
produced_fluids = (
    bores.ProducedFluid(name="Oil", phase=bores.FluidPhase.OIL, specific_gravity=0.85, molecular_weight=180.0),
    bores.ProducedFluid(name="Gas", phase=bores.FluidPhase.GAS, specific_gravity=0.65, molecular_weight=16.04),
    bores.ProducedFluid(name="Water", phase=bores.FluidPhase.WATER, specific_gravity=1.05, molecular_weight=18.015),
)
producer = bores.production_well(
    well_name="P-1", perforating_intervals=[((0, 0, 20), (0, 0, 30))], radius=0.354,
    control=bores.AdaptiveBHPRateControl(target_rate=-150, target_phase="oil", bhp_limit=1500),
    produced_fluids=produced_fluids, skin_factor=0.0, is_active=True,
)
wells = bores.wells_(injectors=None, producers=[producer])

# Rock-fluid
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15, residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25, residual_gas_saturation=0.05,
    wettability=bores.Wettability.WATER_WET, water_exponent=2.0,
    oil_exponent=2.0, gas_exponent=2.0, mixing_rule=bores.eclipse_rule,
)
cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=1.5,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.0, gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)
rock_fluid_tables = bores.RockFluidTables(rel_perm, cap_pressure)

# Config
timer = bores.Timer(
    initial_step_size=bores.Time(hours=2), max_step_size=bores.Time(days=1),
    min_step_size=bores.Time(minutes=5), simulation_time=bores.Time(days=30),
    max_cfl_number=0.9, ramp_up_factor=1.15, backoff_factor=0.5,
)
config = bores.Config(
    timer=timer, wells=wells, rock_fluid_tables=rock_fluid_tables,
    scheme="impes", output_frequency=1, log_interval=10,
    pressure_solver="bicgstab", pressure_preconditioner="ilu", max_iterations=150,
)

# Run
run = bores.Run(model=model, config=config)
print("Starting simulation...")
for state in run():
    if state.step % 10 == 0:
        day = state.time / 86400
        pressure = state.model.fluid_properties.pressure_grid.mean()
        print(f"Day {day:6.1f}: Pressure = {pressure:.1f} psi")
```

---

## Next Steps

Congratulations! You've run your first BORES simulation.

**Try modifying:**

- Change production rate: `target_rate=-200`
- Extend simulation: `simulation_time=bores.Time(days=90)`
- Add more cells: `grid_shape = (1, 1, 100)`

**Continue learning:**

- [Tutorial 2: Building a 3D Model](02-reservoir-model.md)
