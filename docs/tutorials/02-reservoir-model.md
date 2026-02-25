# Tutorial 2: Building a 3D Reservoir Model

Learn how to construct a detailed 3D reservoir model with heterogeneous properties.

---

## What You'll Learn

In this tutorial, you will learn how to:

- Create multi-layer grids where each geological layer has distinct rock properties (thickness, porosity, permeability)
- Apply structural dip to model reservoirs with tilted geological formations
- Set up initial fluid saturations using depth-based fluid contact surfaces (gas-oil contact and oil-water contact)
- Define spatially heterogeneous porosity and permeability distributions that vary by layer
- Configure Brooks-Corey relative permeability models with proper mixing rules for three-phase flow
- Place production wells strategically based on reservoir structure and fluid distribution

---

## Prerequisites

Complete [Tutorial 1: Your First Simulation](01-first-simulation.md) first.

---

## Step 1: Create a Layered Grid

Instead of uniform properties, create a model with 3 distinct layers:

```python
import bores

# Define layer properties
layer_1 = {
    "thickness": 30.0,  # ft
    "porosity": 0.25,
    "permeability": 200.0,  # mD
    "n_cells": 2,  # Number of vertical cells
}

layer_2 = {
    "thickness": 50.0,
    "porosity": 0.20,
    "permeability": 100.0,
    "n_cells": 3,
}

layer_3 = {
    "thickness": 40.0,
    "porosity": 0.18,
    "permeability": 50.0,
    "n_cells": 2,
}

layers = [layer_1, layer_2, layer_3]
```

!!! info "Layer Definition"
    Each layer can have different:

    - Thickness (vertical extent)
    - Porosity (storage capacity)
    - Permeability (flow capacity)
    - Number of vertical cells (resolution)

---

## Step 2: Build the Layered Model

```python
# Grid dimensions
nx, ny = 20, 20
total_nz = sum(layer["n_cells"] for layer in layers)  # 7 cells
grid_shape = (nx, ny, total_nz)

print(f"Grid shape: {nx} × {ny} × {total_nz}")

# Build layer values for each property
thickness_values = []
porosity_values = []
permeability_values = []

for layer in layers:
    n_cells = layer["n_cells"]
    layer_thickness = layer["thickness"] / n_cells

    # Repeat values for each cell in the layer
    thickness_values.extend([layer_thickness] * n_cells)
    porosity_values.extend([layer["porosity"]] * n_cells)
    permeability_values.extend([layer["permeability"]] * n_cells)

# Create grids using BORES helpers
thickness_grid = bores.layered_grid(grid_shape, thickness_values, bores.Orientation.Z)
porosity_grid = bores.layered_grid(grid_shape, porosity_values, bores.Orientation.Z)
permeability_grid = bores.layered_grid(grid_shape, permeability_values, bores.Orientation.Z)

print(f"Created {total_nz} layers using bores.layered_grid()")
```

---

## Step 3: Add Structural Dip

Apply a 5° dip in the x-direction (reservoir slopes):

```python
# Depth parameters
depth_top = 5000.0  # ft (top of reservoir)
cell_size_x = 100.0  # ft
cell_size_y = 100.0  # ft
dip_angle = 5.0  # degrees
dip_azimuth = 90.0  # degrees (dip direction: east/x-direction)

# Build base depth grid from thickness
# This creates depth grid where depth increases with layer index
base_depth_grid = bores.uniform_grid(grid_shape, depth_top)
depth_grid_no_dip = bores.build_depth_grid(thickness_grid)
depth_grid_no_dip += depth_top  # Offset to reservoir top

# Apply structural dip
depth_grid = bores.apply_structural_dip(
    elevation_grid=depth_grid_no_dip,
    cell_dimension=(cell_size_x, cell_size_y),
    elevation_direction="downward",
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
)

print(f"Depth range: {depth_grid.min():.1f} - {depth_grid.max():.1f} ft")
print(f"Dip applied: {dip_angle}° toward azimuth {dip_azimuth}°")
```

!!! tip "Structural Dip"
    Dip affects:

    - Fluid flow direction (gravity segregation)
    - Initial pressure distribution
    - Fluid contact depths

---

## Step 4: Set Up Initial Saturations

Define fluid contacts and compute saturations:

```python
# Fluid contacts
goc_depth = 5050.0  # ft (gas-oil contact)
owc_depth = 5150.0  # ft (oil-water contact)

# Irreducible/residual saturations
swc = 0.20  # Connate water
sgr = 0.05  # Residual gas
sorw = 0.25  # Residual oil to water
sorg = 0.20  # Residual oil to gas

# Create saturation endpoint grids
swc_grid = bores.uniform_grid(grid_shape, swc)
sgr_grid = bores.uniform_grid(grid_shape, sgr)
sorw_grid = bores.uniform_grid(grid_shape, sorw)
sorg_grid = bores.uniform_grid(grid_shape, sorg)

# Build saturations from fluid contacts using BORES helper
water_sat, oil_sat, gas_sat = bores.build_saturation_grids(
    depth_grid=depth_grid,
    gas_oil_contact=goc_depth,
    oil_water_contact=owc_depth,
    connate_water_saturation_grid=swc_grid,
    residual_oil_saturation_water_grid=sorw_grid,
    residual_oil_saturation_gas_grid=sorg_grid,
    residual_gas_saturation_grid=sgr_grid,
    porosity_grid=porosity_grid,
    use_transition_zones=False,  # Sharp contacts
)

print(f"Average saturations:")
print(f"  Oil: {oil_sat.mean():.3f}")
print(f"  Water: {water_sat.mean():.3f}")
print(f"  Gas: {gas_sat.mean():.3f}")
```

---

## Step 5: Create the Reservoir Model

Assemble all components into a complete model:

```python
# Initial pressure (hydrostatic gradient)
reference_depth = 5000.0  # ft
reference_pressure = 3000.0  # psi
pressure_gradient = 0.433  # psi/ft (water gradient)

initial_pressure = reference_pressure + (depth_grid - reference_depth) * pressure_gradient

# Rock-fluid properties
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=swc,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.20,
    residual_gas_saturation=sgr,
    wettability=bores.Wettability.WATER_WET,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=5.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=3.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=rel_perm,
    capillary_pressure_table=cap_pressure,
)

# Create model
model = bores.reservoir_model(
    grid_shape=(nx, ny, total_nz),
    thickness=thickness_grid,
    porosity=porosity_grid,
    permeability=permeability_grid,
    initial_pressure=initial_pressure,
    temperature=180.0,  # °F
    depth=depth_grid,
    initial_oil_saturation=oil_sat,
    initial_water_saturation=water_sat,
    initial_gas_saturation=gas_sat,
    rock_fluid_tables=rock_fluid_tables,
    oil_specific_gravity=0.85,
    gas_specific_gravity=0.65,
    water_salinity=35000.0,  # ppm
)

print(f"Model created with {nx}×{ny}×{total_nz} = {nx*ny*total_nz} cells")
```

---

## Step 6: Visualize the Model

Check the model setup using BORES 2D visualization:

```python
# Create 2D visualizer
viz = bores.plotly2d.DataVisualizer()

# Plot porosity in middle layer
middle_layer = total_nz // 2
fig_porosity = viz.make_plot(
    data=porosity_grid[:, :, middle_layer],
    plot_type="heatmap",
    title=f"Porosity (Layer {middle_layer})",
    x_label="X Index",
    y_label="Y Index",
    colorbar_title="Porosity",
)
fig_porosity.show()

# Oil saturation map
fig_oil_sat = viz.make_plot(
    data=oil_sat[:, :, middle_layer],
    plot_type="heatmap",
    title=f"Oil Saturation (Layer {middle_layer})",
    x_label="X Index",
    y_label="Y Index",
    colorbar_title="So",
)
fig_oil_sat.show()

# Depth map (top layer)
fig_depth = viz.make_plot(
    data=depth_grid[:, :, 0],
    plot_type="heatmap",
    title="Depth to Top of Reservoir",
    x_label="X Index",
    y_label="Y Index",
    colorbar_title="Depth (ft)",
)
fig_depth.show()
```

---

## Step 7: Add a Production Well

Place a well in the oil zone:

```python
# Well location (center of grid, oil zone)
well_i, well_j = nx // 2, ny // 2
oil_zone_layers = []

# Find layers in oil zone
for k in range(total_nz):
    cell_depth = depth_grid[well_i, well_j, k] + thickness_grid[well_i, well_j, k] / 2
    if goc_depth < cell_depth < owc_depth:
        oil_zone_layers.append(k)

print(f"Oil zone layers at well location: {oil_zone_layers}")

# Create well perforations
perforations = [
    ((well_i, well_j, oil_zone_layers[0]),
     (well_i, well_j, oil_zone_layers[-1]))
]

# Production well
clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-150,  # STB/day
        target_phase="oil",
        bhp_limit=1000,  # psi
        clamp=clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-200,  # MCF/day
        target_phase="gas",
        bhp_limit=1000,
        clamp=clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-20,  # STB/day
        target_phase="water",
        bhp_limit=1000,
        clamp=clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=perforations,
    radius=0.3542,  # 8.5" wellbore
    control=control,
    produced_fluids=(
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
    ),
    skin_factor=2.0,
)

wells = bores.wells_(injectors=None, producers=[producer])
```

---

## Step 8: Configure and Run

```python
from pathlib import Path

# Timer
timer = bores.Timer(
    initial_step_size=bores.Time(days=1.0),
    max_step_size=bores.Time(days=10.0),
    min_step_size=bores.Time(hours=6.0),
    simulation_time=bores.Time(days=365),  # 1 year
    max_cfl_number=0.9,
)

# Config
config = bores.Config(
    model=model,
    wells=wells,
    timer=timer,
)

# Create run
run = bores.Run(config=config)

# Save setup
setup_dir = Path("./tutorial_02_setup")
setup_dir.mkdir(exist_ok=True)
model.to_file(setup_dir / "model.h5")
config.to_file(setup_dir / "config.yaml")

# Execute with streaming
store = bores.ZarrStore(setup_dir / "results.zarr")
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=20,
    background_io=True,
)

print("Running simulation...")
with stream:
    stream.consume()

print("Simulation complete!")
```

---

## Step 9: Analyze Results

```python
# Load results
states = list(store.load(bores.ModelState))
analyst = bores.ModelAnalyst(states)

# Recovery factors
print(f"\nRecovery Factors:")
print(f"  Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"  Gas RF: {analyst.gas_recovery_factor:.2%}")

# Production history
print(f"\nProduction History (every 30 days):")
for step, rate in analyst.oil_production_history(interval=30):
    state = analyst.get_state(step)
    time_days = state.time / 86400
    print(f"  Day {time_days:4.0f}: {rate:6.1f} STB/day")

# Final saturations by layer
final_state = analyst.get_state(-1)
final_oil_sat = final_state.model.fluid_properties.saturation_history.oil_saturations[-1]

print(f"\nAverage Oil Saturation by Layer:")
for k in range(total_nz):
    avg_so = final_oil_sat[:, :, k].mean()
    print(f"  Layer {k+1}: {avg_so:.3f}")
```

---

## What You Learned

In this tutorial, you successfully:

- Created a multi-layer reservoir model with heterogeneous rock properties (varying porosity, permeability, and thickness across three distinct layers)
- Applied structural dip to the reservoir model (5-degree dip angle in the x-direction)
- Set up initial saturations with fluid contacts (gas-oil contact at 5050 ft, oil-water contact at 5150 ft)
- Configured Brooks-Corey relative permeability and capillary pressure models
- Placed production wells based on reservoir structure (centered in oil zone between fluid contacts)
- Executed a production simulation and analyzed results using the ModelAnalyst class

---

## Next Steps

- **[Tutorial 3: Water Injection](03-waterflood.md)** - Add water injection for pressure support
- **[Building Models Guide](../guides/building-models.md)** - Advanced model construction techniques
- **[Rock-Fluid Properties Guide](../guides/rock-fluid-properties.md)** - Detailed property configuration
