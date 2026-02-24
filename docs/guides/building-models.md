# Building Models

Learn to construct reservoir models with grids, properties, and structure.

---

## Overview

Building a reservoir model in BORES involves:

1. **Define grid dimensions** - Size and shape
2. **Build property grids** - Porosity, permeability, saturation, etc.
3. **Assemble the model** - Use `bores.reservoir_model()` factory

---

## Grid Dimensions

Every BORES model starts with defining the grid:

```python
# 3D grid
grid_shape = (nx, ny, nz)  # Number of cells in each direction
cell_dimension = (dx, dy)   # Horizontal cell size in feet

# Examples
grid_shape = (20, 20, 10)    # 4000 cells total
cell_dimension = (100.0, 100.0)  # 100ft × 100ft cells
```

!!! info "Vertical Spacing"
    Vertical spacing (dz) comes from the `thickness_grid`, not `cell_dimension`.

---

## Grid Builders

BORES provides three main grid builders:

### 1. Uniform Grids

Same value everywhere:

```python
porosity = bores.uniform_grid(
    grid_shape=(20, 20, 10),
    value=0.20,  # 20% porosity everywhere
)
```

**Use for:**
- Homogeneous properties
- Initial testing
- Simple models

---

### 2. Layered Grids

Different value per layer:

```python
# Porosity decreasing with depth
porosity = bores.layered_grid(
    grid_shape=(20, 20, 10),
    layer_values=bores.array([0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06]),
    orientation=bores.Orientation.Z,  # Stack along z-axis
)
```

**Parameters:**

- `grid_shape`: Grid dimensions
- `layer_values`: One value per layer (length must match)
- `orientation`: `bores.Orientation.X`, `.Y`, or `.Z`

**Use for:**
- Layered reservoirs (most common)
- Properties varying with depth
- Stratigraphic models

!!! tip "Layer Values Length"
    If `orientation=Z` and `nz=10`, you need exactly 10 values in `layer_values`.

---

### 3. Custom Grids

For complex heterogeneity, build grids manually:

```python
import numpy as np

# Start with base
perm = bores.uniform_grid(grid_shape, 100.0)  # mD

# Add high-perm channel
perm[8:12, :, 3] = 500.0  # Horizontal channel at layer 3

# Add low-perm barrier
perm[:, 10, :] = 10.0  # Vertical barrier at y=10

# Add localized high-perm zone
perm[15:18, 15:18, 4:6] = 300.0  # Block in corner
```

**Use for:**
- Complex geology
- Channels, barriers, facies
- Geostatistical realizations

---

## Common Property Grids

### Thickness

Defines vertical cell size:

```python
thickness = bores.layered_grid(
    grid_shape=grid_shape,
    layer_values=bores.array([30, 25, 20, 25, 30, 20, 25, 30, 25, 20]),  # ft
    orientation=bores.Orientation.Z,
)
```

---

### Porosity

Fraction of pore space:

```python
# Uniform
porosity = bores.uniform_grid(grid_shape, 0.20)

# Decreasing with depth
porosity_values = bores.array([0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05])
porosity = bores.layered_grid(grid_shape, porosity_values, bores.Orientation.Z)
```

Typical values:
- Sandstone: 0.15-0.30
- Carbonate: 0.05-0.25
- Shale: 0.01-0.10

---

### Permeability

Use `bores.RockPermeability` for anisotropic permeability:

```python
# Horizontal permeability
kh_values = bores.array([50, 100, 150, 120, 80, 60, 40, 30, 20, 10])  # mD
kx = bores.layered_grid(grid_shape, kh_values, bores.Orientation.Z)

# Y-direction: 80% of X
ky = kx * 0.8

# Vertical: 10% of horizontal (typical for layered rocks)
kz = kx * 0.1

# Combine
permeability = bores.RockPermeability(x=kx, y=ky, z=kz)
```

!!! tip "Anisotropy Ratios"
    - **Isotropic**: kx = ky = kz (rare in nature)
    - **Typical sandstone**: kh:kv = 10:1 to 5:1
    - **Tight layering**: kh:kv can be 100:1 or more

---

### Pressure

Initialize with hydrostatic gradient:

```python
reservoir_top_depth = 8000.0  # ft
pressure_gradient = 0.38  # psi/ft (for oil)

# Calculate layer pressures
layer_depths = reservoir_top_depth + np.cumsum(np.concatenate([[0], thickness_values[:-1]]))
layer_pressures = 14.7 + (layer_depths * pressure_gradient)  # Add atmospheric

pressure_grid = bores.layered_grid(grid_shape, layer_pressures, bores.Orientation.Z)
```

Typical gradients:
- Water: 0.433 psi/ft
- Oil: 0.35-0.40 psi/ft
- Gas: 0.05-0.10 psi/ft

---

### Temperature

Increases with depth:

```python
surface_temp = 60.0  # °F
temp_gradient = 0.015  # °F/ft (typical geothermal)

layer_temps = surface_temp + (layer_depths * temp_gradient)
temperature_grid = bores.layered_grid(grid_shape, layer_temps, bores.Orientation.Z)
```

---

## Saturation Initialization

### Using Fluid Contacts

BORES provides `build_saturation_grids()` to initialize saturations with GOC/OWC:

```python
# Define contacts (depths below reservoir top)
goc_depth = 8060.0  # ft absolute
owc_depth = 8220.0  # ft absolute

# Saturation endpoints
swc = bores.uniform_grid(grid_shape, 0.12)
sorw = bores.uniform_grid(grid_shape, 0.25)
sorg = bores.uniform_grid(grid_shape, 0.15)
sgr = bores.uniform_grid(grid_shape, 0.05)
swi = bores.uniform_grid(grid_shape, 0.15)

# Calculate depth grid
depth_grid = bores.depth_grid(thickness_grid)

# Initialize saturations
water_sat, oil_sat, gas_sat = bores.build_saturation_grids(
    depth_grid=depth_grid,
    gas_oil_contact=goc_depth - reservoir_top_depth,  # Relative to top
    oil_water_contact=owc_depth - reservoir_top_depth,
    connate_water_saturation_grid=swc,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,
    porosity_grid=porosity_grid,
    use_transition_zones=True,
    gas_oil_transition_thickness=8.0,  # ft
    oil_water_transition_thickness=12.0,  # ft
    transition_curvature_exponent=1.2,
)
```

**Transition zones:**
- `use_transition_zones=True`: Smooth saturation gradients (realistic)
- `use_transition_zones=False`: Sharp contacts (simplified)

---

## Structural Dip

Apply tilt to reservoir:

```python
# Calculate depth without dip
depth_grid = bores.depth_grid(thickness_grid)

# Apply dip
dipped_depth = bores.apply_structural_dip(
    elevation_grid=depth_grid,
    cell_dimension=cell_dimension,
    elevation_direction="downward",  # Depth convention
    dip_angle=3.0,      # degrees from horizontal
    dip_azimuth=90.0,   # degrees (0=North, 90=East, 180=South, 270=West)
)
```

**Azimuth convention:**
- 0° = North (+y direction)
- 90° = East (+x direction)
- 180° = South (-y direction)
- 270° = West (-x direction)

---

## Assembling the Model

Once all grids are built, create the model:

```python
model = bores.reservoir_model(
    # Grid definition
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,

    # Pressure and temperature
    pressure_grid=pressure_grid,
    temperature_grid=temperature_grid,

    # Rock properties
    porosity_grid=porosity_grid,
    absolute_permeability=permeability,
    rock_compressibility=4.5e-6,  # 1/psi

    # Saturations
    oil_saturation_grid=oil_sat_grid,
    water_saturation_grid=water_sat_grid,
    gas_saturation_grid=gas_sat_grid,

    # Saturation endpoints
    connate_water_saturation_grid=swc,
    irreducible_water_saturation_grid=swi,
    residual_oil_saturation_water_grid=sorw,
    residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr,

    # Oil PVT
    oil_bubble_point_pressure_grid=bubble_point_grid,
    oil_viscosity_grid=oil_visc_grid,
    oil_compressibility_grid=oil_comp_grid,
    oil_specific_gravity_grid=oil_sg_grid,

    # Gas properties
    gas_gravity_grid=gas_gravity_grid,
    reservoir_gas="methane",  # or "co2", "n2", etc.

    # Optional
    dip_angle=dip_angle,
    dip_azimuth=dip_azimuth,
    net_to_gross_ratio_grid=ntg_grid,
    pvt_tables=pvt_tables,
)
```

---

## Example: Heterogeneous Model

Complete example with realistic heterogeneity:

```python
import bores
import numpy as np

# Grid
grid_shape = (30, 20, 8)
cell_dimension = (150.0, 150.0)

# Thickness (thicker in middle)
thickness_values = bores.array([25, 30, 35, 40, 40, 35, 30, 25])
thickness = bores.layered_grid(grid_shape, thickness_values, bores.Orientation.Z)

# Porosity (decreasing with depth)
poro_values = bores.array([0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10])
porosity = bores.layered_grid(grid_shape, poro_values, bores.Orientation.Z)

# Permeability (with channel)
kh_values = bores.array([80, 120, 150, 180, 160, 120, 80, 50])
kx = bores.layered_grid(grid_shape, kh_values, bores.Orientation.Z)

# Add high-perm channel in layer 3
kx[10:20, :, 3] = 500.0  # Horizontal channel

ky = kx * 0.8
kz = kx * 0.1
perm = bores.RockPermeability(x=kx, y=ky, z=kz)

# Pressure
reservoir_top = 7500.0
depths = reservoir_top + np.cumsum(np.concatenate([[0], thickness_values[:-1]]))
pressures = 14.7 + depths * 0.38
pressure = bores.layered_grid(grid_shape, pressures, bores.Orientation.Z)

# Temperature
temps = 60.0 + depths * 0.015
temperature = bores.layered_grid(grid_shape, temps, bores.Orientation.Z)

# Saturations (using contacts)
goc = 7560.0
owc = 7700.0
swc = bores.uniform_grid(grid_shape, 0.12)
sorw = bores.uniform_grid(grid_shape, 0.25)
sorg = bores.uniform_grid(grid_shape, 0.15)
sgr = bores.uniform_grid(grid_shape, 0.05)
swi = bores.uniform_grid(grid_shape, 0.15)

depth_grid = bores.depth_grid(thickness)
water_sat, oil_sat, gas_sat = bores.build_saturation_grids(
    depth_grid, goc - reservoir_top, owc - reservoir_top,
    swc, sorw, sorg, sgr, porosity, True, 8.0, 12.0,
)

# Oil properties
oil_visc = bores.uniform_grid(grid_shape, 1.8)
oil_comp = bores.uniform_grid(grid_shape, 1.2e-5)
oil_sg = bores.uniform_grid(grid_shape, 0.85)
gas_gravity = bores.uniform_grid(grid_shape, 0.65)
bubble_point = bores.layered_grid(grid_shape, pressures - 350, bores.Orientation.Z)

# Build model
model = bores.reservoir_model(
    grid_shape=grid_shape, cell_dimension=cell_dimension, thickness_grid=thickness,
    pressure_grid=pressure, temperature_grid=temperature, porosity_grid=porosity,
    absolute_permeability=perm, rock_compressibility=4.5e-6,
    oil_saturation_grid=oil_sat, water_saturation_grid=water_sat,
    gas_saturation_grid=gas_sat, oil_bubble_point_pressure_grid=bubble_point,
    oil_viscosity_grid=oil_visc, oil_compressibility_grid=oil_comp,
    oil_specific_gravity_grid=oil_sg, gas_gravity_grid=gas_gravity,
    connate_water_saturation_grid=swc, irreducible_water_saturation_grid=swi,
    residual_oil_saturation_water_grid=sorw, residual_oil_saturation_gas_grid=sorg,
    residual_gas_saturation_grid=sgr, reservoir_gas="methane",
)

print(f"Model built: {grid_shape} cells = {np.prod(grid_shape)} total")
```

---

## Next Steps

- [Rock & Fluid Properties →](rock-fluid-properties.md) - Configure PVT and rel perm
- [Wells & Controls →](wells-and-controls.md) - Add injection/production wells
- [Tutorials →](../tutorials/index.md) - Step-by-step examples
