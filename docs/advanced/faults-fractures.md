# Faults and Fractures

Complete guide to modeling faults and fractures in BORES using the `bores.fractures` module.

---

## Overview

BORES models faults and fractures using the `Fracture` and `FractureGeometry` classes. Faults act as flow barriers or conduits by modifying transmissibility between cells.

**Key Concepts:**

- **Fractures**: General term for any geological discontinuity (includes faults)
- **FractureGeometry**: Defines the spatial extent and orientation
- **Transmissibility multipliers**: Modify cross-fracture flow
- **Permeability/Porosity overrides**: Modify rock properties in damage zones

---

## Quick Start: Helper Functions

BORES provides helper functions for common fault types:

```python
from bores.fractures import vertical_sealing_fault, apply_fracture

# Create a sealing fault at x=25
fault = vertical_sealing_fault(
    fault_id="F-1",
    orientation="x",
    index=25,
    permeability_multiplier=1e-4  # 99.99% sealing
)

# Apply to model
model = apply_fracture(model, fault)
```

**Available helpers:**

- `vertical_sealing_fault()` - Vertical barrier
- `inclined_sealing_fault()` - Dipping fault plane
- `damage_zone_fault()` - Fault with damaged rock properties
- `conductive_fracture_network()` - Enhanced permeability corridors

See module docstring in `bores.fractures` for complete examples.

---

## Manual Construction

### Step 1: Define Geometry

```python
from bores.fractures import FractureGeometry

# Vertical fault at x=25, spanning layers 0-15
geometry = FractureGeometry(
    orientation="x",  # Perpendicular to x-axis
    x_range=(25, 25),  # Single cell plane
    z_range=(0, 15),  # Upper 15 layers only
)
```

**Orientation:**

- `"x"`: Fault plane perpendicular to x-axis (strikes in y-direction, blocks x-flow)
- `"y"`: Fault plane perpendicular to y-axis (strikes in x-direction, blocks y-flow)
- `"z"`: Horizontal fracture (bedding-parallel)

**Range Parameters:**

- For `orientation="x"`: `x_range` is fault location, `y_range` and `z_range` limit extent
- For `orientation="y"`: `y_range` is fault location, `x_range` and `z_range` limit extent
- For `orientation="z"`: `z_range` is fault location, `x_range` and `y_range` limit extent

### Step 2: Create Fracture

```python
from bores.fractures import Fracture

fracture = Fracture(
    id="fault_1",
    geometry=geometry,
    permeability_multiplier=1e-4,  # 99.99% reduction
)
```

**Fracture Parameters:**

**`id`**: Unique identifier

**`geometry`**: `FractureGeometry` object

**`permeability_multiplier`**: Multiplier for cross-fault flow (optional)

- `< 1.0`: Sealing fault (barrier)
- `= 1.0`: No effect
- `> 1.0`: Conductive fault

**`permeability`**: Absolute permeability value for damage zone (mD, optional)

- Mutually exclusive with `permeability_multiplier`
- Use for damage zones with altered rock properties

**`porosity`**: Porosity for damage zone (fraction, optional)

**`conductive`**: Set to `True` for high-permeability conduits

### Step 3: Apply to Model

```python
from bores.fractures import apply_fracture

model = apply_fracture(model, fracture)
```

For multiple fractures:

```python
from bores.fractures import apply_fractures

model = apply_fractures(model, fracture1, fracture2, fracture3)
```

---

## Fault Types

### Sealing Faults

Reduce cross-fault flow:

```python
# Using helper
fault = vertical_sealing_fault(
    fault_id="sealing_fault",
    orientation="x",
    index=10,
    permeability_multiplier=0.01,  # 99% sealing
)

# Or manual construction
geometry = FractureGeometry(
    orientation="x",
    x_range=(10, 10),
)
fault = Fracture(
    id="sealing_fault",
    geometry=geometry,
    permeability_multiplier=0.01,
)

model = apply_fracture(model, fault)
```

**Typical multipliers:**

- Shale-rich faults: `1e-5` to `1e-3`
- Sand-sand with gouge: `0.01` to `0.1`
- Clean sand-sand: `0.1` to `0.5`

### Damage Zones

Faults with altered rock properties:

```python
from bores.fractures import damage_zone_fault

# 5-cell-wide damage zone
fault = damage_zone_fault(
    fault_id="damage_fault",
    orientation="x",
    cell_range=(48, 52),  # Cells 48-52
    permeability_multiplier=1e-5,  # Very low cross-fault flow
    zone_permeability=10.0,  # Damaged rock: 10 mD
    zone_porosity=0.12,  # Reduced porosity
)

model = apply_fracture(model, fault)
```

### Conductive Fractures

Enhanced permeability pathways:

```python
from bores.fractures import conductive_fracture_network

fracture = conductive_fracture_network(
    fracture_id="frac_corridor",
    orientation="y",
    cell_range=(15, 18),  # 3-cell corridor
    fracture_permeability=5000.0,  # High perm: 5 Darcy
    fracture_porosity=0.01,  # Low storage
    permeability_multiplier=10.0,  # 10x enhanced flow
)

model = apply_fracture(model, fracture)
```

### Inclined Faults

Dipping fault planes:

```python
from bores.fractures import inclined_sealing_fault

# Fault dipping at 60° (tan(60°) ≈ 1.73)
fault = inclined_sealing_fault(
    fault_id="dipping_fault",
    orientation="y",  # Strikes in y-direction
    index=30,  # Intersects at y=30
    slope=1.73,  # dz/dx (eastward dip)
    intercept=5.0,  # z=5 when x=0
    permeability_multiplier=1e-4,
)

model = apply_fracture(model, fault)
```

---

## Complete Example

```python
import bores
from bores.fractures import (
    vertical_sealing_fault,
    damage_zone_fault,
    apply_fractures,
)

# Build model (example)
model = bores.reservoir_model(
    grid_shape=(50, 50, 20),
    # ... other parameters
)

# Define fault system
main_fault = vertical_sealing_fault(
    fault_id="main_fault",
    orientation="x",
    index=25,
    permeability_multiplier=1e-4,
)

damage_zone = damage_zone_fault(
    fault_id="damage",
    orientation="y",
    cell_range=(10, 14),
    permeability_multiplier=1e-5,
    zone_permeability=5.0,
    zone_porosity=0.10,
)

# Apply all faults
model = apply_fractures(model, main_fault, damage_zone)
```

---

## Geological Context

**Sealing mechanisms:**

- **Shale smear**: Clay-rich sediments along fault plane
- **Cataclasis**: Fault gouge from rock pulverization
- **Cementation**: Mineral precipitation
- **Juxtaposition**: Impermeable against permeable rock

**Conductive mechanisms:**

- **Fracture networks**: Open fracture systems
- **Breccia zones**: High porosity/permeability damage zones
- **Dissolution**: Enhanced flow paths from mineral dissolution

---

## Best Practices

1. **Use helper functions** for common fault types (simpler, less error-prone)
2. **Calibrate multipliers** from history matching or geological analogs
3. **Start conservative** (more sealing) and adjust based on pressure response
4. **Check grid alignment** - faults should align with cell faces for accuracy
5. **Consider uncertainty** - run sensitivity cases with different multipliers

---

## See Also

- Module documentation: `bores.fractures` module docstring
- Example: [Model Setup](../examples/setup.md) includes fault application
- [Performance Optimization](performance-optimization.md) for large fault systems
