# Boundary Conditions

Configure aquifer support, constant pressure, no-flow, and periodic boundaries.

---

## Overview

Boundary conditions define how the reservoir interacts with its surroundings. BORES supports:

1. **No-flow** (default) - Sealed boundaries
2. **Constant pressure** - Fixed pressure at boundary
3. **Aquifer models** - Carter-Tracy water drive
4. **Periodic** - Wraparound boundaries

---

## Creating Boundary Conditions

Use `bores.BoundaryConditions` to specify conditions for each face:

```python
boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(
            left=left_condition,
            right=right_condition,
            front=front_condition,
            back=back_condition,
            top=top_condition,
            bottom=bottom_condition,
        ),
    },
)
```

---

## No-Flow Boundaries (Default)

Sealed boundaries with no flux:

```python
# No boundary conditions needed - this is the default
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    # boundary_conditions=None,  # No-flow on all sides
)
```

**Use for:**
- Isolated reservoirs
- Tight formations
- Conservative estimates

---

## Constant Pressure Boundaries

Fixed pressure at a boundary:

```python
# Constant pressure on left side
left_bc = bores.ConstantPressureBoundary(pressure=3500.0)  # psi

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(left=left_bc),
    },
)
```

**Use for:**
- Strong aquifer (infinite-acting)
- Pressure-supported reservoirs
- Idealized cases

!!! warning "Infinite Support"
    Constant pressure = infinite fluid supply. Can be unrealistic for long simulations.

---

## Carter-Tracy Aquifer Model

Realistic finite aquifer with time-dependent influx:

```python
# Bottom water drive
bottom_aquifer = bores.CarterTracyAquifer(
    aquifer_permeability=450.0,      # mD
    aquifer_porosity=0.22,            # fraction
    aquifer_compressibility=5e-6,     # psi⁻¹ (rock + water)
    water_viscosity=0.5,              # cP at reservoir temp
    inner_radius=1400.0,              # ft (reservoir-aquifer contact)
    outer_radius=14000.0,             # ft (aquifer extent)
    aquifer_thickness=80.0,           # ft
    initial_pressure=3117.0,          # psi
    angle=360.0,                      # degrees (full circular contact)
)

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(bottom=bottom_aquifer),
    },
)
```

**Parameters:**

- `aquifer_permeability`: Permeability of aquifer rock (mD)
- `aquifer_porosity`: Porosity of aquifer (fraction)
- `aquifer_compressibility`: Total compressibility c_t = c_rock + c_water (1/psi)
- `water_viscosity`: Water viscosity at reservoir temperature (cP)
- `inner_radius`: Radius of reservoir-aquifer contact (ft)
- `outer_radius`: Extent of aquifer (ft) - typically 5-10× inner radius
- `aquifer_thickness`: Thickness of aquifer (ft)
- `initial_pressure`: Initial aquifer pressure (psi)
- `angle`: Contact angle in degrees (360° = full circular, 180° = semicircular)

**Dimensionless Parameters:**

The model uses van Everdingen-Hurst solution:

\\[
t_D = \\frac{\\eta \\cdot t}{r_w^2}
\\]

where \\(\\eta = 0.006328 \\frac{k}{\\phi \\mu c_t}\\)

\\[
r_D = \\frac{r_e}{r_w}
\\]

**Water Influx:**

\\[
Q(t) = B \\sum_{i} \\Delta P(t_i) \\cdot W_D'(t_D - t_{Di})
\\]

where \\(B = 1.119 \\phi c_t (r_e^2 - r_w^2) h \\frac{\\theta}{360°}\\)

**Example: Bottom Water Drive**

```python
# Strong bottom aquifer (common in Middle East)
bottom_aquifer = bores.CarterTracyAquifer(
    aquifer_permeability=500.0,
    aquifer_porosity=0.25,
    aquifer_compressibility=5e-6,
    water_viscosity=0.4,
    inner_radius=2000.0,
    outer_radius=20000.0,  # 10× inner radius
    aquifer_thickness=100.0,
    initial_pressure=3200.0,
    angle=360.0,  # Full bottom contact
)

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(bottom=bottom_aquifer),
    },
)
```

**Example: Edge Water Drive**

```python
# Edge aquifer on left side
edge_aquifer = bores.CarterTracyAquifer(
    aquifer_permeability=300.0,
    aquifer_porosity=0.20,
    aquifer_compressibility=6e-6,
    water_viscosity=0.6,
    inner_radius=1500.0,
    outer_radius=10000.0,
    aquifer_thickness=200.0,  # Same as reservoir
    initial_pressure=3100.0,
    angle=180.0,  # Semicircular edge contact
)

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(left=edge_aquifer),
    },
)
```

**Sizing the Aquifer:**

| Ratio r_e/r_w | Aquifer Strength | Behavior |
|---------------|------------------|----------|
| 2-3 | Weak | Limited support, early decline |
| 5-10 | Moderate | Good support, slower decline |
| 10-20 | Strong | Long-term support |
| > 20 | Very strong | Nearly constant pressure |

!!! tip "Calibration"
    Match aquifer parameters to field history:
    1. Start with r_e/r_w ≈ 10
    2. Adjust permeability to match pressure support
    3. Fine-tune with angle and thickness

---

## Periodic Boundaries

Wraparound boundaries for repeating patterns:

```python
# Periodic in X and Y (for pattern studies)
boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(
            left="periodic",
            right="periodic",
            front="periodic",
            back="periodic",
        ),
    },
)
```

**Use for:**
- Repeating well patterns (5-spot, 7-spot, 9-spot)
- Infinite reservoir approximation
- Symmetry studies

!!! warning "Pairing Required"
    Periodic boundaries must be paired: if left="periodic", then right="periodic" too.

---

## Combining Boundaries

Different conditions on different faces:

```python
# Bottom aquifer + no-flow sides + constant pressure top
bottom_aquifer = bores.CarterTracyAquifer(...)
top_pressure = bores.ConstantPressureBoundary(pressure=3000.0)

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(
            bottom=bottom_aquifer,
            top=top_pressure,
            # left, right, front, back = no-flow (default)
        ),
    },
)
```

---

## Usage in Config

```python
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    boundary_conditions=boundary_conditions,  # ← Add here
)
```

---

## Complete Example

```python
import bores

# Model
model = bores.reservoir_model(...)

# Wells
wells = bores.wells_(...)

# Carter-Tracy bottom aquifer
aquifer = bores.CarterTracyAquifer(
    aquifer_permeability=450.0,
    aquifer_porosity=0.22,
    aquifer_compressibility=5e-6,
    water_viscosity=0.5,
    inner_radius=1400.0,
    outer_radius=14000.0,
    aquifer_thickness=80.0,
    initial_pressure=3117.0,
    angle=360.0,
)

boundary_conditions = bores.BoundaryConditions(
    conditions={
        "pressure": bores.GridBoundaryCondition(bottom=aquifer),
    },
)

# Config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    boundary_conditions=boundary_conditions,
)

# Run
run = bores.Run(model=model, config=config)
for state in run():
    print(f"Day {state.time / 86400:.1f}")
```

---

## Best Practices

### Aquifer Sizing

Start conservative:

```python
# Conservative estimate
inner_radius = sqrt(reservoir_area / pi)
outer_radius = inner_radius * 5  # Moderate support
```

### Pressure Initialization

Match boundary and reservoir:

```python
# Bottom aquifer
aquifer_pressure = pressure_grid[:, :, -1].mean()  # Bottom layer average

aquifer = bores.CarterTracyAquifer(
    ...,
    initial_pressure=aquifer_pressure,  # Match reservoir
)
```

### Validation

Compare against no-flow and constant pressure:

1. Run with **no-flow** (conservative)
2. Run with **constant pressure** (optimistic)
3. Run with **Carter-Tracy** (realistic)
4. Results should be between 1 and 2

---

## Next Steps

- [Faults & Fractures →](faults-fractures.md) - Transmissibility barriers
- [Examples →](../examples/index.md) - See boundary conditions in action
