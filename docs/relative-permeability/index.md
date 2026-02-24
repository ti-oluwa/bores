# Relative Permeability

Comprehensive documentation for BORES relative permeability models and mixing rules for three-phase flow simulations.

## Overview

Relative permeability quantifies how the presence of multiple fluid phases affects the flow of each individual phase through porous rock. In black-oil reservoir simulation, relative permeability curves determine how easily water, oil, and gas can flow through the reservoir at different saturation conditions.

BORES provides two main approaches for computing relative permeabilities:

1. **Analytical Models** - Mathematical functions like Brooks-Corey that compute relative permeability directly from saturations and rock properties
2. **Tabular Models** - Lookup tables interpolated from laboratory measurements or correlations

For three-phase flow (water, oil, gas), BORES uses two-phase relative permeability curves (oil-water and gas-oil) combined with mixing rules to compute the oil relative permeability when all three phases are present.

## Core Concepts

### Two-Phase Systems

In two-phase flow, relative permeability depends only on the saturation of the wetting phase. For example:

- **Oil-Water System**: Water is typically the wetting phase. We measure krw (water relative permeability) and kro (oil relative permeability) as functions of water saturation.
- **Gas-Oil System**: Oil is typically the wetting phase. We measure krg (gas relative permeability) and kro (oil relative permeability) as functions of oil saturation.

### Three-Phase Systems

When all three phases (water, oil, gas) are present simultaneously, computing oil relative permeability becomes more complex. Oil is the intermediate-wettability phase, and its mobility depends on both water and gas saturations.

BORES handles this using **mixing rules** - mathematical formulas that combine the two-phase oil relative permeabilities (kro from oil-water curve and kro from gas-oil curve) into a single three-phase oil relative permeability value.

### Wettability

Wettability describes which fluid preferentially wets (adheres to) the rock surface:

- **Water-Wet**: Water preferentially wets the rock (most common for sandstones)
- **Oil-Wet**: Oil preferentially wets the rock (some carbonates)
- **Mixed-Wet**: Different pore surfaces have different wettability preferences

Wettability affects the shape of relative permeability curves and capillary pressure behavior.

## Available Models

### Brooks-Corey Three-Phase Model

The primary analytical model in BORES. Uses power-law (Corey-type) functions for two-phase curves and applies a mixing rule for three-phase oil relative permeability.

**Documentation**: [Brooks-Corey Model](brooks-corey.md)

**Key Parameters**:

- Residual saturations (Swc, Sorw, Sorg, Sgr)
- Corey exponents (water_exponent, oil_exponent, gas_exponent)
- Wettability type
- Mixing rule selection

**Usage Example** (from scenarios/setup.py):

```python
import bores

relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)
```

### Three-Phase Lookup Table

For cases where you have laboratory-measured or externally-computed relative permeability data, you can use tabular interpolation instead of analytical models.

**Documentation**: [Lookup Tables](tables.md)

**Key Components**:

- Oil-water two-phase table
- Gas-oil two-phase table
- Mixing rule for three-phase oil relative permeability

## Mixing Rules

Mixing rules determine how to compute three-phase oil relative permeability from two-phase data. BORES provides 15 different mixing rules, each with different conservativeness and applicability.

**Complete Documentation**: [Mixing Rules Reference](mixing-rules.md)

### Quick Comparison

| Rule | Conservativeness | Complexity | Typical Use Case |
|------|-----------------|------------|------------------|
| **min_rule** | Very conservative | Simple | Lower bound estimation, safety factor |
| **harmonic_mean_rule** | Very conservative | Simple | Series flow paths, tight formations |
| **geometric_mean_rule** | Conservative | Simple | General-purpose conservative estimate |
| **stone_I_rule** | Moderate | Moderate | Water-wet systems, standard practice |
| **stone_II_rule** | Moderate | Moderate | Industry standard (requires approximation) |
| **eclipse_rule** | Moderate | Moderate | ECLIPSE simulator default (recommended) |
| **arithmetic_mean_rule** | Optimistic | Simple | Upper bound estimation |
| **max_rule** | Very optimistic | Simple | Sensitivity analysis, upper bound |
| **saturation_weighted_interpolation_rule** | Moderate | Moderate | Varying wettability systems |
| **baker_linear_rule** | Moderate | Moderate | Linear saturation weighting |
| **blunt_rule** | Conservative | Moderate | Strongly water-wet systems |
| **hustad_hansen_rule** | Conservative | Moderate | Intermediate wettability |
| **aziz_settari_rule** | Variable | Moderate | Empirical tuning (parameterized) |
| **product_saturation_weighted_rule** | Conservative | Moderate | Low oil saturation emphasis |
| **linear_interpolation_rule** | Moderate | Simple | Simple gas-water ratio weighting |

### Default Recommendation

For most reservoir simulations, use **eclipse_rule** as it provides:

- Moderate conservativeness (neither too optimistic nor too pessimistic)
- Smooth transitions between two-phase limits
- Robust handling of edge cases
- Industry-standard behavior matching commercial simulators

## Quick Start

### Basic Three-Phase Relative Permeability

```python
import bores
import numpy as np

# Create Brooks-Corey model with Eclipse mixing rule
relperm_model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    wettability=bores.WettabilityType.WATER_WET,
    mixing_rule=bores.eclipse_rule,  # Recommended
)

# Compute relative permeabilities at specific saturations
Sw = 0.3  # Water saturation
So = 0.5  # Oil saturation
Sg = 0.2  # Gas saturation

kr = relperm_model(
    water_saturation=Sw,
    oil_saturation=So,
    gas_saturation=Sg,
)

print(f"krw = {kr['water']:.4f}")
print(f"kro = {kr['oil']:.4f}")
print(f"krg = {kr['gas']:.4f}")
```

### Using Custom Mixing Rules

```python
# Try different mixing rules for sensitivity analysis
mixing_rules = [
    ("Conservative (Min)", bores.min_rule),
    ("Industry Standard (Eclipse)", bores.eclipse_rule),
    ("Stone II", bores.stone_II_rule),
    ("Optimistic (Max)", bores.max_rule),
]

for name, rule in mixing_rules:
    model = bores.BrooksCoreyThreePhaseRelPermModel(
        irreducible_water_saturation=0.15,
        residual_oil_saturation_water=0.25,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.045,
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
        mixing_rule=rule,
    )

    kr = model(water_saturation=0.3, oil_saturation=0.5, gas_saturation=0.2)
    print(f"{name}: kro = {kr['oil']:.4f}")
```

### Array Inputs (Grid-Based)

All models support both scalar and array inputs for grid-based simulations:

```python
# Create 3D saturation grids
Sw_grid = bores.uniform_grid(grid_shape=(10, 10, 5), value=0.3)
So_grid = bores.uniform_grid(grid_shape=(10, 10, 5), value=0.5)
Sg_grid = bores.uniform_grid(grid_shape=(10, 10, 5), value=0.2)

# Compute relative permeabilities for entire grid
kr_grid = relperm_model(
    water_saturation=Sw_grid,
    oil_saturation=So_grid,
    gas_saturation=Sg_grid,
)

# kr_grid['water'], kr_grid['oil'], kr_grid['gas'] are all (10, 10, 5) arrays
print(f"Grid shape: {kr_grid['oil'].shape}")
print(f"Average kro: {kr_grid['oil'].mean():.4f}")
```

## Module Contents

- **[Brooks-Corey Model](brooks-corey.md)** - Analytical power-law relative permeability model
- **[Mixing Rules Reference](mixing-rules.md)** - Complete documentation of all 15 mixing rules
- **[Lookup Tables](tables.md)** - Using tabular data for relative permeability
- **[Custom Mixing Rules](custom-rules.md)** - Creating and registering custom mixing rules

## Integration with Simulation

Relative permeability models are specified in the simulation configuration through `RockFluidTables`:

```python
import bores

# Define relative permeability model
relperm_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Define capillary pressure model
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Combine into rock-fluid tables
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relperm_table,
    capillary_pressure_table=capillary_pressure_table,
)

# Use in simulation config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    # ... other config parameters
)
```

## Physical Interpretation

### Relative Permeability Values

Relative permeability values range from 0.0 to 1.0:

- **kr = 0.0**: Phase is immobile (trapped at residual saturation)
- **kr = 1.0**: Phase has maximum mobility (single-phase flow)
- **0.0 < kr < 1.0**: Phase mobility is reduced by presence of other phases

### Effect of Exponents

Higher Corey exponents make curves steeper, meaning:

- **Low exponent (< 2.0)**: Phase becomes mobile quickly as saturation increases
- **High exponent (> 2.0)**: Phase remains relatively immobile until saturation is high

Typical ranges:

- Water exponent: 2.0 - 4.0 (higher for strongly water-wet)
- Oil exponent: 2.0 - 3.0
- Gas exponent: 2.0 - 3.0 (can be higher for low-permeability gas)

### Residual Saturations

- **Swc (Irreducible water)**: Water saturation below which water is immobile (typically 0.10 - 0.25)
- **Sorw (Residual oil to water)**: Oil saturation remaining after waterflooding (typically 0.20 - 0.35)
- **Sorg (Residual oil to gas)**: Oil saturation remaining after gas flooding (typically 0.10 - 0.20)
- **Sgr (Residual gas)**: Gas saturation trapped after imbibition (typically 0.03 - 0.10)

## See Also

- [Capillary Pressure Module](../capillary-pressure/index.md) - Capillary pressure models
- Rock-Fluid Tables (see guides/rock-fluid-properties.md) - Combining relative permeability and capillary pressure
- [Simulation Configuration](../guides/running-simulations.md) - Using rock-fluid tables in simulations
