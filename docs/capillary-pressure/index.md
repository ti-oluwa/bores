# Capillary Pressure

Comprehensive documentation for BORES capillary pressure models for three-phase flow simulations.

## Overview

Capillary pressure is the pressure difference across the interface between two immiscible fluids in porous media. It arises from interfacial tension and wettability effects, and plays a critical role in multiphase flow behavior, especially in:

- Fluid distribution in transition zones
- Imbibition and drainage processes
- Relative permeability hysteresis
- Gravity-capillary equilibrium
- Recovery mechanisms

BORES provides three main capillary pressure models:

1. **Brooks-Corey Model** - Power-law capillary pressure function
2. **van Genuchten Model** - Smooth alternative with better transition behavior
3. **Leverett J-Function** - Dimensionless correlation for scaling across rock types

## Physical Concepts

### Capillary Pressure Definition

Capillary pressure is defined as the pressure difference between non-wetting and wetting phases:

$$P_c = P_{nw} - P_w$$

For three-phase systems (water, oil, gas), we define two capillary pressures:

- **Oil-Water**: $P_{cow} = P_o - P_w$
- **Gas-Oil**: $P_{cgo} = P_g - P_o$

### Wettability and Sign Convention

**Water-Wet Systems** (most common):

- Water preferentially wets rock surface
- $P_{cow} > 0$ (oil pressure higher than water pressure)
- $P_{cgo} > 0$ (gas pressure higher than oil pressure)

**Oil-Wet Systems** (some carbonates):

- Oil preferentially wets rock surface
- $P_{cow} < 0$ (water pressure higher than oil pressure)
- $P_{cgo} > 0$ (gas pressure higher than oil pressure)

**Mixed-Wet Systems**:

- Different pore surfaces have different wettability
- $P_{cow}$ sign varies with saturation
- Weighted combination of water-wet and oil-wet behavior

### Saturation-Capillary Pressure Relationship

Capillary pressure increases as wetting phase saturation decreases:

- **High wetting phase saturation**: $P_c \approx 0$ (large pores filled)
- **Low wetting phase saturation**: $P_c \to \infty$ (only small pores filled)

This inverse relationship is captured by power-law (Brooks-Corey) or exponential (van Genuchten) functions.

## Available Models

### Brooks-Corey Model

Power-law capillary pressure model based on effective saturation. Most widely used in petroleum engineering.

**Documentation**: [Brooks-Corey Model](brooks-corey.md)

**Formula**:
$$P_c = P_d \cdot S_e^{-1/\lambda}$$

where:

- $P_d$ is the displacement (entry) pressure
- $S_e$ is the effective saturation
- $\lambda$ is the pore size distribution index

**Usage Example** (from scenarios/setup.py):

```python
import bores

capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)
```

### van Genuchten Model

Smooth alternative to Brooks-Corey with better behavior near residual saturations. Common in soil science and groundwater modeling.

**Documentation**: [van Genuchten Model](van-genuchten.md)

**Formula**:
$$P_c = \frac{1}{\alpha} \left[(S_e^{-1/m} - 1)^{1/n}\right]$$

where $m = 1 - 1/n$

**Usage Example**:

```python
import bores

capillary_pressure_table = bores.VanGenuchtenCapillaryPressureModel(
    oil_water_alpha_water_wet=0.01,  # 1/psi
    oil_water_n_water_wet=2.0,
    gas_oil_alpha=0.01,  # 1/psi
    gas_oil_n=2.0,
    wettability=bores.WettabilityType.WATER_WET,
)
```

### Leverett J-Function

Dimensionless correlation that scales capillary pressure with rock properties (permeability, porosity) and fluid properties (interfacial tension, contact angle).

**Documentation**: [Leverett J-Function](leverett-j.md)

**Formula**:
$$P_c = \sigma \cdot \cos(\theta) \cdot \sqrt{\frac{\phi}{k}} \cdot J(S_e)$$

**Usage Example**:

```python
import bores

capillary_pressure_table = bores.LeverettJCapillaryPressureModel(
    permeability=100.0,  # mD
    porosity=0.2,
    oil_water_interfacial_tension=30.0,  # dyne/cm
    gas_oil_interfacial_tension=20.0,  # dyne/cm
    j_function_coefficient=0.5,
    j_function_exponent=0.5,
    wettability=bores.WettabilityType.WATER_WET,
)
```

## Quick Comparison

| Model | Smoothness | Parameters | Best For |
|-------|-----------|-----------|----------|
| **Brooks-Corey** | Moderate | Entry pressure, λ | General petroleum applications |
| **van Genuchten** | Smooth | α, n | Near-residual saturations, smooth derivatives |
| **Leverett J** | Moderate | Rock/fluid properties, J-function | Scaling across rock types |

## Quick Start

### Basic Three-Phase Capillary Pressure

```python
import bores

# Create Brooks-Corey model
capillary_pressure_model = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,  # psi
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,  # psi
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.WettabilityType.WATER_WET,
)

# Compute capillary pressures at specific saturations
Sw = 0.3  # Water saturation
So = 0.5  # Oil saturation
Sg = 0.2  # Gas saturation

pc = capillary_pressure_model(
    water_saturation=Sw,
    oil_saturation=So,
    gas_saturation=Sg,
)

print(f"Pcow = {pc['oil_water']:.2f} psi")  # Oil-water capillary pressure
print(f"Pcgo = {pc['gas_oil']:.2f} psi")    # Gas-oil capillary pressure
```

### Using with Relative Permeability in Simulation

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

# Use in simulation config (from scenarios/setup.py)
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",
    # ... other config parameters
)
```

### Array Inputs for Grid-Based Computation

```python
import bores
import numpy as np

# Create saturation grids
Sw_grid = bores.uniform_grid(grid_shape=(10, 10, 5), value=0.3)
So_grid = bores.uniform_grid(grid_shape=(10, 10, 5), value=0.5)
Sg_grid = bores.uniform_grid(grid_shape=(10, 10, 5), value=0.2)

# Compute capillary pressures for entire grid
pc_grid = capillary_pressure_model(
    water_saturation=Sw_grid,
    oil_saturation=So_grid,
    gas_saturation=Sg_grid,
)

# Results are arrays matching input shape
print(f"Pcow grid shape: {pc_grid['oil_water'].shape}")  # (10, 10, 5)
print(f"Average Pcow: {pc_grid['oil_water'].mean():.2f} psi")
print(f"Max Pcgo: {pc_grid['gas_oil'].max():.2f} psi")
```

## Module Contents

- **[Brooks-Corey Model](brooks-corey.md)** - Power-law capillary pressure model
- **[van Genuchten Model](van-genuchten.md)** - Smooth exponential model
- **[Leverett J-Function](leverett-j.md)** - Dimensionless scaling correlation
- **[Lookup Tables](tables.md)** - Using tabular capillary pressure data

## Physical Interpretation

### Entry Pressure

The displacement pressure ($P_d$ or entry pressure) is the minimum capillary pressure required for the non-wetting phase to enter the largest pore throats. It indicates:

- **Low $P_d$ (0.5-2 psi)**: Good permeability, large pores, easy displacement
- **Moderate $P_d$ (2-10 psi)**: Average reservoir rock
- **High $P_d$ (>10 psi)**: Tight rock, small pores, difficult displacement

### Pore Size Distribution Index

The pore size distribution index (λ) characterizes pore throat size variation:

- **High λ (> 3)**: Uniform pore sizes (well-sorted sand)
- **Moderate λ (1-3)**: Moderate sorting (typical reservoirs)
- **Low λ (< 1)**: Wide pore size distribution (poorly sorted)

### Transition Zones

Capillary pressure creates transition zones between fluid contacts:

- **Oil-Water Transition Zone**: Above OWC, where $P_{cow}$ balances gravity
- **Gas-Oil Transition Zone**: Below GOC, where $P_{cgo}$ balances gravity
- **Thickness**: Depends on entry pressure and pore size distribution

Higher capillary pressure (tighter rock) creates thicker transition zones.

## Wettability Types

BORES supports three wettability types:

### Water-Wet (`WettabilityType.WATER_WET`)

**Characteristics**:

- Water preferentially wets rock
- $P_{cow} > 0$ always
- Most common for sandstones
- Favorable for waterflooding

**Typical Parameters**:

```python
oil_water_entry_pressure_water_wet=2.0  # Positive entry pressure
wettability=bores.WettabilityType.WATER_WET
```

### Oil-Wet (`WettabilityType.OIL_WET`)

**Characteristics**:

- Oil preferentially wets rock
- $P_{cow} < 0$ (negative)
- Some carbonates and aged reservoirs
- Less favorable for waterflooding

**Typical Parameters**:

```python
oil_water_entry_pressure_oil_wet=2.0  # Will be negated internally
wettability=bores.WettabilityType.OIL_WET
```

### Mixed-Wet (`WettabilityType.MIXED_WET`)

**Characteristics**:

- Different pore surfaces have different wettability
- $P_{cow}$ varies (positive and negative regions)
- Complex recovery behavior
- Requires careful characterization

**Typical Parameters**:

```python
oil_water_entry_pressure_water_wet=2.0
oil_water_entry_pressure_oil_wet=2.0
mixed_wet_water_fraction=0.5  # 50% water-wet, 50% oil-wet
wettability=bores.WettabilityType.MIXED_WET
```

## Model Selection Guide

**Choose Brooks-Corey when**:

- You have standard reservoir engineering data
- You need industry-standard behavior
- You don't need extreme smoothness near endpoints

**Choose van Genuchten when**:

- You need smooth derivatives for numerical stability
- You're modeling unsaturated flow or near-residual conditions
- You're importing parameters from soil science literature

**Choose Leverett J when**:

- You need to scale capillary pressure across different rock types
- You have rock property grids (permeability, porosity)
- You're using laboratory J-function measurements
- You need to honor spatial heterogeneity in rock properties

## Integration with Simulation

Capillary pressure models are specified in the simulation configuration through `RockFluidTables` (from scenarios/setup.py):

```python
import bores

# Define models
relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
    # ... parameters ...
)

capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    # ... parameters ...
)

# Combine into rock-fluid tables
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relative_permeability_table,
    capillary_pressure_table=capillary_pressure_table,
)

# Use in config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    # ... other parameters ...
)
```

## Effect on Simulation

Capillary pressure affects:

1. **Fluid Distribution**: Establishes equilibrium saturation profiles
2. **Phase Velocities**: Alters pressure gradients driving flow
3. **Imbibition/Drainage**: Controls direction of saturation changes
4. **Transition Zones**: Determines thickness of contact regions
5. **Recovery**: Impacts displacement efficiency and recovery factor

**Note**: Capillary pressure can be neglected for:

- High-permeability systems (k > 1000 mD)
- Large-scale coarse grids (> 100 ft cell size)
- When computational speed is priority over accuracy

## See Also

- [Relative Permeability Module](../relative-permeability/index.md) - Companion relative permeability documentation
- Rock-Fluid Tables (see guides/rock-fluid-properties.md) - Combining relative permeability and capillary pressure
- [Simulation Configuration](../guides/running-simulations.md) - Using rock-fluid tables in simulations
