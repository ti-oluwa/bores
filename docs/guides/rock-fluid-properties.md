# Rock & Fluid Properties

Configure PVT properties, relative permeability, and capillary pressure models.

---

## Overview

BORES requires two types of property definitions:

1. **PVT Properties**: How fluids behave with pressure/temperature changes
2. **Rock-Fluid Interactions**: Relative permeability and capillary pressure

---

## Relative Permeability Models

Relative permeability (kr) describes how the presence of multiple phases reduces the effective permeability to each phase. BORES provides several models.

### Brooks-Corey Three-Phase Model

The most commonly used model in BORES. Provides power-law correlations for three-phase flow.

```python
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation: float,
    residual_oil_saturation_gas: float,
    residual_oil_saturation_water: float,
    residual_gas_saturation: float,
    wettability: bores.WettabilityType,
    water_exponent: float = 2.0,
    oil_exponent: float = 2.0,
    gas_exponent: float = 2.0,
    mixing_rule: Callable = bores.eclipse_rule,
)
```

**Parameters:**

- `irreducible_water_saturation`: Swi - water saturation below which water doesn't flow
- `residual_oil_saturation_gas`: Sorg - oil saturation remaining after gas displacement
- `residual_oil_saturation_water`: Sorw - oil saturation remaining after water displacement
- `residual_gas_saturation`: Sgr - gas saturation that doesn't flow
- `wettability`: `bores.WettabilityType.WATER_WET` or `.OIL_WET`
- `water_exponent`: Corey exponent for water phase (typical: 1.5-4.0)
- `oil_exponent`: Corey exponent for oil phase (typical: 2.0-3.0)
- `gas_exponent`: Corey exponent for gas phase (typical: 1.5-3.0)
- `mixing_rule`: Function for three-phase interpolation (default: `bores.eclipse_rule`)

**Mathematical Form:**

For water-wet systems:

\\[
k_{rw} = k_{rw}^{max} \\left(\\frac{S_w - S_{wi}}{1 - S_{wi} - S_{orw}}\\right)^{n_w}
\\]

\\[
k_{ro} = k_{ro}^{max} \\left(\\frac{1 - S_w - S_g - S_{orw}}{1 - S_{wi} - S_{orw} - S_{gr}}\\right)^{n_o}
\\]

\\[
k_{rg} = k_{rg}^{max} \\left(\\frac{S_g - S_{gr}}{1 - S_{wi} - S_{org} - S_{gr}}\\right)^{n_g}
\\]

**Example:**

```python
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,    # Swi
    residual_oil_saturation_water=0.25,   # Sorw
    residual_oil_saturation_gas=0.15,     # Sorg
    residual_gas_saturation=0.05,         # Sgr
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,   # More linear = easier water flow
    oil_exponent=2.0,     # Standard value
    gas_exponent=2.0,     # Standard value
    mixing_rule=bores.eclipse_rule,
)
```

!!! tip "Choosing Exponents"
    - **Lower exponents (1.5-2.0)**: More linear curves, higher kr at lower saturations
    - **Higher exponents (3.0-4.0)**: More curved, phase flows mainly at high saturations
    - **Typical values**: 2.0 is a good starting point for all phases

### Three-Phase Mixing Rules

BORES provides mixing rules to interpolate between two-phase curves:

```python
# Eclipse/Schlumberger rule (default)
bores.eclipse_rule

# Stone I model
bores.stone1_rule

# Stone II model
bores.stone2_rule

# Harmonic mean
bores.harmonic_mean_rule

# Blunt rule
bores.blunt_rule
```

**Comparison:**

| Rule | Description | Best For |
|------|-------------|----------|
| Eclipse | Saturation-weighted interpolation | General purpose, stable |
| Stone I | Product of two-phase curves | Low gas saturations |
| Stone II | Normalized product | Moderate gas saturations |
| Harmonic Mean | Conservative estimate | Critical applications |
| Blunt | Modified Stone II | High gas saturations |

!!! warning "Mixing Rule Selection"
    The choice of mixing rule can significantly affect results. Eclipse rule is the most stable and widely used.

### Wettability

Wettability determines which fluid preferentially wets the rock surface:

```python
# Water-wet (most common)
wettability = bores.WettabilityType.WATER_WET

# Oil-wet (rare, affects displacement)
wettability = bores.WettabilityType.OIL_WET
```

**Water-wet systems** (most reservoirs):
- Water occupies small pores
- Oil in larger pores
- More efficient waterflooding

**Oil-wet systems** (some carbonates):
- Oil occupies small pores
- Water in larger pores
- Less efficient waterflooding
- Higher residual oil

---

## Capillary Pressure Models

Capillary pressure (Pc) is the pressure difference across the interface between two immiscible fluids.

### Brooks-Corey Capillary Pressure

```python
cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet: float,
    oil_water_pore_size_distribution_index_water_wet: float,
    gas_oil_entry_pressure: float,
    gas_oil_pore_size_distribution_index: float,
    wettability: bores.Wettability,
)
```

**Parameters:**

- `oil_water_entry_pressure_water_wet`: Entry pressure for oil-water system (psi), typical: 1-5 psi
- `oil_water_pore_size_distribution_index_water_wet`: Pore size distribution index (λ), typical: 1.5-3.0
- `gas_oil_entry_pressure`: Entry pressure for gas-oil system (psi), typical: 2-8 psi
- `gas_oil_pore_size_distribution_index`: Pore size distribution index (λ), typical: 1.5-3.0
- `wettability`: `bores.Wettability.WATER_WET` or `.OIL_WET`

**Mathematical Form:**

\\[
P_c = P_e \\left(\\frac{S - S_r}{1 - S_r}\\right)^{-1/\\lambda}
\\]

Where:
- \\(P_e\\) = entry pressure
- \\(S_r\\) = residual saturation
- \\(\\lambda\\) = pore size distribution index

**Example:**

```python
cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,     # psi
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,                  # psi (higher than oil-water)
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)
```

!!! info "Entry Pressure"
    - **Low permeability** (< 50 mD): 5-15 psi entry pressure
    - **Medium permeability** (50-500 mD): 1-5 psi
    - **High permeability** (> 500 mD): 0.5-2 psi

### Van Genuchten Capillary Pressure

Alternative to Brooks-Corey with different functional form:

```python
cap_pressure = bores.VanGenuchtenCapillaryPressureModel(
    oil_water_alpha_water_wet: float,
    oil_water_n_water_wet: float,
    gas_oil_alpha: float,
    gas_oil_n: float,
    wettability: bores.Wettability,
)
```

**Parameters:**

- `oil_water_alpha_water_wet`: Scaling parameter (1/psi), typical: 0.1-0.5
- `oil_water_n_water_wet`: Shape parameter, typical: 1.5-3.0
- `gas_oil_alpha`: Scaling parameter for gas-oil
- `gas_oil_n`: Shape parameter for gas-oil
- `wettability`: Wettability type

**When to use:**
- Fine-grained sediments
- More gradual transitions
- Matching lab data with curved Pc curves

### Leverett J-Function

Dimensionless capillary pressure scaling:

```python
cap_pressure = bores.LeverettJCapillaryPressureModel(
    oil_water_entry_pressure_water_wet: float,
    oil_water_pore_size_distribution_index_water_wet: float,
    gas_oil_entry_pressure: float,
    gas_oil_pore_size_distribution_index: float,
    wettability: bores.Wettability,
    interfacial_tension_oil_water: float = 30.0,     # dyne/cm
    interfacial_tension_gas_oil: float = 25.0,        # dyne/cm
)
```

Automatically scales with permeability and porosity:

\\[
P_c = \\sigma \\sqrt{\\frac{\\phi}{k}} J(S_w)
\\]

**Use when:**
- Heterogeneous permeability
- Want capillary pressure to adapt to local rock quality
- Matching core data across different perm zones

---

## Combining Into Rock-Fluid Tables

Create a `RockFluidTables` object combining both models:

```python
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table: RelativePermeabilityModel,
    capillary_pressure_table: CapillaryPressureModel,
)
```

**Complete Example:**

```python
# Relative permeability
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Capillary pressure
cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Combine
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=rel_perm,
    capillary_pressure_table=cap_pressure,
)

# Use in config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,  # ← Required
    wells=wells,
)
```

---

## Capillary Pressure Effects

### Disabling Capillary Effects

For simple models or debugging:

```python
config = bores.Config(
    ...
    disable_capillary_effects=True,  # Turn off capillary pressure
)
```

### Scaling Capillary Effects

For numerical stability (fine grids, sharp fronts):

```python
config = bores.Config(
    ...
    capillary_strength_factor=0.5,  # Reduce to 50%
)
```

!!! warning "When to Reduce Capillary Strength"
    Capillary gradients can dominate in:
    - Fine meshes (< 10 ft cells)
    - Sharp saturation fronts
    - Low permeability zones

    Reduce `capillary_strength_factor` to 0.3-0.7 to prevent numerical oscillations.

---

## PVT Correlations vs Tables

### Using Correlations (Simple)

Properties calculated on-the-fly during simulation:

```python
# Just specify grid properties
model = bores.reservoir_model(
    ...
    oil_viscosity_grid=oil_visc,
    oil_compressibility_grid=oil_comp,
    gas_gravity_grid=gas_gravity,
    # No pvt_tables needed
)

config = bores.Config(
    ...
    pvt_tables=None,  # Use correlations
)
```

**Pros:**
- Simple setup
- No pre-processing

**Cons:**
- Slower (evaluates correlations every timestep)
- Less flexible (can't use lab data)

### Using PVT Tables (Recommended)

Pre-compute properties for interpolation:

```python
# 1. Build table data
pvt_data = bores.build_pvt_table_data(
    pressures=bores.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]),
    temperatures=bores.array([120, 140, 160, 180, 200, 220]),
    salinities=bores.array([30000, 35000, 40000]),
    oil_specific_gravity=0.845,
    gas_gravity=0.65,
    reservoir_gas="methane",
)

# 2. Create tables
pvt_tables = bores.PVTTables(
    data=pvt_data,
    interpolation_method="linear",  # or "cubic"
)

# 3. Use in config
config = bores.Config(
    ...
    pvt_tables=pvt_tables,
)
```

**Pros:**
- Faster simulation (interpolation vs evaluation)
- Can incorporate lab data
- Thermodynamically consistent

**Cons:**
- Requires setup step
- Memory overhead (small)

!!! tip "When to Use PVT Tables"
    - Simulations > 100 timesteps
    - Large grids (> 10K cells)
    - Need lab data integration
    - Production runs

---

## Best Practices

### Saturation Endpoints

Ensure consistency:

```python
# GOOD: Consistent saturation endpoints
Swi = 0.15  # Irreducible water
Swc = 0.12  # Connate water (must be ≤ Swi)
Sorw = 0.25  # Residual oil to water
Sorg = 0.15  # Residual oil to gas (typically < Sorw)
Sgr = 0.05   # Residual gas

# BAD: Inconsistent saturation endpoints
Swc = 0.18  # ERROR: Can't be > Swi
Sorg = 0.30  # WARNING: Usually < Sorw
```

### Typical Values by Rock Type

**Sandstone (water-wet):**
```python
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
)

cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.5,
    gas_oil_pore_size_distribution_index=2.0,
)
```

**Carbonate (mixed-wet):**
```python
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.10,  # Lower Swi
    residual_oil_saturation_water=0.35,  # Higher Sor
    residual_oil_saturation_gas=0.20,
    residual_gas_saturation=0.05,
    water_exponent=2.5,  # Slightly higher
    oil_exponent=2.5,
    gas_exponent=2.5,
)

cap_pressure = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=5.0,  # Higher entry pressure
    oil_water_pore_size_distribution_index_water_wet=1.8,
    gas_oil_entry_pressure=6.0,
    gas_oil_pore_size_distribution_index=1.8,
)
```

---

## Table-Based Models

Instead of using formula-based models (Brooks-Corey, Van Genuchten, etc.), you can use **tabulated experimental data** for both relative permeability and capillary pressure.

### Why Use Tables?

- **Lab data**: Directly use core flood or centrifuge measurements
- **Complex behavior**: Capture hysteresis, non-monotonic curves
- **No assumptions**: Don't force data to fit a mathematical model
- **Accuracy**: Best representation of actual rock-fluid behavior

### Relative Permeability Tables

#### ThreePhaseRelPermTable

`ThreePhaseRelPermTable` interpolates from tabulated relative permeability data.

**Structure:**

Uses two **two-phase tables** (oil-water and gas-oil) plus a **mixing rule** for three-phase calculations:

```python
import numpy as np
import bores

# Oil-water table data (from lab measurements)
sw_data = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])  # Water saturation
krw_data = np.array([0.00, 0.02, 0.10, 0.25, 0.50, 0.75, 1.00])  # kr_water
kro_ow_data = np.array([1.00, 0.75, 0.50, 0.30, 0.15, 0.05, 0.00])  # kr_oil (vs water)

# Gas-oil table data
so_data = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])  # Oil saturation
kro_go_data = np.array([0.00, 0.05, 0.15, 0.35, 0.60, 0.85, 1.00])  # kr_oil (vs gas)
krg_data = np.array([1.00, 0.80, 0.60, 0.40, 0.20, 0.05, 0.00])  # kr_gas

# Create two-phase tables
oil_water_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_data,
    wetting_phase_relative_permeability=krw_data,
    non_wetting_phase_relative_permeability=kro_ow_data,
)

gas_oil_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_data,
    wetting_phase_relative_permeability=kro_go_data,
    non_wetting_phase_relative_permeability=krg_data,
)

# Create three-phase table with mixing rule
rel_perm_table = bores.ThreePhaseRelPermTable(
    oil_water_table=oil_water_table,
    gas_oil_table=gas_oil_table,
    mixing_rule=bores.stone2_rule,  # or eclipse_rule, stone1_rule, etc.
)
```

**Usage in model:**

```python
# Use table instead of formula model
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=rel_perm_table,
    capillary_pressure_table=cap_pressure_table,  # See below
)

model = bores.reservoir_model(
    # ... grid parameters
    rock_fluid_tables=rock_fluid_tables,  # Provide tables directly
)
```

**Advantages over formula models:**

- Captures exact lab data without curve fitting
- Handles non-monotonic behavior
- No assumptions about pore structure
- Can include hysteresis effects (with separate drainage/imbibition tables)

!!! tip "Data Sources"
    Lab data from:
    - Unsteady-state core floods
    - Steady-state core floods
    - Centrifuge measurements
    - Published correlations (Rock Typing)

### Capillary Pressure Tables

#### ThreePhaseCapillaryPressureTable

`ThreePhaseCapillaryPressureTable` interpolates from tabulated capillary pressure data.

**Structure:**

Uses two **two-phase tables** (oil-water and gas-oil):

```python
import numpy as np
import bores

# Oil-water capillary pressure data (from centrifuge or porous plate)
sw_data = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])  # Water saturation
pcow_data = np.array([15.0, 10.0, 7.0, 5.0, 3.0, 1.5, 0.5])  # Pc_ow = Po - Pw (psi)

# Gas-oil capillary pressure data
so_data = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])  # Oil saturation
pcgo_data = np.array([12.0, 8.0, 5.0, 3.0, 1.5, 0.5, 0.1])  # Pc_go = Pg - Po (psi)

# Create two-phase tables
oil_water_pc_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_data,
    capillary_pressure=pcow_data,
)

gas_oil_pc_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_data,
    capillary_pressure=pcgo_data,
)

# Create three-phase table
cap_pressure_table = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_pc_table,
    gas_oil_table=gas_oil_pc_table,
)
```

**Usage:**

```python
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=rel_perm_table,
    capillary_pressure_table=cap_pressure_table,
)
```

**Advantages:**

- Direct representation of lab measurements
- Captures actual entry pressures and curve shapes
- No assumptions about pore geometry
- Can model fractured or vuggy carbonates

### Complete Table-Based Example

```python
import numpy as np
import bores

# === Relative Permeability Tables (from core flood) ===

# Oil-water data
sw = np.linspace(0.20, 0.80, 20)
krw = (sw - 0.20)**2.5 / (0.60**2.5)  # Normalized
kro_ow = ((0.80 - sw) / 0.60)**2.0

oil_water_relperm = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw,
    wetting_phase_relative_permeability=krw,
    non_wetting_phase_relative_permeability=kro_ow,
)

# Gas-oil data
so = np.linspace(0.20, 0.75, 20)
kro_go = ((so - 0.20) / 0.55)**2.0
krg = ((0.75 - so) / 0.55)**2.5

gas_oil_relperm = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so,
    wetting_phase_relative_permeability=kro_go,
    non_wetting_phase_relative_permeability=krg,
)

# Three-phase table
rel_perm_table = bores.ThreePhaseRelPermTable(
    oil_water_table=oil_water_relperm,
    gas_oil_table=gas_oil_relperm,
    mixing_rule=bores.stone2_rule,
)

# === Capillary Pressure Tables (from centrifuge) ===

# Oil-water Pc
pcow = 15.0 * ((0.80 - sw) / 0.60)**(-0.5)  # Brooks-Corey form

oil_water_pc = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw,
    capillary_pressure=pcow,
)

# Gas-oil Pc
pcgo = 10.0 * ((0.75 - so) / 0.55)**(-0.5)

gas_oil_pc = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so,
    capillary_pressure=pcgo,
)

# Three-phase table
cap_pressure_table = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_pc,
    gas_oil_table=gas_oil_pc,
)

# === Create RockFluidTables ===

rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=rel_perm_table,
    capillary_pressure_table=cap_pressure_table,
)

# === Use in Model ===

model = bores.reservoir_model(
    grid_shape=(20, 20, 5),
    thickness=50.0,
    porosity=0.22,
    permeability=100.0,
    initial_pressure=3000.0,
    temperature=180.0,
    rock_fluid_tables=rock_fluid_tables,  # Use tables
    # ... other parameters
)
```

### Saving and Loading Tables

Tables are serializable like other BORES objects:

```python
from pathlib import Path

# Save rock-fluid tables
rock_fluid_tables.to_file(Path("./rock_fluid_tables.h5"))

# Load
loaded_tables = bores.RockFluidTables.from_file(Path("./rock_fluid_tables.h5"))
```

### Tables vs Models Comparison

| Aspect | Formula Models | Tabulated Data |
| ------ | -------------- | -------------- |
| **Data source** | Mathematical formulas | Lab measurements |
| **Flexibility** | Limited by model form | Arbitrary curves |
| **Interpolation** | Analytic | Linear/spline |
| **Parameters** | 5-10 fitted parameters | Full saturation arrays |
| **Hysteresis** | Requires special handling | Can use separate tables |
| **Memory** | Minimal | ~few KB per table |
| **Speed** | Very fast | Fast (interpolation) |
| **Best for** | Standard rock types | Complex/unusual systems |

!!! tip "When to Use Tables"
    Use tables when:

    - You have high-quality lab data
    - Rock-fluid behavior doesn't fit standard models
    - Hysteresis is important
    - Carbonate or fractured reservoirs

    Use formula models when:

    - Lab data is limited
    - Standard rock types (sandstone)
    - Faster setup and calibration needed
    - Sensitivity studies (easy parameter variation)

---

## Next Steps

- [Wells & Controls →](wells-and-controls.md) - Add injection/production
- [Running Simulations →](running-simulations.md) - Configure and execute
- [PVT Tables Guide →](../advanced/pvt-tables.md) - Advanced PVT usage
