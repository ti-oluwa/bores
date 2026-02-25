# Brooks-Corey Capillary Pressure Model

The Brooks-Corey model is the most widely used capillary pressure model in petroleum engineering. It uses a power-law relationship between capillary pressure and effective saturation.

## Model Description

The Brooks-Corey capillary pressure model relates capillary pressure to effective saturation through a power-law function characterized by two parameters: the displacement (entry) pressure and the pore size distribution index.

### Mathematical Formulation

The general Brooks-Corey capillary pressure equation is:

$$P_c = P_d \cdot S_e^{-1/\lambda}$$

where:
- $P_c$ is the capillary pressure (psi)
- $P_d$ is the displacement (entry) pressure (psi)
- $S_e$ is the effective saturation (dimensionless, 0-1)
- $\lambda$ is the pore size distribution index (dimensionless)

### Effective Saturation

Effective saturation normalizes actual saturation to the mobile pore space:

$$S_e = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{or}}$$

for oil-water systems, where:
- $S_w$ is the actual water saturation
- $S_{wc}$ is the connate (irreducible) water saturation
- $S_{or}$ is the residual oil saturation

## Three-Phase Implementation

For three-phase systems (water, oil, gas), BORES computes two capillary pressures:

### Oil-Water Capillary Pressure

$$P_{cow} = P_o - P_w$$

**Water-Wet**:
$$P_{cow} = P_{d,ow}^{ww} \cdot S_{ew}^{-1/\lambda_{ow}^{ww}}$$

**Oil-Wet**:
$$P_{cow} = -P_{d,ow}^{ow} \cdot S_{ew}^{-1/\lambda_{ow}^{ow}}$$

**Mixed-Wet**:
$$P_{cow} = f_{ww} \cdot P_{cow}^{ww} + (1-f_{ww}) \cdot P_{cow}^{ow}$$

where $f_{ww}$ is the water-wet fraction.

### Gas-Oil Capillary Pressure

$$P_{cgo} = P_g - P_o = P_{d,go} \cdot S_{eg}^{-1/\lambda_{go}}$$

Gas-oil capillary pressure is always positive (gas is always non-wetting to oil).

## API Reference

### Class Definition

From src/bores/capillary_pressures.py:

```python
@attrs.frozen
class BrooksCoreyCapillaryPressureModel:
    """
    Brooks-Corey capillary pressure model for three-phase systems.

    Implements: Pc = Pd * (Se)^(-1/λ)

    Supports water-wet, oil-wet, and mixed-wet systems.
    """

    irreducible_water_saturation: Optional[float] = None
    """Default irreducible water saturation (Swc)."""

    residual_oil_saturation_water: Optional[float] = None
    """Default residual oil saturation after water flood (Sorw)."""

    residual_oil_saturation_gas: Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg)."""

    residual_gas_saturation: Optional[float] = None
    """Default residual gas saturation (Sgr)."""

    oil_water_entry_pressure_water_wet: float = 5.0
    """Entry pressure for oil-water in water-wet system (psi)."""

    oil_water_entry_pressure_oil_wet: float = 5.0
    """Entry pressure for oil-water in oil-wet system (psi)."""

    oil_water_pore_size_distribution_index_water_wet: float = 2.0
    """Pore size distribution index (λ) for oil-water in water-wet."""

    oil_water_pore_size_distribution_index_oil_wet: float = 2.0
    """Pore size distribution index (λ) for oil-water in oil-wet."""

    gas_oil_entry_pressure: float = 1.0
    """Entry pressure for gas-oil (psi)."""

    gas_oil_pore_size_distribution_index: float = 2.0
    """Pore size distribution index (λ) for gas-oil."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (WATER_WET, OIL_WET, or MIXED_WET)."""

    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet (0-1)."""
```

### Parameters

#### Entry Pressures

**oil_water_entry_pressure_water_wet** (`float`)
- Description: Displacement pressure for oil-water in water-wet pores
- Units: psi
- Typical range: 0.5 - 10 psi
- Default: 5.0 psi
- Notes: Higher values indicate tighter rock, more difficult water imbibition

**oil_water_entry_pressure_oil_wet** (`float`)
- Description: Displacement pressure for oil-water in oil-wet pores
- Units: psi
- Typical range: 0.5 - 10 psi
- Default: 5.0 psi
- Notes: Used for oil-wet and mixed-wet systems, sign is handled internally

**gas_oil_entry_pressure** (`float`)
- Description: Displacement pressure for gas-oil system
- Units: psi
- Typical range: 0.5 - 5 psi
- Default: 1.0 psi
- Notes: Usually lower than oil-water (easier gas displacement)

#### Pore Size Distribution Indices

**oil_water_pore_size_distribution_index_water_wet** (`float`)
- Symbol: $\lambda_{ow}^{ww}$
- Description: Controls curvature of Pcow vs Sw curve (water-wet)
- Typical range: 0.5 - 5.0
- Default: 2.0
- Effect: Higher λ = more uniform pore sizes, steeper Pc curve

**oil_water_pore_size_distribution_index_oil_wet** (`float`)
- Symbol: $\lambda_{ow}^{ow}$
- Description: Controls curvature of Pcow vs Sw curve (oil-wet)
- Typical range: 0.5 - 5.0
- Default: 2.0

**gas_oil_pore_size_distribution_index** (`float`)
- Symbol: $\lambda_{go}$
- Description: Controls curvature of Pcgo vs Sg curve
- Typical range: 0.5 - 5.0
- Default: 2.0

#### System Properties

**wettability** (`Wettability`)
- Options: `WATER_WET`, `OIL_WET`, `MIXED_WET`
- Default: `WATER_WET`
- Description: Determines sign and behavior of oil-water capillary pressure

**mixed_wet_water_fraction** (`float`)
- Description: Fraction of pore space that is water-wet
- Range: 0.0 - 1.0
- Default: 0.5 (equal water-wet and oil-wet fractions)
- Notes: Only used when wettability is `MIXED_WET`

## Usage Examples

### Basic Water-Wet System

From scenarios/setup.py:

```python
import bores

# Create Brooks-Corey model for typical water-wet sandstone
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Compute capillary pressures
pc = capillary_pressure_table(
    water_saturation=0.35,
    oil_saturation=0.50,
    gas_saturation=0.15,
)

print(f"Pcow = {pc['oil_water']:.2f} psi")  # Positive value
print(f"Pcgo = {pc['gas_oil']:.2f} psi")    # Positive value
```

### Oil-Wet System

```python
import bores

# Oil-wet carbonate reservoir
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.10,
    residual_oil_saturation_water=0.30,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.04,
    oil_water_entry_pressure_oil_wet=3.0,  # Higher entry pressure
    oil_water_pore_size_distribution_index_oil_wet=1.5,  # Wider pore distribution
    gas_oil_entry_pressure=2.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.OIL_WET,
)

pc = capillary_pressure_table(
    water_saturation=0.25,
    oil_saturation=0.55,
    gas_saturation=0.20,
)

print(f"Pcow = {pc['oil_water']:.2f} psi")  # Negative value (oil-wet)
print(f"Pcgo = {pc['gas_oil']:.2f} psi")    # Positive value
```

### Mixed-Wet System

```python
import bores

# Mixed-wet system (60% water-wet, 40% oil-wet)
capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.12,
    residual_oil_saturation_water=0.28,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_entry_pressure_oil_wet=2.5,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    oil_water_pore_size_distribution_index_oil_wet=1.8,
    gas_oil_entry_pressure=2.0,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.MIXED_WET,
    mixed_wet_water_fraction=0.6,  # 60% water-wet
)

pc = capillary_pressure_table(
    water_saturation=0.30,
    oil_saturation=0.50,
    gas_saturation=0.20,
)

print(f"Pcow = {pc['oil_water']:.2f} psi")  # Mixed sign behavior
print(f"Pcgo = {pc['gas_oil']:.2f} psi")
```

### Override Residual Saturations

```python
import bores

# Model with default residual saturations
model = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Override for specific region (e.g., different rock type)
pc_override = model.get_capillary_pressures(
    water_saturation=0.30,
    oil_saturation=0.50,
    gas_saturation=0.20,
    irreducible_water_saturation=0.20,  # Higher Swc for tight zone
    residual_oil_saturation_water=0.30,  # Higher Sorw
)

print(f"Pcow (override) = {pc_override['oil_water']:.2f} psi")
```

### Grid-Based Computation

```python
import bores
import numpy as np

# Create model
model = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=2.8,
    gas_oil_pore_size_distribution_index=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Create saturation grids
grid_shape = (20, 20, 10)
Sw_grid = bores.uniform_grid(grid_shape, value=0.3)
So_grid = bores.uniform_grid(grid_shape, value=0.5)
Sg_grid = bores.uniform_grid(grid_shape, value=0.2)

# Compute for entire grid
pc_grid = model(
    water_saturation=Sw_grid,
    oil_saturation=So_grid,
    gas_saturation=Sg_grid,
)

print(f"Pcow grid shape: {pc_grid['oil_water'].shape}")  # (20, 20, 10)
print(f"Average Pcow: {pc_grid['oil_water'].mean():.2f} psi")
print(f"Max Pcgo: {pc_grid['gas_oil'].max():.2f} psi")
```

## Parameter Selection Guidelines

### Entry Pressure by Rock Type

**High-Permeability Sandstone** (k > 500 mD):
```python
oil_water_entry_pressure_water_wet=0.5  # Easy displacement
gas_oil_entry_pressure=0.3
```

**Moderate-Permeability Sandstone** (50 < k < 500 mD):
```python
oil_water_entry_pressure_water_wet=2.0  # Standard
gas_oil_entry_pressure=1.5
```

**Low-Permeability Sandstone** (k < 50 mD):
```python
oil_water_entry_pressure_water_wet=5.0  # Difficult displacement
gas_oil_entry_pressure=3.0
```

**Tight Sandstone** (k < 1 mD):
```python
oil_water_entry_pressure_water_wet=15.0  # Very tight
gas_oil_entry_pressure=10.0
```

**Carbonate** (Variable permeability):
```python
oil_water_entry_pressure_water_wet=1.0 - 10.0  # Highly variable
gas_oil_entry_pressure=0.5 - 5.0
```

### Pore Size Distribution Index

The λ parameter controls curve shape:

- **λ = 0.5**: Very wide pore size distribution (poorly sorted)
- **λ = 1.0**: Wide distribution (heterogeneous)
- **λ = 2.0**: Moderate distribution (typical reservoir)
- **λ = 3.0**: Narrow distribution (well-sorted)
- **λ > 4.0**: Very uniform pores (rare)

**General guidelines**:
- Poorly sorted rocks: λ = 0.5 - 1.0
- Typical reservoirs: λ = 1.5 - 2.5
- Well-sorted sands: λ = 2.5 - 4.0

### Wettability Selection

**Indicators of Water-Wet**:
- Sandstones (most common)
- Clean formation (low clay content)
- Fresh reservoir (minimal oil aging)
- High water saturation history

**Indicators of Oil-Wet**:
- Some carbonates
- High asphaltene content
- Long oil contact time (aging)
- Presence of polar compounds

**Indicators of Mixed-Wet**:
- Intermediate contact angles
- Partial oil aging
- Heterogeneous mineralogy
- Non-uniform surface chemistry

## Physical Interpretation

### Capillary Pressure Curves

The Brooks-Corey model produces characteristic capillary pressure curves:

**At Low Wetting Phase Saturation**:
- $S_e \to 0$: $P_c \to \infty$ (infinite suction in smallest pores)
- Steep Pc gradient
- Strong capillary forces

**At High Wetting Phase Saturation**:
- $S_e \to 1$: $P_c \to P_d$ (entry pressure)
- Gentle Pc gradient
- Weak capillary forces

### Transition Zones

Capillary pressure creates transition zones at fluid contacts:

**Thickness estimation**:
$$h_{transition} \approx \frac{P_d}{\Delta \rho \cdot g}$$

where:
- $\Delta \rho$ is the density difference (lbm/ft³)
- $g$ is gravitational acceleration (ft/s²)

**Example**:
- Entry pressure: $P_d = 2$ psi = 288 lbf/ft²
- Density difference: $\Delta \rho = 20$ lbm/ft³
- Transition thickness: $h \approx 288 / (20 \times 32.2) \approx 0.45$ ft

Higher entry pressure → thicker transition zone.

### Effect of λ on Curve Shape

Lower λ (wide pore distribution):
- Gentler Pc curve
- Thicker transition zones
- More gradual saturation changes

Higher λ (narrow pore distribution):
- Steeper Pc curve
- Sharper fluid contacts
- More abrupt saturation changes

## Implementation Notes

The implementation in src/bores/capillary_pressures.py handles:

- Automatic saturation normalization
- Element-wise array operations
- Safe handling of edge cases (zero saturations, division by zero)
- Wettability sign conventions
- Mixed-wet weighted averaging

**Key features**:
- Returns zero capillary pressure when effective saturation ≥ 1
- Handles both scalar and array inputs
- Vectorized computations for grid-based simulations

## See Also

- [van Genuchten Model](van-genuchten.md) - Smooth alternative to Brooks-Corey
- [Leverett J-Function](leverett-j.md) - Dimensionless scaling correlation
- [Capillary Pressure Index](index.md) - Module overview
- [Brooks-Corey Relative Permeability](../relative-permeability/brooks-corey.md) - Companion relative permeability model
