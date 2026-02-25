# van Genuchten Capillary Pressure Model

The van Genuchten model provides a smooth alternative to Brooks-Corey for computing capillary pressure. It uses an exponential formulation that provides better behavior near residual saturations and smoother derivatives.

## Model Description

The van Genuchten model relates capillary pressure to effective saturation through a smooth exponential function. It is widely used in soil science and groundwater modeling, and increasingly in petroleum engineering for applications requiring smooth transitions.

### Mathematical Formulation

The van Genuchten capillary pressure equation is:

$$P_c = \frac{1}{\alpha} \left[(S_e^{-1/m} - 1)^{1/n}\right]$$

where:
- $P_c$ is the capillary pressure (psi)
- $\alpha$ is a scaling parameter (1/psi)
- $S_e$ is the effective saturation (dimensionless, 0-1)
- $n$ is an empirical parameter (> 1)
- $m = 1 - 1/n$ (standard Mualem constraint)

### Comparison with Brooks-Corey

| Feature | Brooks-Corey | van Genuchten |
|---------|--------------|---------------|
| **Formula** | Power-law | Exponential |
| **Parameters** | $P_d$, $\lambda$ | $\alpha$, $n$ |
| **Smoothness** | Moderate | High |
| **Near endpoints** | Can be sharp | Smooth |
| **Derivatives** | Discontinuous at Se=1 | Continuous everywhere |
| **Common use** | Petroleum engineering | Soil science, groundwater |

**When to use van Genuchten over Brooks-Corey**:
- Need smooth derivatives for numerical stability
- Modeling near residual saturations
- Importing from groundwater/soil science literature
- Require continuous second derivatives

## Three-Phase Implementation

For three-phase systems, BORES computes two capillary pressures using van Genuchten formula:

### Oil-Water Capillary Pressure

**Water-Wet**:
$$P_{cow} = \frac{1}{\alpha_{ow}^{ww}} \left[(S_{ew}^{-1/m_{ww}} - 1)^{1/n_{ww}}\right]$$

where $m_{ww} = 1 - 1/n_{ww}$

**Oil-Wet**:
$$P_{cow} = -\frac{1}{\alpha_{ow}^{ow}} \left[(S_{ew}^{-1/m_{ow}} - 1)^{1/n_{ow}}\right]$$

where $m_{ow} = 1 - 1/n_{ow}$

**Mixed-Wet**:
$$P_{cow} = f_{ww} \cdot P_{cow}^{ww} + (1-f_{ww}) \cdot P_{cow}^{ow}$$

### Gas-Oil Capillary Pressure

$$P_{cgo} = \frac{1}{\alpha_{go}} \left[(S_{eg}^{-1/m_{go}} - 1)^{1/n_{go}}\right]$$

where $m_{go} = 1 - 1/n_{go}$

## API Reference

### Class Definition

From src/bores/capillary_pressures.py:

```python
@attrs.frozen
class VanGenuchtenCapillaryPressureModel:
    """
    van Genuchten capillary pressure model for three-phase systems.

    Implements: Pc = (1/α) * [(Se^(-1/m) - 1)^(1/n)] where m = 1 - 1/n

    Provides smoother transitions than Brooks-Corey model.
    """

    irreducible_water_saturation: Optional[float] = None
    """Default irreducible water saturation (Swc)."""

    residual_oil_saturation_water: Optional[float] = None
    """Default residual oil saturation after water flood (Sorw)."""

    residual_oil_saturation_gas: Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg)."""

    residual_gas_saturation: Optional[float] = None
    """Default residual gas saturation (Sgr)."""

    oil_water_alpha_water_wet: float = 0.01
    """van Genuchten α parameter for oil-water (water-wet) [1/psi]."""

    oil_water_alpha_oil_wet: float = 0.01
    """van Genuchten α parameter for oil-water (oil-wet) [1/psi]."""

    oil_water_n_water_wet: float = 2.0
    """van Genuchten n parameter for oil-water (water-wet)."""

    oil_water_n_oil_wet: float = 2.0
    """van Genuchten n parameter for oil-water (oil-wet)."""

    gas_oil_alpha: float = 0.01
    """van Genuchten α parameter for gas-oil [1/psi]."""

    gas_oil_n: float = 2.0
    """van Genuchten n parameter for gas-oil."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (WATER_WET, OIL_WET, or MIXED_WET)."""

    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet (0-1)."""
```

### Parameters

#### Alpha Parameters

**oil_water_alpha_water_wet** (`float`)
- Symbol: $\alpha_{ow}^{ww}$
- Units: 1/psi
- Description: Inverse of characteristic pressure for water-wet oil-water system
- Typical range: 0.001 - 0.1 (1/psi)
- Default: 0.01 (1/psi)
- Notes: Higher α = lower characteristic pressure = easier displacement

**oil_water_alpha_oil_wet** (`float`)
- Symbol: $\alpha_{ow}^{ow}$
- Units: 1/psi
- Typical range: 0.001 - 0.1 (1/psi)
- Default: 0.01 (1/psi)

**gas_oil_alpha** (`float`)
- Symbol: $\alpha_{go}$
- Units: 1/psi
- Typical range: 0.001 - 0.1 (1/psi)
- Default: 0.01 (1/psi)

**Conversion from Brooks-Corey**:

Approximate conversion from Brooks-Corey entry pressure:
$$\alpha \approx \frac{1}{P_d}$$

#### N Parameters

**oil_water_n_water_wet** (`float`)
- Symbol: $n_{ow}^{ww}$
- Description: Controls curve shape for water-wet oil-water system
- Range: Must be > 1.0
- Typical range: 1.5 - 5.0
- Default: 2.0
- Notes: Higher n = steeper curve, more uniform pores

**oil_water_n_oil_wet** (`float`)
- Symbol: $n_{ow}^{ow}$
- Range: Must be > 1.0
- Typical range: 1.5 - 5.0
- Default: 2.0

**gas_oil_n** (`float`)
- Symbol: $n_{go}$
- Range: Must be > 1.0
- Typical range: 1.5 - 5.0
- Default: 2.0

**Relationship to Brooks-Corey**:

The n parameter is related to Brooks-Corey λ:
$$n \approx \lambda + 1$$

#### System Properties

**wettability** (`Wettability`)
- Options: `WATER_WET`, `OIL_WET`, `MIXED_WET`
- Default: `WATER_WET`

**mixed_wet_water_fraction** (`float`)
- Range: 0.0 - 1.0
- Default: 0.5
- Notes: Only used for `MIXED_WET` systems

## Usage Examples

### Basic Water-Wet System

```python
import bores

# Create van Genuchten model
capillary_pressure_table = bores.VanGenuchtenCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_alpha_water_wet=0.02,  # 1/psi
    oil_water_n_water_wet=2.5,
    gas_oil_alpha=0.03,  # 1/psi
    gas_oil_n=2.0,
    wettability=bores.Wettability.WATER_WET,
)

# Compute capillary pressures
pc = capillary_pressure_table(
    water_saturation=0.35,
    oil_saturation=0.50,
    gas_saturation=0.15,
)

print(f"Pcow = {pc['oil_water']:.2f} psi")
print(f"Pcgo = {pc['gas_oil']:.2f} psi")
```

### Converting from Brooks-Corey

```python
import bores

# Brooks-Corey parameters
Pd_ow = 2.0  # psi
lambda_ow = 2.0

Pd_go = 1.5  # psi
lambda_go = 2.0

# Convert to van Genuchten (approximate)
alpha_ow = 1.0 / Pd_ow  # 0.5 (1/psi)
n_ow = lambda_ow + 1.0  # 3.0

alpha_go = 1.0 / Pd_go  # 0.67 (1/psi)
n_go = lambda_go + 1.0  # 3.0

# Create van Genuchten model with converted parameters
capillary_pressure_table = bores.VanGenuchtenCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_alpha_water_wet=alpha_ow,
    oil_water_n_water_wet=n_ow,
    gas_oil_alpha=alpha_go,
    gas_oil_n=n_go,
    wettability=bores.Wettability.WATER_WET,
)
```

### Comparing with Brooks-Corey

```python
import bores
import numpy as np

# Create both models with similar parameters
bc_model = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=1.5,
    gas_oil_pore_size_distribution_index=2.0,
)

vg_model = bores.VanGenuchtenCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_alpha_water_wet=0.5,  # 1/Pd = 1/2.0
    oil_water_n_water_wet=3.0,      # λ + 1 = 2 + 1
    gas_oil_alpha=0.67,  # 1/1.5
    gas_oil_n=3.0,
)

# Compare at various saturations
saturations = np.linspace(0.2, 0.7, 6)

print("Sw    Pcow_BC  Pcow_VG")
for sw in saturations:
    pc_bc = bc_model(water_saturation=sw, oil_saturation=1.0-sw, gas_saturation=0.0)
    pc_vg = vg_model(water_saturation=sw, oil_saturation=1.0-sw, gas_saturation=0.0)
    print(f"{sw:.2f}  {pc_bc['oil_water']:6.2f}  {pc_vg['oil_water']:6.2f}")
```

### Grid-Based Computation

```python
import bores

model = bores.VanGenuchtenCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_alpha_water_wet=0.02,
    oil_water_n_water_wet=2.5,
    gas_oil_alpha=0.03,
    gas_oil_n=2.0,
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

print(f"Average Pcow: {pc_grid['oil_water'].mean():.2f} psi")
```

## Parameter Selection Guidelines

### Alpha Parameter by Rock Type

The alpha parameter is the inverse of a characteristic pressure:

**High-Permeability Rock**:
```python
oil_water_alpha_water_wet=0.1  # 1/psi → characteristic Pc ~ 10 psi
gas_oil_alpha=0.15  # 1/psi
```

**Moderate-Permeability Rock**:
```python
oil_water_alpha_water_wet=0.02  # 1/psi → characteristic Pc ~ 50 psi
gas_oil_alpha=0.03  # 1/psi
```

**Low-Permeability Rock**:
```python
oil_water_alpha_water_wet=0.005  # 1/psi → characteristic Pc ~ 200 psi
gas_oil_alpha=0.01  # 1/psi
```

### N Parameter Selection

The n parameter controls curve steepness:

- **n = 1.5**: Gentle curve, wide pore distribution
- **n = 2.0**: Moderate curve (default)
- **n = 3.0**: Steeper curve, more uniform pores
- **n > 4.0**: Very steep, highly uniform (rare)

**Note**: n must be > 1.0 (enforced by validation)

## Advantages of van Genuchten

### Smooth Derivatives

The van Genuchten model provides continuous first and second derivatives:

$$\frac{dP_c}{dS_e} = -\frac{1}{\alpha m n} S_e^{-(1+m)/m} \left[(S_e^{-1/m} - 1)^{(1-n)/n}\right]$$

This smoothness benefits:
- Numerical stability in simulators
- Newton-Raphson convergence
- Sensitivity analysis

### Better Near-Endpoint Behavior

Unlike Brooks-Corey which can have sharp transitions, van Genuchten provides smooth behavior as $S_e \to 1$:

- Brooks-Corey: $P_c \to P_d$ abruptly
- van Genuchten: $P_c \to 0$ smoothly

This is particularly important for:
- Imbibition processes
- Near-residual saturation conditions
- Capillary end effects in core floods

## Limitations

### Computational Cost

van Genuchten requires more expensive exponential operations compared to Brooks-Corey power-law:

- Brooks-Corey: Simple power operations
- van Genuchten: Exponential + power operations

For large grids, this can impact performance (typically 10-30% slower).

### Parameter Interpretation

van Genuchten parameters (α, n) are less intuitive than Brooks-Corey (Pd, λ):

- α: Inverse pressure (less physical)
- n: Abstract shape parameter

Most petroleum engineers are more familiar with entry pressure and pore size distribution index.

## Physical Interpretation

### Characteristic Pressure

The van Genuchten α parameter defines a characteristic pressure:

$$P_{char} = \frac{1}{\alpha}$$

This is roughly equivalent to Brooks-Corey entry pressure but represents the pressure at which the capillary curve inflects (not the entry threshold).

### Curve Shape

The m parameter (derived from n) controls curvature:

$$m = 1 - \frac{1}{n}$$

- Higher n → higher m → steeper curve at intermediate saturations
- Lower n → lower m → gentler curve throughout

## Implementation Notes

The implementation in src/bores/capillary_pressures.py handles:

- Validation: Ensures n > 1.0, α > 0.0
- Automatic saturation normalization
- Element-wise array operations
- Safe handling of edge cases
- Wettability sign conventions
- Mixed-wet weighted averaging

## See Also

- [Brooks-Corey Model](brooks-corey.md) - Traditional power-law model
- [Leverett J-Function](leverett-j.md) - Dimensionless scaling correlation
- [Capillary Pressure Index](index.md) - Module overview
