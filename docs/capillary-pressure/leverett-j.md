# Leverett J-Function Capillary Pressure Model

The Leverett J-function provides a dimensionless correlation for scaling capillary pressure across different rock types and fluid systems. It explicitly accounts for rock properties (permeability, porosity) and fluid properties (interfacial tension, contact angle).

## Model Description

The Leverett J-function normalizes capillary pressure by rock and fluid properties, creating a universal dimensionless function that can be scaled across heterogeneous reservoirs.

### Mathematical Formulation

The Leverett J-function relates capillary pressure to rock and fluid properties:

$$P_c = \sigma \cdot \cos(\theta) \cdot \sqrt{\frac{\phi}{k}} \cdot J(S_e)$$

where:
- $P_c$ is the capillary pressure (psi)
- $\sigma$ is the interfacial tension (dyne/cm, converted to psi)
- $\theta$ is the contact angle (degrees)
- $\phi$ is the porosity (fraction)
- $k$ is the absolute permeability (mD)
- $J(S_e)$ is the dimensionless Leverett J-function

### Leverett Scaling Factor

The term $\sqrt{\phi/k}$ is the Leverett scaling factor:

$$\text{Leverett Factor} = \sqrt{\frac{\phi}{k}}$$

This factor:
- Has units of $\sqrt{1/\text{mD}}$
- Increases with decreasing permeability (tighter rocks)
- Increases with increasing porosity (more pore space)
- Scales capillary pressure inversely with pore throat size

### J-Function Form

BORES uses a power-law form for the J-function:

$$J(S_e) = a \cdot S_e^{-b}$$

where:
- $a$ is an empirical coefficient (dimensionless)
- $b$ is an empirical exponent (dimensionless)
- $S_e$ is the effective saturation

These parameters are typically fit to laboratory J-function measurements.

## Physical Interpretation

### Interfacial Tension

Interfacial tension ($\sigma$) is the energy per unit area at the fluid interface:

**Typical Values**:
- Oil-Water: 20-40 dyne/cm (typically ~30 dyne/cm)
- Gas-Oil: 15-25 dyne/cm (typically ~20 dyne/cm)
- Gas-Water: 60-75 dyne/cm

Higher interfacial tension → higher capillary pressure

### Contact Angle

Contact angle ($\theta$) measures wettability through the fluid-fluid-solid interface:

**Water-Wet Systems**:
- $\theta = 0°$: Perfect water-wetting
- $\cos(\theta) = 1.0$ (maximum)

**Oil-Wet Systems**:
- $\theta = 180°$: Perfect oil-wetting
- $\cos(\theta) = -1.0$ (reverses sign)

**Neutral-Wet**:
- $\theta = 90°$: No wetting preference
- $\cos(\theta) = 0.0$

**Practical Ranges**:
- Strongly water-wet: $\theta$ = 0-30° ($\cos \theta$ = 0.87-1.0)
- Weakly water-wet: $\theta$ = 30-60° ($\cos \theta$ = 0.5-0.87)
- Intermediate: $\theta$ = 60-120°
- Oil-wet: $\theta$ > 120°

## Three-Phase Implementation

For three-phase systems, BORES computes two capillary pressures:

### Oil-Water Capillary Pressure

$$P_{cow} = \sigma_{ow} \cdot \cos(\theta_{ow}) \cdot \sqrt{\frac{\phi}{k}} \cdot J_{ow}(S_{ew})$$

Sign convention:
- Water-wet: $\cos(\theta_{ow}) > 0 \Rightarrow P_{cow} > 0$
- Oil-wet: $\cos(\theta_{ow}) < 0 \Rightarrow P_{cow} < 0$
- Mixed-wet: Weighted combination

### Gas-Oil Capillary Pressure

$$P_{cgo} = \sigma_{go} \cdot \cos(\theta_{go}) \cdot \sqrt{\frac{\phi}{k}} \cdot J_{go}(S_{eg})$$

Gas is always non-wetting to oil, so $P_{cgo} > 0$ always.

## API Reference

### Class Definition

From src/bores/capillary_pressures.py:

```python
@attrs.frozen
class LeverettJCapillaryPressureModel:
    """
    Leverett J-function capillary pressure model for three-phase systems.

    Uses dimensionless J-function correlation to relate capillary pressure
    to rock and fluid properties: Pc = σ * cos(θ) * sqrt(φ/k) * J(Se)

    Useful when capillary pressure data needs to be scaled across different
    rock types or fluid systems.
    """

    irreducible_water_saturation: Optional[float] = None
    """Default irreducible water saturation (Swc)."""

    residual_oil_saturation_water: Optional[float] = None
    """Default residual oil saturation after water flood (Sorw)."""

    residual_oil_saturation_gas: Optional[float] = None
    """Default residual oil saturation after gas flood (Sorg)."""

    residual_gas_saturation: Optional[float] = None
    """Default residual gas saturation (Sgr)."""

    permeability: float = 100.0
    """Absolute permeability (mD)."""

    porosity: float = 0.2
    """Porosity (fraction, 0-1)."""

    oil_water_interfacial_tension: float = 30.0
    """Oil-water interfacial tension (dyne/cm)."""

    gas_oil_interfacial_tension: float = 20.0
    """Gas-oil interfacial tension (dyne/cm)."""

    contact_angle_oil_water: float = 0.0
    """Oil-water contact angle in degrees (0° = water-wet)."""

    contact_angle_gas_oil: float = 0.0
    """Gas-oil contact angle in degrees (0° = oil-wet to gas)."""

    mixed_wet_water_fraction: float = 0.5
    """Fraction of pore space that is water-wet (0-1)."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (affects sign convention)."""

    j_function_coefficient: float = 0.5
    """Empirical coefficient 'a' in J(Se) = a * Se^(-b)."""

    j_function_exponent: float = 0.5
    """Empirical exponent 'b' in J(Se) = a * Se^(-b)."""
```

### Parameters

#### Rock Properties

**permeability** (`float`)
- Units: mD
- Description: Absolute permeability used in Leverett scaling factor
- Typical range: 0.1 - 10,000 mD
- Default: 100.0 mD
- Notes: Can be cell-specific if using array inputs

**porosity** (`float`)
- Units: Fraction (0-1)
- Description: Porosity used in Leverett scaling factor
- Typical range: 0.05 - 0.35
- Default: 0.2
- Notes: Can be cell-specific if using array inputs

#### Fluid Properties

**oil_water_interfacial_tension** (`float`)
- Units: dyne/cm
- Description: Interfacial tension between oil and water
- Typical range: 20 - 40 dyne/cm
- Default: 30.0 dyne/cm
- Notes: Temperature and composition dependent

**gas_oil_interfacial_tension** (`float`)
- Units: dyne/cm
- Description: Interfacial tension between gas and oil
- Typical range: 15 - 25 dyne/cm
- Default: 20.0 dyne/cm
- Notes: Pressure dependent (decreases with pressure)

#### Contact Angles

**contact_angle_oil_water** (`float`)
- Units: Degrees
- Description: Contact angle for oil-water system
- Range: 0° (water-wet) to 180° (oil-wet)
- Default: 0.0° (perfect water-wet)
- Notes: Typically 0-30° for water-wet sandstones

**contact_angle_gas_oil** (`float`)
- Units: Degrees
- Description: Contact angle for gas-oil system
- Range: 0° (oil-wet to gas) to 180°
- Default: 0.0° (perfect oil-wet relative to gas)
- Notes: Gas is nearly always non-wetting to oil

#### J-Function Parameters

**j_function_coefficient** (`float`)
- Symbol: $a$
- Description: Empirical coefficient in J(Se) = a * Se^(-b)
- Typical range: 0.2 - 1.0
- Default: 0.5
- Notes: Fit to laboratory J-function data

**j_function_exponent** (`float`)
- Symbol: $b$
- Description: Empirical exponent in J(Se) = a * Se^(-b)
- Typical range: 0.3 - 0.7
- Default: 0.5
- Notes: Fit to laboratory J-function data

## Usage Examples

### Basic Usage with Rock Properties

```python
import bores

# Leverett J-function model with rock properties
capillary_pressure_table = bores.LeverettJCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    permeability=100.0,  # mD
    porosity=0.20,  # Fraction
    oil_water_interfacial_tension=30.0,  # dyne/cm
    gas_oil_interfacial_tension=20.0,  # dyne/cm
    contact_angle_oil_water=0.0,  # degrees (water-wet)
    contact_angle_gas_oil=0.0,  # degrees (oil-wet to gas)
    j_function_coefficient=0.5,
    j_function_exponent=0.5,
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

### Scaling Across Rock Types

```python
import bores

# Define base J-function parameters from lab measurements
base_params = {
    "irreducible_water_saturation": 0.15,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.045,
    "oil_water_interfacial_tension": 30.0,
    "gas_oil_interfacial_tension": 20.0,
    "contact_angle_oil_water": 0.0,
    "contact_angle_gas_oil": 0.0,
    "j_function_coefficient": 0.5,
    "j_function_exponent": 0.5,
    "wettability": bores.Wettability.WATER_WET,
}

# High-permeability zone
high_perm_model = bores.LeverettJCapillaryPressureModel(
    **base_params,
    permeability=500.0,  # mD
    porosity=0.25,
)

# Low-permeability zone
low_perm_model = bores.LeverettJCapillaryPressureModel(
    **base_params,
    permeability=20.0,  # mD
    porosity=0.12,
)

# Compare at same saturation
Sw, So, Sg = 0.35, 0.50, 0.15

pc_high = high_perm_model(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)
pc_low = low_perm_model(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)

print(f"High-perm Pcow: {pc_high['oil_water']:.2f} psi")
print(f"Low-perm Pcow: {pc_low['oil_water']:.2f} psi")
print(f"Ratio: {pc_low['oil_water'] / pc_high['oil_water']:.2f}")
```

### Effect of Interfacial Tension

```python
import bores

# Base case
model_base = bores.LeverettJCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    permeability=100.0,
    porosity=0.20,
    oil_water_interfacial_tension=30.0,  # Standard
    gas_oil_interfacial_tension=20.0,
    j_function_coefficient=0.5,
    j_function_exponent=0.5,
)

# Reduced IFT (e.g., surfactant flooding)
model_reduced_ift = bores.LeverettJCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    permeability=100.0,
    porosity=0.20,
    oil_water_interfacial_tension=3.0,  # 10x reduction
    gas_oil_interfacial_tension=2.0,
    j_function_coefficient=0.5,
    j_function_exponent=0.5,
)

# Compare
pc_base = model_base(water_saturation=0.35, oil_saturation=0.50, gas_saturation=0.15)
pc_reduced = model_reduced_ift(water_saturation=0.35, oil_saturation=0.50, gas_saturation=0.15)

print(f"Base IFT Pcow: {pc_base['oil_water']:.2f} psi")
print(f"Reduced IFT Pcow: {pc_reduced['oil_water']:.2f} psi")
print(f"Reduction factor: {pc_base['oil_water'] / pc_reduced['oil_water']:.1f}x")
```

## Parameter Selection Guidelines

### Interfacial Tension by Fluid Type

**Oil-Water Systems**:
- Light oil (> 30 API): 25-35 dyne/cm
- Medium oil (20-30 API): 30-40 dyne/cm
- Heavy oil (< 20 API): 35-45 dyne/cm

**Gas-Oil Systems**:
- High pressure (> 3000 psi): 10-15 dyne/cm
- Moderate pressure (1000-3000 psi): 15-25 dyne/cm
- Low pressure (< 1000 psi): 20-30 dyne/cm

### Contact Angle by Rock Type

**Sandstones** (typically water-wet):
- Clean quartz: 0-10°
- Moderate clay: 10-30°
- High clay: 20-40°

**Carbonates** (variable wettability):
- Fresh core: 20-60° (mixed)
- Aged core: 60-120° (intermediate to oil-wet)
- Strongly oil-wet: > 120°

### J-Function Parameters

Typical values from literature:

**Coefficient (a)**:
- Well-sorted sandstone: 0.4-0.6
- Heterogeneous sandstone: 0.6-0.8
- Carbonate: 0.5-1.0

**Exponent (b)**:
- Typical range: 0.4-0.6
- Default: 0.5 (square root dependence)

**Note**: These parameters should ideally be fit to laboratory J-function data specific to your reservoir.

## Advantages of Leverett J-Function

### Rock Property Scaling

The main advantage is explicit scaling with rock properties:

$$P_c \propto \sqrt{\frac{\phi}{k}}$$

This allows:
1. **Heterogeneous reservoirs**: Different Pc curves for each rock type
2. **Upscaling**: Consistent scaling from core to grid scale
3. **Geological realism**: Honoring spatial permeability/porosity variations

### Fluid Property Dependence

Explicitly accounts for fluid property changes:

$$P_c \propto \sigma \cdot \cos(\theta)$$

This enables:
1. **EOR processes**: Model IFT reduction (surfactants, miscible flooding)
2. **Wettability alteration**: Model contact angle changes
3. **Compositional effects**: Different IFT for different fluids

### Universal J-Function

Once J-function parameters are determined from lab data, the same J(Se) curve can be applied across:
- Different rock types (via permeability/porosity)
- Different fluid systems (via IFT and contact angle)
- Different scales (core to reservoir)

## Limitations

### Empirical J-Function

The power-law J(Se) = a * Se^(-b) is empirical and may not match all rock types perfectly. Some rocks exhibit:
- Non-power-law behavior
- Bimodal pore distributions (dual-porosity)
- Fracture-matrix interactions

For these cases, tabular J-function data may be more appropriate.

### Parameter Uncertainty

Leverett J-function requires more input parameters (k, φ, σ, θ) compared to Brooks-Corey (only Pd, λ). This can introduce:
- Additional uncertainty
- More calibration effort
- Need for laboratory measurements

### Computational Cost

Computing Pc requires:
1. Square root operation ($\sqrt{\phi/k}$)
2. Trigonometric function ($\cos\theta$)
3. Power-law ($S_e^{-b}$)

This is slightly more expensive than Brooks-Corey, though typically negligible for modern computers.

## Unit Conversion Note

The implementation converts interfacial tension from dyne/cm to psi:

$$\text{Conversion factor} = 4.725 \times 10^{-3} \frac{\text{psi}}{\text{dyne/cm}}$$

Internal calculation:
```python
Pc_psi = IFT_dyne_per_cm * 4.725 * cos(theta) * sqrt(phi/k) * J(Se)
```

## Implementation Notes

The implementation in src/bores/capillary_pressures.py handles:

- Unit conversion (dyne/cm to psi)
- Contact angle conversion (degrees to radians)
- Division by zero protection (when k=0 or φ=0)
- Wettability sign conventions
- Mixed-wet weighted averaging
- Array broadcasting for heterogeneous rock properties

## See Also

- [Brooks-Corey Model](brooks-corey.md) - Traditional entry pressure model
- [van Genuchten Model](van-genuchten.md) - Smooth exponential model
- [Capillary Pressure Index](index.md) - Module overview
