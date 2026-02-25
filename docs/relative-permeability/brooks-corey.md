# Brooks-Corey Three-Phase Relative Permeability Model

The Brooks-Corey model is the primary analytical relative permeability model in BORES. It uses power-law (Corey-type) functions to compute water and gas relative permeabilities, and applies a mixing rule to compute three-phase oil relative permeability.

## Model Description

The Brooks-Corey model computes relative permeabilities using normalized (effective) saturations raised to empirical exponents. The model supports both water-wet and oil-wet wettability assumptions.

### Mathematical Formulation

#### Effective Saturation

Effective saturation normalizes actual saturation to the mobile pore space:

$$S_e = \frac{S - S_{residual}}{1 - S_{connate} - S_{residual}}$$

where:

- $S$ is the actual phase saturation
- $S_{residual}$ is the residual (immobile) saturation for that phase
- $S_{connate}$ is the connate (irreducible) saturation of the wetting phase

#### Water-Wet System

For water-wet reservoirs (most common):

**Water Relative Permeability**:
$$k_{rw} = S_{ew}^{n_w}$$

where:
$$S_{ew} = \frac{S_w - S_{wc}}{1 - S_{wc} - S_{orw}}$$

**Gas Relative Permeability**:
$$k_{rg} = S_{eg}^{n_g}$$

where:
$$S_{eg} = \frac{S_g - S_{gr}}{1 - S_{wc} - S_{gr} - S_{org}}$$

**Oil Relative Permeability** (Three-Phase):

Oil relative permeability in a three-phase system is computed using a mixing rule that combines two-phase values:

1. Compute $k_{ro}^{(w)}$ from oil-water curve
2. Compute $k_{ro}^{(g)}$ from gas-oil curve
3. Apply mixing rule: $k_{ro} = f(k_{ro}^{(w)}, k_{ro}^{(g)}, S_w, S_o, S_g)$

The default mixing rule is `eclipse_rule` which provides smooth transitions and matches industry-standard simulator behavior.

#### Oil-Wet System

For oil-wet reservoirs, oil becomes the wetting phase and water becomes the intermediate phase. The model automatically adjusts the formulation based on the specified wettability type.

## API Reference

### Class Definition

From src/bores/relperm.py:

```python
@attrs.frozen
class BrooksCoreyThreePhaseRelPermModel:
    """
    Brooks-Corey-type three-phase relative permeability model.

    Uses the Brooks-Corey model for two-phase relative permeabilities
    and a mixing rule for oil in three-phase system.
    """

    irreducible_water_saturation: Optional[float] = None
    """Irreducible water saturation (Swc)."""

    residual_oil_saturation_water: Optional[float] = None
    """Residual oil saturation after water flood (Sorw)."""

    residual_oil_saturation_gas: Optional[float] = None
    """Residual oil saturation after gas flood (Sorg)."""

    residual_gas_saturation: Optional[float] = None
    """Residual gas saturation (Sgr)."""

    water_exponent: float = 2.0
    """Corey exponent for water relative permeability."""

    oil_exponent: float = 2.0
    """Corey exponent for oil relative permeability."""

    gas_exponent: float = 2.0
    """Corey exponent for gas relative permeability."""

    wettability: Wettability = Wettability.WATER_WET
    """Wettability type (WATER_WET or OIL_WET)."""

    mixing_rule: MixingRule = eclipse_rule
    """Mixing rule function for three-phase oil relative permeability."""
```

### Parameters

#### Residual Saturations

**irreducible_water_saturation** (`float`, optional)

- Symbol: $S_{wc}$ or $S_{wi}$
- Description: Water saturation below which water is immobile (connate water)
- Typical range: 0.10 - 0.25
- Default: None (must be provided)
- Notes: Represents bound water in pore throats and clay minerals

**residual_oil_saturation_water** (`float`, optional)

- Symbol: $S_{orw}$
- Description: Oil saturation remaining after waterflooding to economic limit
- Typical range: 0.20 - 0.35
- Default: None (must be provided)
- Notes: Higher values indicate less efficient water displacement

**residual_oil_saturation_gas** (`float`, optional)

- Symbol: $S_{org}$
- Description: Oil saturation remaining after gas flooding to economic limit
- Typical range: 0.10 - 0.20
- Default: None (must be provided)
- Notes: Usually lower than Sorw because gas displacement is more efficient

**residual_gas_saturation** (`float`, optional)

- Symbol: $S_{gr}$
- Description: Gas saturation trapped after imbibition (water or oil displacing gas)
- Typical range: 0.03 - 0.10
- Default: None (must be provided)
- Notes: Represents gas bubbles trapped in pore bodies

#### Corey Exponents

**water_exponent** (`float`)

- Symbol: $n_w$
- Description: Power-law exponent for water relative permeability curve
- Typical range: 2.0 - 4.0
- Default: 2.0
- Effect: Higher values create steeper curves (slower krw increase with Sw)
- Notes: Higher for strongly water-wet systems

**oil_exponent** (`float`)

- Symbol: $n_o$
- Description: Power-law exponent for oil relative permeability curve
- Typical range: 2.0 - 3.0
- Default: 2.0
- Effect: Higher values create steeper curves (slower kro increase with So)
- Notes: Affects Stone mixing rule blending behavior

**gas_exponent** (`float`)

- Symbol: $n_g$
- Description: Power-law exponent for gas relative permeability curve
- Typical range: 2.0 - 3.0
- Default: 2.0
- Effect: Higher values create steeper curves (slower krg increase with Sg)
- Notes: Can be higher (3-4) for low-permeability gas reservoirs

#### System Properties

**wettability** (`Wettability`)

- Options: `Wettability.WATER_WET`, `Wettability.OIL_WET`
- Default: `Wettability.WATER_WET`
- Description: Controls which phase is treated as the wetting phase
- Notes: Affects curve shapes and endpoint behavior

**mixing_rule** (`MixingRule`)

- Default: `eclipse_rule`
- Description: Function to combine two-phase oil kr values in three-phase system
- Options: See [Mixing Rules Reference](mixing-rules.md)
- Recommendation: Use `eclipse_rule` for most applications

### Methods

**get_relative_permeabilities()**

Compute relative permeabilities for water, oil, and gas phases.

```python
def get_relative_permeabilities(
    self,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
    irreducible_water_saturation: Optional[float] = None,
    residual_oil_saturation_water: Optional[float] = None,
    residual_oil_saturation_gas: Optional[float] = None,
    residual_gas_saturation: Optional[float] = None,
) -> RelativePermeabilities:
    """
    Compute relative permeabilities for water, oil, and gas.

    Supports both scalar and array inputs for saturations.

    Parameters:
        water_saturation: Water saturation (0-1) - scalar or array
        oil_saturation: Oil saturation (0-1) - scalar or array
        gas_saturation: Gas saturation (0-1) - scalar or array
        irreducible_water_saturation: Optional override for Swc
        residual_oil_saturation_water: Optional override for Sorw
        residual_oil_saturation_gas: Optional override for Sorg
        residual_gas_saturation: Optional override for Sgr

    Returns:
        Dictionary with keys 'water', 'oil', 'gas' containing relative
        permeabilities (same shape as input saturations)
    """
```

****call**()**

Convenience method that calls `get_relative_permeabilities()`. Allows using the model as a callable.

```python
kr = model(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)
```

## Usage Examples

### Basic Usage

From scenarios/setup.py:

```python
import bores

# Create model with typical sandstone parameters
relative_permeability_table = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.Wettability.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Compute relative permeabilities at specific saturations
kr = relative_permeability_table(
    water_saturation=0.35,
    oil_saturation=0.50,
    gas_saturation=0.15,
)

print(f"krw = {kr['water']:.4f}")  # Water relative permeability
print(f"kro = {kr['oil']:.4f}")    # Oil relative permeability
print(f"krg = {kr['gas']:.4f}")    # Gas relative permeability
```

### Sensitivity to Exponents

```python
import bores
import numpy as np

# Base parameters
base_params = {
    "irreducible_water_saturation": 0.15,
    "residual_oil_saturation_water": 0.25,
    "residual_oil_saturation_gas": 0.15,
    "residual_gas_saturation": 0.045,
    "mixing_rule": bores.eclipse_rule,
}

# Test different water exponents
water_exponents = [1.5, 2.0, 2.5, 3.0, 4.0]
Sw_values = np.linspace(0.15, 0.75, 50)  # From Swc to 1-Sorw

for n_w in water_exponents:
    model = bores.BrooksCoreyThreePhaseRelPermModel(
        **base_params,
        water_exponent=n_w,
        oil_exponent=2.0,
        gas_exponent=2.0,
    )

    # Compute krw curve for oil-water system (Sg = 0)
    krw_values = []
    for Sw in Sw_values:
        kr = model(water_saturation=Sw, oil_saturation=1.0-Sw, gas_saturation=0.0)
        krw_values.append(kr['water'])

    print(f"n_w = {n_w}: krw at Sw=0.5 is {krw_values[len(krw_values)//2]:.4f}")
```

### Override Residual Saturations Per Call

```python
import bores

# Create model with default residual saturations
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Use default values
kr_default = model(water_saturation=0.3, oil_saturation=0.5, gas_saturation=0.2)

# Override for specific computation (e.g., different rock type region)
kr_override = model.get_relative_permeabilities(
    water_saturation=0.3,
    oil_saturation=0.5,
    gas_saturation=0.2,
    irreducible_water_saturation=0.20,  # Higher Swc
    residual_oil_saturation_water=0.30,  # Higher Sorw
)

print(f"Default: kro = {kr_default['oil']:.4f}")
print(f"Override: kro = {kr_override['oil']:.4f}")
```

### Grid-Based Computation

```python
import bores
import numpy as np

# Create saturation grids (e.g., from simulation state)
grid_shape = (20, 20, 10)
Sw_grid = bores.uniform_grid(grid_shape, value=0.3)
So_grid = bores.uniform_grid(grid_shape, value=0.5)
Sg_grid = bores.uniform_grid(grid_shape, value=0.2)

# Create model
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Compute relative permeabilities for entire grid
kr_grid = model(
    water_saturation=Sw_grid,
    oil_saturation=So_grid,
    gas_saturation=Sg_grid,
)

# Results are arrays matching input shape
print(f"krw grid shape: {kr_grid['water'].shape}")  # (20, 20, 10)
print(f"Average kro: {kr_grid['oil'].mean():.4f}")
print(f"Max krg: {kr_grid['gas'].max():.4f}")
```

## Parameter Selection Guidelines

### Typical Values by Rock Type

**Sandstone (Water-Wet)**:

```python
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,      # Moderate
    residual_oil_saturation_water=0.25,      # Moderate
    residual_oil_saturation_gas=0.15,        # Good gas displacement
    residual_gas_saturation=0.05,            # Low trapping
    water_exponent=2.5,                       # Moderate curve
    oil_exponent=2.0,                         # Standard
    gas_exponent=2.0,                         # Standard
    wettability=bores.Wettability.WATER_WET,
    mixing_rule=bores.eclipse_rule,
)
```

**Tight Sandstone (Water-Wet)**:

```python
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,      # Higher Swc
    residual_oil_saturation_water=0.35,      # Poor recovery
    residual_oil_saturation_gas=0.20,        # Poor gas sweep
    residual_gas_saturation=0.08,            # More trapping
    water_exponent=3.5,                       # Steep curve
    oil_exponent=2.5,                         # Reduced mobility
    gas_exponent=3.0,                         # Reduced gas mobility
    wettability=bores.Wettability.WATER_WET,
    mixing_rule=bores.eclipse_rule,
)
```

**Carbonate (Oil-Wet)**:

```python
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.10,      # Lower Swc
    residual_oil_saturation_water=0.30,      # Poor waterflood
    residual_oil_saturation_gas=0.15,        # Moderate gas recovery
    residual_gas_saturation=0.04,            # Low trapping
    water_exponent=2.0,                       # Flatter curve
    oil_exponent=2.0,                         # Standard
    gas_exponent=2.0,                         # Standard
    wettability=bores.Wettability.OIL_WET,
    mixing_rule=bores.eclipse_rule,
)
```

### Exponent Selection

The Corey exponent controls curve steepness:

- **n = 1.0**: Linear relationship (rarely used)
- **n = 2.0**: Standard quadratic (most common baseline)
- **n = 3.0**: Steeper curve, slower phase mobility increase
- **n = 4.0**: Very steep, appropriate for strongly preferential wetting

**General guidelines**:

- Start with n = 2.0 for all phases
- Increase water exponent (2.5-3.5) for strongly water-wet systems
- Increase gas exponent (2.5-3.0) for low-permeability gas
- Keep oil exponent around 2.0 unless matching lab data

### Mixing Rule Selection

See [Mixing Rules Reference](mixing-rules.md) for detailed comparison. Quick recommendations:

- **General use**: `eclipse_rule` (industry standard, robust)
- **Conservative estimate**: `geometric_mean_rule` or `harmonic_mean_rule`
- **History matching**: Try `stone_I_rule` or `stone_II_rule`
- **Sensitivity analysis**: Compare `min_rule` (lower bound) vs `max_rule` (upper bound)

## Physical Interpretation

### Curve Shapes

The Brooks-Corey model produces smooth, monotonic curves:

- **At residual saturation**: kr = 0 (phase is immobile)
- **At maximum saturation**: kr approaches 1.0 (single-phase behavior)
- **Between extremes**: Power-law relationship creates characteristic concave shape

### Three-Phase Interference

When all three phases are present, oil relative permeability is reduced below either two-phase value. This represents the physical reality that oil has reduced flow paths when squeezed between water (in small pores) and gas (in large pores).

The mixing rule quantifies this reduction. More conservative rules (like `harmonic_mean_rule`) predict greater interference, while optimistic rules (like `arithmetic_mean_rule`) predict less.

### Wettability Effects

**Water-Wet Systems**:

- Water preferentially occupies small pores
- Gas preferentially occupies large pores
- Oil occupies intermediate pores
- Water mobility increases quickly with saturation
- Gas has high mobility even at low saturation

**Oil-Wet Systems**:

- Oil preferentially occupies small pores
- Water is pushed to larger pores
- Water mobility reduced compared to water-wet case
- Oil retains mobility to lower saturations
- Typically lower waterflood recovery

## Implementation Notes

The implementation in src/bores/relperm.py uses Numba JIT compilation for performance. The core computation function `compute_corey_three_phase_relative_permeabilities()` is optimized for both scalar and array inputs.

**Key features**:

- Automatic saturation normalization (ensures Sw + So + Sg = 1)
- Element-wise array operations for grid-based computation
- Safe handling of edge cases (zero denominators, residual saturations)
- Validation of input saturations and residual saturation constraints

## See Also

- [Mixing Rules Reference](mixing-rules.md) - Complete documentation of all mixing rules
- [Lookup Tables](tables.md) - Using tabular relative permeability data
- [Capillary Pressure Models](../capillary-pressure/index.md) - Companion capillary pressure documentation
- Rock-Fluid Tables (see guides/rock-fluid-properties.md) - Integrating relative permeability into simulations
