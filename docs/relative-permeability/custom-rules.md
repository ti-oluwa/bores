# Creating Custom Mixing Rules

Guide to creating, registering, and using custom mixing rules for three-phase relative permeability in BORES.

## Overview

While BORES provides 15 built-in mixing rules, you may need custom mixing rules to:
- Implement published correlations not included in BORES
- Match specific laboratory or field data
- Explore new theoretical formulations
- Adapt mixing rules to unique reservoir characteristics

BORES makes it easy to create custom mixing rules using the `@mixing_rule` decorator, with full support for Numba JIT compilation for performance.

## Basic Custom Mixing Rule

### Template

```python
import bores
from bores.types import FloatOrArray
import numba
import numpy as np

@bores.mixing_rule(name="my_custom_rule")
@numba.njit(cache=True)
def my_custom_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Custom mixing rule description.

    Parameters:
        kro_w: Oil relative permeability from oil-water curve
        kro_g: Oil relative permeability from gas-oil curve
        water_saturation: Current water saturation
        oil_saturation: Current oil saturation
        gas_saturation: Current gas saturation

    Returns:
        Three-phase oil relative permeability
    """
    # Your custom formula here
    result = (kro_w + kro_g) / 2.0  # Example: arithmetic mean
    return result
```

### Usage

```python
import bores

# Use your custom rule in a model
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=my_custom_rule,  # Your custom rule
)

kr = model(water_saturation=0.3, oil_saturation=0.5, gas_saturation=0.2)
print(f"kro with custom rule: {kr['oil']:.4f}")
```

## Real-World Examples

### Example 1: Weighted Mixing Rule

A custom rule that weights the two-phase values by user-defined factors:

```python
import bores
from bores.types import FloatOrArray
import numba
import numpy as np

@bores.mixing_rule(name="weighted_70_30")
@numba.njit(cache=True)
def weighted_70_30(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Weighted mixing rule: 70% from oil-water, 30% from gas-oil.

    Useful when oil-water behavior dominates (e.g., strong waterflooding).
    """
    return 0.7 * kro_w + 0.3 * kro_g

# Usage
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=weighted_70_30,
)
```

### Example 2: Saturation-Dependent Weighting

A rule that adjusts weighting based on which displacing phase is dominant:

```python
@bores.mixing_rule(name="adaptive_saturation_weighted")
@numba.njit(cache=True)
def adaptive_saturation_weighted(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Adaptive weighting based on displacing phase saturations.

    Emphasizes oil-water curve when water saturation is high,
    and gas-oil curve when gas saturation is high.
    """
    # Total displacing phases
    total_displacing = water_saturation + gas_saturation

    # Compute weights (avoid division by zero)
    water_weight = np.where(
        total_displacing > 0.0,
        water_saturation / total_displacing,
        0.5  # Equal weighting if no displacing phases
    )
    gas_weight = np.where(
        total_displacing > 0.0,
        gas_saturation / total_displacing,
        0.5
    )

    # Weighted combination
    result = kro_w * water_weight + kro_g * gas_weight

    # Handle pure oil case
    return np.where(
        total_displacing > 0.0,
        result,
        np.maximum(kro_w, kro_g)  # Use maximum when only oil present
    )
```

### Example 3: Threshold-Based Rule

A rule that switches behavior based on saturation thresholds:

```python
@bores.mixing_rule(name="threshold_based")
@numba.njit(cache=True)
def threshold_based(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Threshold-based mixing rule.

    - If Sw > 0.5: Use oil-water curve (waterflooding dominant)
    - If Sg > 0.3: Use gas-oil curve (gas flooding dominant)
    - Otherwise: Use geometric mean
    """
    # Geometric mean as baseline
    geometric_mean = np.sqrt(kro_w * kro_g)

    # Apply thresholds
    result = np.where(
        water_saturation > 0.5,
        kro_w,  # Water dominant
        np.where(
            gas_saturation > 0.3,
            kro_g,  # Gas dominant
            geometric_mean  # Balanced
        )
    )

    return result
```

## Parameterized Mixing Rules

For mixing rules with adjustable parameters, use a factory function:

### Template

```python
import bores
from bores.types import FloatOrArray, MixingRule
import numba
import numpy as np
import typing

def weighted_mixing_rule(alpha: float = 0.5) -> MixingRule:
    """
    Factory function for weighted mixing rule.

    Parameters:
        alpha: Weight for oil-water curve (0-1). Remaining weight (1-alpha) goes to gas-oil.

    Returns:
        A mixing rule function with the specified weighting.
    """

    def serializer(rule: MixingRule, recurse: bool = True) -> typing.Dict[str, float]:
        """Serialize the rule parameters."""
        return {"alpha": alpha}

    def deserializer(data: typing.Dict[str, float]) -> MixingRule:
        """Deserialize the rule from parameters."""
        if not isinstance(data, dict) or "alpha" not in data:
            raise ValueError("Invalid data for weighted mixing rule deserialization.")
        return weighted_mixing_rule(alpha=data["alpha"])

    @bores.mixing_rule(
        name=f"weighted(alpha={alpha})",
        serializer=serializer,
        deserializer=deserializer,
    )
    @numba.njit(cache=True)
    def _rule(
        kro_w: FloatOrArray,
        kro_g: FloatOrArray,
        water_saturation: FloatOrArray,
        oil_saturation: FloatOrArray,
        gas_saturation: FloatOrArray,
    ) -> FloatOrArray:
        """Weighted combination with parameter alpha."""
        return alpha * kro_w + (1.0 - alpha) * kro_g

    _rule.__name__ = f"weighted_mixing_rule(alpha={alpha})"
    return _rule
```

### Usage

```python
# Create rule with specific parameter
custom_rule = weighted_mixing_rule(alpha=0.7)

# Use in model
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=custom_rule,
)

# Test different parameter values
for alpha_val in [0.3, 0.5, 0.7]:
    rule = weighted_mixing_rule(alpha=alpha_val)
    model_test = bores.BrooksCoreyThreePhaseRelPermModel(
        irreducible_water_saturation=0.15,
        residual_oil_saturation_water=0.25,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.045,
        water_exponent=2.0,
        oil_exponent=2.0,
        gas_exponent=2.0,
        mixing_rule=rule,
    )

    kr = model_test(water_saturation=0.3, oil_saturation=0.5, gas_saturation=0.2)
    print(f"alpha={alpha_val}: kro={kr['oil']:.4f}")
```

## Implementation Requirements

### Function Signature

All mixing rules MUST have this exact signature:

```python
def mixing_rule_function(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
```

**Parameters**:
- `kro_w`: Oil relative permeability from oil-water two-phase curve
- `kro_g`: Oil relative permeability from gas-oil two-phase curve
- `water_saturation`: Current water saturation (0-1)
- `oil_saturation`: Current oil saturation (0-1)
- `gas_saturation`: Current gas saturation (0-1)

**Return**:
- Three-phase oil relative permeability (0-1)

### Input Types

All parameters can be:
- **Scalars** (`float`)
- **Arrays** (`np.ndarray` of any shape)

Your implementation must handle both cases correctly using NumPy broadcasting.

### Output Requirements

1. **Range**: Output must be in [0, 1]
2. **Type**: Output type must match input type (scalar → scalar, array → array)
3. **Shape**: For array inputs, output shape must match input shapes (after broadcasting)
4. **Edge cases**: Handle zero values, pure phases, numerical stability

### Numba Compatibility

For performance, use `@numba.njit(cache=True)`:

**Allowed operations**:
- NumPy array operations (`np.maximum`, `np.minimum`, `np.sqrt`, `np.where`, etc.)
- Standard arithmetic (`+`, `-`, `*`, `/`, `**`)
- Comparison operators (`>`, `<`, `==`, etc.)
- Mathematical functions from NumPy (`np.exp`, `np.log`, `np.sin`, etc.)

**Disallowed**:
- Python lists, dictionaries, sets
- String operations (except in docstrings)
- Complex control flow that Numba can't compile
- Non-Numba-compatible libraries

## Best Practices

### 1. Handle Edge Cases

```python
@bores.mixing_rule(name="robust_custom_rule")
@numba.njit(cache=True)
def robust_custom_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """Example of handling edge cases."""

    # Protect against division by zero
    safe_kro_w = np.maximum(kro_w, 1e-12)
    safe_kro_g = np.maximum(kro_g, 1e-12)

    # Compute result
    result = (safe_kro_w * safe_kro_g) / (safe_kro_w + safe_kro_g)

    # Handle case where both inputs are zero
    result = np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)

    # Clamp to valid range
    result = np.clip(result, 0.0, 1.0)

    return result
```

### 2. Document Physical Meaning

```python
@bores.mixing_rule(name="documented_rule")
@numba.njit(cache=True)
def documented_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Custom mixing rule with clear physical interpretation.

    Formula:
        kro = kro_w * f_w + kro_g * f_g

    where:
        f_w = So / (So + Sg)   (oil fraction in oil-gas system)
        f_g = So / (So + Sw)   (oil fraction in oil-water system)

    Physical Meaning:
        Weights two-phase values by oil's fractional presence in
        each respective two-phase system. Emphasizes oil-water curve
        when gas saturation is high, and gas-oil curve when water
        saturation is high.

    Suitable For:
        - Moderate interference assumption
        - Systems with balanced water and gas flooding
        - General three-phase flow

    References:
        Based on ECLIPSE simulator default (similar behavior).
    """
    # ... implementation ...
```

### 3. Validate Inputs (Development Only)

During development, add validation (remove for production):

```python
@bores.mixing_rule(name="validated_rule")
@numba.njit(cache=True)
def validated_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """Rule with input validation (for development)."""

    # Development: Check kr bounds
    # assert np.all((kro_w >= 0.0) & (kro_w <= 1.0)), "kro_w out of range"
    # assert np.all((kro_g >= 0.0) & (kro_g <= 1.0)), "kro_g out of range"

    # Development: Check saturation bounds
    # assert np.all((water_saturation >= 0.0) & (water_saturation <= 1.0))
    # assert np.all((oil_saturation >= 0.0) & (oil_saturation <= 1.0))
    # assert np.all((gas_saturation >= 0.0) & (gas_saturation <= 1.0))

    # Compute result
    result = np.minimum(kro_w, kro_g)
    return result
```

### 4. Test Thoroughly

```python
import bores
import numpy as np

# Create your custom rule
@bores.mixing_rule(name="test_rule")
@numba.njit(cache=True)
def test_rule(kro_w, kro_g, Sw, So, Sg):
    return (kro_w + kro_g) / 2.0

# Test 1: Scalar inputs
kr = test_rule(kro_w=0.5, kro_g=0.3, Sw=0.3, So=0.5, Sg=0.2)
assert 0.0 <= kr <= 1.0, "Output out of range"
print(f"Test 1 (scalar): kro = {kr:.4f}")

# Test 2: Array inputs
kro_w_arr = np.array([0.5, 0.6, 0.4])
kro_g_arr = np.array([0.3, 0.4, 0.2])
Sw_arr = np.array([0.3, 0.3, 0.3])
So_arr = np.array([0.5, 0.5, 0.5])
Sg_arr = np.array([0.2, 0.2, 0.2])

kr_arr = test_rule(kro_w_arr, kro_g_arr, Sw_arr, So_arr, Sg_arr)
assert kr_arr.shape == kro_w_arr.shape, "Shape mismatch"
assert np.all((kr_arr >= 0.0) & (kr_arr <= 1.0)), "Array output out of range"
print(f"Test 2 (array): kro = {kr_arr}")

# Test 3: Two-phase limits
kr_ow = test_rule(kro_w=0.5, kro_g=1.0, Sw=0.3, So=0.7, Sg=0.0)  # Sg=0, should → kro_w
kr_og = test_rule(kro_w=1.0, kro_g=0.3, Sw=0.0, So=0.7, Sg=0.3)  # Sw=0, should → kro_g
print(f"Test 3 (two-phase limits): kr_ow={kr_ow:.4f}, kr_og={kr_og:.4f}")

# Test 4: In model
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=test_rule,
)

kr_model = model(water_saturation=0.3, oil_saturation=0.5, gas_saturation=0.2)
print(f"Test 4 (in model): kro = {kr_model['oil']:.4f}")
```

## Debugging Tips

### 1. Start Simple

Begin with a simple formula, verify it works, then add complexity:

```python
# Step 1: Basic arithmetic mean (simple)
@bores.mixing_rule(name="debug_v1")
@numba.njit(cache=True)
def debug_v1(kro_w, kro_g, Sw, So, Sg):
    return (kro_w + kro_g) / 2.0

# Step 2: Add saturation weighting
@bores.mixing_rule(name="debug_v2")
@numba.njit(cache=True)
def debug_v2(kro_w, kro_g, Sw, So, Sg):
    total = Sw + Sg
    if total > 0.0:
        w_weight = Sw / total
        g_weight = Sg / total
    else:
        w_weight = 0.5
        g_weight = 0.5
    return kro_w * w_weight + kro_g * g_weight

# Step 3: Convert to NumPy (for arrays)
@bores.mixing_rule(name="debug_v3")
@numba.njit(cache=True)
def debug_v3(kro_w, kro_g, Sw, So, Sg):
    total = Sw + Sg
    w_weight = np.where(total > 0.0, Sw / total, 0.5)
    g_weight = np.where(total > 0.0, Sg / total, 0.5)
    return kro_w * w_weight + kro_g * g_weight
```

### 2. Disable Numba Temporarily

For debugging, remove `@numba.njit` to use Python debugger:

```python
@bores.mixing_rule(name="debug_no_numba")
# @numba.njit(cache=True)  # Comment out during debugging
def debug_no_numba(kro_w, kro_g, Sw, So, Sg):
    # Add print statements or breakpoints here
    print(f"kro_w={kro_w}, kro_g={kro_g}, Sw={Sw}, So={So}, Sg={Sg}")
    result = (kro_w + kro_g) / 2.0
    print(f"result={result}")
    return result
```

## Advanced Topics

### Saturation History Dependence

Some advanced mixing rules might depend on saturation history (e.g., hysteresis). While the current mixing rule signature doesn't support this directly, you can use closures:

```python
def history_dependent_mixing_rule():
    """Factory for mixing rule with internal state (advanced)."""
    # Note: This won't work directly with Numba
    # Consider using grid-based state tracking instead

    previous_saturations = {"Sw": 0.0, "So": 1.0, "Sg": 0.0}

    @bores.mixing_rule(name="history_dependent")
    def _rule(kro_w, kro_g, Sw, So, Sg):
        # Access previous state
        dSw = Sw - previous_saturations["Sw"]

        # Update state
        previous_saturations["Sw"] = Sw
        previous_saturations["So"] = So
        previous_saturations["Sg"] = Sg

        # Compute based on direction (imbibition vs drainage)
        if dSw > 0:  # Imbibition
            return kro_w * 0.8 + kro_g * 0.2
        else:  # Drainage
            return kro_w * 0.2 + kro_g * 0.8

    return _rule
```

## See Also

- [Mixing Rules Reference](mixing-rules.md) - Documentation of all built-in mixing rules
- [Brooks-Corey Model](brooks-corey.md) - Using mixing rules with analytical models
- [Lookup Tables](tables.md) - Using mixing rules with tabular data
- [Relative Permeability Index](index.md) - Module overview
