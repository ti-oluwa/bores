# Mixing Rules Reference

Complete documentation of all 15 mixing rules available in BORES for computing three-phase oil relative permeability.

## Overview

In three-phase flow simulations, computing oil relative permeability requires combining information from two-phase relative permeability curves (oil-water and gas-oil). Mixing rules are mathematical formulas that perform this combination.

### The Problem

Given:
- $k_{ro}^{(w)}$: Oil relative permeability from oil-water curve at current water saturation
- $k_{ro}^{(g)}$: Oil relative permeability from gas-oil curve at current gas saturation
- $S_w$, $S_o$, $S_g$: Current water, oil, and gas saturations

Find:
- $k_{ro}$: Three-phase oil relative permeability

### Physical Meaning

Mixing rules quantify how much the presence of both water and gas reduces oil mobility compared to two-phase systems. More conservative rules predict greater mobility reduction (interference), while optimistic rules predict less.

## Quick Reference Table

| Rule | Formula | Conservativeness | Use Case |
|------|---------|------------------|----------|
| [min_rule](#min-rule) | $\min(k_{ro}^{(w)}, k_{ro}^{(g)})$ | Very conservative | Lower bound, safety analysis |
| [harmonic_mean_rule](#harmonic-mean-rule) | $\frac{2}{1/k_{ro}^{(w)} + 1/k_{ro}^{(g)}}$ | Very conservative | Tight rocks, series flow |
| [geometric_mean_rule](#geometric-mean-rule) | $\sqrt{k_{ro}^{(w)} \cdot k_{ro}^{(g)}}$ | Conservative | General conservative estimate |
| [stone_I_rule](#stone-i-rule) | $\frac{k_{ro}^{(w)} \cdot k_{ro}^{(g)}}{k_{ro}^{(w)} + k_{ro}^{(g)} - k_{ro}^{(w)} \cdot k_{ro}^{(g)}}$ | Moderate | Water-wet systems |
| [stone_II_rule](#stone-ii-rule) | $k_{ro}^{(w)} + k_{ro}^{(g)} - 1$ (clamped) | Moderate | Industry standard (approximate) |
| [eclipse_rule](#eclipse-rule) | Saturation-weighted | Moderate | **Recommended** - simulator standard |
| [arithmetic_mean_rule](#arithmetic-mean-rule) | $(k_{ro}^{(w)} + k_{ro}^{(g)})/2$ | Optimistic | Upper bound estimate |
| [max_rule](#max-rule) | $\max(k_{ro}^{(w)}, k_{ro}^{(g)})$ | Very optimistic | Sensitivity analysis upper bound |

## Detailed Documentation

### min_rule

**Formula**:
$$k_{ro} = \min(k_{ro}^{(w)}, k_{ro}^{(g)})$$

**Description**:
The most conservative mixing rule. Takes the minimum of the two two-phase oil relative permeabilities. Represents maximum three-phase interference where oil mobility is limited by the worst-case two-phase system.

**When to Use**:
- Conservative estimates for production forecasting
- Safety factor in recovery calculations
- Lower bound in sensitivity analysis
- Risk-averse project economics

**Characteristics**:
- Always gives the lowest kro prediction
- Simple to understand and explain
- No dependence on saturation ratios (only on kr values)
- Can be overly pessimistic for most real systems

**Implementation** (src/bores/relperm.py:236-248):
```python
@mixing_rule
@numba.njit(cache=True)
def min_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """Conservative rule: kro = min(kro_w, kro_g)"""
    return np.minimum(kro_w, kro_g)
```

**Example**:
```python
import bores

model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.min_rule,  # Most conservative
)

kr = model(water_saturation=0.3, oil_saturation=0.5, gas_saturation=0.2)
print(f"kro (min rule) = {kr['oil']:.4f}")
```

---

### harmonic_mean_rule

**Formula**:
$$k_{ro} = \frac{2}{1/k_{ro}^{(w)} + 1/k_{ro}^{(g)}}$$

(Returns 0 if either input is 0)

**Description**:
Computes the harmonic mean of two-phase oil relative permeabilities. Heavily weighted by the smaller value. Appropriate for series flow paths where fluids must pass through both water-wet and gas-wet regions sequentially.

**When to Use**:
- Tight formations with tortuous flow paths
- Layered systems with series flow
- Conservative estimates (second most conservative after min_rule)
- Systems where oil must navigate both water and gas barriers

**Characteristics**:
- Very sensitive to low kr values
- If either $k_{ro}^{(w)}$ or $k_{ro}^{(g)}$ is zero, result is zero
- More conservative than geometric mean
- Physical interpretation: resistance-in-series analogy

**Implementation** (src/bores/relperm.py:381-409):
```python
@mixing_rule
@numba.njit(cache=True)
def harmonic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Harmonic mean: kro = 2 / (1/kro_w + 1/kro_g)
    Most conservative of the mean rules.
    """
    epsilon = 1e-30
    safe_kro_w = np.maximum(kro_w, epsilon)
    safe_kro_g = np.maximum(kro_g, epsilon)

    result = 2.0 / ((1.0 / safe_kro_w) + (1.0 / safe_kro_g))

    # Return 0 if either original value was zero
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)
```

**Example**:
```python
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.25,  # Tight rock
    residual_oil_saturation_water=0.35,
    residual_oil_saturation_gas=0.20,
    residual_gas_saturation=0.08,
    water_exponent=3.5,  # Steep curves for tight rock
    oil_exponent=2.5,
    gas_exponent=3.0,
    mixing_rule=bores.harmonic_mean_rule,  # Conservative for tight rocks
)
```

---

### geometric_mean_rule

**Formula**:
$$k_{ro} = \sqrt{k_{ro}^{(w)} \cdot k_{ro}^{(g)}}$$

**Description**:
Computes the geometric mean of two-phase oil relative permeabilities. Provides a conservative estimate that is less severe than harmonic mean. Good general-purpose conservative mixing rule.

**When to Use**:
- General-purpose conservative estimate
- When specific system characteristics are unknown
- Balanced between arithmetic and harmonic means
- Moderate interference assumption

**Characteristics**:
- If either input is zero, result is zero
- Always between harmonic and arithmetic means
- Symmetric treatment of both phases
- Smooth variation with saturation

**Implementation** (src/bores/relperm.py:359-378):
```python
@mixing_rule
@numba.njit(cache=True)
def geometric_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Geometric mean: kro = sqrt(kro_w * kro_g)
    More conservative than arithmetic mean.
    """
    return np.sqrt(kro_w * kro_g)
```

**Example**:
```python
# Conservative general-purpose estimate
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.geometric_mean_rule,
)
```

---

### stone_I_rule

**Formula**:
$$k_{ro} = \frac{k_{ro}^{(w)} \cdot k_{ro}^{(g)}}{k_{ro}^{(w)} + k_{ro}^{(g)} - k_{ro}^{(w)} \cdot k_{ro}^{(g)}}$$

(Returns 0 if both inputs are 0)

**Description**:
Stone's Model I (1970) - one of the earliest and most widely used three-phase relative permeability correlations. Provides moderate conservativeness and smooth transitions between two-phase limits. Well-suited for water-wet systems.

**When to Use**:
- Water-wet reservoirs (most common)
- History matching reservoir performance
- Industry-standard alternative to Eclipse rule
- When Stone II approximation is inappropriate

**Characteristics**:
- Reduces to correct two-phase limits (kro_w when Sg=0, kro_g when Sw=0)
- Moderate interference prediction
- Well-tested in field applications
- Numerically stable

**Implementation** (src/bores/relperm.py:251-268):
```python
@mixing_rule
@numba.njit(cache=True)
def stone_I_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone I rule (1970):
    kro = (kro_w * kro_g) / (kro_w + kro_g - kro_w * kro_g)
    """
    denom = np.maximum(((kro_w + kro_g) - (kro_w * kro_g)), 1e-12)
    result = (kro_w * kro_g) / denom
    return np.where((kro_w <= 0.0) & (kro_g <= 0.0), 0.0, result)
```

**Example**:
```python
# Water-wet sandstone with Stone I
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.05,
    water_exponent=2.5,
    oil_exponent=2.0,
    gas_exponent=2.0,
    wettability=bores.WettabilityType.WATER_WET,
    mixing_rule=bores.stone_I_rule,
)
```

---

### stone_II_rule

**Formula** (Approximation):
$$k_{ro} = k_{ro}^{(w)} + k_{ro}^{(g)} - 1$$

(Clamped to [0, ∞), returns 0 if either input is 0)

**Description**:
Stone's Model II (1973) - widely used in industry. **Note**: The BORES implementation uses an approximation because the mixing rule signature doesn't include water and gas relative permeabilities. The approximation assumes $k_{rw} \approx 1 - k_{ro}^{(w)}$ and $k_{rg} \approx 1 - k_{ro}^{(g)}$, which is valid for normalized unit-endpoint tables but can be inaccurate for highly non-linear Corey-type models.

**When to Use**:
- Industry-standard practice
- Normalized relative permeability tables
- Linear or near-linear kr curves
- When approximation is acceptable

**When NOT to Use**:
- Highly non-linear Corey models (exponents >> 2.0)
- Non-normalized tables
- When precision is critical

**Characteristics**:
- Can give negative values (clamped to 0)
- Approximation breaks down with high Corey exponents
- Well-known in industry
- Simple formula

**Implementation** (src/bores/relperm.py:270-334):
```python
@mixing_rule
@numba.njit(cache=True)
def stone_II_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Stone II rule (1973) - approximated form.

    Original: kro = (kro_w + krw) * (kro_g + krg) - krw - krg
    Approximation: krw ≈ 1 - kro_w, krg ≈ 1 - kro_g
    Result: kro = kro_w + kro_g - 1
    """
    krw_approx = 1.0 - kro_w
    krg_approx = 1.0 - kro_g

    result = (kro_w + krw_approx) * (kro_g + krg_approx) - krw_approx - krg_approx
    # Simplifies to: kro = kro_w + kro_g - 1

    result = np.maximum(result, 0.0)  # Clamp negative values
    return np.where((kro_w <= 0.0) | (kro_g <= 0.0), 0.0, result)
```

**Example**:
```python
# Use with caution - best for moderate exponents
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,  # Keep moderate
    oil_exponent=2.0,    # Not too high
    gas_exponent=2.0,
    mixing_rule=bores.stone_II_rule,
)
```

---

### eclipse_rule

**Formula** (Saturation-weighted):
$$k_{ro} = k_{ro}^{(w)} \cdot f_w + k_{ro}^{(g)} \cdot f_g$$

where:
$$f_w = \frac{S_o}{S_o + S_g}, \quad f_g = \frac{S_o}{S_o + S_w}$$

**Description**:
The ECLIPSE simulator default three-phase mixing rule. Provides saturation-weighted combination of two-phase values with smooth transitions. **Recommended as the default mixing rule for most applications.**

**When to Use**:
- General reservoir simulation (recommended default)
- Matching commercial simulator behavior
- Smooth phase transitions required
- Robust handling of edge cases

**Characteristics**:
- Moderate conservativeness
- Smooth saturation dependence
- Reduces correctly to two-phase limits
- Industry-proven in commercial simulators
- Robust numerical behavior

**Implementation** (src/bores/relperm.py:590-625):
```python
@mixing_rule
@numba.njit(cache=True)
def eclipse_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    ECLIPSE simulator default three-phase rule.

    kro = kro_w * f_w + kro_g * f_g
    where f_w = So/(So+Sg), f_g = So/(So+Sw)
    """
    total_mobile = oil_saturation + water_saturation + gas_saturation

    denom_w = oil_saturation + gas_saturation
    f_w = np.where(denom_w > 0.0, oil_saturation / denom_w, 0.0)

    denom_g = oil_saturation + water_saturation
    f_g = np.where(denom_g > 0.0, oil_saturation / denom_g, 0.0)

    result = (kro_w * f_w) + (kro_g * f_g)
    return np.where(total_mobile > 0.0, result, 0.0)
```

**Example** (from scenarios/setup.py):
```python
# Recommended default configuration
model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_gas=0.15,
    residual_oil_saturation_water=0.25,
    residual_gas_saturation=0.045,
    wettability=bores.WettabilityType.WATER_WET,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,  # Industry standard
)
```

---

### arithmetic_mean_rule

**Formula**:
$$k_{ro} = \frac{k_{ro}^{(w)} + k_{ro}^{(g)}}{2}$$

**Description**:
Simple arithmetic average of two-phase oil relative permeabilities. Tends to overestimate kro compared to other methods. Useful as an upper bound estimate or for sensitivity analysis.

**When to Use**:
- Upper bound estimation
- Sensitivity analysis (optimistic case)
- Quick approximate calculations
- Comparison benchmark

**Characteristics**:
- Simple to understand
- No saturation dependence
- Optimistic predictions
- Rarely used for production forecasting

**Implementation** (src/bores/relperm.py:337-356):
```python
@mixing_rule
@numba.njit(cache=True)
def arithmetic_mean_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """
    Simple arithmetic mean: kro = (kro_w + kro_g) / 2
    Tends to overestimate kro.
    """
    return (kro_w + kro_g) / 2.0
```

---

### max_rule

**Formula**:
$$k_{ro} = \max(k_{ro}^{(w)}, k_{ro}^{(g)})$$

**Description**:
The most optimistic mixing rule. Takes the maximum of the two two-phase oil relative permeabilities. Represents minimal three-phase interference where oil mobility is limited only by the best-case two-phase system.

**When to Use**:
- Upper bound in sensitivity analysis
- Optimistic production forecasts
- Theoretical maximum recovery estimates
- Rarely used in practice (too optimistic)

**Characteristics**:
- Always gives the highest kro prediction
- No three-phase interference
- Simple to understand
- Generally unrealistic for most systems

**Implementation** (src/bores/relperm.py:627-647):
```python
@mixing_rule
@numba.njit(cache=True)
def max_rule(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """Maximum rule - most optimistic: kro = max(kro_w, kro_g)"""
    return np.maximum(kro_w, kro_g)
```

---

### Other Mixing Rules

#### saturation_weighted_interpolation_rule

Interpolates based on water-to-gas saturation ratio. Similar to Eclipse rule but uses different weighting.

```python
kro = kro_w * (Sw / (Sw + Sg)) + kro_g * (Sg / (Sw + Sg))
```

#### baker_linear_rule

Baker's linear interpolation (1988). Saturation-weighted combination similar to saturation_weighted_interpolation_rule.

```python
kro = (Sw * kro_w + Sg * kro_g) / (Sw + Sg)
```

#### blunt_rule

Developed for strongly water-wet systems. Accounts for pore-level displacement.

```python
kro = kro_w * kro_g * (2 - kro_w - kro_g)
```

#### hustad_hansen_rule

Hustad-Hansen rule (1995). Conservative estimate for intermediate wettability.

```python
kro = (kro_w * kro_g) / max(kro_w, kro_g)
```

#### aziz_settari_rule

Parameterized empirical correlation. Default exponents are a=0.5, b=0.5.

```python
kro = kro_w^a * kro_g^b
```

Usage:
```python
# Create parameterized rule
custom_rule = bores.aziz_settari_rule(a=0.6, b=0.4)

model = bores.BrooksCoreyThreePhaseRelPermModel(
    # ... other parameters ...
    mixing_rule=custom_rule,
)
```

## Comparison and Selection Guide

### By Conservativeness (Most to Least)

1. **min_rule** - Most conservative, lower bound
2. **harmonic_mean_rule** - Very conservative, series flow
3. **geometric_mean_rule** - Conservative, general purpose
4. **blunt_rule** - Conservative, water-wet
5. **hustad_hansen_rule** - Conservative, intermediate wettability
6. **stone_I_rule** - Moderate, industry standard
7. **eclipse_rule** - Moderate, recommended default
8. **stone_II_rule** - Moderate, approximation
9. **saturation_weighted_interpolation_rule** - Moderate
10. **baker_linear_rule** - Moderate
11. **linear_interpolation_rule** - Moderate
12. **product_saturation_weighted_rule** - Moderate to optimistic
13. **arithmetic_mean_rule** - Optimistic
14. **max_rule** - Most optimistic, upper bound

### Selection Flowchart

**For production forecasting**:
- Start with **eclipse_rule** (recommended)
- If too optimistic, try **stone_I_rule** or **geometric_mean_rule**
- If matching specific simulator, check its default (usually eclipse_rule or stone_II_rule)

**For sensitivity analysis**:
- Lower bound: **min_rule** or **harmonic_mean_rule**
- Base case: **eclipse_rule** or **stone_I_rule**
- Upper bound: **arithmetic_mean_rule** or **max_rule**

**For specific systems**:
- Water-wet sandstone: **eclipse_rule** or **stone_I_rule**
- Oil-wet carbonate: **eclipse_rule** with oil-wet wettability
- Tight rock: **harmonic_mean_rule** or **geometric_mean_rule**
- Unknown system: **eclipse_rule** (safest default)

## Custom Mixing Rules

You can create custom mixing rules using the `@mixing_rule` decorator. See [Custom Mixing Rules](custom-rules.md) for detailed instructions.

**Basic example**:
```python
import bores
from bores.types import FloatOrArray
import numba
import numpy as np

@bores.mixing_rule(name="custom_weighted")
@numba.njit(cache=True)
def custom_weighted(
    kro_w: FloatOrArray,
    kro_g: FloatOrArray,
    water_saturation: FloatOrArray,
    oil_saturation: FloatOrArray,
    gas_saturation: FloatOrArray,
) -> FloatOrArray:
    """Custom weighted average with 70/30 split."""
    return 0.7 * kro_w + 0.3 * kro_g

# Use in model
model = bores.BrooksCoreyThreePhaseRelPermModel(
    # ... parameters ...
    mixing_rule=custom_weighted,
)
```

## See Also

- [Brooks-Corey Model](brooks-corey.md) - Using mixing rules with analytical models
- [Lookup Tables](tables.md) - Using mixing rules with tabular data
- [Custom Mixing Rules](custom-rules.md) - Creating and registering custom rules
- [Relative Permeability Index](index.md) - Module overview
