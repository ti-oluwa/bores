# Constants and Configuration

Physical constants, unit conversions, and configuration settings in BORES.

---

## Physical Constants

BORES provides a `constants` module (accessible as `bores.c` or `bores.constants`) with physical constants and conversion factors.

### Accessing Constants

```python
import bores

# Access constants module
c = bores.c  # Short alias
# or
c = bores.constants  # Full name

# Use constants
gas_constant = c.GAS_CONSTANT  # psi·ft³/(lb-mol·°R)
gravity = c.STANDARD_GRAVITY  # ft/s²
```

### Common Constants

**Universal Constants:**

```python
c.GAS_CONSTANT  # Universal gas constant: 10.7316 psi·ft³/(lb-mol·°R)
c.STANDARD_GRAVITY  # Gravitational acceleration: 32.174 ft/s²
```

**Standard Conditions:**

```python
c.STANDARD_TEMPERATURE  # 60°F (Standard temperature)
c.STANDARD_PRESSURE  # 14.696 psi (1 atm)
```

**Fluid Properties:**

```python
c.WATER_DENSITY_SC  # Water density at SC: 62.4 lbm/ft³
c.AIR_MOLECULAR_WEIGHT  # Air molecular weight: 28.97 lbm/lb-mol
```

**Time Conversions:**

```python
c.SECONDS_PER_DAY  # 86400 seconds
c.DAYS_PER_YEAR  # 365.25 days
c.HOURS_PER_DAY  # 24 hours
```

**Numerical Tolerances:**

```python
c.MINIMUM_SATURATION  # Minimum saturation value (prevents division by zero)
c.MINIMUM_TRANSMISSIBILITY_FACTOR  # Minimum transmissibility multiplier
c.EPSILON  # Small number for numerical comparisons
```

---

## Configuration System

BORES uses a `Config` class to configure simulation settings.

### Creating Configuration

```python
import bores

# Create timer
timer = bores.Timer(
    initial_step_size=bores.Time(hours=2),
    max_step_size=bores.Time(days=1),
    min_step_size=bores.Time(minutes=5),
    simulation_time=bores.Time(days=30),
)

# Create configuration
config = bores.Config(
    timer=timer,
    wells=wells,
    rock_fluid_tables=rock_fluid_tables,
    scheme="impes",  # or "explicit"
    pressure_solver="bicgstab",
    pressure_preconditioner="ilu",
    max_iterations=150,
)
```

### Configuration Parameters

**Numerical Scheme:**

```python
config = bores.Config(
    scheme="impes",  # Options: "impes" (default), "explicit"
    # ...
)
```

- `"impes"`: Implicit Pressure, Explicit Saturation (recommended)
- `"explicit"`: Fully explicit scheme (rarely used)

**Pressure Solver:**

```python
config = bores.Config(
    pressure_solver="bicgstab",  # Solver type
    pressure_preconditioner="ilu",  # Preconditioner
    max_iterations=150,  # Max solver iterations
    pressure_tolerance=1e-6,  # Convergence tolerance
    # ...
)
```

**Available solvers:**

- `"bicgstab"`: BiConjugate Gradient Stabilized (default, good for most cases)
- `"gmres"`: Generalized Minimal Residual (good for difficult problems)
- `"cg"`: Conjugate Gradient (symmetric problems only)
- `"lgmres"`: LGMRES with restarts
- `"direct"`: Direct solver (small grids only)

**Available preconditioners:**

- `"ilu"`: Incomplete LU (default, good balance)
- `"amg"`: Algebraic Multigrid (excellent for large grids)
- `"cpr"`: Constrained Pressure Residual (best for multiphase flow)
- `"diagonal"`: Diagonal (cheap, for well-conditioned systems)
- `"block_jacobi"`: Block Jacobi (good for coupled systems)
- `"polynomial"`: Polynomial (low-cost alternative to diagonal)

**Timestep Control:**

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=2),
    max_step_size=bores.Time(days=1),
    min_step_size=bores.Time(minutes=5),
    simulation_time=bores.Time(days=365),
    max_cfl_number=0.9,  # CFL limit for stability
    ramp_up_factor=1.15,  # Increase factor on success
    backoff_factor=0.5,  # Decrease factor on failure
    max_rejects=10,  # Max consecutive rejections
)
```

**Output Control:**

```python
config = bores.Config(
    output_frequency=1,  # Save every N timesteps
    log_interval=10,  # Log progress every N steps
    # ...
)
```

### Modifying Configuration

Configuration is immutable (frozen attrs class). Use `copy()` to create modified versions:

```python
# Create base config
config = bores.Config(
    timer=timer,
    wells=wells,
    # ...
)

# Create modified copy
new_config = config.copy(
    max_iterations=200,  # Increase max iterations
    pressure_tolerance=1e-7,  # Tighter tolerance
)

# Or use with_updates (alternative method)
new_config = config.with_updates(
    max_iterations=200,
    pressure_tolerance=1e-7,
)
```

---

## Precision Control

BORES supports both 32-bit and 64-bit floating-point precision.

### Setting Precision

```python
import bores

# Use 32-bit precision (default, faster, less memory)
bores.use_32bit_precision()

# Use 64-bit precision (slower, more memory, higher accuracy)
bores.use_64bit_precision()
```

**32-bit (float32):**

- Memory: ~4 bytes per value
- Precision: ~7 decimal digits
- Speed: Faster on most hardware
- Use for: Most reservoir simulations

**64-bit (float64):**

- Memory: ~8 bytes per value
- Precision: ~15 decimal digits
- Speed: Slower (2x memory, ~1.5x slower)
- Use for: High-precision requirements, numerical sensitivity studies

### Checking Current Precision

```python
from bores._precision import get_dtype, get_floating_point_info

# Get current dtype
dtype = get_dtype()  # numpy.float32 or numpy.float64

# Get floating point info
info = get_floating_point_info()
print(f"Epsilon: {info.eps}")  # Machine epsilon
print(f"Max value: {info.max}")  # Maximum representable value
```

---

## Context Management

BORES uses context variables for thread-safe global state.

### Precision Context

```python
from bores._precision import precision_context, Precision

# Temporarily use 64-bit precision
with precision_context(Precision.FLOAT64):
    # Operations here use float64
    model = bores.reservoir_model(...)
# Back to default precision
```

---

## Unit Conversions

While BORES primarily uses field units (psi, ft, STB, °F), you may need conversions:

**Temperature:**

```python
# °F to °R (Rankine)
temp_R = temp_F + 459.67

# °F to °C
temp_C = (temp_F - 32) * 5/9
```

**Pressure:**

```python
# psi to Pa
pressure_Pa = pressure_psi * 6894.76

# psi to bar
pressure_bar = pressure_psi * 0.0689476
```

**Volume:**

```python
# ft³ to m³
volume_m3 = volume_ft3 * 0.0283168

# bbl to m³
volume_m3 = volume_bbl * 0.158987
```

---

## Best Practices

1. **Use constants module** for physical constants (don't hardcode values)
2. **32-bit precision** is sufficient for most simulations
3. **Use 64-bit** only when numerical issues arise
4. **Immutable config** - always create new config with `copy()` or `with_updates()`
5. **CFL number** - keep < 1.0 for stability (0.9 is safe default)
6. **Solver selection** - start with `bicgstab` + `ilu`, upgrade to `cpr` if needed

---

## See Also

- [Running Simulations](../guides/running-simulations.md) - Configuring simulations
- [Performance Optimization](../advanced/performance-optimization.md) - Solver tuning
- [Best Practices: Solver Selection](../best-practices/solver-selection.md) - Choosing solvers
