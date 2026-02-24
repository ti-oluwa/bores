# Relative Permeability Lookup Tables

Documentation for using tabular (lookup table) relative permeability data in BORES instead of analytical models.

## Overview

While analytical models like Brooks-Corey provide convenient parameterized functions, laboratory measurements and external simulators often produce relative permeability data as discrete tables of saturation vs. relative permeability values. BORES supports direct use of such tabular data through interpolation-based relative permeability tables.

### When to Use Tables

Use tabular relative permeability instead of analytical models when:

- You have laboratory core flood measurements
- You have externally-computed kr curves (from specialized software)
- Your system doesn't match standard analytical model assumptions
- You need to match specific measured data exactly
- You're importing data from another simulator

### When to Use Analytical Models

Use analytical models (Brooks-Corey) instead of tables when:

- You don't have measured data
- You want parameterized sensitivity analysis (varying exponents, residuals)
- You need smooth derivatives for numerical stability
- You're performing preliminary screening studies

## Table Structure

BORES uses a hierarchical table structure:

1. **TwoPhaseRelPermTable** - Single two-phase system (oil-water OR gas-oil)
2. **ThreePhaseRelPermTable** - Three-phase system (combines two TwoPhaseRelPermTable + mixing rule)

## Two-Phase Tables

### Class Definition

From src/bores/relperm.py:

```python
@attrs.frozen
class TwoPhaseRelPermTable:
    """
    Two-phase relative permeability lookup table.

    Interpolates relative permeabilities for two fluid phases based on
    saturation values using fast linear interpolation (np.interp).

    Supports both scalar and array inputs up to 3D.
    """

    wetting_phase: FluidPhase
    """The wetting fluid phase (e.g., 'water' or 'oil')."""

    non_wetting_phase: FluidPhase
    """The non-wetting fluid phase (e.g., 'oil' or 'gas')."""

    wetting_phase_saturation: np.ndarray
    """Saturation values for wetting phase (must be monotonically increasing)."""

    wetting_phase_relative_permeability: np.ndarray
    """Relative permeability values for wetting phase."""

    non_wetting_phase_relative_permeability: np.ndarray
    """Relative permeability values for non-wetting phase."""
```

### Creating Two-Phase Tables

**Oil-Water System** (water is wetting phase):

```python
import bores
import numpy as np

# Laboratory measurements: Sw vs krw and kro
sw_data = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
krw_data = np.array([0.00, 0.01, 0.04, 0.10, 0.20, 0.35, 0.55])
kro_data = np.array([0.90, 0.70, 0.50, 0.30, 0.15, 0.05, 0.00])

oil_water_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_data,
    wetting_phase_relative_permeability=krw_data,
    non_wetting_phase_relative_permeability=kro_data,
)

# Query at specific saturation
krw = oil_water_table.get_wetting_phase_relative_permeability(0.50)
kro = oil_water_table.get_non_wetting_phase_relative_permeability(0.50)
print(f"At Sw=0.50: krw={krw:.4f}, kro={kro:.4f}")
```

**Gas-Oil System** (oil is wetting phase):

```python
# Laboratory measurements: So vs krg and kro
so_data = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
kro_gas_data = np.array([0.00, 0.02, 0.08, 0.18, 0.32, 0.50, 0.70, 0.90])
krg_data = np.array([0.80, 0.60, 0.42, 0.28, 0.16, 0.08, 0.02, 0.00])

gas_oil_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_data,
    wetting_phase_relative_permeability=kro_gas_data,
    non_wetting_phase_relative_permeability=krg_data,
)
```

### Interpolation Behavior

The table uses linear interpolation (`np.interp`) between data points:

- **Between points**: Linear interpolation
- **Below minimum saturation**: Constant extrapolation (uses first value)
- **Above maximum saturation**: Constant extrapolation (uses last value)

**Example**:

```python
# Query at various saturations
saturations = np.array([0.10, 0.20, 0.40, 0.60, 0.80])
krw_values = oil_water_table.get_wetting_phase_relative_permeability(saturations)

for sw, krw in zip(saturations, krw_values):
    print(f"Sw={sw:.2f}: krw={krw:.4f}")
```

### Array Inputs

Tables support both scalar and multi-dimensional array inputs:

```python
# 3D grid of saturations
sw_grid = np.random.uniform(0.15, 0.75, size=(10, 10, 5))

# Compute krw for entire grid (vectorized)
krw_grid = oil_water_table.get_wetting_phase_relative_permeability(sw_grid)

print(f"Input shape: {sw_grid.shape}")    # (10, 10, 5)
print(f"Output shape: {krw_grid.shape}")  # (10, 10, 5)
```

## Three-Phase Tables

### Class Definition

From src/bores/relperm.py:

```python
@attrs.frozen
class ThreePhaseRelPermTable:
    """
    Three-phase relative permeability lookup table.

    Uses two two-phase tables (oil-water and gas-oil) and applies a mixing
    rule to compute oil relative permeability in three-phase systems.
    """

    oil_water_table: TwoPhaseRelPermTable
    """Oil-water two-phase table (water=wetting OR oil=wetting)."""

    gas_oil_table: TwoPhaseRelPermTable
    """Gas-oil two-phase table (oil=wetting)."""

    mixing_rule: Optional[MixingRule] = None
    """Mixing rule for three-phase oil kr (default: min_rule)."""
```

### Creating Three-Phase Tables

Combine two two-phase tables with a mixing rule:

```python
import bores

# Create oil-water table (from measurements)
oil_water_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_data,
    wetting_phase_relative_permeability=krw_data,
    non_wetting_phase_relative_permeability=kro_water_data,
)

# Create gas-oil table (from measurements)
gas_oil_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_data,
    wetting_phase_relative_permeability=kro_gas_data,
    non_wetting_phase_relative_permeability=krg_data,
)

# Combine into three-phase table
three_phase_table = bores.ThreePhaseRelPermTable(
    oil_water_table=oil_water_table,
    gas_oil_table=gas_oil_table,
    mixing_rule=bores.eclipse_rule,  # Recommended
)

# Query three-phase relative permeabilities
kr = three_phase_table(
    water_saturation=0.30,
    oil_saturation=0.50,
    gas_saturation=0.20,
)

print(f"krw = {kr['water']:.4f}")
print(f"kro = {kr['oil']:.4f}")
print(f"krg = {kr['gas']:.4f}")
```

### Mixing Rule Selection

The mixing rule determines how to combine two-phase oil kr values when all three phases are present. See [Mixing Rules Reference](mixing-rules.md) for detailed comparison.

**Recommended**:
```python
mixing_rule=bores.eclipse_rule  # Industry standard
```

**Conservative**:
```python
mixing_rule=bores.geometric_mean_rule  # Conservative estimate
```

**If None**:
```python
mixing_rule=None  # Defaults to min_rule (very conservative)
```

## Practical Workflow

### 1. Import Data from Laboratory

```python
import bores
import numpy as np
import pandas as pd

# Load laboratory data (example: CSV file)
ow_data = pd.read_csv("oil_water_flood_data.csv")
go_data = pd.read_csv("gas_oil_flood_data.csv")

# Extract columns
sw_lab = ow_data["Sw"].values
krw_lab = ow_data["krw"].values
kro_water_lab = ow_data["kro"].values

so_lab = go_data["So"].values
kro_gas_lab = go_data["kro"].values
krg_lab = go_data["krg"].values

# Create tables
oil_water_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_lab,
    wetting_phase_relative_permeability=krw_lab,
    non_wetting_phase_relative_permeability=kro_water_lab,
)

gas_oil_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_lab,
    wetting_phase_relative_permeability=kro_gas_lab,
    non_wetting_phase_relative_permeability=krg_lab,
)

three_phase_table = bores.ThreePhaseRelPermTable(
    oil_water_table=oil_water_table,
    gas_oil_table=gas_oil_table,
    mixing_rule=bores.eclipse_rule,
)
```

### 2. Use in Simulation Configuration

```python
import bores

# Create rock-fluid tables using tabular relative permeability
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=three_phase_table,  # Tabular data
    capillary_pressure_table=capillary_pressure_table,
)

# Use in simulation config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    # ... other config parameters ...
)
```

### 3. Compare with Analytical Model

```python
import bores
import numpy as np

# Analytical model
analytical_model = bores.BrooksCoreyThreePhaseRelPermModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    water_exponent=2.0,
    oil_exponent=2.0,
    gas_exponent=2.0,
    mixing_rule=bores.eclipse_rule,
)

# Test saturations
Sw = 0.35
So = 0.50
Sg = 0.15

# Compare results
kr_table = three_phase_table(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)
kr_analytical = analytical_model(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)

print("Tabular vs Analytical:")
print(f"krw: {kr_table['water']:.4f} vs {kr_analytical['water']:.4f}")
print(f"kro: {kr_table['oil']:.4f} vs {kr_analytical['oil']:.4f}")
print(f"krg: {kr_table['gas']:.4f} vs {kr_analytical['gas']:.4f}")
```

## Data Requirements

### Minimum Data Points

For stable interpolation, provide at least:
- **5-7 points** for simple curves
- **10-15 points** for non-linear curves
- **20+ points** for high-fidelity matching

### Saturation Range

Tables should span the full expected saturation range:

**Oil-Water Table**:
- Start: $S_w = S_{wc}$ (irreducible water)
- End: $S_w = 1 - S_{orw}$ (residual oil to water)

**Gas-Oil Table**:
- Start: $S_o = S_{org}$ (residual oil to gas)
- End: $S_o = 1 - S_{wc} - S_{gr}$ (maximum oil saturation)

### Endpoint Consistency

Ensure endpoint values are physically consistent:

```python
# Oil-water table endpoints
assert krw_data[0] == 0.0    # krw = 0 at Swc
assert kro_data[-1] == 0.0   # kro = 0 at Sorw

# Gas-oil table endpoints
assert kro_gas_data[0] == 0.0  # kro = 0 at Sorg
assert krg_data[-1] == 0.0     # krg = 0 at Sgr
```

## Validation

### Check Monotonicity

Relative permeability curves should be monotonic:

```python
# Water kr should increase with Sw
assert np.all(np.diff(krw_data) >= 0), "krw must be monotonically increasing"

# Oil kr should decrease with Sw (in oil-water system)
assert np.all(np.diff(kro_data) <= 0), "kro must be monotonically decreasing"
```

### Check Endpoints

```python
# Check that saturation arrays are sorted
assert np.all(np.diff(sw_data) > 0), "Sw must be monotonically increasing"
assert np.all(np.diff(so_data) > 0), "So must be monotonically increasing"

# Check kr bounds
assert np.all((krw_data >= 0) & (krw_data <= 1)), "krw must be in [0, 1]"
assert np.all((kro_data >= 0) & (kro_data <= 1)), "kro must be in [0, 1]"
assert np.all((krg_data >= 0) & (krg_data <= 1)), "krg must be in [0, 1]"
```

## Advanced Usage

### Spatially Varying Tables

For heterogeneous reservoirs with different rock types:

```python
import bores

# Define multiple tables for different rock types
rock_type_1_table = bores.ThreePhaseRelPermTable(
    oil_water_table=ow_table_type1,
    gas_oil_table=go_table_type1,
    mixing_rule=bores.eclipse_rule,
)

rock_type_2_table = bores.ThreePhaseRelPermTable(
    oil_water_table=ow_table_type2,
    gas_oil_table=go_table_type2,
    mixing_rule=bores.eclipse_rule,
)

# In simulation, select table based on grid cell properties
# (implementation depends on your workflow)
```

### Generating Tables from Analytical Models

You can create tables by sampling analytical models:

```python
import bores
import numpy as np

# Create analytical model
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

# Generate oil-water table from model
sw_points = np.linspace(0.15, 0.75, 20)
krw_values = []
kro_values = []

for sw in sw_points:
    kr = model(water_saturation=sw, oil_saturation=1.0-sw, gas_saturation=0.0)
    krw_values.append(kr['water'])
    kro_values.append(kr['oil'])

# Create table from generated data
oil_water_table = bores.TwoPhaseRelPermTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_points,
    wetting_phase_relative_permeability=np.array(krw_values),
    non_wetting_phase_relative_permeability=np.array(kro_values),
)
```

## Performance Considerations

### Interpolation Speed

Linear interpolation using `np.interp` is very fast (< 1 μs per lookup). For grid-based simulations with thousands of cells, vectorized operations maintain excellent performance.

**Benchmark**:
```python
import time
import numpy as np

# Large grid
grid_size = (100, 100, 50)  # 500,000 cells
sw_grid = np.random.uniform(0.15, 0.75, size=grid_size)

# Measure interpolation time
start = time.time()
krw_grid = oil_water_table.get_wetting_phase_relative_permeability(sw_grid)
elapsed = time.time() - start

print(f"Interpolated {grid_size[0]*grid_size[1]*grid_size[2]} values in {elapsed:.3f} seconds")
# Typical: ~0.01-0.05 seconds for 500k cells
```

### Memory Usage

Tables store saturation and kr arrays. Memory usage is negligible compared to grid data:
- ~1 KB per two-phase table (20 points × 3 arrays × 8 bytes)
- Entire three-phase table: ~2 KB

## See Also

- [Brooks-Corey Model](brooks-corey.md) - Analytical relative permeability model
- [Mixing Rules Reference](mixing-rules.md) - Choosing the right mixing rule for tables
- [Relative Permeability Index](index.md) - Module overview
