# Capillary Pressure Lookup Tables

Documentation for using tabular (lookup table) capillary pressure data in BORES instead of analytical models.

## Overview

While analytical models (Brooks-Corey, van Genuchten, Leverett J) provide convenient parameterized functions, laboratory measurements and specialized software often produce capillary pressure data as discrete tables. BORES supports direct use of such tabular data through interpolation.

### When to Use Tables

Use tabular capillary pressure instead of analytical models when:

- You have laboratory capillary pressure measurements (mercury injection, centrifuge, etc.)
- You have externally-computed Pc curves (from specialized software)
- Your system doesn't match standard analytical model assumptions
- You need to match specific measured data exactly
- You're importing data from another simulator

### When to Use Analytical Models

Use analytical models instead of tables when:

- You don't have measured data
- You want parameterized sensitivity analysis
- You need smooth derivatives for numerical stability
- You're performing preliminary screening studies
- You need to scale across rock types (Leverett J)

## Table Structure

BORES uses a hierarchical table structure:

1. **TwoPhaseCapillaryPressureTable** - Single two-phase system (oil-water OR gas-oil)
2. **ThreePhaseCapillaryPressureTable** - Three-phase system (combines two TwoPhaseCapillaryPressureTable)

## Two-Phase Tables

### Class Definition

From src/bores/capillary_pressures.py:

```python
@attrs.frozen
class TwoPhaseCapillaryPressureTable:
    """
    Two-phase capillary pressure lookup table.

    Interpolates capillary pressure for two fluid phases based on
    saturation values using fast linear interpolation (np.interp).

    Supports both scalar and array inputs up to 3D.
    """

    wetting_phase: FluidPhase
    """The wetting fluid phase (e.g., 'water' or 'oil')."""

    non_wetting_phase: FluidPhase
    """The non-wetting fluid phase (e.g., 'oil' or 'gas')."""

    wetting_phase_saturation: np.ndarray
    """Saturation values for wetting phase (must be monotonically increasing)."""

    capillary_pressure: np.ndarray
    """Capillary pressure values (Pc = P_non-wetting - P_wetting) in psi."""
```

### Creating Two-Phase Tables

**Oil-Water System** (water is wetting phase):

```python
import bores
import numpy as np

# Laboratory measurements: Sw vs Pcow
sw_data = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
pcow_data = np.array([15.0, 8.0, 4.5, 2.5, 1.2, 0.5, 0.1])  # psi

oil_water_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_data,
    capillary_pressure=pcow_data,
)

# Query at specific saturation
pcow = oil_water_table(wetting_phase_saturation=0.40)
print(f"At Sw=0.40: Pcow = {pcow:.2f} psi")
```

**Gas-Oil System** (oil is wetting phase):

```python
# Laboratory measurements: So vs Pcgo
so_data = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
pcgo_data = np.array([12.0, 7.0, 4.0, 2.2, 1.0, 0.4, 0.0])  # psi

gas_oil_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_data,
    capillary_pressure=pcgo_data,
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
saturations = np.array([0.10, 0.20, 0.50, 0.70, 0.85])
pcow_values = oil_water_table(wetting_phase_saturation=saturations)

for sw, pcow in zip(saturations, pcow_values):
    print(f"Sw={sw:.2f}: Pcow={pcow:.2f} psi")
```

### Array Inputs

Tables support both scalar and multi-dimensional array inputs:

```python
# 3D grid of saturations
sw_grid = np.random.uniform(0.15, 0.75, size=(10, 10, 5))

# Compute Pcow for entire grid (vectorized)
pcow_grid = oil_water_table(wetting_phase_saturation=sw_grid)

print(f"Input shape: {sw_grid.shape}")    # (10, 10, 5)
print(f"Output shape: {pcow_grid.shape}")  # (10, 10, 5)
```

## Three-Phase Tables

### Class Definition

From src/bores/capillary_pressures.py:

```python
@attrs.frozen
class ThreePhaseCapillaryPressureTable:
    """
    Three-phase capillary pressure lookup table.

    Uses two two-phase tables (oil-water and gas-oil) to compute
    capillary pressures in a three-phase system.

    Pcow = Po - Pw (oil-water capillary pressure)
    Pcgo = Pg - Po (gas-oil capillary pressure)
    """

    oil_water_table: TwoPhaseCapillaryPressureTable
    """Capillary pressure table for oil-water system."""

    gas_oil_table: TwoPhaseCapillaryPressureTable
    """Capillary pressure table for gas-oil system."""
```

### Creating Three-Phase Tables

Combine two two-phase tables:

```python
import bores
import numpy as np

# Create oil-water table (from measurements)
sw_data = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
pcow_data = np.array([15.0, 8.0, 4.5, 2.5, 1.2, 0.5, 0.1])

oil_water_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_data,
    capillary_pressure=pcow_data,
)

# Create gas-oil table (from measurements)
so_data = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
pcgo_data = np.array([12.0, 7.0, 4.0, 2.2, 1.0, 0.4, 0.0])

gas_oil_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_data,
    capillary_pressure=pcgo_data,
)

# Combine into three-phase table
three_phase_table = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_table,
    gas_oil_table=gas_oil_table,
)

# Query three-phase capillary pressures
pc = three_phase_table(
    water_saturation=0.30,
    oil_saturation=0.50,
    gas_saturation=0.20,
)

print(f"Pcow = {pc['oil_water']:.2f} psi")
print(f"Pcgo = {pc['gas_oil']:.2f} psi")
```

## Practical Workflow

### 1. Import Data from Laboratory

```python
import bores
import numpy as np
import pandas as pd

# Load laboratory data (example: CSV file from mercury injection)
ow_data = pd.read_csv("mercury_injection_oil_water.csv")
go_data = pd.read_csv("mercury_injection_gas_oil.csv")

# Extract columns
sw_lab = ow_data["Sw"].values
pcow_lab = ow_data["Pc_psi"].values

so_lab = go_data["So"].values
pcgo_lab = go_data["Pc_psi"].values

# Create tables
oil_water_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_lab,
    capillary_pressure=pcow_lab,
)

gas_oil_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.OIL,
    non_wetting_phase=bores.FluidPhase.GAS,
    wetting_phase_saturation=so_lab,
    capillary_pressure=pcgo_lab,
)

three_phase_table = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=oil_water_table,
    gas_oil_table=gas_oil_table,
)
```

### 2. Use in Simulation Configuration

```python
import bores

# Create rock-fluid tables using tabular capillary pressure
rock_fluid_tables = bores.RockFluidTables(
    relative_permeability_table=relperm_table,
    capillary_pressure_table=three_phase_table,  # Tabular data
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
analytical_model = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=1.5,
    gas_oil_pore_size_distribution_index=2.0,
)

# Test saturations
Sw, So, Sg = 0.35, 0.50, 0.15

# Compare results
pc_table = three_phase_table(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)
pc_analytical = analytical_model(water_saturation=Sw, oil_saturation=So, gas_saturation=Sg)

print("Tabular vs Analytical:")
print(f"Pcow: {pc_table['oil_water']:.2f} vs {pc_analytical['oil_water']:.2f} psi")
print(f"Pcgo: {pc_table['gas_oil']:.2f} vs {pc_analytical['gas_oil']:.2f} psi")
```

## Data Requirements

### Minimum Data Points

For stable interpolation, provide at least:
- **7-10 points** for simple curves
- **15-20 points** for non-linear curves
- **30+ points** for high-fidelity matching

### Saturation Range

Tables should span the full expected saturation range:

**Oil-Water Table**:
- Start: $S_w = S_{wc}$ (irreducible water)
- End: $S_w = 1 - S_{orw}$ (residual oil to water)

**Gas-Oil Table**:
- Start: $S_o = S_{org}$ (residual oil to gas)
- End: $S_o = 1 - S_{wc} - S_{gr}$ (maximum oil saturation)

### Capillary Pressure Behavior

Capillary pressure tables should exhibit physically consistent behavior:

```python
# Oil-water table validation
assert np.all(np.diff(pcow_data) <= 0), "Pcow must decrease with increasing Sw"
assert pcow_data[0] > pcow_data[-1], "Pcow highest at low Sw"

# Gas-oil table validation
assert np.all(np.diff(pcgo_data) <= 0), "Pcgo must decrease with increasing So"
assert pcgo_data[0] > pcgo_data[-1], "Pcgo highest at low So"
```

## Validation

### Check Monotonicity

Capillary pressure must be monotonically decreasing with increasing wetting phase saturation:

```python
# Check saturation arrays are sorted
assert np.all(np.diff(sw_data) > 0), "Sw must be monotonically increasing"
assert np.all(np.diff(so_data) > 0), "So must be monotonically increasing"

# Check Pc arrays are decreasing
assert np.all(np.diff(pcow_data) <= 0), "Pcow must be monotonically decreasing"
assert np.all(np.diff(pcgo_data) <= 0), "Pcgo must be monotonically decreasing"
```

### Check Physical Bounds

```python
# Check saturation bounds
assert np.all((sw_data >= 0) & (sw_data <= 1)), "Sw must be in [0, 1]"
assert np.all((so_data >= 0) & (so_data <= 1)), "So must be in [0, 1]"

# Check Pc non-negativity (for water-wet systems)
assert np.all(pcow_data >= 0), "Pcow must be non-negative for water-wet"
assert np.all(pcgo_data >= 0), "Pcgo must be non-negative"
```

## Advanced Usage

### Spatially Varying Tables

For heterogeneous reservoirs with different rock types:

```python
import bores

# Define multiple tables for different rock types
rock_type_1_table = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=ow_table_type1,
    gas_oil_table=go_table_type1,
)

rock_type_2_table = bores.ThreePhaseCapillaryPressureTable(
    oil_water_table=ow_table_type2,
    gas_oil_table=go_table_type2,
)

# In simulation, select table based on grid cell rock type
# (implementation depends on your workflow)
```

### Generating Tables from Analytical Models

You can create tables by sampling analytical models:

```python
import bores
import numpy as np

# Create analytical model
model = bores.BrooksCoreyCapillaryPressureModel(
    irreducible_water_saturation=0.15,
    residual_oil_saturation_water=0.25,
    residual_oil_saturation_gas=0.15,
    residual_gas_saturation=0.045,
    oil_water_entry_pressure_water_wet=2.0,
    oil_water_pore_size_distribution_index_water_wet=2.0,
    gas_oil_entry_pressure=1.5,
    gas_oil_pore_size_distribution_index=2.0,
)

# Generate oil-water table from model
sw_points = np.linspace(0.15, 0.75, 20)
pcow_values = []

for sw in sw_points:
    pc = model(water_saturation=sw, oil_saturation=1.0-sw, gas_saturation=0.0)
    pcow_values.append(pc['oil_water'])

# Create table from generated data
oil_water_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_points,
    capillary_pressure=np.array(pcow_values),
)
```

### Converting Laboratory Data

Mercury injection capillary pressure (MICP) data requires conversion:

```python
import numpy as np

# MICP data (mercury-air system)
sw_micp = np.array([...])  # From MICP test
pc_hg_air = np.array([...])  # psi, mercury-air

# Convert to reservoir conditions (oil-water)
# Leverett J-function scaling
sigma_hg_air = 480.0  # dyne/cm (mercury-air IFT)
sigma_ow = 30.0  # dyne/cm (oil-water IFT)
theta_hg = 140.0  # degrees (mercury contact angle)
theta_ow = 0.0  # degrees (oil-water contact angle, water-wet)

# Conversion factor
conversion = (sigma_ow * np.cos(np.deg2rad(theta_ow))) / \
             (sigma_hg_air * np.cos(np.deg2rad(theta_hg)))

pc_ow = pc_hg_air * conversion

# Create table with converted data
oil_water_table = bores.TwoPhaseCapillaryPressureTable(
    wetting_phase=bores.FluidPhase.WATER,
    non_wetting_phase=bores.FluidPhase.OIL,
    wetting_phase_saturation=sw_micp,
    capillary_pressure=pc_ow,
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
pcow_grid = oil_water_table(wetting_phase_saturation=sw_grid)
elapsed = time.time() - start

print(f"Interpolated {grid_size[0]*grid_size[1]*grid_size[2]} values in {elapsed:.3f} seconds")
# Typical: ~0.01-0.05 seconds for 500k cells
```

### Memory Usage

Tables store saturation and Pc arrays. Memory usage is negligible:
- ~1 KB per two-phase table (20 points × 2 arrays × 8 bytes)
- Entire three-phase table: ~2 KB

## See Also

- [Brooks-Corey Model](brooks-corey.md) - Analytical power-law model
- [van Genuchten Model](van-genuchten.md) - Smooth exponential model
- [Leverett J-Function](leverett-j.md) - Dimensionless scaling correlation
- [Capillary Pressure Index](index.md) - Module overview
- [Relative Permeability Tables](../relative-permeability/tables.md) - Companion relative permeability tables
