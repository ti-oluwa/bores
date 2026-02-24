# PVT Tables

Comprehensive guide to using PVT (Pressure-Volume-Temperature) tables instead of correlations.

---

## Overview

BORES supports two methods for computing fluid properties:

1. **Correlations** (default): Empirical correlations (Standing, Vasquez-Beggs, etc.)
2. **PVT Tables**: Tabulated data from lab measurements or equation-of-state (EOS) models

PVT tables provide:

- **Higher accuracy**: Based on lab PVT experiments or tuned EOS
- **Complex fluids**: Handle non-typical fluids (CO2, H2S, heavy oil)
- **Fast lookup**: O(1) interpolation vs O(n) correlation calculations
- **Memory vs compute trade-off**: Tables use more memory but faster

Defined in `bores.tables.pvt`.

---

## PVT Table Structure

### PVTTableData

`PVTTableData` holds raw tabulated data:

**Base grids** (independent variables):
- `pressures` - 1D array of pressures (psi)
- `temperatures` - 1D array of temperatures (°F)
- `salinities` - Optional 1D array of salinities (ppm NaCl) for water
- `solution_gas_oil_ratios` - Optional 1D array of Rs values (SCF/STB)

**Oil properties** (2D tables: n_pressures × n_temperatures):
- `oil_viscosity_table` - μo(P,T) in cP
- `oil_compressibility_table` - co(P,T) in psi⁻¹
- `oil_specific_gravity_table` - γo(P,T) (dimensionless)
- `oil_api_gravity_table` - °API(P,T)
- `oil_density_table` - ρo(P,T) in lbm/ft³
- `oil_formation_volume_factor_table` - Bo(P,T) in bbl/STB
- `solution_gas_to_oil_ratio_table` - Rs(P,T) in SCF/STB
- `bubble_point_pressures` - Pb(T) or Pb(Rs,T)

**Gas properties** (2D tables: n_pressures × n_temperatures):
- `gas_viscosity_table` - μg(P,T) in cP
- `gas_compressibility_table` - cg(P,T) in psi⁻¹
- `gas_gravity_table` - γg(P,T) (dimensionless, air=1)
- `gas_molecular_weight_table` - Mg(P,T) in lbm/lb-mol
- `gas_density_table` - ρg(P,T) in lbm/ft³
- `gas_formation_volume_factor_table` - Bg(P,T) in bbl/SCF
- `gas_compressibility_factor_table` - z(P,T) (dimensionless)

**Water properties** (3D tables: n_pressures × n_temperatures × n_salinities):
- `water_viscosity_table` - μw(P,T,S) in cP
- `water_compressibility_table` - cw(P,T,S) in psi⁻¹
- `water_density_table` - ρw(P,T,S) in lbm/ft³
- `water_formation_volume_factor_table` - Bw(P,T,S) in bbl/STB
- `water_bubble_point_pressure_table` - Pb,w(P,T,S) in psi
- `gas_solubility_in_water_table` - Rsw(P,T,S) in SCF/STB

---

## Creating PVT Tables

### From Lab Data

```python
import numpy as np
import bores

# Define pressure and temperature grids
pressures = np.linspace(1000, 5000, 50)  # 50 points from 1000-5000 psi
temperatures = np.linspace(100, 250, 20)  # 20 points from 100-250 °F

# Create 2D tables (P x T) from lab measurements
# Example: Oil viscosity
P_grid, T_grid = np.meshgrid(pressures, temperatures, indexing='ij')
oil_visc = compute_oil_viscosity_from_lab(P_grid, T_grid)  # Your function

# Create PVT table data
pvt_data = bores.PVTTableData(
    pressures=pressures,
    temperatures=temperatures,
    oil_viscosity_table=oil_visc,
    # ... add other properties
)
```

### From Correlations

Build tables from correlations for specific fluid:

```python
import bores

# Build PVT table data from correlations for specific reservoir conditions
pvt_data = bores.build_pvt_table_data(
    pressure_range=(1000.0, 5000.0),
    temperature_range=(100.0, 250.0),
    n_pressures=50,
    n_temperatures=20,
    oil_specific_gravity=0.845,
    gas_specific_gravity=0.65,
    water_salinity=35000.0,  # ppm NaCl
    separator_pressure=100.0,  # psi
    separator_temperature=70.0,  # °F
)

# Save to file
pvt_data.to_file("pvt_tables.h5")
```

### Complete Table Definition

```python
import numpy as np
import bores

# Pressure and temperature grids
pressures = np.linspace(1000, 5000, 50)  # psi
temperatures = np.linspace(100, 250, 20)  # °F
salinities = np.array([0, 35000, 100000])  # ppm NaCl (fresh, seawater, brine)

# 2D oil tables (P x T)
P, T = np.meshgrid(pressures, temperatures, indexing='ij')

# Example: Bo from correlation (for demonstration)
oil_api = 35.0
gas_sg = 0.65
Rs = 500 + 10 * (P / 1000)  # Simplified Rs(P)
Bo = 1.0 + 1.2e-4 * Rs  # Simplified Bo(Rs)

# Example: Gas compressibility factor from EOS
z_factor = compute_z_factor_eos(P, T, gas_sg)  # Your EOS function

# Create PVT table data
pvt_data = bores.PVTTableData(
    pressures=pressures,
    temperatures=temperatures,
    salinities=salinities,
    # Oil properties
    oil_formation_volume_factor_table=Bo,
    solution_gas_to_oil_ratio_table=Rs,
    # Gas properties
    gas_compressibility_factor_table=z_factor,
    # Add more tables as needed
)

# Save
pvt_data.to_file("pvt_tables.h5")
```

---

## Using PVT Tables

### Build Interpolators

Convert `PVTTableData` to `PVTTables` for fast lookup:

```python
# Load table data
pvt_data = bores.PVTTableData.from_file("pvt_tables.h5")

# Build interpolators
pvt_tables = bores.PVTTables(
    pvt_table_data=pvt_data,
    interpolation_method="linear",  # or "cubic"
)
```

**Interpolation methods**:

- `"linear"`: Fast, 1st-order accurate
- `"cubic"`: Slower, 3rd-order accurate (smooth derivatives)

### Query Properties

```python
# Single point query
P = 3000.0  # psi
T = 180.0   # °F

oil_visc = pvt_tables.oil_viscosity(P, T)
print(f"μo = {oil_visc:.3f} cP")

# Grid query
P_grid = np.array([[2000, 2500], [3000, 3500]])  # (2, 2) grid
T_grid = np.array([[150, 160], [170, 180]])

Bo_grid = pvt_tables.oil_formation_volume_factor(P_grid, T_grid)
print(f"Bo shape: {Bo_grid.shape}")  # (2, 2)
```

### Available Lookup Functions

**Oil properties**:
- `pvt_tables.oil_viscosity(P, T)` → μo (cP)
- `pvt_tables.oil_compressibility(P, T)` → co (psi⁻¹)
- `pvt_tables.oil_density(P, T)` → ρo (lbm/ft³)
- `pvt_tables.oil_formation_volume_factor(P, T)` → Bo (bbl/STB)
- `pvt_tables.solution_gas_to_oil_ratio(P, T)` → Rs (SCF/STB)
- `pvt_tables.oil_bubble_point_pressure(T)` → Pb (psi)

**Gas properties**:
- `pvt_tables.gas_viscosity(P, T)` → μg (cP)
- `pvt_tables.gas_compressibility(P, T)` → cg (psi⁻¹)
- `pvt_tables.gas_density(P, T)` → ρg (lbm/ft³)
- `pvt_tables.gas_formation_volume_factor(P, T)` → Bg (bbl/SCF)
- `pvt_tables.gas_compressibility_factor(P, T)` → z (dimensionless)
- `pvt_tables.gas_molecular_weight(P, T)` → Mg (lbm/lb-mol)

**Water properties**:
- `pvt_tables.water_viscosity(P, T, S)` → μw (cP)
- `pvt_tables.water_compressibility(P, T, S)` → cw (psi⁻¹)
- `pvt_tables.water_density(P, T, S)` → ρw (lbm/ft³)
- `pvt_tables.water_formation_volume_factor(P, T, S)` → Bw (bbl/STB)
- `pvt_tables.gas_solubility_in_water(P, T, S)` → Rsw (SCF/STB)

---

## Integration with Reservoir Model

### Use Tables Instead of Correlations

```python
import bores
from pathlib import Path

# Load or create PVT tables
pvt_data = bores.PVTTableData.from_file("pvt_tables.h5")
pvt_tables = bores.PVTTables(pvt_table_data=pvt_data)

# Create model with PVT tables
model = bores.reservoir_model(
    grid_shape=(20, 20, 5),
    thickness=50.0,
    porosity=0.22,
    permeability=100.0,
    initial_pressure=3000.0,
    temperature=180.0,
    pvt_tables=pvt_tables,  # Use tables instead of correlations
    # ... other parameters
)

# Create config
config = bores.Config(
    model=model,
    # ... other config
)

# Save PVT tables with config for reproducibility
pvt_data.to_file(Path("./setup/pvt_tables.h5"))
```

### Load from Files

```python
# Load run with PVT tables
run = bores.Run.from_files(
    model_path=Path("./setup/model.h5"),
    config_path=Path("./setup/config.yaml"),
    pvt_table_path=Path("./setup/pvt_tables.h5"),  # Specify PVT tables
)
```

---

## Property Clamping

PVT tables automatically clamp interpolated values to physically valid ranges:

```python
# Default clamps (from DEFAULT_PROPERTY_CLAMPS)
clamps = {
    "oil_viscosity": (1e-6, 1e4),           # cP
    "oil_compressibility": (0.0, 0.1),      # psi⁻¹
    "oil_density": (1.0, 80.0),             # lbm/ft³
    "oil_formation_volume_factor": (0.5, 5.0),  # bbl/STB
    "solution_gas_to_oil_ratio": (0.0, 5000.0),  # SCF/STB
    "gas_viscosity": (1e-6, 1e2),           # cP
    "gas_compressibility": (0.0, 0.1),      # psi⁻¹
    "gas_density": (0.001, 50.0),           # lbm/ft³
    "gas_compressibility_factor": (0.1, 3.0),  # dimensionless
    "water_viscosity": (1e-6, 10.0),        # cP
    "water_compressibility": (0.0, 0.01),   # psi⁻¹
    "water_density": (30.0, 80.0),          # lbm/ft³
}

# Values outside these ranges are clipped
```

!!! warning "Extrapolation"
    When querying pressures/temperatures outside the table range, interpolators extrapolate. Clamping prevents unphysical values, but accuracy degrades. Ensure table ranges cover expected simulation conditions.

---

## Performance Considerations

### Memory Usage

PVT tables use memory proportional to grid size:

```
Memory ≈ n_properties × n_pressures × n_temperatures × sizeof(float)
```

Example:
- 10 properties
- 50 pressures × 20 temperatures = 1000 points per property
- Float32: 4 bytes
- Memory ≈ 10 × 1000 × 4 bytes = 40 KB per reservoir cell

For large reservoirs (100k+ cells), this can be significant. Consider:

1. **Sparse tables**: Only populate needed properties
2. **Coarser grids**: Reduce n_pressures/n_temperatures (interpolation smooths)
3. **32-bit precision**: Use `bores.use_32bit_precision()` (default)

### Lookup Speed

Interpolation is O(1) but slower than direct calculation for simple correlations. Use tables when:

1. **Complex fluids**: EOS calculations are expensive
2. **Lab data**: No correlation available
3. **Large grids**: Amortize table build cost over many queries

Benchmarks (typical):
- Correlation: ~10 μs per cell
- Linear interpolation: ~5 μs per cell
- Cubic interpolation: ~15 μs per cell

For 100k cells:
- Correlation: ~1 second
- Linear table: ~0.5 seconds
- Cubic table: ~1.5 seconds

---

## Complete Workflow Example

```python
import numpy as np
import bores
from pathlib import Path

# Step 1: Create PVT table data from lab measurements
pressures = np.linspace(1000, 5000, 60)
temperatures = np.linspace(100, 250, 25)
P, T = np.meshgrid(pressures, temperatures, indexing='ij')

# Load lab data (example: from CSV or database)
oil_visc = load_lab_oil_viscosity(P, T)  # Your function
oil_bo = load_lab_oil_fvf(P, T)
gas_z = load_lab_gas_z_factor(P, T)

# Create table data
pvt_data = bores.PVTTableData(
    pressures=pressures,
    temperatures=temperatures,
    oil_viscosity_table=oil_visc,
    oil_formation_volume_factor_table=oil_bo,
    gas_compressibility_factor_table=gas_z,
    # ... add other properties
)

# Save to disk
pvt_data.to_file(Path("./pvt_data.h5"))
print("PVT table data saved.")

# Step 2: Build interpolators
pvt_tables = bores.PVTTables(
    pvt_table_data=pvt_data,
    interpolation_method="cubic",  # Smooth derivatives
)
print("Interpolators built.")

# Step 3: Test lookup
P_test = 3000.0
T_test = 180.0
print(f"\nTest lookup at P={P_test} psi, T={T_test} °F:")
print(f"  μo = {pvt_tables.oil_viscosity(P_test, T_test):.3f} cP")
print(f"  Bo = {pvt_tables.oil_formation_volume_factor(P_test, T_test):.4f} bbl/STB")
print(f"  z  = {pvt_tables.gas_compressibility_factor(P_test, T_test):.4f}")

# Step 4: Use in reservoir model
model = bores.reservoir_model(
    grid_shape=(20, 20, 5),
    thickness=50.0,
    porosity=0.22,
    permeability=100.0,
    initial_pressure=3000.0,
    temperature=180.0,
    pvt_tables=pvt_tables,
    # ... other parameters
)

print(f"\nModel created with PVT tables.")
print(f"Model grid shape: {model.grid_shape}")

# Step 5: Save for simulation
model.to_file(Path("./model.h5"))
pvt_data.to_file(Path("./pvt.h5"))
print("Model and PVT data saved for simulation.")
```

---

## Best Practices

### Table Resolution

1. **Pressure**: 40-60 points covering expected range (±20% margin)
2. **Temperature**: 15-25 points (temperature varies less than pressure)
3. **Salinity**: 3-5 points (fresh, seawater, high salinity)

### Interpolation Method

1. **Linear**: Default, fast, sufficient for most cases
2. **Cubic**: Use when smooth derivatives needed (e.g., compressibility from FVF)

### Table Range

1. **Lower bound**: Min expected P - 20% (account for drawdown)
2. **Upper bound**: Max expected P + 20% (account for injection)
3. **Check extrapolation**: Monitor warnings during simulation

### Validation

1. **Plot tables**: Visual inspection for anomalies
2. **Compare to correlations**: Verify consistency
3. **Check derivatives**: Ensure smooth transitions (especially near bubble point)

```python
# Plot oil FVF
import matplotlib.pyplot as plt

P_range = np.linspace(1000, 5000, 100)
T_fixed = 180.0
Bo_curve = pvt_tables.oil_formation_volume_factor(P_range, np.full_like(P_range, T_fixed))

plt.plot(P_range, Bo_curve)
plt.xlabel("Pressure (psi)")
plt.ylabel("Bo (bbl/STB)")
plt.title(f"Oil FVF at T = {T_fixed}°F")
plt.grid(True)
plt.show()
```

---

## Next Steps

- [Building Models](../guides/building-models.md) - Using PVT tables in models
- [Rock-Fluid Properties](../guides/rock-fluid-properties.md) - Relative permeability and capillary pressure tables
- [Storage and Serialization](storage-serialization.md) - Saving and loading tables
