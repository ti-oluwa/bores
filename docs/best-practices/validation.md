# Validation Best Practices

Guidelines for validating and quality-checking BORES simulations.

---

## Overview

Validation ensures simulation results are:

- **Physically plausible**: Follow reservoir engineering principles
- **Numerically accurate**: Grid-independent and converged
- **Internally consistent**: Mass balance, energy balance
- **Reasonable**: Match expectations and analogues

**Goal**: Build confidence in simulation results before making decisions.

---

## Pre-Simulation Validation

### Check Initial Conditions

```python
import bores
import numpy as np

# Load model
model = bores.reservoir_model(...)

# 1. Saturation sum = 1.0
so = model.fluid_properties.saturation_history.oil_saturations[-1]
sw = model.fluid_properties.saturation_history.water_saturations[-1]
sg = model.fluid_properties.saturation_history.gas_saturations[-1]

sat_sum = so + sw + sg
assert np.allclose(sat_sum, 1.0), f"Saturation sum error: {sat_sum.min():.3f} to {sat_sum.max():.3f}"
print("✅ Saturation sum = 1.0")

# 2. Saturations in valid range
assert so.min() >= 0 and so.max() <= 1, "Oil saturation out of range"
assert sw.min() >= 0 and sw.max() <= 1, "Water saturation out of range"
assert sg.min() >= 0 and sg.max() <= 1, "Gas saturation out of range"
print("✅ Saturations in [0, 1]")

# 3. Pressure reasonable
pressure = model.fluid_properties.pressure_grid
assert pressure.min() > 0, "Negative pressure!"
assert pressure.max() < 10000, "Unrealistic high pressure"
print(f"✅ Pressure range: {pressure.min():.0f} - {pressure.max():.0f} psi")

# 4. Porosity reasonable
porosity = model.rock_properties.porosity_grid
assert porosity.min() >= 0 and porosity.max() <= 0.5, "Unrealistic porosity"
print(f"✅ Porosity range: {porosity.min():.3f} - {porosity.max():.3f}")

# 5. Permeability positive
perm = model.rock_properties.permeability.x_direction
assert perm.min() > 0, "Non-positive permeability"
print(f"✅ Permeability range: {perm.min():.1f} - {perm.max():.1f} mD")
```

### Visualize Initial State

```python
import matplotlib.pyplot as plt

# Plot initial saturations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

layer = so.shape[2] // 2  # Middle layer

axes[0].imshow(so[:, :, layer].T, origin='lower', cmap='Oranges', vmin=0, vmax=1)
axes[0].set_title('Oil Saturation')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].imshow(sw[:, :, layer].T, origin='lower', cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Water Saturation')

axes[2].imshow(sg[:, :, layer].T, origin='lower', cmap='Greens', vmin=0, vmax=1)
axes[2].set_title('Gas Saturation')

plt.tight_layout()
plt.show()

# Visual inspection: Look for unrealistic patterns
```

---

## During-Simulation Monitoring

### Material Balance

Track total mass in system:

```python
volumes = []

for state in run():
    # Compute total hydrocarbon volume
    pv = (state.model.rock_properties.porosity_grid *
          state.model.thickness_grid *
          100 * 100)  # Cell volume

    so_current = state.model.fluid_properties.saturation_history.oil_saturations[-1]
    sw_current = state.model.fluid_properties.saturation_history.water_saturations[-1]
    sg_current = state.model.fluid_properties.saturation_history.gas_saturations[-1]

    oil_vol = np.sum(pv * so_current)
    water_vol = np.sum(pv * sw_current)
    gas_vol = np.sum(pv * sg_current)

    volumes.append({
        "step": state.step,
        "time": state.time / 86400,
        "oil": oil_vol,
        "water": water_vol,
        "gas": gas_vol,
        "total": oil_vol + water_vol + gas_vol,
    })

# Check total pore volume conservation
initial_total = volumes[0]["total"]
final_total = volumes[-1]["total"]
change = abs(final_total - initial_total) / initial_total

if change < 0.01:
    print(f"✅ Pore volume conserved: {change*100:.3f}% change")
else:
    print(f"⚠️ Pore volume violation: {change*100:.1f}% change")
```

### Saturation Bounds

```python
for state in run():
    so = state.model.fluid_properties.saturation_history.oil_saturations[-1]
    sw = state.model.fluid_properties.saturation_history.water_saturations[-1]
    sg = state.model.fluid_properties.saturation_history.gas_saturations[-1]

    # Check bounds
    if so.min() < 0 or so.max() > 1:
        print(f"⚠️ Step {state.step}: Oil saturation out of bounds!")

    if sw.min() < 0 or sw.max() > 1:
        print(f"⚠️ Step {state.step}: Water saturation out of bounds!")

    if sg.min() < 0 or sg.max() > 1:
        print(f"⚠️ Step {state.step}: Gas saturation out of bounds!")

    # Check sum
    sat_sum = so + sw + sg
    if not np.allclose(sat_sum, 1.0, atol=1e-3):
        print(f"⚠️ Step {state.step}: Saturation sum violation!")
```

---

## Post-Simulation Validation

### Grid Convergence

Test that results don't change with finer grid:

```python
# Run 3 grids
grids = [
    (10, 10, 5),    # Coarse
    (20, 20, 10),   # Medium
    (40, 40, 20),   # Fine
]

results = []

for grid_shape in grids:
    model = bores.reservoir_model(grid_shape=grid_shape, ...)
    # ... run simulation
    analyst = bores.ModelAnalyst(states)
    results.append(analyst.oil_recovery_factor)

# Check convergence
print("Grid Convergence:")
for grid, rf in zip(grids, results):
    print(f"  {grid}: RF = {rf:.2%}")

# Convergence criterion: <1% change from medium to fine
change = abs(results[2] - results[1]) / results[1]
if change < 0.01:
    print(f"✅ Grid converged: {change*100:.2f}% change")
else:
    print(f"⚠️ Not converged: {change*100:.1f}% change - refine further")
```

### Timestep Independence

Test that results don't change with smaller timesteps:

```python
timestep_configs = [
    {"max_step_size": bores.Time(days=60)},  # Coarse
    {"max_step_size": bores.Time(days=30)},  # Medium
    {"max_step_size": bores.Time(days=15)},  # Fine
]

results = []

for config in timestep_configs:
    timer = bores.Timer(**config, ...)
    # ... run simulation
    results.append(oil_recovery_factor)

# Check convergence
change = abs(results[2] - results[1]) / results[1]
if change < 0.02:  # 2% tolerance (looser than grid)
    print(f"✅ Timestep independent: {change*100:.2f}% change")
```

### Recovery Factor Sanity Check

Compare to typical values:

```python
rf = analyst.oil_recovery_factor

# Primary depletion
if scenario == "primary":
    if 0.05 <= rf <= 0.35:
        print(f"✅ Recovery factor reasonable: {rf:.2%}")
    else:
        print(f"⚠️ Unusual recovery factor: {rf:.2%}")
        print("   Expected: 5-35% for primary depletion")

# Waterflooding
elif scenario == "waterflood":
    if 0.30 <= rf <= 0.65:
        print(f"✅ Recovery factor reasonable: {rf:.2%}")
    else:
        print(f"⚠️ Unusual recovery factor: {rf:.2%}")
        print("   Expected: 30-65% for waterflooding")

# Miscible CO2
elif scenario == "co2_miscible":
    if 0.50 <= rf <= 0.80:
        print(f"✅ Recovery factor reasonable: {rf:.2%}")
    else:
        print(f"⚠️ Unusual recovery factor: {rf:.2%}")
        print("   Expected: 50-80% for miscible EOR")
```

### Water Cut Evolution

```python
# Check water cut progression
wc_history = []

for step in range(0, analyst.max_step, 20):
    rates = analyst.instantaneous_production_rates(step)
    wc_history.append(rates.water_cut)

# Water cut should increase monotonically (waterflood)
if scenario == "waterflood":
    increasing = all(wc_history[i] <= wc_history[i+1] for i in range(len(wc_history)-1))

    if increasing:
        print("✅ Water cut increases monotonically")
    else:
        print("⚠️ Water cut decreases - check for numerical issues")
```

### GOR Trends

```python
# GOR should be relatively stable (primary depletion above bubble point)
# Or increase (gas breakthrough)

gors = []
for step in range(0, analyst.max_step, 20):
    rates = analyst.instantaneous_production_rates(step)
    gors.append(rates.gas_oil_ratio)

initial_gor = gors[0]
final_gor = gors[-1]

# Check for unrealistic GOR
if final_gor > 50000:  # SCF/STB
    print(f"⚠️ Very high GOR: {final_gor:.0f} SCF/STB - check for gas breakout")
elif final_gor < 0:
    print("❌ Negative GOR - numerical error!")
else:
    print(f"✅ GOR reasonable: {initial_gor:.0f} → {final_gor:.0f} SCF/STB")
```

---

## Physics Validation

### Darcy's Law Check

Verify flow follows pressure gradient:

```python
# Get final state
final_state = analyst.get_state(-1)
pressure = final_state.model.fluid_properties.pressure_grid

# Producer and injector pressures
producer_p = pressure[producer_i, producer_j, producer_k]
injector_p = pressure[injector_i, injector_j, injector_k]

# Flow should be from high to low pressure
if injector_p > producer_p:
    print(f"✅ Flow direction correct: {injector_p:.0f} → {producer_p:.0f} psi")
else:
    print(f"⚠️ Flow direction wrong: Injector P < Producer P")
```

### Buoyancy Check

Oil should rise, water should sink:

```python
# Average saturations by layer
for k in range(nz):
    so_layer = final_oil_sat[:, :, k].mean()
    sw_layer = final_water_sat[:, :, k].mean()

    print(f"Layer {k}: So = {so_layer:.3f}, Sw = {sw_layer:.3f}")

# For water-wet, expect:
# - More water in bottom layers
# - More oil in top layers (if gas present, oil in middle)
```

---

## Comparison Validation

### Analogue Fields

Compare to similar reservoirs:

```python
# Your reservoir
your_perm = 100  # mD
your_depth = 5000  # ft
your_rf = 0.42  # Recovery factor

# Analogue from literature
analogue_perm = 120  # mD
analogue_depth = 5200  # ft
analogue_rf = 0.38  # Recovery factor

# Rough comparison
if abs(your_rf - analogue_rf) < 0.1:
    print("✅ Recovery similar to analogue")
else:
    print(f"⚠️ Recovery differs from analogue: {your_rf:.2%} vs {analogue_rf:.2%}")
```

### Material Balance Equation

For depletion:

```python
# Simple material balance check
N = analyst.stoiip  # Initial oil (STB)
Np = analyst.cumulative_oil_produced  # Produced oil (STB)
remaining = N - Np

# Compute from final state
final_state = analyst.get_state(-1)
final_oil_pv = compute_oil_in_place(final_state)  # STB

# Compare
error = abs(final_oil_pv - remaining) / N

if error < 0.05:
    print(f"✅ Material balance: {error*100:.2f}% error")
else:
    print(f"⚠️ Material balance violation: {error*100:.1f}% error")
```

---

## Validation Checklist

### Pre-Simulation

✅ **Initial saturations**: Sum = 1.0, range [0, 1]
✅ **Initial pressure**: Positive, reasonable range
✅ **Porosity**: 0.05 - 0.40 (typical)
✅ **Permeability**: Positive, reasonable range
✅ **Fluid contacts**: OWC, GOC at correct depths
✅ **Well locations**: Inside reservoir, correct layers

### During Simulation

✅ **Saturation bounds**: Always in [0, 1]
✅ **Saturation sum**: Always = 1.0 ± 0.001
✅ **Pressure**: Positive, no oscillations
✅ **Convergence**: <10% timestep rejection rate
✅ **Mass balance**: Total pore volume conserved

### Post-Simulation

✅ **Grid convergence**: <1% change medium → fine
✅ **Timestep independence**: <2% change
✅ **Recovery factor**: Within expected range
✅ **Water cut**: Monotonic increase (waterflood)
✅ **GOR**: Reasonable values and trends
✅ **Material balance**: <5% error
✅ **Physics**: Flow follows pressure gradient
✅ **Comparison**: Matches analogues

---

## Troubleshooting

### Unphysical Results

**Symptoms**: Negative saturations, recovery >100%, etc.

**Causes**:

- Numerical instability
- Grid too coarse
- Timestep too large
- Solver not converging

**Solutions**:

1. Refine grid (especially near wells)
2. Reduce max_cfl_number
3. Use tighter solver tolerance
4. Check initial conditions

### Poor Material Balance

**Symptoms**: Oil in ≠ oil out + remaining

**Causes**:

- Solver tolerance too loose
- Boundary condition leaks
- Timestep rejections losing mass

**Solutions**:

1. Tighten solver tolerance (1e-7)
2. Check boundary conditions
3. Reduce max timestep
4. Monitor rejections

### Results Don't Match Expectations

**Symptoms**: Recovery too high/low, wrong breakthrough time

**Causes**:

- Wrong relative permeability
- Wrong PVT properties
- Wrong well controls
- Grid too coarse

**Solutions**:

1. Validate input data against lab
2. Check rock-fluid properties
3. Verify well specifications
4. Run grid convergence study

---

## Validation Report Template

```markdown
# Simulation Validation Report

## Model Description
- Grid: 50×50×15 = 37,500 cells
- Simulation time: 10 years
- Recovery method: Waterflooding

## Pre-Simulation Checks
✅ Initial saturations valid
✅ Pressure range: 2800-3100 psi
✅ Porosity: 0.18-0.25
✅ Permeability: 50-200 mD

## Simulation Quality
✅ Timestep rejection rate: 3.2%
✅ Average solver iterations: 45
✅ Mass balance error: 0.8%

## Grid Convergence
| Grid | Cells | Oil RF |
|------|-------|--------|
| Coarse | 9,000 | 38.2% |
| Medium | 37,500 | 41.5% |
| Fine | 150,000 | 42.1% |

Change (medium → fine): 1.4% ✅ Acceptable

## Results Validation
✅ Oil RF: 41.5% (expected 35-50% for waterflood)
✅ Water breakthrough: 180 days (reasonable)
✅ Final water cut: 85% (typical)
✅ GOR stable: 500-600 SCF/STB

## Conclusions
Simulation results are validated and reliable for decision-making.
```

---

## Next Steps

- [Grid Design](grid-design.md) - Optimize grid for accuracy
- [Solver Selection](solver-selection.md) - Ensure convergence
- [Timestep Control](timestep-control.md) - Maintain stability
