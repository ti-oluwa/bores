# Model Stabilization

Equilibrate the reservoir model before production begins.

---

## Objective

Run a short equilibration period to stabilize the model:

- **Duration**: 100 days (no wells)
- **Purpose**: Let initial conditions settle
- **Result**: Stable pressure and saturation distributions
- **Use**: Provides starting point for all production scenarios

This is the **second step** in the workflow:

1. [Setup](setup.md) ← Create initial model
2. **Stabilization** ← YOU ARE HERE
3. [Primary Depletion](primary-depletion.md) ← Add production
4. [Waterflood](waterflood-pattern.md), [Gas Injection](gas-cap-expansion.md), [CO2 EOR](miscible-eor.md)

---

## Why Stabilization?

Initial reservoir models often have small inconsistencies:

- **Pressure not perfectly hydrostatic**: Small variations in initial pressure
- **Saturations not perfectly equilibrated**: Capillary-gravity equilibrium takes time
- **PVT state adjustment**: Fluids adjust to P-T conditions

**Without stabilization**: First production timesteps may show artificial transients

**With stabilization**: Start from a physically consistent state

!!! tip "When to Skip"
    For conceptual studies or when initial conditions are already perfectly equilibrated, you can skip this step and use the setup model directly.

---

## Complete Code

```python
import logging
from pathlib import Path
import numpy as np
import bores

logging.basicConfig(level=logging.INFO)
np.set_printoptions(threshold=np.inf)
bores.use_32bit_precision()

# Register cached preconditioner
ilu_preconditioner = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,
    recompute_threshold=0.3,
)
ilu_preconditioner.register(override=True)

#=== Load Setup Files ===
setup_dir = Path("./scenarios/runs/setup")
stabilization_dir = Path("./scenarios/runs/stabilization")
stabilization_dir.mkdir(parents=True, exist_ok=True)

# Load model, config, and PVT tables from setup
run = bores.Run.from_files(
    model_path=setup_dir / "model.h5",
    config_path=setup_dir / "config.yaml",
    pvt_table_path=setup_dir / "pvt.h5",
)

print("Loaded initial model from setup")
print(f"  Grid shape: {run.config.model.grid_shape}")
print(f"  Initial avg pressure: {run.config.model.fluid_properties.pressure_grid.mean():.1f} psi")

#=== Configure Stabilization Run ===
# Short timer for equilibration (no wells, so should be fast)
timer = bores.Timer(
    initial_step_size=bores.Time(days=1.0),
    max_step_size=bores.Time(days=10.0),
    min_step_size=bores.Time(hours=6.0),
    simulation_time=bores.Time(days=100),  # 100 days equilibration
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
)

# Update config for stabilization (still no wells)
run.config = run.config.with_updates(
    timer=timer,
    wells=None,  # No wells during stabilization
    pressure_solver="bicgstab",
    pressure_preconditioner="cached_ilu",  # Use cached preconditioner
)

# Save stabilization config
run.config.to_file(stabilization_dir / "config.yaml")

#=== Execute Stabilization ===
results_dir = stabilization_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

store = bores.ZarrStore(results_dir / "stabilization.zarr")

stream = bores.StateStream(
    run(),
    store=store,
    batch_size=10,  # Small batch for short run
    background_io=True,
)

print("\nStarting stabilization run (100 days, no wells)...")

with stream:
    for state in stream:
        if state.step % 5 == 0:
            time_days = state.time / 86400
            avg_pressure = state.model.fluid_properties.pressure_grid.mean()
            print(f"  Day {time_days:3.0f}: P_avg = {avg_pressure:6.1f} psi")

    # Get final state
    final_state = stream.last()

print("\nStabilization complete!")

#=== Save Stabilized Model ===
# Save final stabilized model for use in production runs
final_state.model.to_file(results_dir / "model.h5")

print(f"\n{'='*60}")
print("STABILIZATION RESULTS")
print(f"{'='*60}")

# Compare initial vs final
initial_pressure = run.config.model.fluid_properties.pressure_grid.mean()
final_pressure = final_state.model.fluid_properties.pressure_grid.mean()
pressure_change = final_pressure - initial_pressure

print(f"Initial avg pressure: {initial_pressure:.1f} psi")
print(f"Final avg pressure: {final_pressure:.1f} psi")
print(f"Pressure change: {pressure_change:+.1f} psi ({pressure_change/initial_pressure*100:+.2f}%)")

# Check saturation stability
initial_oil_sat = run.config.model.fluid_properties.saturation_history.oil_saturations[-1].mean()
final_oil_sat = final_state.model.fluid_properties.saturation_history.oil_saturations[-1].mean()
sat_change = final_oil_sat - initial_oil_sat

print(f"\nInitial avg oil saturation: {initial_oil_sat:.4f}")
print(f"Final avg oil saturation: {final_oil_sat:.4f}")
print(f"Saturation change: {sat_change:+.4f}")

# Stability check
if abs(pressure_change / initial_pressure) < 0.01 and abs(sat_change) < 0.001:
    print("\nGOOD: Model is well-stabilized (< 1% change)")
else:
    print(f"\nWARNING: Model still adjusting - consider longer stabilization")

print(f"\nStabilized model saved to: {results_dir / 'model.h5'}")
print(f"\nNext step: Run primary depletion")
print(f"  → Use stabilized model: {results_dir / 'model.h5'}")
```

---

## What Happens During Stabilization

### 1. Pressure Redistribution

Initial pressure may not be perfectly hydrostatic due to:

- Discrete gridding
- Numerical interpolation
- Vertical permeability barriers

**Result**: Small pressure adjustments (<1% typically)

### 2. Capillary-Gravity Equilibrium

Saturations adjust to balance:

```
Capillary pressure = Gravity head difference
```

**Result**: Small saturation redistributions near fluid contacts

### 3. PVT Equilibration

Fluids adjust properties to local P-T conditions:

- Solution gas in oil equilibrates
- Gas compressibility adjusts
- Water formation volume factor stabilizes

**Result**: Consistent fluid properties throughout

---

## Expected Results

### Pressure Changes

Typical pressure changes during stabilization:

```
ΔP / P_initial < 1%
```

**Example**:

- Initial: 3000 psi
- Final: 2995 psi
- Change: -5 psi (-0.17%)

**If change > 1%**: Extend stabilization time or check initial conditions

### Saturation Changes

Typical saturation changes:

```
ΔS_o < 0.001 (0.1%)
```

**Example**:

- Initial So: 0.6523
- Final So: 0.6519
- Change: -0.0004

**If change > 0.01**: Check fluid contacts and capillary pressure

---

## Monitoring Stabilization

### Pressure Evolution

```python
# Load states
store = bores.ZarrStore("./scenarios/runs/stabilization/results/stabilization.zarr")
states = list(store.load(bores.ModelState))

# Track average pressure
import matplotlib.pyplot as plt

times = []
pressures = []

for state in states:
    times.append(state.time / 86400)  # days
    pressures.append(state.model.fluid_properties.pressure_grid.mean())

plt.figure(figsize=(10, 6))
plt.plot(times, pressures, 'b-', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Average Pressure (psi)')
plt.title('Pressure Evolution During Stabilization')
plt.grid(True, alpha=0.3)
plt.show()

# Pressure should asymptote to stable value
```

### Saturation Evolution

```python
oil_saturations = []
water_saturations = []

for state in states:
    oil_sat = state.model.fluid_properties.saturation_history.oil_saturations[-1]
    water_sat = state.model.fluid_properties.saturation_history.water_saturations[-1]

    oil_saturations.append(oil_sat.mean())
    water_saturations.append(water_sat.mean())

plt.figure(figsize=(10, 6))
plt.plot(times, oil_saturations, 'g-', linewidth=2, label='Oil')
plt.plot(times, water_saturations, 'b-', linewidth=2, label='Water')
plt.xlabel('Time (days)')
plt.ylabel('Average Saturation')
plt.title('Saturation Evolution During Stabilization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Troubleshooting

### Large Pressure Changes (>5%)

**Cause**: Initial pressure not hydrostatic

**Solution**:

```python
# Recalculate initial pressure more carefully
depth = model.depth
reference_depth = depth.min()
reference_pressure = 3000.0  # psi at shallowest point
pressure_gradient = 0.433  # psi/ft for water

# Hydrostatic pressure
initial_pressure = reference_pressure + (depth - reference_depth) * pressure_gradient
```

### Large Saturation Changes (>1%)

**Cause**: Saturations not in capillary-gravity equilibrium

**Solution**:

- Extend stabilization time (200-300 days)
- Check capillary pressure model parameters
- Verify fluid contact depths

### Oscillations

**Cause**: Timestep too large or solver not converging

**Solution**:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=12.0),  # Smaller initial step
    max_step_size=bores.Time(days=5.0),  # Smaller max step
    max_cfl_number=0.7,  # More conservative
)
```

---

## Comparison: With vs Without Stabilization

### Without Stabilization

```python
# Skip stabilization, go straight to production
run = bores.Run.from_files(
    model_path=Path("./scenarios/runs/setup/model.h5"),
    # ... add wells and run
)

# First 10 days may show:
# - Pressure spikes/dips
# - Saturation oscillations
# - Artificial transients
```

### With Stabilization

```python
# Use stabilized model
run = bores.Run.from_files(
    model_path=Path("./scenarios/runs/stabilization/results/model.h5"),
    # ... add wells and run
)

# Production starts from stable state:
# - Smooth pressure decline
# - Physical saturation changes
# - No artificial transients
```

---

## Files Created

After stabilization:

```
scenarios/runs/stabilization/
├── config.yaml                  # Stabilization config
└── results/
    ├── stabilization.zarr/      # Full state history
    └── model.h5                 # Final stabilized model ← Use this for production!
```

**Use for all production scenarios**:

- [Primary Depletion](primary-depletion.md): Uses `stabilization/results/model.h5`
- [Waterflood](waterflood-pattern.md): Uses `primary_depletion/results/model.h5` (which started from stabilized)
- [Gas Injection](gas-cap-expansion.md): Uses `primary_depletion/results/model.h5`
- [CO2 EOR](miscible-eor.md): Uses `primary_depletion/results/model.h5`

---

## Analysis

### Verify Stabilization

```python
from pathlib import Path
import bores

# Load initial and final states
initial_model = bores.ReservoirModel.from_file(
    Path("./scenarios/runs/setup/model.h5")
)

stabilized_model = bores.ReservoirModel.from_file(
    Path("./scenarios/runs/stabilization/results/model.h5")
)

# Compare
p_initial = initial_model.fluid_properties.pressure_grid
p_final = stabilized_model.fluid_properties.pressure_grid

print("Pressure Statistics:")
print(f"  Initial: {p_initial.mean():.1f} ± {p_initial.std():.1f} psi")
print(f"  Final: {p_final.mean():.1f} ± {p_final.std():.1f} psi")

p_change = p_final - p_initial
print(f"  Max change: {p_change.max():.1f} psi")
print(f"  Mean change: {p_change.mean():.1f} psi")

# Saturation comparison
so_initial = initial_model.fluid_properties.saturation_history.oil_saturations[-1]
so_final = stabilized_model.fluid_properties.saturation_history.oil_saturations[-1]

print("\nOil Saturation Statistics:")
print(f"  Initial: {so_initial.mean():.4f}")
print(f"  Final: {so_final.mean():.4f}")
print(f"  Change: {(so_final - so_initial).mean():+.4f}")
```

---

## Recommended Duration

| Reservoir Type | Duration | Reason |
| -------------- | -------- | ------ |
| **Homogeneous, shallow** | 30-50 days | Fast equilibration |
| **Standard (like example)** | 100 days | Default |
| **Heterogeneous** | 200-300 days | Longer equilibration time |
| **Deep, tight** | 300-500 days | Very slow pressure diffusion |

**Rule of thumb**: Run until changes < 0.1% per 10 days

---

## What You Learned

In this example, you successfully:

- Created a stabilized reservoir model by running gravity-capillary equilibration with no wells
- Monitored pressure and saturation equilibration to ensure model reaches steady state
- Verified model stability by checking that saturation and pressure changes are below convergence thresholds
- Generated a starting point for production scenarios that represents true thermodynamic equilibrium

---

## Next Steps

- **[Primary Depletion →](primary-depletion.md)** - Add producer and deplete reservoir (uses stabilized model)
- [Setup](setup.md) - Review initial model creation
- [Running Simulations Guide](../guides/running-simulations.md) - Configuration details
