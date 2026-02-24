# Tutorial 3: Water Injection

Learn how to set up and optimize waterflooding for secondary oil recovery.

---

## What You'll Learn

In this tutorial, you will learn how to:

- Add water injection wells to a reservoir model for secondary recovery
- Configure injection controls with both rate targets and bottomhole pressure (BHP) constraints
- Balance injection and production rates to maintain reservoir pressure (voidage replacement ratio)
- Monitor water breakthrough time and water cut evolution during waterflooding
- Analyze sweep efficiency to understand how effectively injected water contacts oil-bearing rock
- Optimize well placement patterns (5-spot, inverted 9-spot) for maximum oil recovery

---

## Prerequisites

Complete [Tutorial 2: Building a 3D Reservoir Model](02-reservoir-model.md) first.

---

## Overview

Waterflooding is the most common secondary recovery method. Injected water:

- Maintains reservoir pressure
- Displaces oil toward producers
- Recovers 30-60% of original oil (vs 15-25% primary recovery)

**Key challenge**: Optimize sweep efficiency (contact all oil-bearing rock)

---

## Step 1: Load Depleted Model

Start from a depleted model (after primary production):

```python
import bores
from pathlib import Path

# Load model from Tutorial 2 or primary depletion
run = bores.Run.from_files(
    model_path=Path("./primary_depletion/model.h5"),
    config_path=Path("./primary_depletion/config.yaml"),
)

# Check current state
print(f"Initial pressure: {run.config.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

!!! info "Starting Point"
    We assume the reservoir has been depleted by primary production. Pressure has dropped and production rates are declining.

---

## Step 2: Design Injection Pattern

Use a **5-spot pattern**: 4 corner injectors + 1 central producer.

```python
# Grid dimensions
nx, ny, nz = run.config.model.grid_shape

# Well locations
injector_locations = [
    (3, 3),      # Top-left
    (3, nx-4),   # Top-right
    (ny-4, 3),   # Bottom-left
    (ny-4, nx-4),  # Bottom-right
]

producer_location = (nx//2, ny//2)  # Center

print(f"Injector pattern: 4-spot corners")
print(f"Producer location: Center ({producer_location})")
```

---

## Step 3: Create Water Injection Wells

```python
# Injection control (rate-based with BHP limit)
injection_clamp = bores.InjectionClamp()
injection_control = bores.AdaptiveBHPRateControl(
    target_rate=400,  # STB/day per injector
    target_phase="water",
    bhp_limit=3500,  # psi (below fracture pressure)
    clamp=injection_clamp,
)

# Perforate all oil-bearing layers (layers 2-5)
injection_perfs = [((i, j, 1), (i, j, 5))]  # Layers 2-6 (0-indexed)

# Create injectors
injectors = []
for idx, (i, j) in enumerate(injector_locations, start=1):
    injector = bores.injection_well(
        well_name=f"WI-{idx}",
        perforating_intervals=[((i, j, 1), (i, j, 5))],
        radius=0.3542,  # 8.5" wellbore
        control=injection_control,
        injected_fluid=bores.InjectedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.05,  # Slightly saline
            molecular_weight=18.015,
        ),
        is_active=True,
        skin_factor=0.0,  # No damage (new well)
    )
    injectors.append(injector)

print(f"Created {len(injectors)} water injection wells")
```

!!! warning "BHP Limit"
    Set BHP limit below **fracture pressure** to avoid:

    - Hydraulic fracturing
    - Channeling (water bypasses oil)
    - Formation damage

---

## Step 4: Update Production Well

Increase production capacity to handle water production:

```python
# Higher water rate target (expect water breakthrough)
production_clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-200,  # STB/day (increased from 150)
        target_phase="oil",
        bhp_limit=800,  # psi
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-300,  # MCF/day
        target_phase="gas",
        bhp_limit=800,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-1500,  # STB/day (high for water production)
        target_phase="water",
        bhp_limit=800,
        clamp=production_clamp,
    ),
)

# Update existing producer or create new one
producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((producer_location[0], producer_location[1], 2),
                            (producer_location[0], producer_location[1], 5))],
    radius=0.3542,
    control=control,
    produced_fluids=(
        bores.ProducedFluid(name="Oil", phase=bores.FluidPhase.OIL,
                           specific_gravity=0.85, molecular_weight=180.0),
        bores.ProducedFluid(name="Gas", phase=bores.FluidPhase.GAS,
                           specific_gravity=0.65, molecular_weight=16.04),
        bores.ProducedFluid(name="Water", phase=bores.FluidPhase.WATER,
                           specific_gravity=1.05, molecular_weight=18.015),
    ),
    skin_factor=2.5,
    is_active=True,
)

wells = bores.wells_(injectors=injectors, producers=[producer])
```

---

## Step 5: Calculate Voidage Replacement Ratio

Ensure injection balances production (VRR ≈ 1.0 for pressure maintenance):

```python
# Total injection
total_injection = len(injectors) * 400  # STB/day

# Expected production (approximate)
expected_oil = 200  # STB/day
expected_water = 200  # STB/day (initial, will increase)
expected_gas = 300 * 1000 / 5.615  # Convert MCF to barrels (approx)

total_production = expected_oil + expected_water + expected_gas

# Voidage replacement ratio
vrr = total_injection / total_production
print(f"Target VRR: {vrr:.2f}")
print(f"Total injection: {total_injection} STB/day")
print(f"Expected production: {total_production:.0f} res barrels/day")
```

!!! tip "VRR Guidelines"
    - **VRR < 1.0**: Under-injection (pressure declines)
    - **VRR = 1.0**: Balanced (pressure stable)
    - **VRR > 1.0**: Over-injection (pressure increases)

    For waterflooding, target VRR = 1.0 - 1.2

---

## Step 6: Configure Simulation

```python
# Timer for 2 years of waterflooding
timer = bores.Timer(
    initial_step_size=bores.Time(days=2.0),
    max_step_size=bores.Time(days=10.0),
    min_step_size=bores.Time(hours=12.0),
    simulation_time=bores.Time(days=2 * bores.c.DAYS_PER_YEAR),  # 2 years
    max_cfl_number=0.9,
)

# Update config
run.config = run.config.with_updates(
    wells=wells,
    timer=timer,
)

# Save setup
setup_dir = Path("./tutorial_03_waterflood")
setup_dir.mkdir(exist_ok=True)
run.to_file(setup_dir / "run.h5")
```

---

## Step 7: Execute Simulation

```python
# Storage
store = bores.ZarrStore(setup_dir / "results.zarr")

# Stream execution
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=20,
    background_io=True,
)

print("Starting waterflood simulation...")
with stream:
    for state in stream:
        if state.step % 50 == 0:
            progress = stream.progress()
            time_days = state.time / 86400
            avg_pressure = state.model.fluid_properties.pressure_grid.mean()
            print(f"Day {time_days:4.0f}: P_avg = {avg_pressure:6.1f} psi, "
                  f"Saved {progress['saved_count']} states")

print("Simulation complete!")
```

---

## Step 8: Analyze Results

### Production History

```python
# Load results
states = list(store.load(bores.ModelState))
analyst = bores.ModelAnalyst(states)

# Oil and water production
import matplotlib.pyplot as plt

steps = []
oil_rates = []
water_rates = []
water_cuts = []

for step in range(0, analyst.max_step, 10):
    rates = analyst.instantaneous_production_rates(step)
    state = analyst.get_state(step)
    time_days = state.time / 86400

    steps.append(time_days)
    oil_rates.append(rates.oil_rate)
    water_rates.append(rates.water_rate)
    water_cuts.append(rates.water_cut * 100)  # Convert to %

# Plot production
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Rates
axes[0].plot(steps, oil_rates, 'g-', linewidth=2, label='Oil')
axes[0].plot(steps, water_rates, 'b--', linewidth=2, label='Water')
axes[0].set_ylabel('Rate (STB/day)')
axes[0].set_title('Production Rates')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Water cut
axes[1].plot(steps, water_cuts, 'r-', linewidth=2)
axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% WC')
axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% WC (Economic Limit)')
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Water Cut (%)')
axes[1].set_title('Water Cut Evolution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Recovery Factors

```python
# Final recovery
print(f"\nRecovery Factors:")
print(f"  Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"  Cumulative oil: {analyst.cumulative_oil_produced:,.0f} STB")

# Voidage replacement ratio (injection volume / production volume at reservoir conditions)
vrr_actual = analyst.voidage_replacement_ratio(step=-1)
print(f"\nActual VRR: {vrr_actual:.2f}")

# Sweep efficiency
sweep = analyst.sweep_efficiency_analysis(step=-1, displacing_phase="water")
print(f"\nSweep Efficiency:")
print(f"  Volumetric sweep: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"  Displacement efficiency: {sweep.displacement_efficiency:.2%}")
print(f"  Areal sweep: {sweep.areal_sweep_efficiency:.2%}")
```

### Water Saturation Maps

```python
# Initial and final water saturation
initial_state = analyst.get_state(0)
final_state = analyst.get_state(-1)

sw_initial = initial_state.model.fluid_properties.saturation_history.water_saturations[-1]
sw_final = final_state.model.fluid_properties.saturation_history.water_saturations[-1]

# Plot middle layer
layer = nz // 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(sw_initial[:, :, layer].T, origin='lower', cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Initial Water Saturation')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0], label='Sw')

# Mark wells
for i, j in injector_locations:
    axes[0].plot(i, j, 'rs', markersize=10, label='Injector' if (i,j) == injector_locations[0] else '')
axes[0].plot(producer_location[0], producer_location[1], 'go', markersize=10, label='Producer')

im2 = axes[1].imshow(sw_final[:, :, layer].T, origin='lower', cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Final Water Saturation (Water Flood)')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1], label='Sw')

# Mark wells
for i, j in injector_locations:
    axes[1].plot(i, j, 'rs', markersize=10)
axes[1].plot(producer_location[0], producer_location[1], 'go', markersize=10)

axes[0].legend()
plt.tight_layout()
plt.show()
```

---

## Optimization Tips

### 1. Adjust Injection Rates

If water cuts too high:

```python
# Reduce injection rates
new_control = bores.AdaptiveBHPRateControl(
    target_rate=300,  # Reduced from 400
    target_phase="water",
    bhp_limit=3500,
)
```

### 2. Pattern Modification

Convert to **inverted 9-spot** (more injectors):

```python
# Add 4 edge injectors
additional_locations = [
    (nx//2, 3),      # Top edge
    (nx//2, ny-4),   # Bottom edge
    (3, ny//2),      # Left edge
    (nx-4, ny//2),   # Right edge
]
```

### 3. Water Quality

Reduce injectivity loss:

```python
injected_fluid=bores.InjectedFluid(
    name="Filtered Water",
    phase=bores.FluidPhase.WATER,
    specific_gravity=1.02,  # Lower salinity
    molecular_weight=18.015,
)

# Also reduce skin factor
skin_factor=0.0  # Well stimulated/no damage
```

---

## Common Issues

### Water Breakthrough Too Early

**Cause**: High permeability streaks, poor vertical sweep

**Solutions**:
- Reduce injection rates
- Add polymer to water (increase viscosity)
- Use conformance control (future feature)

### Pressure Too High

**Cause**: Over-injection (VRR > 1.5)

**Solutions**:
- Reduce injection rates
- Add more producers
- Lower BHP limit

### Low Oil Recovery

**Cause**: Poor sweep efficiency

**Solutions**:
- Optimize well placement
- Increase pattern density (more wells)
- Check for vertical barriers (low vertical permeability)

---

## What You Learned

In this tutorial, you successfully:

- Designed a 5-spot waterflood pattern with 4 corner injectors and 1 central producer for optimal sweep efficiency
- Configured water injection wells with adaptive BHP/rate controls that maintain pressure below fracture pressure
- Calculated and monitored voidage replacement ratio (VRR) to balance injection and production rates for pressure maintenance
- Tracked water breakthrough time and monitored water cut evolution throughout the waterflood
- Analyzed sweep efficiency metrics including volumetric, areal, and displacement efficiency
- Visualized saturation changes over time using 2D maps and time-series plots

---

## Next Steps

- **[Tutorial 4: Gas Injection](04-gas-injection.md)** - Pressure maintenance with gas
- **[Wells & Controls Guide](../guides/wells-and-controls.md)** - Advanced well configuration
- **[Waterflood Pattern Example](../examples/waterflood-pattern.md)** - Complete production example
