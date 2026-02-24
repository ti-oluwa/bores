# Tutorial 4: Gas Injection

Learn how to implement gas injection for pressure maintenance and enhanced recovery.

---

## What You'll Learn

In this tutorial, you will learn how to:

- Set up gas injection wells with proper controls for pressure maintenance and enhanced oil recovery
- Choose between associated gas recycling (reinjecting produced gas) and external gas injection (purchasing CO2 or nitrogen)
- Configure the Todd-Longstaff miscibility model to simulate partial mixing between injected gas and reservoir oil based on pressure and MMP
- Monitor gas breakthrough at production wells by tracking gas-oil ratio (GOR) evolution over time
- Optimize injection strategy by adjusting injection rates, well placement, and considering water-alternating-gas (WAG) patterns
- Understand the difference between miscible and immiscible gas injection based on reservoir pressure relative to minimum miscibility pressure (MMP)

---

## Prerequisites

Complete [Tutorial 3: Water Injection](03-waterflood.md) first.

---

## Overview

Gas injection is used for:

1. **Pressure Maintenance**: Prevent pressure decline
2. **Gas Cap Expansion**: Utilize existing gas cap
3. **Enhanced Recovery**: Miscible or immiscible displacement
4. **Associated Gas Recycling**: Reinject produced gas

**Key parameters**:
- Minimum Miscibility Pressure (MMP)
- Todd-Longstaff omega (mixing parameter)
- Injection rate vs reservoir voidage

---

## Gas Injection Types

### Immiscible Gas Injection

Below MMP - gas displaces oil without mixing:

- **Methane (CH4)**: MMP = 3000-5000 psi
- **Associated gas**: MMP varies with composition
- **Recovery**: 10-20% incremental

### Miscible Gas Injection

Above MMP - gas mixes with oil:

- **CO2**: MMP = 1500-2500 psi
- **Enriched gas**: MMP = 1000-2000 psi
- **LPG**: MMP = 500-1500 psi
- **Recovery**: 15-30% incremental

---

## Step 1: Choose Injection Gas

For this tutorial, we'll use **associated gas recycling** (methane-rich):

```python
import bores
from pathlib import Path

# Gas properties (from produced gas analysis)
gas_specific_gravity = 0.68  # Slightly heavier than pure CH4
gas_molecular_weight = 18.5  # g/mol (mixture)

# Estimate MMP (for methane-rich gas)
reservoir_temp = 180.0  # °F
mmp_estimate = 4000.0  # psi (high, so mostly immiscible)

print(f"Gas: Associated gas (γg = {gas_specific_gravity})")
print(f"MMP estimate: {mmp_estimate} psi")
print(f"Reservoir pressure: ~2500 psi (below MMP → immiscible)")
```

---

## Step 2: Load Depleted Model

```python
# Load from waterflood or primary depletion
run = bores.Run.from_files(
    model_path=Path("./primary_depletion/model.h5"),
    config_path=Path("./primary_depletion/config.yaml"),
)

# Check current state
current_pressure = run.config.model.fluid_properties.pressure_grid.mean()
print(f"Current average pressure: {current_pressure:.1f} psi")
```

---

## Step 3: Create Gas Injection Wells

Place injectors in **updip locations** (gas migrates up):

```python
# Grid dimensions
nx, ny, nz = run.config.model.grid_shape

# Injection well locations (corners, updip if dipping reservoir)
injector_locations = [
    (3, 3),      # Top-left corner
    (3, nx-4),   # Top-right corner
]

# Injection control
injection_clamp = bores.InjectionClamp()
control = bores.AdaptiveBHPRateControl(
    target_rate=100_000,  # SCF/day (note: gas in SCF, not STB!)
    target_phase="gas",
    bhp_limit=3000,  # psi (below MMP, but maintain pressure)
    clamp=injection_clamp,
)

# Create injectors
injectors = []
for idx, (i, j) in enumerate(injector_locations, start=1):
    injector = bores.injection_well(
        well_name=f"GI-{idx}",
        perforating_intervals=[((i, j, 0), (i, j, 2))],  # Top layers (gas rises)
        radius=0.3542,
        control=control,
        injected_fluid=bores.InjectedFluid(
            name="methane",  # CoolProp-supported fluid name
            phase=bores.FluidPhase.GAS,
            specific_gravity=gas_specific_gravity,
            molecular_weight=gas_molecular_weight,
            minimum_miscibility_pressure=mmp_estimate,
            todd_longstaff_omega=0.33,  # Low mixing (mostly immiscible)
            is_miscible=True,  # Enable Todd-Longstaff model
        ),
        is_active=True,
        skin_factor=1.0,
    )
    injectors.append(injector)

print(f"Created {len(injectors)} gas injection wells")
```

!!! tip "Gas Injection Best Practices"
    - **Updip wells**: Gas rises, so inject at structural high
    - **Top layers**: Perforate upper layers to minimize gravity override
    - **BHP limit**: Stay below fracture pressure, but high enough for good injectivity

---

## Step 4: Update Producer

Increase gas handling capacity:

```python
production_clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-150,  # STB/day
        target_phase="oil",
        bhp_limit=1200,  # psi
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-500,  # MCF/day (increased for injected gas)
        target_phase="gas",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-20,  # STB/day
        target_phase="water",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
)

# Producer location (downdip or center)
producer_loc = (nx//2, ny//2)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((producer_loc[0], producer_loc[1], 3),
                            (producer_loc[0], producer_loc[1], nz-1))],  # Lower layers
    radius=0.3542,
    control=control,
    produced_fluids=(
        bores.ProducedFluid(name="Oil", phase=bores.FluidPhase.OIL,
                           specific_gravity=0.85, molecular_weight=180.0),
        bores.ProducedFluid(name="Gas", phase=bores.FluidPhase.GAS,
                           specific_gravity=0.68, molecular_weight=18.5),
        bores.ProducedFluid(name="Water", phase=bores.FluidPhase.WATER,
                           specific_gravity=1.05, molecular_weight=18.015),
    ),
    skin_factor=2.5,
)

wells = bores.wells_(injectors=injectors, producers=[producer])
```

---

## Step 5: Enable Todd-Longstaff Model

Critical for capturing gas-oil mixing effects:

```python
# Timer
timer = bores.Timer(
    initial_step_size=bores.Time(hours=24.0),
    max_step_size=bores.Time(days=10.0),
    min_step_size=bores.Time(hours=6.0),
    simulation_time=bores.Time(days=3 * bores.c.DAYS_PER_YEAR),  # 3 years
    max_cfl_number=0.9,
)

# Update config with Todd-Longstaff
run.config = run.config.with_updates(
    wells=wells,
    timer=timer,
    miscibility_model="todd_longstaff",  # CRITICAL: Enable mixing model
)

print("Todd-Longstaff miscibility model enabled")
```

!!! warning "Miscibility Model"
    Always enable `miscibility_model="todd_longstaff"` when `is_miscible=True` in `InjectedFluid`. Otherwise, miscibility effects are ignored!

---

## Step 6: Run Simulation

```python
# Save setup
setup_dir = Path("./tutorial_04_gas_injection")
setup_dir.mkdir(exist_ok=True)
run.to_file(setup_dir / "run.h5")

# Execute
store = bores.ZarrStore(setup_dir / "results.zarr")
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=20,
    background_io=True,
)

print("Starting gas injection simulation...")
with stream:
    for state in stream:
        if state.step % 50 == 0:
            time_days = state.time / 86400
            avg_pressure = state.model.fluid_properties.pressure_grid.mean()
            print(f"Day {time_days:4.0f}: P_avg = {avg_pressure:6.1f} psi")

    last_state = stream.last()

print("Simulation complete!")
```

---

## Step 7: Analyze Results

### Pressure Response

```python
states = list(store.load(bores.ModelState))
analyst = bores.ModelAnalyst(states)

# Pressure history
import matplotlib.pyplot as plt

times = []
pressures = []

for step in range(0, analyst.max_step, 10):
    state = analyst.get_state(step)
    times.append(state.time / 86400)  # Days
    pressures.append(state.model.fluid_properties.pressure_grid.mean())

plt.figure(figsize=(10, 6))
plt.plot(times, pressures, 'b-', linewidth=2)
plt.axhline(y=mmp_estimate, color='r', linestyle='--', label=f'MMP ({mmp_estimate} psi)')
plt.xlabel('Time (days)')
plt.ylabel('Average Pressure (psi)')
plt.title('Pressure Maintenance with Gas Injection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Gas-Oil Ratio (GOR)

```python
# GOR history
gors = []
for step in range(0, analyst.max_step, 10):
    rates = analyst.instantaneous_production_rates(step)
    gors.append(rates.gas_oil_ratio)

plt.figure(figsize=(10, 6))
plt.plot(times, gors, 'g-', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('GOR (SCF/STB)')
plt.title('Gas-Oil Ratio Evolution')
plt.grid(True, alpha=0.3)
plt.show()

# Check for gas breakthrough
initial_gor = gors[0]
final_gor = gors[-1]
gor_increase = (final_gor - initial_gor) / initial_gor * 100

print(f"\nGOR Evolution:")
print(f"  Initial GOR: {initial_gor:.1f} SCF/STB")
print(f"  Final GOR: {final_gor:.1f} SCF/STB")
print(f"  Increase: {gor_increase:.1f}%")

if final_gor > 2 * initial_gor:
    print("  WARNING: Significant gas breakthrough detected!")
```

### Recovery Factor

```python
# Oil recovery
print(f"\nRecovery Factor:")
print(f"  Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"  Cumulative oil: {analyst.cumulative_oil_produced:,.0f} STB")

# Gas utilization
gas_injected = analyst.gas_injected(0, -1)  # SCF
gas_produced = analyst.cumulative_free_gas_produced  # SCF
gas_retained = gas_injected - gas_produced

print(f"\nGas Balance:")
print(f"  Injected: {gas_injected/1e6:.1f} MMSCF")
print(f"  Produced: {gas_produced/1e6:.1f} MMSCF")
print(f"  Retained: {gas_retained/1e6:.1f} MMSCF ({gas_retained/gas_injected*100:.1f}%)")
```

### Sweep Efficiency

```python
# Sweep analysis
sweep = analyst.sweep_efficiency_analysis(step=-1, displacing_phase="gas")
print(f"\nSweep Efficiency:")
print(f"  Volumetric: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"  Displacement: {sweep.displacement_efficiency:.2%}")
print(f"  Vertical: {sweep.vertical_sweep_efficiency:.2%}")
```

---

## Optimization Strategies

### 1. Gas Recycling

Reinject produced gas to reduce costs:

```python
# Calculate recycling ratio
recycling_ratio = gas_produced / gas_injected
print(f"Gas recycling: {recycling_ratio*100:.1f}% of injected gas is produced")

# If recycling > 80%, consider reducing injection or adding gas compression
```

### 2. WAG (Water-Alternating-Gas)

Improve sweep efficiency by alternating water and gas:

```python
# Create schedule for WAG
well_schedule = bores.WellSchedule()

# Inject gas for 90 days
well_schedule["gas_cycle"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=90)),
    action=bores.update_well(is_active=False),  # Turn off after 90 days
)

# Inject water for 90 days (would need separate water injector)
# Then repeat...
```

### 3. Gravity Segregation Management

If gas override is severe:

```python
# Use horizontal wells in lower layers
producer_hz = bores.production_well(
    well_name="P-HZ",
    perforating_intervals=[
        ((nx//2, j, nz-2), (nx//2, j, nz-2))
        for j in range(ny//4, 3*ny//4)  # Horizontal extent
    ],
    # ... other parameters
)
```

---

## Common Issues

### Early Gas Breakthrough

**Cause**: Gravity override, high-permeability channels

**Solutions**:
- Reduce injection rate
- Inject in lower layers
- Use WAG (water blocks gas channels)
- Add horizontal producers

### Insufficient Pressure Support

**Cause**: Low injection rate, gas leaking to aquifer

**Solutions**:
- Increase injection rate
- Add more injectors
- Seal gas cap with barriers

### High Gas Recycling Costs

**Cause**: Poor sweep, gas cycling through high-perm zones

**Solutions**:
- Optimize well placement
- Use conformance control
- Switch to water injection in high-perm zones

---

## What You Learned

In this tutorial, you successfully:

- Set up gas injection wells for pressure maintenance with appropriate well placement (updip locations where gas naturally migrates)
- Configured the Todd-Longstaff miscibility model to capture partial mixing between injected gas and reservoir oil
- Monitored reservoir pressure response to gas injection and identified when pressure approaches or exceeds minimum miscibility pressure (MMP)
- Tracked gas-oil ratio (GOR) evolution to detect gas breakthrough at production wells
- Analyzed sweep efficiency and gas utilization factor to understand displacement effectiveness
- Identified optimization opportunities including gas recycling, water-alternating-gas (WAG), and gravity segregation management

---

## Next Steps

- **[Tutorial 5: Miscible Gas Flooding](05-miscible-flooding.md)** - CO2 miscible EOR
- **[Miscible EOR Example](../examples/miscible-eor.md)** - Complete CO2 flooding scenario
- **[Rock-Fluid Properties Guide](../guides/rock-fluid-properties.md)** - Todd-Longstaff model details
