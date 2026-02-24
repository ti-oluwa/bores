# Waterflood Pattern Example

Complete working example of waterflooding with 5-spot injection pattern.

---

## Objective

Simulate waterflooding for secondary recovery:

- 4 water injectors (corner pattern)
- 1 producer (center)
- 5-spot injection pattern
- Pressure maintenance and oil displacement
- 3 years of waterflooding

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

# Cached preconditioner
ilu_preconditioner = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,
    recompute_threshold=0.3,
)
ilu_preconditioner.register(override=True)

# Load depleted model from primary depletion run
run = bores.Run.from_files(
    model_path=Path("./scenarios/runs/primary_depletion/results/model.h5"),
    config_path=Path("./scenarios/runs/setup/config.yaml"),
    pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
)

#=== Water Injection Wells (4-Spot Pattern) ===
injection_clamp = bores.InjectionClamp()
control = bores.AdaptiveBHPRateControl(
    target_rate=500,  # STB/day per injector
    target_phase="water",
    bhp_limit=3000,  # psi maximum
    clamp=injection_clamp,
)

# Corner 1: Top-left
water_injector_1 = bores.injection_well(
    well_name="WI-1",
    perforating_intervals=[((3, 3, 2), (3, 3, 4))],  # 3 layers
    radius=0.3542,  # 8.5" wellbore
    control=control,
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.05,
        molecular_weight=18.015,
    ),
    is_active=True,
    skin_factor=2.0,
)

# Corner 2: Top-right
water_injector_2 = water_injector_1.duplicate(
    name="WI-2",
    perforating_intervals=[((3, 16, 2), (3, 16, 4))]
)

# Corner 3: Bottom-left
water_injector_3 = water_injector_1.duplicate(
    name="WI-3",
    perforating_intervals=[((16, 3, 2), (16, 3, 4))]
)

# Corner 4: Bottom-right
water_injector_4 = water_injector_1.duplicate(
    name="WI-4",
    perforating_intervals=[((16, 16, 2), (16, 16, 4))]
)

injectors = [water_injector_1, water_injector_2, water_injector_3, water_injector_4]

#=== Production Well (Center) ===
production_clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-200,  # STB/day
        target_phase="oil",
        bhp_limit=800,  # psi minimum
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-300,  # MCF/day
        target_phase="gas",
        bhp_limit=800,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-1000,  # STB/day (higher for water production)
        target_phase="water",
        bhp_limit=800,
        clamp=production_clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((10, 10, 2), (10, 10, 4))],  # Center location
    radius=0.3542,
    control=control,
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.845,
            molecular_weight=180.0,
        ),
        bores.ProducedFluid(
            name="Gas",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=16.04,
        ),
        bores.ProducedFluid(
            name="Water",
            phase=bores.FluidPhase.WATER,
            specific_gravity=1.05,
            molecular_weight=18.015,
        ),
    ),
    skin_factor=2.5,
    is_active=True,
)

producers = [producer]
wells = bores.wells_(injectors=injectors, producers=producers)

#=== Timer ===
timer = bores.Timer(
    initial_step_size=bores.Time(hours=24.0),
    max_step_size=bores.Time(days=7.0),
    min_step_size=bores.Time(hours=6.0),
    simulation_time=bores.Time(days=3 * bores.c.DAYS_PER_YEAR),  # 3 years
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
    max_rejects=20,
)

#=== Update Config ===
run.config = run.config.with_updates(
    wells=wells,
    timer=timer,
)

# Save run
run.to_file(Path("./scenarios/runs/waterflood/run.h5"))

#=== Execute with Streaming ===
store = bores.ZarrStore(
    store=Path("./scenarios/runs/waterflood/results/waterflood.zarr")
)

stream = bores.StateStream(
    run(),
    store=store,
    batch_size=30,
    background_io=True,
)

with stream:
    stream.consume()  # Run to completion

print("Simulation complete!")
```

---

## Key Features

### 5-Spot Injection Pattern

Optimal well pattern for uniform sweep:

```
WI-1 -------- WI-2
  |            |
  |     P-1    |
  |            |
WI-3 -------- WI-4
```

Four corner injectors displace oil toward central producer.

**Advantages**:

- Uniform sweep from all directions
- Balanced pressure distribution
- Minimal bypass of oil-rich zones
- Standard industry practice

**Pattern geometry**:

- Injector spacing: ~500 ft (depends on grid)
- Producer at geometric center
- Perforate same layers (vertical continuity)

### Water Injection Control

```python
bores.AdaptiveBHPRateControl(
    target_rate=500,  # STB/day per injector (2000 STB/day total)
    target_phase="water",
    bhp_limit=3000,  # Below fracture pressure
)
```

**Key parameters**:

- **Injection rate**: 500 STB/day per well (total 2000 STB/day)
- **BHP limit**: 3000 psi (below fracture gradient)
- **Total liquid injection**: Should exceed total production for pressure maintenance

### Voidage Replacement

Ensure VRR ≥ 1.0 for pressure maintenance:

```
VRR = (Water Injection) / (Oil + Water + Gas Production)
```

For this example:

- Total injection: 2000 STB/day
- Target production: ~200 STB/day oil + 1000 STB/day water = 1200 STB/day liquid
- VRR ≈ 1.67 (over-injection → pressure increase)

---

## Expected Results

### Pressure Response

**Phase 1 (0-100 days)**: Pressure buildup near injectors

**Phase 2 (100-365 days)**: Pressure stabilizes, water front advances

**Phase 3 (1-3 years)**: Breakthrough at producer, increasing water cut

### Water Breakthrough

Water reaches producer after:

- **Early breakthrough** (< 1 year): High permeability channels, vertical fractures
- **Normal breakthrough** (1-2 years): Typical for 500 ft spacing
- **Late breakthrough** (> 2 years): Low permeability, poor connectivity

### Water Cut Evolution

Fraction of water in produced liquids:

```
Water Cut = Water Rate / (Oil Rate + Water Rate)
```

Typical progression:

- **0-1 year**: 0-10% (mostly oil)
- **1-2 years**: 10-50% (water breakthrough)
- **2-3 years**: 50-90% (high water cut)
- **3+ years**: 90-98% (economic limit)

### Incremental Recovery

Waterflooding typically recovers:

- **Primary recovery**: 15-25% OOIP
- **Waterflood incremental**: 15-30% OOIP
- **Total recovery**: 30-55% OOIP

Expected for this example:

- Primary RF: ~20%
- Waterflood RF: ~40% (additional 20%)

---

## Analysis

```python
# Load results
store = bores.ZarrStore("./scenarios/runs/waterflood/results/waterflood.zarr")
states = list(store.load(bores.ModelState))

# Create analyst (provide initial volumes from primary run)
analyst = bores.ModelAnalyst(
    states,
    initial_stoiip=1_500_000,  # STB (from primary run)
)

# Recovery factor
oil_rf = analyst.oil_recovery_factor
print(f"Oil RF: {oil_rf:.2%}")

# Cumulative production
cum_oil = analyst.cumulative_oil_produced
cum_water = analyst.cumulative_water_produced
print(f"Oil produced: {cum_oil:,.0f} STB")
print(f"Water produced: {cum_water:,.0f} STB")

# Water cut history
print("\nWater cut history:")
for step, rates in enumerate(analyst.instantaneous_production_rates):
    if step % 50 == 0:
        wc = rates.water_cut
        time_days = analyst.get_state(step).time / 86400
        print(f"Day {time_days:.0f}: Water cut = {wc:.2%}")

# Cumulative injection
water_inj = analyst.water_injected(0, -1)
print(f"\nWater injected: {water_inj:,.0f} STB")

# Voidage replacement ratio
vrr = analyst.voidage_replacement_ratio(step=-1)
print(f"VRR: {vrr:.2f}")

# Sweep efficiency
sweep = analyst.sweep_efficiency_analysis(step=-1, displacing_phase="water")
print(f"\nVolumetric sweep: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"Displacement efficiency: {sweep.displacement_efficiency:.2%}")
print(f"Areal sweep: {sweep.areal_sweep_efficiency:.2%}")
print(f"Vertical sweep: {sweep.vertical_sweep_efficiency:.2%}")

# Material balance
mb = analyst.material_balance_analysis(step=-1)
print(f"\nFinal pressure: {mb.pressure:.1f} psi")
print(f"Water drive index: {mb.water_drive_index:.2%}")
```

### Water Cut Plot

```python
import matplotlib.pyplot as plt

steps = []
water_cuts = []
for step in range(0, analyst.max_step, 10):
    rates = analyst.instantaneous_production_rates(step)
    steps.append(step)
    water_cuts.append(rates.water_cut * 100)  # Convert to %

plt.plot(steps, water_cuts, linewidth=2)
plt.axhline(y=50, color='r', linestyle='--', label='50% Water Cut')
plt.axhline(y=90, color='orange', linestyle='--', label='90% Water Cut (Economic Limit)')
plt.xlabel("Time Step")
plt.ylabel("Water Cut (%)")
plt.title("Water Cut Evolution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Optimization Strategies

### Injection Rate Adjustment

Reduce water cut by optimizing rates:

```python
# Reduce rate in high water-cut well
well_schedule = bores.WellSchedule()
well_schedule["reduce_rate"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=365)),
    action=bores.update_well_control(target_rate=300),  # Reduce from 500
)
```

### Pattern Balancing

Monitor individual well performance:

```python
# Injection rates by well
for well_name in ["WI-1", "WI-2", "WI-3", "WI-4"]:
    for step, rate in analyst.water_injection_history(interval=50, well_name=well_name):
        print(f"{well_name} Step {step}: {rate:.0f} STB/day")
```

Adjust rates to balance sweep:

- **High rate** in slow-sweep direction
- **Low rate** where breakthrough occurs early

### Conformance Control

Use polymer or gel treatments (future enhancement):

```python
# Polymer injection (concept)
polymer_fluid = bores.InjectedFluid(
    name="Polymer Solution",
    phase=bores.FluidPhase.WATER,
    specific_gravity=1.05,
    molecular_weight=18.015,
    # viscosity_multiplier=5.0,  # Future feature
)
```

---

## Variations

### Line Drive Pattern

Instead of 5-spot, use line drive (parallel injector/producer rows):

```python
# Injectors (row i=3)
injector_1 = bores.injection_well(..., perforating_intervals=[((3, 5, 2), (3, 5, 4))])
injector_2 = bores.injection_well(..., perforating_intervals=[((3, 10, 2), (3, 10, 4))])
injector_3 = bores.injection_well(..., perforating_intervals=[((3, 15, 2), (3, 15, 4))])

# Producers (row i=16)
producer_1 = bores.production_well(..., perforating_intervals=[((16, 5, 2), (16, 5, 4))])
producer_2 = bores.production_well(..., perforating_intervals=[((16, 10, 2), (16, 10, 4))])
producer_3 = bores.production_well(..., perforating_intervals=[((16, 15, 2), (16, 15, 4))])
```

### Inverted 5-Spot

Central injector, corner producers:

```python
# Central injector
injector = bores.injection_well(..., perforating_intervals=[((10, 10, 2), (10, 10, 4))])

# Corner producers
producer_1 = bores.production_well(..., perforating_intervals=[((3, 3, 2), (3, 3, 4))])
producer_2 = bores.production_well(..., perforating_intervals=[((3, 16, 2), (3, 16, 4))])
producer_3 = bores.production_well(..., perforating_intervals=[((16, 3, 2), (16, 3, 4))])
producer_4 = bores.production_well(..., perforating_intervals=[((16, 16, 2), (16, 16, 4))])
```

**Use case**: Low permeability reservoirs, maximize drawdown

### Staggered Line Drive

Offset rows for better sweep:

```
I   P   I   P   I
  P   I   P   I
I   P   I   P   I
```

---

## Next Steps

- [Miscible CO2 EOR](miscible-eor.md) - Enhanced recovery with CO2
- [Immiscible Gas Injection](gas-cap-expansion.md) - Gas injection
- [Analyzing Results](../guides/analyzing-results.md) - Detailed analysis methods
- [Wells and Controls](../guides/wells-and-controls.md) - Well pattern design
