# Miscible CO2 EOR Example

Complete working example of miscible CO2 flooding for enhanced oil recovery.

---

## Objective

Simulate miscible CO2 flooding with Todd-Longstaff model:

- 2 CO2 gas injectors (high rate)
- 1 producer (shut-in initially, activated after 100 days)
- Miscible displacement (pressure above MMP)
- Todd-Longstaff mixing model
- 2 years + 100 days total simulation

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

#=== CO2 Injection Wells ===
injection_clamp = bores.InjectionClamp()
control = bores.AdaptiveBHPRateControl(
    target_rate=1_000_000,  # SCF/day (high rate for miscible flooding)
    target_phase="gas",
    bhp_limit=3500,  # psi maximum
    clamp=injection_clamp,
)

gas_injector_1 = bores.injection_well(
    well_name="GI-1",
    perforating_intervals=[((16, 3, 1), (16, 3, 3))],  # Corner location
    radius=0.3542,  # 8.5" wellbore
    control=control,
    injected_fluid=bores.InjectedFluid(
        name="CO2",
        phase=bores.FluidPhase.GAS,
        specific_gravity=1.52,  # Relative to air (CO2 is dense)
        molecular_weight=44.01,  # g/mol
        viscosity=0.05,  # cP at reservoir conditions (from lab or EOS)
        density=35.0,  # lbm/ft³ at reservoir P&T (NOT correlation value!)
        minimum_miscibility_pressure=2200.0,  # psi
        todd_longstaff_omega=0.67,  # Moderate mixing parameter for CO2
        is_miscible=True,  # Enable Todd-Longstaff model
        concentration=1.0,  # 100% CO2
    ),
    is_active=True,
    skin_factor=2.0,
)

# Duplicate for second injector
gas_injector_2 = gas_injector_1.duplicate(
    name="GI-2",
    perforating_intervals=[((16, 16, 1), (16, 16, 3))]  # Opposite corner
)
injectors = [gas_injector_1, gas_injector_2]

#=== Production Well ===
production_clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-150,  # STB/day
        target_phase="oil",
        bhp_limit=1200,  # psi
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-500,  # MCF/day
        target_phase="gas",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-10,  # STB/day
        target_phase="water",
        bhp_limit=1200,
        clamp=production_clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((14, 10, 3), (14, 10, 4))],
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
            specific_gravity=0.65,  # Mixture of reservoir gas and CO2
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
    is_active=False,  # Start shut-in
)

#=== Well Schedule: Activate Producer at 100 days ===
well_schedule = bores.WellSchedule()
well_schedule["open_well"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=100)),
    action=bores.update_well(is_active=True),
)

well_schedules = bores.WellSchedules()
well_schedules[producer.name] = well_schedule

producers = [producer]
wells = bores.wells_(injectors=injectors, producers=producers)

#=== Timer ===
timer = bores.Timer(
    initial_step_size=bores.Time(hours=30.0),
    max_step_size=bores.Time(days=5.0),
    min_step_size=bores.Time(minutes=10),
    simulation_time=bores.Time(days=(bores.c.DAYS_PER_YEAR * 2) + 100),  # 2 years + 100 days
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
    max_rejects=20,
)

#=== Update Config ===
run.config = run.config.with_updates(
    wells=wells,
    well_schedules=well_schedules,
    timer=timer,
    miscibility_model="todd_longstaff",  # Enable Todd-Longstaff model
    constants=bores.Constants(),
)

# Save run
run.to_file(Path("./scenarios/runs/co2_injection/run.h5"))

#=== Execute with Streaming ===
store = bores.ZarrStore(
    store=Path("./scenarios/runs/co2_injection/results/co2_injection.zarr")
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

### CO2 Injection with Density Override

CO2 properties deviate from standard correlations, so specify explicitly:

```python
bores.InjectedFluid(
    name="CO2",
    phase=bores.FluidPhase.GAS,
    specific_gravity=1.52,      # CO2 is 1.52× air density
    molecular_weight=44.01,
    viscosity=0.05,             # cP (from lab or EOS, NOT correlation)
    density=35.0,               # lbm/ft³ (from lab or EOS, NOT correlation)
    minimum_miscibility_pressure=2200.0,  # psi
    todd_longstaff_omega=0.67,            # Moderate mixing for CO2
    is_miscible=True,
    concentration=1.0,                    # 100% CO2
)
```

!!! warning "CO2 Property Overrides"
    Standard correlations give incorrect CO2 properties:

    - Correlation density: ~3-7 lbm/ft³
    - Actual density: ~35 lbm/ft³ at reservoir conditions

    - Correlation viscosity: ~0.01-0.02 cP
    - Actual viscosity: ~0.05 cP

    **Always specify `density` and `viscosity` explicitly for CO2!**

### Todd-Longstaff Model Configuration

Must enable in config:

```python
run.config = run.config.with_updates(
    miscibility_model="todd_longstaff",
)
```

### High Injection Rate

Miscible flooding requires high rates to maintain pressure above MMP:

```python
target_rate=1_000_000  # SCF/day (20× higher than immiscible)
bhp_limit=3500         # psi (above MMP of 2200 psi)
```

---

## Todd-Longstaff Parameters

### Omega (Mixing Parameter)

Controls degree of mixing between solvent and oil:

```python
todd_longstaff_omega=0.67  # Typical for CO2
```

- **0.0**: Fully segregated flow (no mixing)
- **0.33**: Partial mixing (CH4, light gases)
- **0.67**: Moderate mixing (CO2, typical value)
- **0.80**: High mixing (enriched gas, NGL)
- **1.0**: Fully mixed (homogeneous)

Effective viscosity formula:

```
μ_eff = μ_mix^ω × μ_seg^(1-ω)
```

where:
- `μ_mix` = Volume-weighted average viscosity (full mixing)
- `μ_seg` = Segregated viscosity (no mixing)

### Minimum Miscibility Pressure (MMP)

Pressure threshold for miscibility:

```python
minimum_miscibility_pressure=2200.0  # psi
```

- **P > MMP**: First-contact or multi-contact miscible (efficient displacement)
- **P ≈ MMP**: Transition zone (partial miscibility)
- **P < MMP**: Immiscible (poor sweep efficiency)

CO2 MMP typical range: **1500-2500 psi** depending on:

- Oil composition (lighter oils have lower MMP)
- Temperature (higher temp increases MMP)
- Reservoir depth

Correlations for estimating MMP:

- **Alston et al.**: `MMP = f(T, C7+ MW, volatile fraction)`
- **Yellig-Metcalfe**: `MMP = f(T, C5+ MW)`
- **Lab measurements**: Slim-tube or rising-bubble tests

---

## Expected Results

### Pressure Maintenance

CO2 injection maintains reservoir pressure above MMP:

- Primary depletion: P drops to ~2100 psi
- After CO2 injection: P rises to 2500-3000 psi (above MMP)

### Miscible Displacement

Near-piston-like displacement with:

- High displacement efficiency (70-90%)
- Low residual oil saturation in swept zone
- Reduced viscosity (oil-CO2 mixing)

### Oil Swelling

CO2 dissolves in oil, causing:

- Volume expansion (5-15%)
- Viscosity reduction (2-5×)
- Improved mobility

### Incremental Recovery

Miscible CO2 EOR recovers an additional:

- **15-25%** of OOIP (over primary + secondary)
- **Total RF**: 50-75% (vs 20-35% for waterflooding)

---

## Analysis

```python
# Load results
store = bores.ZarrStore("./scenarios/runs/co2_injection/results/co2_injection.zarr")
states = list(store.load(bores.ModelState))

# Create analyst (provide initial volumes from primary run)
analyst = bores.ModelAnalyst(
    states,
    initial_stoiip=1_500_000,  # STB (from primary run)
    initial_stgiip=8_000_000_000,  # SCF
)

# Recovery factor (incremental over primary)
oil_rf = analyst.oil_recovery_factor
print(f"Total Oil RF: {oil_rf:.2%}")

# Cumulative injection
gas_inj = analyst.gas_injected(0, -1)
print(f"CO2 injected: {gas_inj/1e9:.2f} BSCF")

# Sweep efficiency
sweep = analyst.sweep_efficiency_analysis(step=-1, displacing_phase="gas")
print(f"Volumetric sweep: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"Displacement efficiency: {sweep.displacement_efficiency:.2%}")
print(f"Recovery efficiency: {sweep.recovery_efficiency:.2%}")

# Voidage replacement ratio
vrr = analyst.voidage_replacement_ratio(step=-1)
print(f"VRR: {vrr:.2f}")  # Should be > 1.0 for pressure maintenance

# Material balance
mb = analyst.material_balance_analysis(step=-1)
print(f"Final pressure: {mb.pressure:.1f} psi")
print(f"Solution gas drive: {mb.solution_gas_drive_index:.2%}")
print(f"Gas injection drive: {mb.gas_cap_drive_index:.2%}")

# Production history
print("\nProduction history:")
for step, rate in analyst.oil_production_history(interval=20):
    state = analyst.get_state(step)
    time_days = state.time / 86400
    print(f"Day {time_days:.0f}: {rate:.1f} STB/day")
```

---

## Comparison: Miscible vs Immiscible

| Parameter | Immiscible (CH4) | Miscible (CO2) |
|-----------|------------------|----------------|
| Injection rate | 50,000 SCF/day | 1,000,000 SCF/day |
| BHP limit | 1500 psi | 3500 psi |
| MMP | 4000 psi (high) | 2200 psi (achievable) |
| Omega | 0.33 | 0.67 |
| Reservoir pressure | Below MMP | Above MMP |
| Displacement | Immiscible (poor) | Miscible (efficient) |
| Oil RF (total) | 25-35% | 50-75% |
| Incremental RF | 10-15% | 20-40% |
| Sweep efficiency | 40-60% | 70-90% |
| Gas breakthrough | Early | Delayed |

---

## Variations

### Higher Pressure Limit

Ensure miscibility throughout:

```python
bhp_limit=4000  # psi (well above MMP)
```

### WAG (Water-Alternating-Gas)

Improve sweep efficiency:

```python
# Add water injector
water_injector = bores.injection_well(
    well_name="WI-1",
    perforating_intervals=[...],
    radius=0.3542,
    control=bores.AdaptiveBHPRateControl(
        target_rate=500,  # STB/day
        target_phase="water",
        bhp_limit=3500,
        clamp=bores.InjectionClamp(),
    ),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.05,
        molecular_weight=18.015,
    ),
    is_active=True,
)

# Schedule alternating injection (CO2 → Water → CO2 → ...)
# Use well schedules to toggle wells on/off
```

### Multiple Producers

5-spot or line-drive pattern:

```python
producer_2 = producer.duplicate(name="P-2", perforating_intervals=[...])
producer_3 = producer.duplicate(name="P-3", perforating_intervals=[...])
producers = [producer, producer_2, producer_3]
```

---

## Next Steps

- [Immiscible Gas Injection](gas-cap-expansion.md) - CH4 injection comparison
- [Waterflood Pattern](waterflood-pattern.md) - Waterflooding example
- [Todd-Longstaff Model](../guides/rock-fluid-properties.md#todd-longstaff-miscibility) - Theory details
- [Analyzing Results](../guides/analyzing-results.md) - Detailed analysis methods
