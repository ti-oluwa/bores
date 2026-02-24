# Immiscible Gas Injection Example

Complete working example of immiscible methane gas injection with Todd-Longstaff model.

---

## Objective

Simulate gas injection for pressure maintenance and enhanced recovery:

- 2 gas injectors (CH4)
- 1 producer (initially shut-in)
- Well scheduling (activate producer after 100 days)
- Miscible model enabled but low MMP (partial miscibility)
- 5 years of production

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

# Load depleted model
run = bores.Run.from_files(
    model_path=Path("./scenarios/runs/primary_depletion/results/model.h5"),
    config_path=Path("./scenarios/runs/setup/config.yaml"),
    pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
)

#=== Gas Injection Wells ===
injection_clamp = bores.InjectionClamp()
control = bores.AdaptiveBHPRateControl(
    target_rate=50000,  # SCF/day
    target_phase="gas",
    bhp_limit=1500,  # psi maximum
    clamp=injection_clamp,
)

gas_injector_1 = bores.injection_well(
    well_name="GI-1",
    perforating_intervals=[((16, 3, 1), (16, 3, 3))],  # Corner location
    radius=0.3542,  # 8.5" wellbore
    control=control,
    injected_fluid=bores.InjectedFluid(
        name="Methane",
        phase=bores.FluidPhase.GAS,
        specific_gravity=0.65,  # Relative to air
        molecular_weight=16.04,  # g/mol
        minimum_miscibility_pressure=4000.0,  # psi (high, so mostly immiscible)
        todd_longstaff_omega=0.33,  # Mixing parameter
        is_miscible=True,  # Enable Todd-Longstaff model
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
    max_step_size=bores.Time(days=10.0),
    min_step_size=bores.Time(hours=6.0),
    simulation_time=bores.Time(days=(bores.c.DAYS_PER_YEAR * 5) + 100),  # 5 years + 100 days
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
)

#=== Update Config ===
run.config = run.config.with_updates(
    wells=wells,
    well_schedules=well_schedules,
    timer=timer,
    miscibility_model="todd_longstaff",  # Enable Todd-Longstaff model
)

# Save run
run.to_file(Path("./scenarios/runs/ch4_injection/run.h5"))

#=== Execute with Streaming ===
store = bores.ZarrStore(
    store=Path("./scenarios/runs/ch4_injection/results/ch4_injection.zarr")
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

**`InjectedFluid` with Miscibility Parameters**:

```python
bores.InjectedFluid(
    name="Methane",
    phase=bores.FluidPhase.GAS,
    specific_gravity=0.65,
    molecular_weight=16.04,
    minimum_miscibility_pressure=4000.0,  # High MMP = mostly immiscible
    todd_longstaff_omega=0.33,             # Low omega = more segregated flow
    is_miscible=True,                      # Enable Todd-Longstaff
)
```

**Well Scheduling**:

```python
well_schedule = bores.WellSchedule()
well_schedule["open_well"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=100)),
    action=bores.update_well(is_active=True),
)

well_schedules = bores.WellSchedules()
well_schedules[producer.name] = well_schedule
```

**Todd-Longstaff Model in Config**:

```python
run.config = run.config.with_updates(
    miscibility_model="todd_longstaff",  # Must enable explicitly
)
```

---

## Todd-Longstaff Parameters

**omega** (mixing parameter):

- **0.0**: Fully segregated flow (no mixing)
- **0.33**: Partial mixing (used here for CH4)
- **0.67**: Moderate mixing (typical for CO2)
- **1.0**: Fully mixed

**Minimum Miscibility Pressure (MMP)**:

- Above MMP: Miscible displacement
- Below MMP: Immiscible with some mixing
- CH4 MMP is typically 3000-5000 psi (high)
- Reservoir pressure often below MMP = mostly immiscible

---

## Expected Results

**Pressure Maintenance**: Injection prevents further pressure decline

**Gas Breakthrough**: Injected gas eventually reaches producer

**Incremental Recovery**: 10-20% additional oil recovery vs primary depletion

---

## Analysis

```python
# Load results
store = bores.ZarrStore("./scenarios/runs/ch4_injection/results/ch4_injection.zarr")
states = list(store)
analyst = bores.ModelAnalyst(states)

# Gas injection vs production
for step, gas_inj in analyst.gas_injection_history(interval=20):
    gas_prod = analyst.gas_injected(0, step)
    print(f"Step {step}: Injected {gas_inj/1e6:.1f} MMSCF")

# Sweep efficiency
sweep = analyst.sweep_efficiency_analysis(step=-1, displacing_phase="gas")
print(f"Volumetric sweep: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"Displacement efficiency: {sweep.displacement_efficiency:.2%}")

# Voidage replacement ratio
vrr = analyst.voidage_replacement_ratio(step=-1)
print(f"VRR: {vrr:.2f}")  # > 1.0 = pressure maintenance
```

---

## Variations

**Higher Injection Rate**:

```python
target_rate=100000  # SCF/day instead of 50000
```

**More Injectors** (4-spot pattern):

```python
gas_injector_3 = gas_injector_1.duplicate(name="GI-3", perforating_intervals=[...])
gas_injector_4 = gas_injector_1.duplicate(name="GI-4", perforating_intervals=[...])
injectors = [gas_injector_1, gas_injector_2, gas_injector_3, gas_injector_4]
```

**Immediate Production** (no delay):

```python
is_active=True  # Don't use well schedule
```

---

## Next Steps

- [Miscible CO2 Flooding](miscible-eor.md) - Miscible displacement
- [Well Schedules Guide](../guides/wells-and-controls.md#well-scheduling) - Advanced scheduling
- [Todd-Longstaff Model](../guides/rock-fluid-properties.md#todd-longstaff-miscibility) - Theory
