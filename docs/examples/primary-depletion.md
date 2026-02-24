# Primary Depletion Example

Complete working example of natural depletion with solution gas drive.

---

## Objective

Simulate a reservoir under primary depletion (no injection) with:

- Single production well
- Multi-phase rate control (oil, gas, water)
- Bottom-hole pressure limit
- 2 years of production
- State streaming to Zarr storage

---

## Complete Code

```python
import logging
from pathlib import Path
import numpy as np
import bores

# Setup logging
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

# Load model from setup (assumes you've run setup.py first)
run = bores.Run.from_files(
    model_path=Path("./scenarios/runs/stabilization/results/model.h5"),
    config_path=Path("./scenarios/runs/setup/config.yaml"),
    pvt_table_path=Path("./scenarios/runs/setup/pvt.h5"),
)

# Production well with multi-phase control
clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-100,  # STB/day
        target_phase="oil",
        bhp_limit=800,  # psi minimum
        clamp=clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-100,  # MCF/day (thousand cubic feet)
        target_phase="gas",
        bhp_limit=800,
        clamp=clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-10,  # STB/day
        target_phase="water",
        bhp_limit=800,
        clamp=clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((14, 10, 3), (14, 10, 4))],  # 2 perforated layers
    radius=0.3542,  # ft (8.5" wellbore)
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

wells = bores.wells_(injectors=None, producers=[producer])

# Timer for 2 years
timer = bores.Timer(
    initial_step_size=bores.Time(hours=20),
    max_step_size=bores.Time(days=5),
    min_step_size=bores.Time(minutes=10.0),
    simulation_time=bores.Time(days=2 * bores.c.DAYS_PER_YEAR),  # 730 days
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
    max_rejects=20,
)

# Update config
run.config = run.config.with_updates(wells=wells, timer=timer)

# Save run
run.to_file(Path("./scenarios/runs/primary_depletion/run.h5"))

# Create storage
store = bores.ZarrStore(
    store=Path("./scenarios/runs/primary_depletion/results/primary_depletion.zarr"),
)

# Execute with streaming
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=30,  # Write every 30 states
    background_io=True,
)

with stream:
    last_state = stream.last()

# Save final model state
last_state.model.to_file(
    Path("./scenarios/runs/primary_depletion/results/model.h5")
)

print(f"Simulation complete. Final pressure: {last_state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

---

## Key Features

**Multi-Phase Rate Control**: Each phase has independent target rate and BHP limit

```python
bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(...),
    gas_control=bores.AdaptiveBHPRateControl(...),
    water_control=bores.AdaptiveBHPRateControl(...),
)
```

**Production Clamp**: Prevents injection during production

```python
clamp = bores.ProductionClamp()
# Returns 0 if rate > 0 (injection)
```

**ProducedFluid Specification**: Required for production wells

```python
produced_fluids=(
    bores.ProducedFluid(name="Oil", phase=bores.FluidPhase.OIL, ...),
    bores.ProducedFluid(name="Gas", phase=bores.FluidPhase.GAS, ...),
    bores.ProducedFluid(name="Water", phase=bores.FluidPhase.WATER, ...),
)
```

**State Streaming**: Efficient storage of large results

```python
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=30,  # Batch writes for efficiency
    background_io=True,  # Asynchronous I/O
)

with stream:
    last_state = stream.last()  # Only keeps last state in memory
```

---

## Expected Results

**Pressure Decline**: ~3100 psi → ~2100 psi over 2 years

**Gas Evolution**: Free gas saturation increases as pressure drops below bubble point

**Recovery Factor**: Typically 5-15% for solution gas drive

---

## Analysis

After running, analyze results:

```python
# Load states
store = bores.ZarrStore("./scenarios/runs/primary_depletion/results/primary_depletion.zarr")
states = list(store)

# Create analyst
analyst = bores.ModelAnalyst(states)

# Recovery factors
print(f"Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"Gas RF: {analyst.gas_recovery_factor:.2%}")

# Production history
for step, oil_rate in analyst.oil_production_history(interval=10):
    print(f"Step {step}: Oil rate = {oil_rate:.1f} STB/day")

# Decline curve
decline = analyst.decline_curve_analysis(phase="oil", decline_type="exponential")
print(f"Decline rate: {decline.decline_rate_per_timestep:.4f}")
print(f"R²: {decline.r_squared:.3f}")
```

---

## Variations

**Higher Production Rate**:
```python
target_rate=-200  # Instead of -100
```

**Longer Simulation**:
```python
simulation_time=bores.Time(days=5 * bores.c.DAYS_PER_YEAR)  # 5 years
```

**Lower BHP Limit** (more aggressive):
```python
bhp_limit=500  # Instead of 800 psi
```

---

## Next Steps

- [Immiscible Gas Injection](gas-cap-expansion.md) - Add pressure support
- [Waterflood](waterflood-pattern.md) - Waterflooding example
- [Analysis Guide](../guides/analyzing-results.md) - Detailed analysis methods
