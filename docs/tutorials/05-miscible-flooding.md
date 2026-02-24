# Tutorial 5: Miscible Gas Flooding

Learn how to implement CO2 miscible flooding for enhanced oil recovery (EOR).

---

## What You'll Learn

In this tutorial, you will learn how to:

- Design a miscible CO2 flooding project by assessing reservoir suitability (estimating minimum miscibility pressure and checking if it's achievable at reservoir conditions)
- Configure CO2 fluid properties correctly by overriding standard gas correlations with accurate density and viscosity values for CO2 at reservoir conditions
- Set up high-pressure injection operations that maintain reservoir pressure above MMP to ensure miscible displacement
- Monitor the physical effects of CO2 miscibility including oil swelling, viscosity reduction, and enhanced displacement efficiency
- Achieve recovery factors exceeding 50% of original oil in place (OOIP) through miscible EOR, significantly higher than primary (15-25%) or waterflood (30-40%) recovery

---

## Prerequisites

Complete [Tutorial 4: Gas Injection](04-gas-injection.md) first.

---

## Overview

**Miscible flooding** is the most effective EOR technique:

- **Mechanism**: CO2 dissolves in oil, reducing viscosity and swelling volume
- **MMP**: Pressure must exceed Minimum Miscibility Pressure
- **Recovery**: 40-70% of OOIP (vs 20-30% for waterflooding)
- **Cost**: Higher than waterflooding, but better economics for high-value oil

**Key difference from immiscible**:

| Parameter | Immiscible (CH4) | Miscible (CO2) |
| --------- | ---------------- | -------------- |
| Pressure | Below MMP | Above MMP |
| Displacement | Poor | Excellent |
| Oil Recovery | +10-20% | +20-40% |
| Mixing | Limited | Extensive |

---

## Step 1: Check Reservoir Suitability

Not all reservoirs are suitable for miscible CO2:

```python
import bores
from pathlib import Path

# Load depleted model
run = bores.Run.from_files(
    model_path=Path("./primary_depletion/model.h5"),
    config_path=Path("./primary_depletion/config.yaml"),
)

# Check conditions
avg_pressure = run.config.model.fluid_properties.pressure_grid.mean()
avg_depth = run.config.model.depth.mean() if run.config.model.depth is not None else 5000
temperature = run.config.model.temperature

print("Reservoir Conditions:")
print(f"  Current pressure: {avg_pressure:.1f} psi")
print(f"  Depth: {avg_depth:.1f} ft")
print(f"  Temperature: {temperature:.1f} °F")

# Estimate MMP (correlation for light-medium oil)
# MMP increases with temperature, decreases with light ends
oil_api = 35.0  # Assume medium oil
mmp_estimate = 1200 + 20 * (temperature - 100)  # Simplified correlation
print(f"\nEstimated MMP: {mmp_estimate:.0f} psi")

# Check feasibility
if mmp_estimate < 3000:
    print("GOOD: Reservoir is a good candidate for CO2 miscible flooding")
    target_pressure = mmp_estimate * 1.2  # 20% above MMP
    print(f"   Target injection pressure: {target_pressure:.0f} psi")
else:
    print("WARNING: MMP may be too high - consider enriched gas or thermal EOR")
```

!!! info "MMP Correlations"
    Actual MMP should be determined by:

    - Slim-tube experiments
    - PVT cell tests
    - Commercial correlations (Alston, Yellig-Metcalfe)

    Rule of thumb: MMP = 1000-1500 psi for light oil, 2000-3000 psi for medium oil

---

## Step 2: Configure CO2 Properties

**Critical**: CO2 properties deviate from standard correlations:

```python
# Standard correlation (WRONG for CO2!)
# Would give density ~3-7 lbm/ft³
# Actual CO2 density at reservoir conditions: ~35 lbm/ft³

# Correct CO2 properties
co2_density = 35.0  # lbm/ft³ at 2500 psi, 180°F (from tables or EOS)
co2_viscosity = 0.05  # cP at reservoir conditions
co2_sg = 1.52  # Specific gravity (relative to air)
co2_mw = 44.01  # Molecular weight

print("CO2 Properties (at reservoir conditions):")
print(f"  Density: {co2_density} lbm/ft³")
print(f"  Viscosity: {co2_viscosity} cP")
print(f"  Specific gravity: {co2_sg} (air=1)")
```

---

## Step 3: Create High-Pressure CO2 Injectors

```python
# Grid dimensions
nx, ny, nz = run.config.model.grid_shape

# Injector locations (updip, corners for 4-spot)
injector_locations = [
    (3, 3),
    (3, nx-4),
    (nx-4, 3),
    (nx-4, nx-4),
]

# High-rate injection control
injection_clamp = bores.InjectionClamp()
control = bores.AdaptiveBHPRateControl(
    target_rate=1_000_000,  # SCF/day (high rate for miscible)
    target_phase="gas",
    bhp_limit=3500,  # psi (well above MMP)
    clamp=injection_clamp,
)

# Create CO2 injectors
injectors = []
for idx, (i, j) in enumerate(injector_locations, start=1):
    injector = bores.injection_well(
        well_name=f"CO2-{idx}",
        perforating_intervals=[((i, j, 1), (i, j, 3))],  # Top layers
        radius=0.3542,
        control=control,
        injected_fluid=bores.InjectedFluid(
            name="CO2",
            phase=bores.FluidPhase.GAS,
            specific_gravity=co2_sg,
            molecular_weight=co2_mw,
            density=co2_density,  # CRITICAL: Override correlation
            viscosity=co2_viscosity,  # CRITICAL: Override correlation
            minimum_miscibility_pressure=mmp_estimate,
            todd_longstaff_omega=0.67,  # Moderate mixing for CO2
            is_miscible=True,
            concentration=1.0,  # 100% CO2
        ),
        is_active=True,
        skin_factor=1.0,
    )
    injectors.append(injector)

print(f"Created {len(injectors)} CO2 injection wells")
```

!!! warning "Property Overrides"
    **Always specify `density` and `viscosity` for CO2!**

    Without overrides:
    - Correlation density: ~5 lbm/ft³ (WRONG)
    - Actual density: ~35 lbm/ft³ (CORRECT)
    - Error: ~600% (UNACCEPTABLE)

---

## Step 4: Update Producer for High GOR

```python
# Central producer
producer_loc = (nx//2, ny//2)

production_clamp = bores.ProductionClamp()
control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-200,  # STB/day
        target_phase="oil",
        bhp_limit=1500,  # Higher BHP for miscible (prevent gas breakout)
        clamp=production_clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-1000,  # MCF/day (very high for CO2 breakthrough)
        target_phase="gas",
        bhp_limit=1500,
        clamp=production_clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-30,  # STB/day
        target_phase="water",
        bhp_limit=1500,
        clamp=production_clamp,
    ),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((producer_loc[0], producer_loc[1], 2),
                            (producer_loc[0], producer_loc[1], 5))],
    radius=0.3542,
    control=control,
    produced_fluids=(
        bores.ProducedFluid(name="Oil", phase=bores.FluidPhase.OIL,
                           specific_gravity=0.85, molecular_weight=180.0),
        bores.ProducedFluid(name="Gas", phase=bores.FluidPhase.GAS,
                           specific_gravity=1.52, molecular_weight=44.01),  # CO2 properties
        bores.ProducedFluid(name="Water", phase=bores.FluidPhase.WATER,
                           specific_gravity=1.05, molecular_weight=18.015),
    ),
    skin_factor=2.5,
)

wells = bores.wells_(injectors=injectors, producers=[producer])
```

---

## Step 5: Configure Simulation

```python
# Timer for 2-year miscible flood
timer = bores.Timer(
    initial_step_size=bores.Time(hours=24.0),
    max_step_size=bores.Time(days=7.0),
    min_step_size=bores.Time(hours=12.0),
    simulation_time=bores.Time(days=2 * bores.c.DAYS_PER_YEAR),
    max_cfl_number=0.9,
)

# Enable Todd-Longstaff
run.config = run.config.with_updates(
    wells=wells,
    timer=timer,
    miscibility_model="todd_longstaff",
)

# Save
setup_dir = Path("./tutorial_05_co2_flood")
setup_dir.mkdir(exist_ok=True)
run.to_file(setup_dir / "run.h5")
```

---

## Step 6: Execute Simulation

```python
store = bores.ZarrStore(setup_dir / "results.zarr")
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=20,
    background_io=True,
)

print("Starting CO2 miscible flooding...")
with stream:
    for state in stream:
        if state.step % 50 == 0:
            time_days = state.time / 86400
            avg_p = state.model.fluid_properties.pressure_grid.mean()
            print(f"Day {time_days:4.0f}: P = {avg_p:.1f} psi")

print("Simulation complete!")
```

---

## Step 7: Analyze Results

### Recovery Factor

```python
import matplotlib.pyplot as plt

states = list(store.load(bores.ModelState))
analyst = bores.ModelAnalyst(states, initial_stoiip=1_500_000)  # From primary run

# Final recovery
print(f"\nRecovery Factors:")
print(f"  Total Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"  Cumulative oil: {analyst.cumulative_oil_produced:,.0f} STB")

# Compare to primary/waterflood
primary_rf = 0.20  # Assume 20% from primary
waterflood_rf = 0.40  # Assume 40% from waterflood
incremental_co2 = analyst.oil_recovery_factor - waterflood_rf

print(f"\nIncremental Recovery:")
print(f"  Over primary: +{(analyst.oil_recovery_factor - primary_rf)*100:.1f}%")
print(f"  Over waterflood: +{incremental_co2*100:.1f}%")
```

### Sweep Efficiency

```python
sweep = analyst.sweep_efficiency_analysis(step=-1, displacing_phase="gas")
print(f"\nSweep Efficiency:")
print(f"  Volumetric: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"  Displacement: {sweep.displacement_efficiency:.2%}")
print(f"  Recovery efficiency: {sweep.recovery_efficiency:.2%}")

# Miscible flooding should have high displacement efficiency (>80%)
if sweep.displacement_efficiency > 0.8:
    print("  EXCELLENT: High displacement efficiency indicates miscible displacement")
else:
    print("  WARNING: Low displacement efficiency - verify pressure maintained above MMP")
```

### CO2 Utilization

```python
co2_injected = analyst.gas_injected(0, -1) / 1e6  # MMSCF
incremental_oil = (analyst.oil_recovery_factor - waterflood_rf) * 1_500_000  # STB

co2_utilization = co2_injected / incremental_oil if incremental_oil > 0 else float('inf')

print(f"\nCO2 Utilization:")
print(f"  CO2 injected: {co2_injected:.1f} MMSCF")
print(f"  Incremental oil: {incremental_oil:,.0f} STB")
print(f"  CO2/oil ratio: {co2_utilization:.2f} MSCF/STB")

# Typical: 4-10 MSCF/STB for miscible flooding
if co2_utilization < 10:
    print("  GOOD: Efficient CO2 utilization within typical range")
else:
    print("  WARNING: High CO2 usage - check for channeling or poor sweep")
```

### Pressure Maintenance

```python
times = []
pressures = []

for step in range(0, analyst.max_step, 10):
    state = analyst.get_state(step)
    times.append(state.time / 86400)
    pressures.append(state.model.fluid_properties.pressure_grid.mean())

plt.figure(figsize=(10, 6))
plt.plot(times, pressures, 'b-', linewidth=2, label='Reservoir Pressure')
plt.axhline(y=mmp_estimate, color='r', linestyle='--', linewidth=2,
            label=f'MMP ({mmp_estimate:.0f} psi)')
plt.axhline(y=target_pressure, color='g', linestyle=':', linewidth=2,
            label=f'Target ({target_pressure:.0f} psi)')
plt.xlabel('Time (days)')
plt.ylabel('Pressure (psi)')
plt.title('Pressure History - CO2 Miscible Flooding')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check if pressure maintained above MMP
avg_pressure_final = pressures[-1]
if avg_pressure_final > mmp_estimate:
    print(f"GOOD: Pressure maintained above MMP ({avg_pressure_final:.0f} > {mmp_estimate:.0f} psi)")
else:
    print(f"WARNING: Pressure below MMP ({avg_pressure_final:.0f} < {mmp_estimate:.0f} psi) - not fully miscible!")
```

---

## Optimization Tips

### 1. Maximize Miscibility

Ensure pressure > MMP throughout flood:

```python
# Increase injection BHP limit
bhp_limit=4000  # Higher pressure

# Or increase injection rate
target_rate=1_500_000  # SCF/day
```

### 2. Control CO2 Cycling

High CO2 recycling reduces economics:

```python
# Use WAG (Water-Alternating-Gas)
# Or reduce injection rate after breakthrough
# Or drill infill producers
```

### 3. Improve Vertical Sweep

CO2 overrides due to low density:

```python
# Use horizontal wells
# Inject in lower layers
# Add conformance control
```

---

## Economic Analysis

```python
# Simplified economics
oil_price = 70.0  # $/bbl
co2_cost = 25.0  # $/MSCF
opex = 15.0  # $/bbl oil

# Revenue
oil_revenue = incremental_oil * oil_price

# Costs
co2_cost_total = co2_injected * 1000 * co2_cost / 1e6  # Convert MMSCF to $
opex_total = incremental_oil * opex

# Profit
profit = oil_revenue - co2_cost_total - opex_total

print(f"\nSimplified Economics:")
print(f"  Oil revenue: ${oil_revenue/1e6:.1f} MM")
print(f"  CO2 cost: ${co2_cost_total/1e6:.1f} MM")
print(f"  OPEX: ${opex_total/1e6:.1f} MM")
print(f"  Profit: ${profit/1e6:.1f} MM")
print(f"  NPV per barrel: ${profit/incremental_oil:.2f}/bbl")
```

---

## What You Learned

In this tutorial, you successfully:

- Designed a complete CO2 miscible flooding project including reservoir suitability assessment and well pattern design
- Correctly configured CO2 fluid properties with density and viscosity overrides (critical for accurate simulation since CO2 deviates significantly from standard gas correlations)
- Set up high-pressure injection operations to maintain reservoir pressure above minimum miscibility pressure (MMP) throughout the flood
- Monitored pressure maintenance and verified miscibility conditions by tracking average reservoir pressure relative to MMP
- Analyzed sweep efficiency metrics and CO2 utilization factor to assess displacement effectiveness and project economics
- Calculated simplified economic metrics including revenue, costs, and net present value per incremental barrel recovered

---

## Congratulations!

You have successfully completed all 5 BORES tutorials and learned:

1. Basic simulation setup (1D primary depletion)
2. Building 3D heterogeneous reservoir models with layered properties and structural dip
3. Waterflooding operations with injection/production balance and sweep analysis
4. Immiscible gas injection for pressure maintenance
5. Miscible CO2 enhanced oil recovery (EOR) with Todd-Longstaff model

---

## Next Steps

- **[User Guides](../guides/index.md)** - Deep dive into each module
- **[Examples](../examples/index.md)** - Complete production scenarios
- **[Best Practices](../best-practices/index.md)** - Optimization techniques
- **[API Reference](../reference/api.md)** - Detailed API documentation
