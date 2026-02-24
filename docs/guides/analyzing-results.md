# Analyzing Results

Comprehensive guide to analyzing reservoir simulation results with BORES.

---

## Overview

The `bores.ModelAnalyst` class provides extensive analysis capabilities for evaluating reservoir performance over simulation states. It supports:

- Volumetric calculations (oil/gas/water in place)
- Recovery factors and cumulative production
- Production/injection rate histories
- Material balance analysis
- Decline curve analysis
- Sweep efficiency analysis
- Well productivity analysis
- Front tracking

---

## Creating an Analyst

Load states from a store and create an analyst:

```python
import bores

# Load states from storage
store = bores.ZarrStore("./results/simulation.zarr")
states = list(store.load(bores.ModelState))

# Create analyst
analyst = bores.ModelAnalyst(states)
```

For EOR or continuation runs where initial state (step 0) is not available:

```python
# Provide initial volumes explicitly
analyst = bores.ModelAnalyst(
    states,
    initial_stoiip=1_500_000,  # STB
    initial_stgiip=8_000_000_000,  # SCF
    initial_stwiip=500_000,  # STB
)
```

!!! tip "State Loading"
    The analyst converts the iterable to an internal dictionary indexed by step number, so you can pass any iterable (list, generator, or store loader).

---

## Volumetric Analysis

### Initial Volumes

Get initial oil, gas, and water in place:

```python
# Stock tank oil initially in place (STOIIP)
stoiip = analyst.stoiip  # STB
print(f"STOIIP: {stoiip:,.0f} STB")

# Stock tank gas initially in place (STGIIP)
stgiip = analyst.stgiip  # SCF
print(f"STGIIP: {stgiip:,.0f} SCF")

# Stock tank water initially in place
stwiip = analyst.stock_tank_water_initially_in_place  # STB
```

Alternative names:

```python
stoiip = analyst.stock_tank_oil_initially_in_place
stgiip = analyst.stock_tank_gas_initially_in_place
```

### Volumes at Any Step

Get volumes at a specific timestep:

```python
# Oil in place at step 100
oil_step_100 = analyst.oil_in_place(step=100)  # STB

# Gas in place at final step
gas_final = analyst.gas_in_place(step=-1)  # SCF

# Water in place
water = analyst.water_in_place(step=50)  # STB
```

### Volumetric Summary

Get complete volumetric summary at a step:

```python
volumes = analyst.volumetrics(step=-1)
print(f"Oil in place: {volumes.oil_in_place:,.0f} STB")
print(f"Gas in place: {volumes.gas_in_place:,.0f} SCF")
print(f"Water in place: {volumes.water_in_place:,.0f} STB")
print(f"Pore volume: {volumes.pore_volume:,.0f} ft³")
print(f"HCPV: {volumes.hydrocarbon_pore_volume:,.0f} ft³")
```

Returns a `ReservoirVolumetrics` object with:

- `oil_in_place` - Oil in place (STB)
- `gas_in_place` - Gas in place (SCF)
- `water_in_place` - Water in place (STB)
- `pore_volume` - Total pore volume (ft³)
- `hydrocarbon_pore_volume` - Hydrocarbon pore volume (ft³)

---

## Recovery Factors

### Current Recovery Factors

Get recovery factors for the entire simulation:

```python
# Oil recovery factor
oil_rf = analyst.oil_recovery_factor
print(f"Oil RF: {oil_rf:.2%}")

# Gas recovery factor
gas_rf = analyst.gas_recovery_factor
print(f"Gas RF: {gas_rf:.2%}")

# Water recovery factor
water_rf = analyst.water_recovery_factor
print(f"Water RF: {water_rf:.2%}")
```

!!! info "Recovery Factor Definition"
    Recovery Factor (RF) = Cumulative Produced / Initially in Place

    Typical ranges:

    - **Primary recovery**: 5-30% (solution gas drive) to 35-75% (water drive)
    - **Secondary recovery**: 30-60% (waterflooding) to 40-70% (gas injection)
    - **Enhanced recovery**: 40-80% (thermal, miscible EOR)

### Recovery Factor Between Steps

Compute incremental recovery between any two steps:

```python
# Recovery between step 0 and step 100
rf_0_to_100 = analyst.oil_recovery_factor_between(start_step=0, end_step=100)

# Recovery in last 50 steps
rf_recent = analyst.oil_recovery_factor_between(start_step=-50, end_step=-1)
```

---

## Cumulative Production

### Total Cumulative

Get total cumulative production from first to last state:

```python
# Cumulative oil produced
cum_oil = analyst.cumulative_oil_produced  # STB
# or
cum_oil = analyst.No

# Cumulative gas produced
cum_gas = analyst.cumulative_free_gas_produced  # SCF
# or
cum_gas = analyst.Ng

# Cumulative water produced
cum_water = analyst.cumulative_water_produced  # STB
# or
cum_water = analyst.Nw
```

### Cumulative Between Steps

Get cumulative production between any two steps:

```python
# Oil produced from step 0 to 100
oil_prod = analyst.oil_produced(start_step=0, end_step=100)  # STB

# Gas produced in last 50 steps
gas_prod = analyst.free_gas_produced(start_step=-50, end_step=-1)  # SCF

# Water produced
water_prod = analyst.water_produced(start_step=0, end_step=-1)  # STB
```

### Cumulative Summary

Get complete cumulative production summary:

```python
cum_prod = analyst.cumulative_production(start_step=0, end_step=-1)
print(f"Cumulative oil: {cum_prod.cumulative_oil:,.0f} STB")
print(f"Cumulative gas: {cum_prod.cumulative_free_gas:,.0f} SCF")
print(f"Cumulative water: {cum_prod.cumulative_water:,.0f} STB")
print(f"Oil RF: {cum_prod.oil_recovery_factor:.2%}")
print(f"Gas RF: {cum_prod.gas_recovery_factor:.2%}")
```

Returns a `CumulativeProduction` object.

---

## Production Rate Histories

### Instantaneous Rates

Get production rates at a specific step:

```python
rates = analyst.instantaneous_production_rates(step=-1, well_name="P-1")
print(f"Oil rate: {rates.oil_rate:.1f} STB/day")
print(f"Gas rate: {rates.gas_rate:.0f} SCF/day")
print(f"Water rate: {rates.water_rate:.1f} STB/day")
print(f"Total liquid: {rates.total_liquid_rate:.1f} STB/day")
print(f"GOR: {rates.gas_oil_ratio:.1f} SCF/STB")
print(f"Water cut: {rates.water_cut:.2%}")
```

Returns an `InstantaneousRates` object with:

- `oil_rate` - Oil production rate (STB/day)
- `gas_rate` - Gas production rate (SCF/day)
- `water_rate` - Water production rate (STB/day)
- `total_liquid_rate` - Total liquid rate (STB/day)
- `gas_oil_ratio` - GOR (SCF/STB)
- `water_cut` - Water cut (0 to 1)

### Rate Histories

Get production history over time:

```python
# Oil production history (every 10 steps)
for step, rate in analyst.oil_production_history(interval=10, well_name="P-1"):
    print(f"Step {step}: {rate:.1f} STB/day")

# Gas production history
for step, rate in analyst.gas_production_history(interval=5):
    print(f"Step {step}: {rate:.0f} SCF/day")

# Water production history
for step, rate in analyst.water_production_history(interval=10):
    print(f"Step {step}: {rate:.1f} STB/day")
```

!!! tip "Field vs. Well Rates"
    Omit `well_name` to get field-wide rates (all wells combined).

### Plotting Production History

```python
import matplotlib.pyplot as plt

# Collect data
steps = []
oil_rates = []
for step, rate in analyst.oil_production_history(interval=5):
    steps.append(step)
    oil_rates.append(rate)

# Plot
plt.plot(steps, oil_rates)
plt.xlabel("Time Step")
plt.ylabel("Oil Rate (STB/day)")
plt.title("Oil Production History")
plt.show()
```

---

## Injection Rate Histories

Similar to production, but for injection wells:

```python
# Instantaneous injection rates
inj_rates = analyst.instantaneous_injection_rates(step=-1, well_name="GI-1")
print(f"Gas injection: {inj_rates.gas_rate:.0f} SCF/day")

# Injection history
for step, rate in analyst.gas_injection_history(interval=10, well_name="GI-1"):
    print(f"Step {step}: {rate:.0f} SCF/day")

# Water injection history
for step, rate in analyst.water_injection_history(interval=10, well_name="WI-1"):
    print(f"Step {step}: {rate:.1f} STB/day")
```

### Cumulative Injection

```python
# Gas injected from step 0 to 100
gas_inj = analyst.gas_injected(start_step=0, end_step=100)  # SCF

# Water injected
water_inj = analyst.water_injected(start_step=0, end_step=-1)  # STB
```

### Voidage Replacement Ratio

Calculate voidage replacement ratio (VRR):

```python
vrr = analyst.voidage_replacement_ratio(step=-1)
print(f"VRR: {vrr:.2f}")
```

!!! info "VRR Interpretation"
    - **VRR > 1.0**: Over-injecting (pressure maintenance or increase)
    - **VRR = 1.0**: Balanced voidage replacement
    - **VRR < 1.0**: Under-injecting (pressure decline)

---

## Decline Curve Analysis

Fit production data to decline curve models:

```python
# Exponential decline
decline = analyst.decline_curve_analysis(
    phase="oil",
    decline_type="exponential",
    start_step=0,
    end_step=-1
)
print(f"Initial rate: {decline.initial_rate:.1f} STB/day")
print(f"Decline rate: {decline.decline_rate_per_timestep:.4f}")
print(f"R²: {decline.r_squared:.3f}")

# Hyperbolic decline
hyperbolic = analyst.decline_curve_analysis(
    phase="oil",
    decline_type="hyperbolic",
    start_step=0,
    end_step=-1
)
print(f"b-factor: {hyperbolic.b_factor:.3f}")

# Harmonic decline
harmonic = analyst.decline_curve_analysis(
    phase="oil",
    decline_type="harmonic"
)
```

Returns a `DeclineCurveResult` object with:

- `decline_type` - Type of decline ("exponential", "hyperbolic", "harmonic")
- `initial_rate` - Initial production rate (STB/day or SCF/day)
- `decline_rate_per_timestep` - Decline rate per timestep
- `b_factor` - Hyperbolic decline exponent (0 = exponential, 1 = harmonic)
- `r_squared` - Coefficient of determination (goodness of fit)
- `steps` - Time steps used
- `actual_rates` - Actual production rates
- `predicted_rates` - Predicted rates from decline curve

!!! info "Decline Types"
    - **Exponential**: Constant % decline (most conservative)
        - Formula: `q(t) = q_i * exp(-D * t)`
        - Common in tight reservoirs

    - **Hyperbolic**: Variable decline (most common)
        - Formula: `q(t) = q_i / (1 + b * D * t)^(1/b)`
        - `b` between 0 and 1

    - **Harmonic**: Declining % decline (most optimistic)
        - Formula: `q(t) = q_i / (1 + D * t)`
        - `b = 1` (special case of hyperbolic)

### Plotting Decline Curve

```python
import matplotlib.pyplot as plt

decline = analyst.decline_curve_analysis(phase="oil", decline_type="exponential")

plt.plot(decline.steps, decline.actual_rates, 'o', label="Actual")
plt.plot(decline.steps, decline.predicted_rates, '-', label="Fit")
plt.xlabel("Time Step")
plt.ylabel("Oil Rate (STB/day)")
plt.title(f"Exponential Decline (R² = {decline.r_squared:.3f})")
plt.legend()
plt.show()
```

---

## Material Balance Analysis

Analyze reservoir drive mechanisms:

```python
mb = analyst.material_balance_analysis(step=-1)
print(f"Pressure: {mb.pressure:.1f} psi")
print(f"Oil expansion factor: {mb.oil_expansion_factor:.3f}")
print(f"Solution gas drive index: {mb.solution_gas_drive_index:.2%}")
print(f"Gas cap drive index: {mb.gas_cap_drive_index:.2%}")
print(f"Water drive index: {mb.water_drive_index:.2%}")
print(f"Compaction drive index: {mb.compaction_drive_index:.2%}")
print(f"Aquifer influx: {mb.aquifer_influx:.0f} STB")
```

Returns a `MaterialBalanceAnalysis` object.

!!! info "Drive Indices"
    Drive indices sum to 1.0 (100%) and indicate the relative contribution of each drive mechanism:

    - **Solution gas drive**: Gas expansion as pressure drops
    - **Gas cap drive**: Gas cap expansion displaces oil
    - **Water drive**: Aquifer influx maintains pressure
    - **Compaction drive**: Rock and fluid compressibility

---

## Sweep Efficiency Analysis

Analyze how effectively the displacing phase contacts and recovers oil:

```python
sweep = analyst.sweep_efficiency_analysis(
    step=-1,
    displacing_phase="gas",  # or "water"
    threshold=0.05  # 5% saturation threshold
)
print(f"Volumetric sweep: {sweep.volumetric_sweep_efficiency:.2%}")
print(f"Displacement efficiency: {sweep.displacement_efficiency:.2%}")
print(f"Recovery efficiency: {sweep.recovery_efficiency:.2%}")
print(f"Contacted oil: {sweep.contacted_oil:,.0f} STB")
print(f"Uncontacted oil: {sweep.uncontacted_oil:,.0f} STB")
print(f"Areal sweep: {sweep.areal_sweep_efficiency:.2%}")
print(f"Vertical sweep: {sweep.vertical_sweep_efficiency:.2%}")
```

Returns a `SweepEfficiencyAnalysis` object.

!!! info "Sweep Efficiency Components"
    **Volumetric Sweep** = Fraction of reservoir contacted by displacing phase

    **Displacement Efficiency** = Fraction of oil recovered in contacted zones

    **Recovery Efficiency** = Volumetric Sweep × Displacement Efficiency

    Factors affecting sweep:

    - **Areal**: Well pattern, mobility ratio, permeability heterogeneity
    - **Vertical**: Layering, gravity segregation, permeability contrasts
    - **Displacement**: Viscosity ratio, interfacial tension, wettability

---

## Well Productivity Analysis

Analyze individual well performance:

```python
productivity = analyst.productivity_analysis(step=-1, well_name="P-1")
print(f"Flow rate: {productivity.total_flow_rate:.1f} STB/day")
print(f"Average reservoir pressure: {productivity.average_reservoir_pressure:.1f} psi")
print(f"Skin factor: {productivity.skin_factor:.2f}")
print(f"Flow efficiency: {productivity.flow_efficiency:.2%}")
print(f"Well index: {productivity.well_index:.2f} rb/day/psi")
print(f"Average mobility: {productivity.average_mobility:.4f} 1/cp")
```

Returns a `ProductivityAnalysis` object.

!!! tip "Skin Factor Interpretation"
    - **s < 0**: Stimulated well (fracture, acidizing)
    - **s = 0**: Ideal well (no damage or stimulation)
    - **s > 0**: Damaged well (drilling mud, scale, paraffin)

    Flow efficiency = `exp(-s)`, so:
    - s = 0 → 100% efficiency
    - s = 2 → 13.5% efficiency
    - s = 5 → 0.7% efficiency

---

## Front Tracking

Track displacement fronts:

```python
# Water front position
water_front = analyst.track_front(
    step=-1,
    phase="water",
    saturation_threshold=0.5,  # 50% water saturation
    direction="x"  # Track along x-axis
)
print(f"Water front position: {water_front:.1f} ft")

# Gas front position
gas_front = analyst.track_front(
    step=-1,
    phase="gas",
    saturation_threshold=0.3,
    direction="vertical"
)
print(f"Gas front position: {gas_front:.1f} ft (from top)")
```

---

## Cell-Level Analysis

Analyze specific cells or regions:

```python
from bores import Cells, CellFilter

# Analyze specific cells
cells = Cells([(10, 10, 3), (10, 10, 4)])
oil_in_cells = analyst.oil_in_place(step=-1, cells=cells)

# Analyze region using filter
cell_filter = CellFilter(x_range=(0, 10), y_range=(0, 10))
oil_in_region = analyst.oil_in_place(step=-1, cells=cell_filter)
```

---

## Accessing States

Get individual states for custom analysis:

```python
# Get specific state
state = analyst.get_state(step=100)

# Get final state
final_state = analyst.get_state(step=-1)

# Access state attributes
pressure = final_state.model.fluid_properties.pressure_grid
oil_sat = final_state.model.fluid_properties.saturation_history.oil_saturations[-1]

# Check available steps
print(f"Available steps: {analyst.available_steps}")
print(f"Min step: {analyst.min_step}")
print(f"Max step: {analyst.max_step}")
```

---

## Complete Analysis Example

```python
import bores

# Load results
store = bores.ZarrStore("./results/primary_depletion.zarr")
states = list(store.load(bores.ModelState))

# Create analyst
analyst = bores.ModelAnalyst(states)

# Initial volumes
print(f"\n{'='*60}")
print("INITIAL VOLUMES")
print(f"{'='*60}")
print(f"STOIIP: {analyst.stoiip:,.0f} STB")
print(f"STGIIP: {analyst.stgiip:,.0f} SCF")
print(f"STWIIP: {analyst.stock_tank_water_initially_in_place:,.0f} STB")

# Recovery factors
print(f"\n{'='*60}")
print("RECOVERY FACTORS")
print(f"{'='*60}")
print(f"Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"Gas RF: {analyst.gas_recovery_factor:.2%}")
print(f"Water RF: {analyst.water_recovery_factor:.2%}")

# Cumulative production
print(f"\n{'='*60}")
print("CUMULATIVE PRODUCTION")
print(f"{'='*60}")
print(f"Oil: {analyst.No:,.0f} STB")
print(f"Gas: {analyst.Ng:,.0f} SCF")
print(f"Water: {analyst.Nw:,.0f} STB")

# Current rates
rates = analyst.instantaneous_production_rates(step=-1)
print(f"\n{'='*60}")
print("CURRENT RATES")
print(f"{'='*60}")
print(f"Oil: {rates.oil_rate:.1f} STB/day")
print(f"Gas: {rates.gas_rate:,.0f} SCF/day")
print(f"Water: {rates.water_rate:.1f} STB/day")
print(f"GOR: {rates.gas_oil_ratio:.1f} SCF/STB")
print(f"Water cut: {rates.water_cut:.2%}")

# Decline curve
decline = analyst.decline_curve_analysis(phase="oil", decline_type="exponential")
print(f"\n{'='*60}")
print("DECLINE ANALYSIS")
print(f"{'='*60}")
print(f"Initial rate: {decline.initial_rate:.1f} STB/day")
print(f"Decline rate: {decline.decline_rate_per_timestep:.4f} per step")
print(f"R²: {decline.r_squared:.3f}")

# Material balance
mb = analyst.material_balance_analysis(step=-1)
print(f"\n{'='*60}")
print("MATERIAL BALANCE")
print(f"{'='*60}")
print(f"Pressure: {mb.pressure:.1f} psi")
print(f"Solution gas drive: {mb.solution_gas_drive_index:.2%}")
print(f"Gas cap drive: {mb.gas_cap_drive_index:.2%}")
print(f"Water drive: {mb.water_drive_index:.2%}")

# Volumetrics at final step
volumes = analyst.volumetrics(step=-1)
print(f"\n{'='*60}")
print("FINAL VOLUMETRICS")
print(f"{'='*60}")
print(f"Oil in place: {volumes.oil_in_place:,.0f} STB")
print(f"Gas in place: {volumes.gas_in_place:,.0f} SCF")
print(f"HCPV: {volumes.hydrocarbon_pore_volume:,.0f} ft³")
```

---

## Next Steps

- [States, Streams, and Stores](states-streams-stores.md) - State management and storage
- [Visualization](visualization.md) - Plotting and 3D visualization
- [Examples](../examples/primary-depletion.md) - Complete working examples
