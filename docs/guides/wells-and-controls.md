# Wells & Controls

Comprehensive guide to production wells, injection wells, well controls, and scheduling.

---

## Overview

BORES wells consist of:

1. **Well location** - Perforating intervals in grid
2. **Physical properties** - Radius, skin factor
3. **Control strategy** - How the well operates (rate, BHP, adaptive)
4. **Fluids** - What's injected or produced
5. **Scheduling** (optional) - Time-dependent changes

---

## Production Wells

### Creating a Production Well

```python
producer = bores.production_well(
    well_name: str,
    perforating_intervals: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    radius: float,
    control: WellControl,
    produced_fluids: tuple[ProducedFluid, ...],
    skin_factor: float = 0.0,
    is_active: bool = True,
)
```

**Parameters:**

- `well_name`: Unique identifier (e.g., "PROD-1", "P-1")
- `perforating_intervals`: List of `((x1, y1, z1), (x2, y2, z2))` cell ranges
- `radius`: Wellbore radius in feet (e.g., 0.354 ft = 8.5" hole)
- `control`: Well control object (rate, BHP, or adaptive)
- `produced_fluids`: Tuple of `ProducedFluid` objects for oil, gas, water
- `skin_factor`: Dimensionless (positive = damage, negative = stimulation, 0 = no damage)
- `is_active`: Whether well is open (True) or shut-in (False)

**Complete Example:**

```python
import bores

producer = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[
        ((10, 15, 5), (10, 15, 8))  # Vertical well, layers 5-8
    ],
    radius=0.354,  # 8.5" wellbore
    control=bores.AdaptiveBHPRateControl(
        target_rate=-500,  # STB/day (negative = production)
        target_phase="oil",
        bhp_limit=1000,  # psi minimum
        clamp=bores.ProductionClamp(),
    ),
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.85,
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
    skin_factor=3.0,  # Damaged well
    is_active=True,
)
```

### ProducedFluid

All production wells must specify produced fluids:

```python
bores.ProducedFluid(
    name: str,                    # Identifier (e.g., "Oil", "Gas", "Water")
    phase: FluidPhase,            # OIL, GAS, or WATER
    specific_gravity: float,      # Relative to water (oil, water) or air (gas)
    molecular_weight: float,      # g/mol
)
```

**Typical Values:**

| Fluid | Phase | Specific Gravity | Molecular Weight |
|-------|-------|------------------|------------------|
| Oil (36° API) | OIL | 0.845 | 150-200 |
| Gas (methane) | GAS | 0.55-0.75 | 16-20 |
| Water (brine) | WATER | 1.01-1.10 | 18.015 |

---

## Injection Wells

### Creating an Injection Well

```python
injector = bores.injection_well(
    well_name: str,
    perforating_intervals: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    radius: float,
    control: WellControl,
    injected_fluid: InjectedFluid,
    skin_factor: float = 0.0,
    is_active: bool = True,
)
```

**Complete Example - Water Injection:**

```python
water_injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((5, 5, 3), (5, 5, 6))],
    radius=0.354,
    control=bores.ConstantRateControl(
        target_rate=1000,  # STB/day (positive = injection)
        target_phase="water",
        bhp_limit=3500,  # psi maximum
    ),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.05,
        molecular_weight=18.015,
        salinity=35000,  # ppm NaCl
    ),
    skin_factor=0.0,
    is_active=True,
)
```

**Complete Example - Gas Injection (Immiscible):**

```python
gas_injector = bores.injection_well(
    well_name="GI-1",
    perforating_intervals=[((15, 15, 2), (15, 15, 4))],
    radius=0.354,
    control=bores.AdaptiveBHPRateControl(
        target_rate=100000,  # SCF/day
        target_phase="gas",
        bhp_limit=4000,  # psi maximum
        clamp=bores.InjectionClamp(),
    ),
    injected_fluid=bores.InjectedFluid(
        name="Methane",
        phase=bores.FluidPhase.GAS,
        specific_gravity=0.65,
        molecular_weight=16.04,
        is_miscible=False,  # Immiscible
    ),
    skin_factor=1.0,
    is_active=True,
)
```

**Complete Example - Miscible CO2 Injection:**

```python
co2_injector = bores.injection_well(
    well_name="CO2-1",
    perforating_intervals=[((18, 8, 1), (18, 8, 3))],
    radius=0.354,
    control=bores.AdaptiveBHPRateControl(
        target_rate=500000,  # SCF/day
        target_phase="gas",
        bhp_limit=4500,
        clamp=bores.InjectionClamp(),
    ),
    injected_fluid=bores.InjectedFluid(
        name="CO2",
        phase=bores.FluidPhase.GAS,
        specific_gravity=1.52,  # CO2 is heavier than air
        molecular_weight=44.01,
        viscosity=0.05,  # cP - override correlation
        density=35.0,  # lbm/ft³ - override correlation (CO2 is dense!)
        is_miscible=True,
        minimum_miscibility_pressure=2200.0,  # psi
        todd_longstaff_omega=0.67,  # Mixing parameter
        miscibility_transition_width=500.0,  # psi
        concentration=1.0,  # Pure CO2
    ),
    skin_factor=0.0,
    is_active=True,
)
```

### InjectedFluid

```python
bores.InjectedFluid(
    name: str,
    phase: FluidPhase,                   # GAS or WATER only
    specific_gravity: float,
    molecular_weight: float,
    salinity: Optional[float] = None,    # ppm NaCl (for water)
    is_miscible: bool = False,
    todd_longstaff_omega: float = 0.67,  # 0-1, mixing parameter
    minimum_miscibility_pressure: Optional[float] = None,  # psi
    miscibility_transition_width: float = 500.0,  # psi
    concentration: float = 1.0,          # 0-1
    density: Optional[float] = None,     # lbm/ft³ (override correlation)
    viscosity: Optional[float] = None,   # cP (override correlation)
)
```

**Miscibility Parameters:**

- `is_miscible`: Enable Todd-Longstaff model (requires `minimum_miscibility_pressure` and `todd_longstaff_omega`)
- `todd_longstaff_omega`:
  - 0.0 = Fully segregated flow
  - 0.33 = Partial mixing (CH4)
  - 0.67 = Moderate mixing (CO2, typical)
  - 1.0 = Fully mixed
- `minimum_miscibility_pressure`: Pressure above which fluid becomes miscible with oil
- `miscibility_transition_width`: Smooth transition zone around MMP
- `concentration`: Volume fraction (1.0 = pure, < 1.0 = mixed)

**Density/Viscosity Overrides:**

For non-ideal gases like CO2, specify measured values:

```python
density=35.0,    # CO2 at reservoir conditions (NOT 3-7 from correlation!)
viscosity=0.05,  # CO2 viscosity
```

---

## Well Controls

### ConstantRateControl

Fixed injection or production rate:

```python
control = bores.ConstantRateControl(
    target_rate: float,                # STB/day or SCF/day (negative = production)
    target_phase: str,                 # "oil", "water", or "gas"
    bhp_limit: Optional[float] = None, # psi
    clamp: Optional[RateClamp] = None,
)
```

**Example:**

```python
control = bores.ConstantRateControl(
    target_rate=500,  # STB/day injection
    target_phase="water",
    bhp_limit=4000,  # Don't exceed 4000 psi
)
```

**Behavior:**
- Maintains `target_rate` unless BHP limit would be violated
- If BHP limit reached, rate becomes 0 (well shuts in)
- No automatic rate adjustment

---

### AdaptiveBHPRateControl

Rate control with automatic BHP adjustment:

```python
control = bores.AdaptiveBHPRateControl(
    target_rate: float,
    target_phase: str,
    bhp_limit: float,  # Required (not optional)
    clamp: Optional[RateClamp] = None,
)
```

**Example - Production:**

```python
control = bores.AdaptiveBHPRateControl(
    target_rate=-300,  # STB/day
    target_phase="oil",
    bhp_limit=1000,  # psi minimum
    clamp=bores.ProductionClamp(),
)
```

**Example - Injection:**

```python
control = bores.AdaptiveBHPRateControl(
    target_rate=50000,  # SCF/day
    target_phase="gas",
    bhp_limit=3500,  # psi maximum
    clamp=bores.InjectionClamp(),
)
```

**Behavior:**
- Tries to maintain `target_rate`
- If BHP limit would be violated, reduces rate to stay within limit
- More realistic than ConstantRateControl

---

### BHPControl

Fixed bottom-hole pressure:

```python
control = bores.BHPControl(
    bhp: float,  # psi
    clamp: Optional[RateClamp] = None,
)
```

**Example:**

```python
control = bores.BHPControl(
    bhp=2500,  # psi
    clamp=bores.ProductionClamp(),
)
```

**Behavior:**
- Maintains constant BHP
- Rate varies based on reservoir pressure
- Flow rate computed from Darcy's law

---

### MultiPhaseRateControl

Independent controls for each phase:

```python
control = bores.MultiPhaseRateControl(
    oil_control: WellControl,
    gas_control: WellControl,
    water_control: WellControl,
)
```

**Complete Example:**

```python
clamp = bores.ProductionClamp()

control = bores.MultiPhaseRateControl(
    oil_control=bores.AdaptiveBHPRateControl(
        target_rate=-500,
        target_phase="oil",
        bhp_limit=1000,
        clamp=clamp,
    ),
    gas_control=bores.AdaptiveBHPRateControl(
        target_rate=-1000,  # MCF/day
        target_phase="gas",
        bhp_limit=1000,
        clamp=clamp,
    ),
    water_control=bores.AdaptiveBHPRateControl(
        target_rate=-50,
        target_phase="water",
        bhp_limit=1000,
        clamp=clamp,
    ),
)
```

**Use Cases:**
- Different rate targets per phase
- Different BHP limits per phase
- Complex production constraints

---

## Rate Clamping

Prevent unphysical flow directions:

### ProductionClamp

```python
clamp = bores.ProductionClamp(value=0.0)
```

- Clamps rate to 0 if rate > 0 (injection during production)
- Clamps BHP if bhp > reservoir pressure

### InjectionClamp

```python
clamp = bores.InjectionClamp(value=0.0)
```

- Clamps rate to 0 if rate < 0 (production during injection)
- Clamps BHP if bhp < reservoir pressure

**Usage:**

```python
control = bores.AdaptiveBHPRateControl(
    target_rate=-200,
    target_phase="oil",
    bhp_limit=800,
    clamp=bores.ProductionClamp(),  # Prevent injection
)
```

---

## Well Scheduling

Change well properties during simulation:

### Creating a Schedule

```python
# Create schedule
well_schedule = bores.WellSchedule()

# Add event: open well at day 100
well_schedule["open_well"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=100)),
    action=bores.update_well(is_active=True),
)

# Add event: increase rate at day 365
well_schedule["increase_rate"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=365)),
    action=bores.update_well(
        control=bores.AdaptiveBHPRateControl(
            target_rate=-800,  # Increase from -500
            target_phase="oil",
            bhp_limit=1000,
        )
    ),
)

# Add to WellSchedules
well_schedules = bores.WellSchedules()
well_schedules[producer.name] = well_schedule

# Use in config
config = bores.Config(
    timer=timer,
    wells=wells,
    well_schedules=well_schedules,
    ...
)
```

### TimePredicate

Trigger at specific time or timestep:

```python
# By simulation time
predicate = bores.time_predicate(time=bores.Time(days=365))

# By timestep number
predicate = bores.time_predicate(time_step=100)
```

### UpdateAction

Change well properties:

```python
action = bores.update_well(
    control: Optional[WellControl] = None,
    skin_factor: Optional[float] = None,
    is_active: Optional[bool] = None,
    injected_fluid: Optional[InjectedFluid] = None,
    produced_fluids: Optional[tuple[ProducedFluid, ...]] = None,
)
```

**Complete Scheduling Example:**

```python
# Producer starts shut-in, opens at 100 days, rate increase at 1 year
producer = bores.production_well(
    well_name="P-1",
    ...,
    is_active=False,  # Start shut-in
)

schedule = bores.WellSchedule()

# Event 1: Open at 100 days
schedule["open"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=100)),
    action=bores.update_well(is_active=True),
)

# Event 2: Increase rate at 1 year
schedule["increase"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=365)),
    action=bores.update_well(
        control=bores.AdaptiveBHPRateControl(
            target_rate=-800,
            target_phase="oil",
            bhp_limit=800,
        )
    ),
)

# Event 3: Workover at 2 years (improve skin)
schedule["workover"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=730)),
    action=bores.update_well(skin_factor=0.0),  # Remove damage
)

well_schedules = bores.WellSchedules()
well_schedules["P-1"] = schedule
```

---

## Combining Wells

```python
wells = bores.wells_(
    injectors: Optional[list[InjectionWell]] = None,
    producers: Optional[list[ProductionWell]] = None,
)
```

**Example:**

```python
wells = bores.wells_(
    injectors=[water_inj_1, water_inj_2, gas_inj_1],
    producers=[prod_1, prod_2, prod_3, prod_4],
)

config = bores.Config(
    timer=timer,
    wells=wells,
    ...
)
```

---

## Well Duplication

Create multiple similar wells:

```python
injector_1 = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((5, 5, 3), (5, 5, 6))],
    ...,
)

# Duplicate with different location and name
injector_2 = injector_1.duplicate(
    name="INJ-2",
    perforating_intervals=[((15, 5, 3), (15, 5, 6))],
)

injector_3 = injector_1.duplicate(
    name="INJ-3",
    perforating_intervals=[((5, 15, 3), (5, 15, 6))],
)

injectors = [injector_1, injector_2, injector_3]
```

---

## Well Physics

### Peaceman Well Index

BORES uses the Peaceman model:

\\[
WI = \\frac{2\\pi k h}{\\ln(r_e/r_w) + s}
\\]

Where:
- \\(k\\) = permeability (mD)
- \\(h\\) = interval thickness (ft)
- \\(r_e\\) = effective drainage radius (ft)
- \\(r_w\\) = wellbore radius (ft)
- \\(s\\) = skin factor

### Effective Drainage Radius (3D)

For vertical wells (Z-oriented):

\\[
r_e = 0.28 \\sqrt{\\frac{\\Delta x^2 + \\Delta y^2}{\\sqrt{k_x/k_y} + \\sqrt{k_y/k_x}}}
\\]

For horizontal wells, formula adjusts based on well orientation (X or Y).

### Flow Rate Calculation

**Oil/Water:**

\\[
q = 0.00708 \\cdot WI \\cdot \\lambda \\cdot (P_{bhp} - P)
\\]

Where \\(\\lambda = k_r/(\\mu B)\\) is phase mobility.

**Gas (Pseudo-pressure):**

\\[
q = 0.01988 \\cdot \\frac{T_{sc}}{P_{sc}} \\cdot \\frac{WI}{T} \\cdot (m(P_{bhp}) - m(P))
\\]

Where \\(m(P)\\) is gas pseudo-pressure.

---

## Complete Example: 5-Spot Pattern

```python
import bores

# Central injector
injector = bores.injection_well(
    well_name="INJ-1",
    perforating_intervals=[((10, 10, 3), (10, 10, 6))],
    radius=0.354,
    control=bores.ConstantRateControl(
        target_rate=2000,
        target_phase="water",
        bhp_limit=4000,
    ),
    injected_fluid=bores.InjectedFluid(
        name="Water",
        phase=bores.FluidPhase.WATER,
        specific_gravity=1.05,
        molecular_weight=18.015,
        salinity=35000,
    ),
    skin_factor=0.0,
    is_active=True,
)

# 4 corner producers
clamp = bores.ProductionClamp()
prod_control = bores.AdaptiveBHPRateControl(
    target_rate=-400,
    target_phase="oil",
    bhp_limit=1000,
    clamp=clamp,
)

producer_1 = bores.production_well(
    well_name="PROD-1",
    perforating_intervals=[((3, 3, 3), (3, 3, 6))],
    radius=0.354,
    control=prod_control,
    produced_fluids=(
        bores.ProducedFluid(
            name="Oil",
            phase=bores.FluidPhase.OIL,
            specific_gravity=0.85,
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
    skin_factor=2.0,
    is_active=True,
)

producer_2 = producer_1.duplicate(name="PROD-2", perforating_intervals=[((17, 3, 3), (17, 3, 6))])
producer_3 = producer_1.duplicate(name="PROD-3", perforating_intervals=[((3, 17, 3), (3, 17, 6))])
producer_4 = producer_1.duplicate(name="PROD-4", perforating_intervals=[((17, 17, 3), (17, 17, 6))])

wells = bores.wells_(
    injectors=[injector],
    producers=[producer_1, producer_2, producer_3, producer_4],
)
```

---

## Next Steps

- [Running Simulations](running-simulations.md) - Use wells in config
- [Examples](../examples/index.md) - Complete working scenarios
