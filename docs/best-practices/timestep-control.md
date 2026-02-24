# Timestep Control Best Practices

Guidelines for optimal adaptive timestep control in BORES simulations.

---

## Overview

Adaptive timestep control automatically adjusts timestep size to:

- **Maintain stability**: Prevent numerical errors
- **Maximize speed**: Use large steps when possible
- **Ensure accuracy**: Use small steps when needed

**Key concept**: CFL (Courant-Friedrichs-Lewy) condition controls timestep based on flow velocity and grid size.

---

## Timer Configuration

```python
import bores

timer = bores.Timer(
    initial_step_size=bores.Time(days=1.0),
    max_step_size=bores.Time(days=30.0),
    min_step_size=bores.Time(hours=1.0),
    simulation_time=bores.Time(days=365 * 5),  # 5 years
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
    aggressive_backoff_factor=0.25,
    max_rejects=20,
)
```

---

## Key Parameters

### Initial Step Size

Start conservatively:

```python
# Default (general purpose)
initial_step_size=bores.Time(days=1.0)  # 1 day

# Fast changes expected (injection start, well activation)
initial_step_size=bores.Time(hours=12.0)  # 0.5 days

# Slow changes (depletion only)
initial_step_size=bores.Time(days=5.0)  # 5 days
```

### Max Step Size

Limit maximum timestep:

```python
# Standard production
max_step_size=bores.Time(days=30.0)  # 1 month

# Need fine temporal resolution
max_step_size=bores.Time(days=10.0)  # 10 days

# Long-term forecasting (>20 years)
max_step_size=bores.Time(days=90.0)  # 3 months
```

!!! tip "Max Step Size Guidelines"
    - **Injection/production**: 10-30 days
    - **Depletion only**: 30-90 days
    - **EOR studies**: 7-30 days
    - **Short-term forecasts**: 1-10 days

### Min Step Size

Safety net for difficult periods:

```python
# Standard (hours to days)
min_step_size=bores.Time(hours=6.0)  # 0.25 days

# Very difficult (well events, gas breakthrough)
min_step_size=bores.Time(hours=1.0)  # 1 hour

# Give up faster
min_step_size=bores.Time(hours=12.0)  # 0.5 days
```

**If timestep hits min_step_size repeatedly**: Check for:

- Solver convergence issues
- Physical instabilities (negative saturations)
- Grid quality problems

---

## CFL Number

Controls timestep based on fluid velocity:

```
Δt ≤ CFL × (Δx / v)
```

where:
- Δt = timestep
- Δx = cell size
- v = fluid velocity
- CFL = target CFL number (typically 0.5-1.0)

### CFL Selection

```python
# Conservative (stable, slower)
max_cfl_number=0.5  # Small timesteps

# Balanced (recommended)
max_cfl_number=0.9  # Default

# Aggressive (fast, may be unstable)
max_cfl_number=1.5  # Larger timesteps
```

**Guidelines**:

| Scenario | CFL | Reasoning |
| -------- | --- | --------- |
| **Waterflooding** | 0.9 | Standard |
| **Gas injection** | 0.7 | Gas moves faster |
| **Primary depletion** | 1.2 | Slow, stable |
| **Miscible flooding** | 0.8 | Complex physics |
| **Difficult convergence** | 0.5 | More stable |

---

## Adaptive Control

### Ramp-Up Factor

Increase timestep when stable:

```python
# Conservative
ramp_up_factor=1.1  # 10% increase per step

# Standard
ramp_up_factor=1.2  # 20% increase (default)

# Aggressive
ramp_up_factor=1.5  # 50% increase per step
```

**Effect**: Larger values reach `max_step_size` faster.

### Backoff Factor

Reduce timestep on rejection:

```python
# Conservative (small reduction)
backoff_factor=0.7  # Reduce to 70% of previous

# Standard
backoff_factor=0.5  # Halve timestep (default)

# Aggressive (large reduction)
backoff_factor=0.3  # Reduce to 30%
```

### Aggressive Backoff

Use when repeated rejections occur:

```python
# After 2-3 rejections, cut timestep more aggressively
aggressive_backoff_factor=0.25  # Reduce to 25%
```

---

## Timestep Rejection

### Rejection Criteria

Timestep rejected if:

1. **Solver fails to converge**
2. **Negative saturations** (So, Sw, or Sg < 0)
3. **Saturation sum violation** (So + Sw + Sg ≠ 1)
4. **Pressure oscillations**
5. **CFL violation** (velocity too high)

### Max Rejects

Limit consecutive rejections before giving up:

```python
# Persistent (keeps trying)
max_rejects=50  # Many retries

# Standard
max_rejects=20  # Default

# Give up quickly
max_rejects=5  # Fewer retries
```

!!! warning "Max Rejects Reached"
    If simulation stops due to `max_rejects`:

    1. **Check solver**: Try different solver/preconditioner
    2. **Refine grid**: Bad grid → instability
    3. **Reduce max_cfl**: Start with 0.5
    4. **Check initial conditions**: Ensure valid saturations

---

## Optimization Strategies

### Fast Production Run

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=5.0),  # Start large
    max_step_size=bores.Time(days=60.0),  # Allow big steps
    min_step_size=bores.Time(days=1.0),  # Don't go too small
    max_cfl_number=1.2,  # Aggressive
    ramp_up_factor=1.3,  # Quick ramp-up
    backoff_factor=0.5,
    max_rejects=10,  # Give up faster
)
```

### High-Accuracy Study

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=12.0),  # Start small
    max_step_size=bores.Time(days=10.0),  # Limit max
    min_step_size=bores.Time(hours=1.0),  # Allow very small
    max_cfl_number=0.7,  # Conservative
    ramp_up_factor=1.1,  # Slow ramp-up
    backoff_factor=0.5,
    aggressive_backoff_factor=0.2,  # Aggressive reduction
    max_rejects=30,  # Persistent
)
```

### Well Event (Injection Start)

```python
# Use well schedule to temporarily reduce timestep
well_schedule = bores.WellSchedule()

# At injection start, force small timestep
well_schedule["start_injection"] = bores.WellEvent(
    predicate=bores.time_predicate(time=bores.Time(days=100)),
    action=bores.update_well(is_active=True),
)

# Timer will automatically handle transient with small steps
timer = bores.Timer(
    initial_step_size=bores.Time(hours=6.0),  # Small for transient
    max_step_size=bores.Time(days=30.0),
    min_step_size=bores.Time(hours=1.0),
    max_cfl_number=0.8,  # Conservative during event
)
```

---

## Monitoring Timestep Behavior

### Track Timestep Sizes

```python
import matplotlib.pyplot as plt

times = []
step_sizes = []

for state in run():
    times.append(state.time / 86400)  # Convert to days
    step_sizes.append(state.step_size / 86400)

# Plot timestep evolution
plt.figure(figsize=(10, 6))
plt.plot(times, step_sizes, 'b-', linewidth=2)
plt.xlabel('Simulation Time (days)')
plt.ylabel('Timestep Size (days)')
plt.title('Adaptive Timestep Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

### Track Rejections

```python
rejections = []

for state in run():
    if state.timer_state:
        rejections.append(state.timer_state.total_rejects)

print(f"Total timestep rejections: {rejections[-1]}")
print(f"Rejection rate: {rejections[-1] / state.step * 100:.1f}%")

# High rejection rate (>10%) indicates problems
```

---

## Common Issues

### Too Many Small Timesteps

**Symptom**: Simulation slow, timesteps remain small

**Causes**:

- max_cfl too low
- Solver not converging
- Physical instabilities

**Solutions**:

```python
# Increase max_cfl
max_cfl_number=1.0  # From 0.7

# Faster ramp-up
ramp_up_factor=1.3  # From 1.1

# Check solver convergence
pressure_solver="gmres"  # More robust
```

### Frequent Rejections

**Symptom**: Many timestep cuts, slow progress

**Causes**:

- CFL too high
- Poor initial conditions
- Grid quality issues

**Solutions**:

```python
# Reduce max_cfl
max_cfl_number=0.7  # From 0.9

# Smaller initial step
initial_step_size=bores.Time(hours=12.0)

# More aggressive backoff
aggressive_backoff_factor=0.2  # From 0.25
```

### Oscillating Timesteps

**Symptom**: Timestep grows then cuts repeatedly

**Causes**:

- ramp_up_factor too large
- Physics approaching stability limit

**Solutions**:

```python
# Slower ramp-up
ramp_up_factor=1.1  # From 1.3

# Lower max_cfl
max_cfl_number=0.8  # From 1.0
```

---

## Advanced: Phase-Specific CFL

Different phases have different velocities:

```
v_water ≈ 1 ft/day (slow)
v_oil ≈ 2 ft/day (medium)
v_gas ≈ 10 ft/day (fast)
```

For gas-dominated systems, use lower CFL:

```python
# Gas injection or gas cap drive
max_cfl_number=0.7  # Lower than default 0.9
```

---

## Recommended Configurations

### Primary Depletion

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=1.0),
    max_step_size=bores.Time(days=30.0),
    min_step_size=bores.Time(hours=6.0),
    max_cfl_number=1.0,  # Stable flow
    ramp_up_factor=1.3,  # Quick ramp-up
)
```

### Waterflooding

```python
timer = bores.Timer(
    initial_step_size=bores.Time(days=1.0),
    max_step_size=bores.Time(days=15.0),  # Moderate max
    min_step_size=bores.Time(hours=6.0),
    max_cfl_number=0.9,  # Standard
    ramp_up_factor=1.2,
)
```

### Gas Injection

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=12.0),  # Small start
    max_step_size=bores.Time(days=10.0),  # Smaller max
    min_step_size=bores.Time(hours=1.0),
    max_cfl_number=0.7,  # Conservative (gas is fast)
    ramp_up_factor=1.15,  # Slow ramp-up
)
```

### Miscible CO2 EOR

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=24.0),
    max_step_size=bores.Time(days=7.0),  # Small for complex physics
    min_step_size=bores.Time(hours=2.0),
    max_cfl_number=0.8,
    ramp_up_factor=1.2,
)
```

---

## Checklist

✅ **initial_step_size**: Conservative (1 day or less)
✅ **max_step_size**: Appropriate for problem (10-30 days)
✅ **max_cfl_number**: 0.7-1.0 for most cases
✅ **Monitor rejections**: <10% rejection rate
✅ **Check timestep plot**: Should grow smoothly to max

---

## Next Steps

- [Solver Selection](solver-selection.md) - Choose appropriate solver
- [Grid Design](grid-design.md) - Optimize grid for stability
- [Performance Optimization](../advanced/performance-optimization.md) - Speed up simulations
