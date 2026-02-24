# Running Simulations

Configure solvers, timesteps, and execute simulations with the Run class.

---

## Overview

Running a BORES simulation involves three steps:

1. **Configure**: Create a `Config` object with simulation parameters
2. **Create Run**: Instantiate `Run` with model and config
3. **Execute**: Iterate through timesteps and process results

---

## The Config Class

`bores.Config` controls **how** the simulation runs - solvers, tolerances, timesteps, and physics options.

### Minimal Configuration

```python
config = bores.Config(
    timer=timer,                    # Required: controls timesteps
    rock_fluid_tables=rock_fluid_tables,  # Required: rel perm & cap pressure
)
```

### Complete Configuration

```python
config = bores.Config(
    # ===== Required =====
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,

    # ===== Optional: Wells & Boundaries =====
    wells=wells,                    # Production/injection wells
    well_schedules=well_schedules,  # Time-dependent well controls
    boundary_conditions=boundary_conditions,  # Aquifer support, etc.

    # ===== PVT =====
    pvt_tables=pvt_tables,          # Use tables instead of correlations
    use_pseudo_pressure=True,       # Enable gas pseudo-pressure (default: True)

    # ===== Numerical Scheme =====
    scheme="impes",                 # "impes", "explicit", or "implicit"

    # ===== Solvers =====
    pressure_solver="bicgstab",     # Pressure solver
    saturation_solver="bicgstab",   # Saturation solver (for implicit scheme)
    pressure_preconditioner="ilu",  # Preconditioner for pressure
    saturation_preconditioner="ilu",  # Preconditioner for saturation

    # ===== Convergence Tolerances =====
    pressure_convergence_tolerance=1e-6,      # Relative tolerance
    saturation_convergence_tolerance=1e-4,    # Relative tolerance
    max_iterations=250,                       # Max solver iterations

    # ===== CFL Thresholds (for 'explicit' scheme) =====
    pressure_cfl_threshold=0.9,     # Max pressure CFL
    saturation_cfl_threshold=0.6,   # Max saturation CFL

    # ===== Timestep Rejection Criteria =====
    max_pressure_change=100.0,      # psi per timestep
    max_oil_saturation_change=0.5,  # Absolute saturation change
    max_water_saturation_change=0.4,
    max_gas_saturation_change=0.85,

    # ===== Capillary Effects =====
    disable_capillary_effects=False,  # Set True to turn off capillary pressure
    capillary_strength_factor=1.0,    # Scale capillary effects (0-1)

    # ===== Miscibility =====
    miscibility_model="immiscible",  # "immiscible" or "todd_longstaff"

    # ===== Output & Logging =====
    output_frequency=1,              # Yield state every N timesteps
    log_interval=5,                  # Log progress every N timesteps
    warn_well_anomalies=True,        # Warn about unusual well behavior

    # ===== Advanced =====
    disable_structural_dip=False,    # Disable dip effects
    phase_appearance_tolerance=1e-6,  # Tolerance for phase appearance/disappearance
)
```

---

## Evolution Schemes

BORES supports three numerical schemes for solving the black-oil equations.

### IMPES (Recommended)

**IMplicit Pressure, Explicit Saturation**

```python
config = bores.Config(
    scheme="impes",
    ...
)
```

**How it works:**

1. Solve pressure **implicitly** (linear system solve)
2. Update saturations **explicitly** (direct calculation)
3. Update well rates
4. Advance timestep

**Advantages:**

- **Fast**: Only one linear solve per timestep
- **Stable**: Implicit pressure handles large pressure changes
- **Memory efficient**: Smaller matrices
- **Well-tested**: Industry standard for waterfloods

**Disadvantages:**

- **CFL limited**: Requires adaptive timesteps for saturation stability
- **Explicit saturations**: Can be less stable for high-velocity flow

**Best for:**

- Waterfloods
- Gas injection (immiscible)
- Primary depletion
- Most production scenarios
- **Default choice for 95% of simulations**

!!! success "IMPES is the Default"
    Use IMPES unless you have a specific reason not to. It's fast, stable, and handles most problems well.

---

### Explicit Scheme

**Fully explicit in pressure and saturation**

```python
config = bores.Config(
    scheme="explicit",
    pressure_cfl_threshold=0.9,     # Important!
    saturation_cfl_threshold=0.6,   # Important!
    ...
)
```

**How it works:**

1. Update pressure **explicitly** (direct calculation)
2. Update saturations **explicitly** (direct calculation)
3. Update well rates
4. Advance timestep

**Advantages:**

- **Fastest per timestep**: No linear solves
- **Simple**: Straightforward implementation
- **Good for testing**: Quick iterations

**Disadvantages:**

- **Very CFL limited**: Requires very small timesteps
- **Less stable**: Both pressure and saturation explicit
- **Slow overall**: Many small timesteps needed
- **Conditional stability**: Can diverge if dt too large

**Best for:**

- Testing and debugging
- Very simple problems
- Short simulations with small grids

!!! warning "Use with Caution"
    Explicit scheme is rarely the best choice for production runs. It requires extremely small timesteps to remain stable.

**CFL Thresholds:**

The explicit scheme requires careful CFL control:

```python
config = bores.Config(
    scheme="explicit",
    pressure_cfl_threshold=0.9,   # Keep < 1.0 for stability
    saturation_cfl_threshold=0.6,  # Conservative (< pressure CFL)
)
```

Lower thresholds = more stable but slower.

---

### Implicit Scheme (Not Yet Implemented)

**Fully implicit in pressure and saturation**

```python
config = bores.Config(
    scheme="implicit",  # Not yet available
    ...
)
```

!!! info "Future Feature"
    Fully implicit scheme is **planned but not yet implemented**. It will provide:

    - Unconditional stability
    - Large timesteps
    - Better for challenging problems (high-velocity, compositional)

    Implementation is complex and will be added in future releases.

---

## Scheme Comparison Table

| Feature | IMPES | Explicit | Implicit (Future) |
|---------|-------|----------|-------------------|
| **Speed per step** | Medium | Fast | Slow |
| **Overall speed** | **Fast** | Slow | Medium-Fast |
| **Stability** | **Good** | Poor | Excellent |
| **Timestep size** | **Medium** | Very small | Large |
| **CFL constraint** | Moderate | Strict | None |
| **Memory** | **Low** | Very low | High |
| **Complexity** | Low | Very low | High |
| **Best for** | **Production runs** | Testing | Challenging physics |

!!! tip "Rule of Thumb"
    - **Use IMPES** for 95% of cases
    - **Use Explicit** only for quick tests or debugging
    - **Wait for Implicit** for compositional/thermal/high-velocity flows

---

## Pressure Solvers

BORES uses iterative solvers from SciPy for linear systems. Choose based on your problem.

### Available Solvers

```python
config = bores.Config(
    pressure_solver="bicgstab",  # Choose one
    ...
)
```

| Solver | Description | Speed | Memory | Best For |
|--------|-------------|-------|--------|----------|
| `"bicgstab"` | **BiConjugate Gradient Stabilized** | Fast | Low | General purpose (**default**) |
| `"gmres"` | Generalized Minimal Residual | Medium | Medium | Non-symmetric systems |
| `"lgmres"` | GMRES with deflation | Medium | Medium | Ill-conditioned matrices |
| `"cgs"` | Conjugate Gradient Squared | Fast | Low | Well-conditioned systems |
| `"qmr"` | Quasi-Minimal Residual | Slow | Medium | Difficult convergence |

!!! success "BiCGSTAB is Default"
    `bicgstab` is a good all-around solver. Only change if you have convergence issues.

### Solver Sequences

You can try multiple solvers in sequence:

```python
config = bores.Config(
    pressure_solver=["bicgstab", "gmres", "lgmres"],
)
```

BORES will:

1. Try `bicgstab` first
2. If it fails, try `gmres`
3. If that fails, try `lgmres`

---

## Preconditioners

Preconditioners transform the linear system to improve convergence.

### Available Preconditioners

```python
config = bores.Config(
    pressure_preconditioner="ilu",  # Choose one
    ...
)
```

| Preconditioner | Description | Setup Time | Memory | Best For |
|----------------|-------------|------------|--------|----------|
| `"ilu"` | Incomplete LU | Fast | Low | General purpose (**default**) |
| `"amg"` | Algebraic Multigrid | Slow | High | Large systems, high anisotropy |
| `"diagonal"` | Diagonal (Jacobi) | Very fast | Very low | Well-conditioned systems |
| `"cpr"` | Constrained Pressure Residual | Medium | Medium | Pressure-dominated flow |
| `None` | No preconditioning | Instant | None | Testing, small systems |

### Cached Preconditioners (Recommended)

Preconditioners are expensive to build but can be reused:

```python
# Register cached preconditioner
cached_precond = bores.CachedPreconditionerFactory(
    factory="ilu",              # Base preconditioner
    name="cached_ilu",          # Unique name
    update_frequency=10,        # Rebuild every 10 timesteps
    recompute_threshold=0.3,    # Rebuild if matrix changes > 30%
)
cached_precond.register(override=True)

# Use in config
config = bores.Config(
    pressure_preconditioner="cached_ilu",
    ...
)
```

**Benefits:**

- **20-40% faster** than rebuilding every timestep
- Automatically rebuilds when needed
- Transparent to user

!!! tip "Always Use Cached Preconditioners"
    For simulations > 50 timesteps, cached preconditioners are worth it.

---

## Convergence Tolerances

Control solver accuracy vs speed.

```python
config = bores.Config(
    pressure_convergence_tolerance=1e-6,      # Stricter
    saturation_convergence_tolerance=1e-4,    # Looser (default)
    max_iterations=250,                       # Max iterations
    ...
)
```

**Pressure tolerance:**

- `1e-6` (default): Good balance
- `1e-7` to `1e-8`: High accuracy (slower)
- `1e-5`: Faster but less accurate

**Saturation tolerance:**

- `1e-4` (default): Adequate for most cases
- `1e-5`: Higher accuracy
- `1e-3`: Fast but may affect material balance

**Max iterations:**

- `250` (default): Sufficient for most problems
- Increase to `500` for difficult convergence
- If solver doesn't converge within max_iterations, timestep is rejected

!!! warning "Convergence Failure"
    If solver consistently fails to converge:

    1. Check model setup (unrealistic properties)
    2. Try different solver/preconditioner
    3. Reduce timestep size
    4. Increase `max_iterations`

---

## Timestep Rejection Criteria

BORES rejects timesteps that violate physical constraints.

```python
config = bores.Config(
    max_pressure_change=100.0,        # psi per timestep
    max_oil_saturation_change=0.5,    # Absolute change
    max_water_saturation_change=0.4,
    max_gas_saturation_change=0.85,   # Gas can change more
    ...
)
```

**What happens when a criterion is violated:**

1. Timestep is **rejected**
2. Step size reduced (by `timer.backoff_factor`)
3. Timestep **retried** with smaller dt

**Typical values:**

- **Waterfloods**: Oil 0.3-0.5, Water 0.3-0.4, Pressure 50-100 psi
- **Gas injection**: Gas 0.7-0.9, Oil 0.4-0.6, Pressure 100-200 psi
- **Primary depletion**: Oil 0.5-0.7, Pressure 100-150 psi

!!! tip "Tuning Rejection Criteria"
    - **Too strict**: Many rejections, slow progress
    - **Too loose**: Instability, poor material balance
    - Start with defaults, adjust if needed

---

## The Run Class

`bores.Run` executes the simulation and yields results.

### Creating a Run

```python
run = bores.Run(
    model: ReservoirModel[ThreeDimensions],
    config: Config,
    name: Optional[str] = None,
    description: Optional[str] = None,
)
```

**Parameters:**

- `model`: Reservoir model from `bores.reservoir_model()`
- `config`: Configuration from `bores.Config()`
- `name`: Optional human-readable name
- `description`: Optional description

**Example:**

```python
run = bores.Run(
    model=model,
    config=config,
    name="5-Year Waterflood",
    description="Five-spot pattern with 1 injector, 4 producers",
)
```

---

### Executing a Run

Call `run()` to get an iterator:

```python
# Basic execution
for state in run():
    # Process each timestep
    print(f"Time: {state.time / 86400:.1f} days")
    print(f"Pressure: {state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

**The `state` object:**

```python
state.step          # Timestep index (0, 1, 2, ...)
state.time          # Simulation time (seconds)
state.step_size     # Timestep size used (seconds)
state.model         # Updated ReservoirModel
state.wells         # Wells configuration at this state
state.injection     # Injection rate grids (by phase)
state.production    # Production rate grids (by phase)
state.relative_permeabilities  # kr grids (krw, kro, krg)
state.relative_mobilities      # Mobility grids
state.capillary_pressures      # Pc grids (Pcow, Pcgo)
state.timer_state              # Timer state (for diagnostics)
```

---

### Processing Results

#### Example 1: Monitor Pressure Decline

```python
pressures = []
times = []

for state in run():
    avg_pressure = state.model.fluid_properties.pressure_grid.mean()
    pressures.append(avg_pressure)
    times.append(state.time / 86400)  # Convert to days

    if state.step % 10 == 0:
        print(f"Day {times[-1]:.1f}: Pressure = {pressures[-1]:.1f} psi")
```

#### Example 2: Track Production Rates

```python
oil_rates = []

for state in run():
    # Sum oil production from all cells
    total_oil_rate = state.production.oil.sum()  # STB/day
    oil_rates.append(total_oil_rate)

    if state.step % 5 == 0:
        print(f"Step {state.step}: Oil rate = {oil_rates[-1]:.1f} STB/day")
```

#### Example 3: Collect All States

```python
states = list(run())  # Collect all states (WARNING: high memory usage!)

# Access any timestep
final_state = states[-1]
mid_state = states[len(states) // 2]
```

!!! warning "Memory Consideration"
    Collecting all states uses memory: `~100 MB per 10K cells per 100 timesteps`. For large simulations, use `StateStream` instead (see [Storage & Serialization](../advanced/storage-serialization.md)).

---

### Saving and Loading Runs

Save run configuration:

```python
run.to_file("my_run.h5")
```

Load and continue:

```python
run = bores.Run.from_files(
    model_path="model.h5",
    config_path="config.yaml",
    pvt_table_path="pvt.h5",  # Optional
)

# Execute
for state in run():
    ...
```

---

## Complete Example

```python
import bores

# Model (simplified)
model = bores.reservoir_model(...)

# Wells
producer = bores.production_well(...)
injector = bores.injection_well(...)
wells = bores.wells_(injectors=[injector], producers=[producer])

# Rock-fluid properties
rel_perm = bores.BrooksCoreyThreePhaseRelPermModel(...)
cap_pressure = bores.BrooksCoreyCapillaryPressureModel(...)
rock_fluid_tables = bores.RockFluidTables(rel_perm, cap_pressure)

# PVT tables
pvt_data = bores.build_pvt_table_data(...)
pvt_tables = bores.PVTTables(data=pvt_data)

# Cached preconditioner
precond = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,
)
precond.register(override=True)

# Timer
timer = bores.Timer(
    initial_step_size=bores.Time(hours=4),
    max_step_size=bores.Time(days=5),
    min_step_size=bores.Time(minutes=10),
    simulation_time=bores.Time(days=365 * 5),  # 5 years
    max_cfl_number=0.9,
    ramp_up_factor=1.2,
    backoff_factor=0.5,
)

# Config
config = bores.Config(
    timer=timer,
    rock_fluid_tables=rock_fluid_tables,
    wells=wells,
    pvt_tables=pvt_tables,
    scheme="impes",                    # IMPES scheme
    pressure_solver="bicgstab",
    pressure_preconditioner="cached_ilu",
    pressure_convergence_tolerance=1e-6,
    saturation_convergence_tolerance=1e-4,
    max_iterations=200,
    output_frequency=1,
    log_interval=5,
    max_pressure_change=100.0,
    max_oil_saturation_change=0.5,
)

# Run
run = bores.Run(
    model=model,
    config=config,
    name="5-Year Waterflood",
)

# Execute
print("Starting simulation...")
for state in run():
    if state.step % 20 == 0:
        day = state.time / 86400
        pressure = state.model.fluid_properties.pressure_grid.mean()
        oil_sat = state.model.fluid_properties.oil_saturation_grid.mean()
        print(f"Day {day:6.1f}: P = {pressure:.1f} psi, So = {oil_sat:.3f}")

print("Simulation complete!")
```

---

## Next Steps

- [Analyzing Results →](analyzing-results.md) - Extract production data and calculate metrics
- [Visualization →](visualization.md) - Create plots and 3D visualizations
- [Storage & Serialization →](../advanced/storage-serialization.md) - Save large simulations efficiently
