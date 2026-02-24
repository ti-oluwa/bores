# Solver Selection Best Practices

Guidelines for choosing pressure solvers and preconditioners.

---

## Overview

BORES uses **iterative Krylov solvers** for the pressure equation. Choice affects:

- **Convergence rate**: Iterations to solution
- **Robustness**: Success rate for difficult problems
- **Memory usage**: Solver workspace
- **CPU time**: Per-iteration cost

**Goal**: Match solver to problem characteristics for best performance.

---

## Available Solvers

### Solver Comparison

| Solver | Speed | Robustness | Memory | Best For |
| ------ | ----- | ---------- | ------ | -------- |
| **BiCGSTAB** | Fast | Good | Low | General purpose (default) |
| **GMRES** | Medium | Excellent | High | Difficult problems |
| **LGMRES** | Medium | Very good | Medium | Complex wells |
| **CGS** | Fast | Fair | Low | Well-conditioned |
| **TFQMR** | Fast | Good | Low | Alternative to BiCGSTAB |
| **CG** | Fastest | Fair | Lowest | Symmetric (no wells) |
| **Direct** | Slow | Perfect | Very high | Small grids only |

---

## Solver Selection Guide

### By Grid Size

```python
import bores

# Small grid (<10k cells)
config = bores.Config(
    pressure_solver="direct",  # Guaranteed convergence
    # ... other parameters
)

# Medium grid (10k-100k cells)
config = bores.Config(
    pressure_solver="bicgstab",  # Fast, reliable (default)
    pressure_solver_tolerance=1e-6,
)

# Large grid (>100k cells)
config = bores.Config(
    pressure_solver="gmres",  # Most robust
    pressure_solver_tolerance=1e-5,  # Relaxed tolerance
    gmres_restart=30,  # Restart parameter
)
```

### By Problem Type

**Standard waterflooding**:

```python
config = bores.Config(
    pressure_solver="bicgstab",  # Default, works well
    pressure_preconditioner="ilu",  # ILU preconditioner
)
```

**Gas injection (compressible)**:

```python
config = bores.Config(
    pressure_solver="gmres",  # More robust for gas
    pressure_preconditioner="amg",  # AMG handles anisotropy
    gmres_restart=50,
)
```

**Highly heterogeneous**:

```python
config = bores.Config(
    pressure_solver="lgmres",  # Handles heterogeneity well
    pressure_preconditioner="cpr",  # CPR for multi-phase
)
```

**Many wells**:

```python
config = bores.Config(
    pressure_solver="lgmres",  # Good for well constraints
    pressure_preconditioner="ilu",
    pressure_solver_max_iterations=500,  # Allow more iterations
)
```

---

## Preconditioners

Preconditioners transform the linear system to improve convergence.

### Preconditioner Comparison

| Preconditioner | Setup Cost | Per-Iter Cost | Effectiveness | Best For |
| -------------- | ---------- | ------------- | ------------- | -------- |
| **ILU** | Low | Low | Good | General (default) |
| **AMG** | High | Medium | Excellent | Anisotropic grids |
| **CPR** | Medium | Medium | Very good | Multi-phase |
| **Diagonal** | None | Very low | Fair | Well-conditioned |
| **Block Jacobi** | Low | Low | Fair | Parallel (future) |
| **Polynomial** | Low | Low | Good | Large grids |

### Preconditioner Selection

**Default (works for most cases)**:

```python
config = bores.Config(
    pressure_preconditioner="ilu",  # Fast, reliable
)
```

**Anisotropic permeability** (kx ≠ ky ≠ kz):

```python
config = bores.Config(
    pressure_preconditioner="amg",  # Algebraic multigrid
)
```

**Three-phase flow**:

```python
config = bores.Config(
    pressure_preconditioner="cpr",  # Constrained Pressure Residual
)
```

**Large grids** (>200k cells):

```python
config = bores.Config(
    pressure_preconditioner="polynomial",  # Low memory
)
```

---

## Cached Preconditioners

Reuse preconditioner across timesteps for 20-40% speedup:

```python
import bores

# Register cached ILU preconditioner
ilu_cached = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,  # Rebuild every 10 timesteps
    recompute_threshold=0.3,  # Rebuild if matrix changes >30%
)
ilu_cached.register(override=True)

# Use in config
config = bores.Config(
    pressure_preconditioner="cached_ilu",  # Use cached version
    # ... other parameters
)
```

**Benefits**:

- 20-40% faster for typical simulations
- Automatic rebuilding when needed
- No accuracy loss

**Drawbacks**:

- Not effective if matrix changes rapidly
- Slightly more memory

---

## Convergence Tolerances

### Pressure Solver Tolerance

```python
# Tight tolerance (accurate, slower)
config = bores.Config(
    pressure_solver_tolerance=1e-8,  # Very accurate
)

# Standard tolerance (recommended)
config = bores.Config(
    pressure_solver_tolerance=1e-6,  # Good balance
)

# Relaxed tolerance (fast, less accurate)
config = bores.Config(
    pressure_solver_tolerance=1e-4,  # Faster convergence
)
```

**Guidelines**:

- **1e-8**: Research, verification
- **1e-6**: Production runs (default)
- **1e-4**: Scoping studies, conceptual models

### Maximum Iterations

```python
# Standard
config = bores.Config(
    pressure_solver_max_iterations=200,  # Default
)

# Difficult problems
config = bores.Config(
    pressure_solver_max_iterations=500,  # Allow more iterations
)
```

---

## Solver-Specific Parameters

### GMRES Restart

GMRES memory grows with iterations - restart to limit:

```python
config = bores.Config(
    pressure_solver="gmres",
    gmres_restart=30,  # Restart after 30 iterations
)
```

**Guidelines**:

- **restart=20**: Low memory (large grids)
- **restart=30**: Balanced (default)
- **restart=50**: Better convergence (small grids)

---

## Troubleshooting

### Solver Fails to Converge

**Symptoms**:

- "Maximum iterations reached"
- Timestep rejection
- Slow progress

**Solutions**:

1. **Switch solver**:

    ```python
    # If BiCGSTAB fails, try GMRES
    config = bores.Config(
        pressure_solver="gmres",  # More robust
    )
    ```

2. **Improve preconditioner**:

    ```python
    config = bores.Config(
        pressure_preconditioner="amg",  # Better than ILU
    )
    ```

3. **Increase iterations**:

    ```python
    config = bores.Config(
        pressure_solver_max_iterations=500,
    )
    ```

4. **Relax tolerance**:

    ```python
    config = bores.Config(
        pressure_solver_tolerance=1e-5,  # Less strict
    )
    ```

### Solver Too Slow

**Symptoms**:

- Many iterations per timestep (>100)
- Long wall-clock time

**Solutions**:

1. **Use cached preconditioner**:

    ```python
    ilu_cached = bores.CachedPreconditionerFactory(
        factory="ilu",
        name="cached_ilu",
        update_frequency=10,
    )
    ilu_cached.register(override=True)
    ```

2. **Tighten convergence criteria** (counterintuitive but can help):

    ```python
    config = bores.Config(
        pressure_solver_tolerance=1e-7,  # Tighter tolerance
        # Prevents drift and cumulative errors
    )
    ```

3. **Coarsen timesteps**:

    ```python
    timer = bores.Timer(
        max_step_size=bores.Time(days=30),  # Larger steps
    )
    ```

---

## Performance Monitoring

### Track Solver Iterations

```python
# Access solver stats from state
for state in run():
    if state.timer_state:
        iters = state.timer_state.solver_iterations
        if iters > 100:
            print(f"Step {state.step}: High iteration count ({iters})")
```

### Benchmarking Solvers

Test different solvers for your specific problem:

```python
import time

solvers = ["bicgstab", "gmres", "lgmres"]
results = {}

for solver in solvers:
    config = bores.Config(
        pressure_solver=solver,
        # ... same other parameters
    )

    start = time.time()
    run = bores.Run(config=config)
    for state in run():
        pass  # Consume states
    elapsed = time.time() - start

    results[solver] = elapsed
    print(f"{solver}: {elapsed:.1f} seconds")

# Use fastest solver
best_solver = min(results, key=results.get)
print(f"Best solver: {best_solver}")
```

---

## Advanced: Custom Solver Settings

For expert users, fine-tune solver behavior:

```python
config = bores.Config(
    pressure_solver="gmres",
    pressure_solver_tolerance=1e-6,
    pressure_solver_max_iterations=300,
    pressure_preconditioner="amg",

    # GMRES-specific
    gmres_restart=40,

    # Preconditioner-specific
    # (future: AMG parameters, ILU fill level, etc.)
)
```

---

## Recommended Configurations

### Quick Scoping Study

```python
config = bores.Config(
    pressure_solver="bicgstab",
    pressure_solver_tolerance=1e-5,  # Relaxed
    pressure_preconditioner="diagonal",  # Fast
)
```

### Production Run (Balanced)

```python
# Register cached preconditioner
bores.CachedPreconditionerFactory(
    factory="ilu", name="cached_ilu", update_frequency=10
).register(override=True)

config = bores.Config(
    pressure_solver="bicgstab",
    pressure_solver_tolerance=1e-6,
    pressure_preconditioner="cached_ilu",  # 20-40% speedup
)
```

### High-Accuracy Research

```python
config = bores.Config(
    pressure_solver="gmres",
    pressure_solver_tolerance=1e-8,  # Tight
    pressure_preconditioner="amg",  # Best preconditioner
    gmres_restart=50,
    pressure_solver_max_iterations=500,
)
```

### Large Field Model (>200k cells)

```python
config = bores.Config(
    pressure_solver="gmres",
    pressure_solver_tolerance=1e-5,  # Relaxed for speed
    pressure_preconditioner="polynomial",  # Low memory
    gmres_restart=20,  # Low memory
)
```

---

## Checklist

✅ **Start with defaults**: BiCGSTAB + ILU
✅ **Enable caching**: 20-40% speedup for free
✅ **Monitor convergence**: Track iteration counts
✅ **Switch if needed**: GMRES for robustness
✅ **Benchmark**: Test solvers for your problem

---

## Next Steps

- [Timestep Control](timestep-control.md) - Optimize timestep sizes
- [Performance Optimization](../advanced/performance-optimization.md) - Advanced speedup techniques
- [Running Simulations](../guides/running-simulations.md) - Complete configuration guide
