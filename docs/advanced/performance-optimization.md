# Performance Optimization

Tips and techniques for faster simulations and lower memory usage.

---

## Overview

BORES simulations can be optimized for:

1. **Execution speed** - Reduce runtime
2. **Memory usage** - Handle larger models
3. **Convergence** - Fewer solver iterations

---

## Quick Wins

### 1. Use 32-bit Precision

**Impact**: 50% memory reduction, 10-30% faster

```python
bores.use_32bit_precision()  # Default, but be explicit
```

**When to use 64-bit:**
- Very long simulations (> 10 years)
- Need high numerical accuracy
- Debugging convergence issues

---

### 2. Use PVT Tables

**Impact**: 20-40% faster than correlations

```python
# Build once
pvt_data = bores.build_pvt_table_data(...)
pvt_tables = bores.PVTTables(data=pvt_data)

# Use in all runs
config = bores.Config(..., pvt_tables=pvt_tables)
```

---

### 3. Cached Preconditioners

**Impact**: 20-40% faster for long simulations

```python
precond = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,  # Rebuild every 10 steps
    recompute_threshold=0.3,
)
precond.register(override=True)

config = bores.Config(..., pressure_preconditioner="cached_ilu")
```

---

### 4. Reduce Output Frequency

**Impact**: Faster execution, less I/O

```python
config = bores.Config(
    ...,
    output_frequency=10,  # Yield every 10 steps instead of 1
)
```

---

### 5. Use State Streaming

**Impact**: Constant memory usage for large simulations

```python
store = bores.ZarrStore("results.zarr")
stream = bores.StateStream(
    run(),
    store=store,
    batch_size=50,  # Write every 50 states
    background_io=True,   # Async disk I/O
)

with stream:
    last_state = stream.last()  # Only keeps last state in memory
```

---

## Grid Optimization

### Cell Count

**Target**: 10K-50K cells for workstation, 100K-500K for server

```python
# Calculate total cells
total_cells = nx * ny * nz

# Example
grid_shape = (30, 20, 10)  # 6000 cells - good for testing
grid_shape = (50, 50, 20)  # 50K cells - production
grid_shape = (100, 100, 30)  # 300K cells - large model
```

### Aspect Ratio

**Target**: 1:1:1 to 10:1 (horizontal:vertical)

```python
# Good
cell_dimension = (100.0, 100.0)  # dx, dy
thickness = 20.0  # dz ≈ 20 ft → ratio 5:1 ✓

# Bad
cell_dimension = (500.0, 500.0)
thickness = 10.0  # ratio 50:1 - too high!
```

### Local Refinement

Refine near wells, not everywhere:

```python
# Coarse base grid
grid_shape = (40, 40, 10)  # 16K cells

# Refine near wells in post-processing, or use
# smaller cells only where needed
```

---

## Solver Optimization

### Solver Selection

**Fastest** to **most robust**:

1. `"cgs"` - Fastest, needs good preconditioner
2. `"bicgstab"` - **Best balance** (default)
3. `"gmres"` - Slower, more robust
4. `"lgmres"` - Slowest, most robust

```python
config = bores.Config(
    pressure_solver="bicgstab",  # Default, good choice
)
```

### Preconditioner Selection

**By grid size:**

| Grid Size | Preconditioner | Why |
|-----------|----------------|-----|
| < 10K cells | `"diagonal"` or `"ilu"` | Fast setup |
| 10K-100K cells | `"ilu"` (cached) | **Best balance** |
| > 100K cells | `"amg"` | Scales better |

```python
# Small grids
config = bores.Config(pressure_preconditioner="ilu")

# Large grids
config = bores.Config(pressure_preconditioner="amg")
```

### Tolerance Tuning

Relax tolerances for speed:

```python
config = bores.Config(
    pressure_convergence_tolerance=1e-5,  # Default: 1e-6
    saturation_convergence_tolerance=1e-3,  # Default: 1e-4
)
```

!!! warning "Material Balance"
    Looser tolerances may affect material balance. Monitor cumulative production.

---

## Timestep Optimization

### Larger Initial Step

Start with larger timestep:

```python
timer = bores.Timer(
    initial_step_size=bores.Time(hours=12),  # Instead of 2-4 hours
    max_step_size=bores.Time(days=10),  # Instead of 5 days
)
```

### Relax CFL

For IMPES, allow higher CFL:

```python
timer = bores.Timer(
    max_cfl_number=0.95,  # Default: 0.9 (can push to 0.99)
)
```

### Faster Ramp-up

Grow timestep more aggressively:

```python
timer = bores.Timer(
    ramp_up_factor=1.3,  # Default: 1.2
)
```

---

## Memory Optimization

### Estimated Memory Usage

\\[
\\text{Memory (MB)} \\approx \\frac{\\text{cells} \\times \\text{properties} \\times \\text{bytes}}{10^6}
\\]

**Example** (32-bit):
- Grid: 50K cells
- Properties: ~30 grids (pressure, saturation, etc.)
- Bytes: 4 bytes per float32
- **Memory**: 50K × 30 × 4 / 1M = **6 MB** per state

**For 1000 timesteps**: 6 GB if all states kept in memory!

### Solution: Stream to Disk

```python
store = bores.ZarrStore("results.zarr")
stream = bores.StateStream(run(), store=store, batch_size=100)

with stream:
    # Memory usage: ~6 MB (only current state)
    for state in stream:
        pass  # State automatically written to disk
```

### Solution: Reduce Output

```python
config = bores.Config(output_frequency=20)  # Every 20 steps
# 1000 steps → only 50 states saved → 300 MB total
```

---

## Parallel Processing

### Run Multiple Scenarios

```python
from multiprocessing import Pool

def run_scenario(params):
    model = bores.reservoir_model(..., **params)
    run = bores.Run(model, config)
    return list(run())

# Run 4 scenarios in parallel
with Pool(4) as pool:
    results = pool.map(run_scenario, param_list)
```

### NumPy Threading

BORES uses NumPy - control thread count:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python simulation.py
```

---

## Benchmarking

### Measure Performance

```python
import time

start = time.time()

run = bores.Run(model, config)
states = list(run())

elapsed = time.time() - start
print(f"Runtime: {elapsed:.1f} seconds")
print(f"Steps: {len(states)}")
print(f"Time per step: {elapsed / len(states):.2f} s")
```

### Profile Hotspots

```python
import cProfile

cProfile.run('list(run())', 'profile.stats')

# Analyze
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

---

## Performance Checklist

Before running large simulations:

- [ ] 32-bit precision enabled
- [ ] PVT tables built
- [ ] Cached preconditioner configured
- [ ] Grid size reasonable (< 500K cells)
- [ ] Output frequency set (> 1)
- [ ] State streaming enabled
- [ ] Timestep sizes tuned
- [ ] Solver/preconditioner appropriate for grid size

---

## Typical Performance

**Workstation** (Intel i7, 16 GB RAM):

| Grid Size | Timesteps | Runtime | Steps/sec |
|-----------|-----------|---------|-----------|
| 1K cells | 100 | 10 sec | 10 |
| 10K cells | 100 | 2 min | 0.8 |
| 50K cells | 100 | 15 min | 0.1 |
| 100K cells | 100 | 45 min | 0.04 |

**Server** (32 cores, 128 GB RAM):

| Grid Size | Timesteps | Runtime | Steps/sec |
|-----------|-----------|---------|-----------|
| 50K cells | 1000 | 30 min | 0.5 |
| 300K cells | 1000 | 4 hours | 0.07 |
| 1M cells | 1000 | 18 hours | 0.015 |

---

## Next Steps

- [Storage & Serialization →](storage-serialization.md) - Efficient data management
- [Best Practices →](../best-practices/index.md) - General optimization tips
