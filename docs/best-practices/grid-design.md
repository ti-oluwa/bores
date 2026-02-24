# Grid Design Best Practices

Guidelines for designing efficient and accurate reservoir grids.

---

## Overview

Grid design significantly affects:

- **Accuracy**: Fine grids capture heterogeneity
- **Performance**: Coarse grids run faster
- **Convergence**: Poor grids cause solver issues
- **Memory**: Grid size determines RAM requirements

**Goal**: Balance accuracy and performance for your specific application.

---

## Grid Resolution

### Cell Count Guidelines

| Reservoir Type | Typical Grid | Cell Count | Use Case |
| -------------- | ------------ | ---------- | -------- |
| **Conceptual study** | 10×10×3 | ~300 | Scoping, screening |
| **Sector model** | 30×30×10 | ~9,000 | Pattern optimization |
| **Full field** | 100×100×20 | ~200,000 | Development planning |
| **High-resolution** | 200×200×50 | ~2,000,000 | Complex reservoirs |

!!! tip "Start Small"
    Begin with coarse grid (10×10×5), verify physics, then refine.

### Horizontal Resolution

**Rule of thumb**: 1 cell ≈ 50-200 ft in each horizontal direction

```python
import bores

# Reservoir dimensions
length_x = 2000  # ft
length_y = 2000  # ft

# Cell size 100 ft × 100 ft
nx = length_x // 100  # 20 cells
ny = length_y // 100  # 20 cells

model = bores.reservoir_model(
    grid_shape=(nx, ny, 5),
    # ... other parameters
)
```

**Factors affecting horizontal resolution**:

- **Well spacing**: Need 3-5 cells between wells
- **Heterogeneity**: Finer grid for variable properties
- **Fluid contacts**: 2-3 cells across transition zone
- **Computational budget**: Memory/time constraints

### Vertical Resolution

**Rule of thumb**: 1 layer ≈ 5-20 ft thickness

```python
# Layered grid with varying resolution
layers = [
    {"thickness": 30, "n_cells": 6},  # 5 ft per cell (high resolution)
    {"thickness": 50, "n_cells": 5},  # 10 ft per cell (moderate)
    {"thickness": 40, "n_cells": 2},  # 20 ft per cell (coarse)
]
```

**Factors affecting vertical resolution**:

- **Layer boundaries**: Honor geological layers
- **Fluid contacts**: Fine grid near OWC/GOC (5-10 ft cells)
- **Permeability contrasts**: Capture flow barriers
- **Gravity segregation**: Finer for buoyancy-driven flow

---

## Aspect Ratio

**Definition**: Ratio of cell dimensions (Δx : Δy : Δz)

**Guidelines**:

- **Ideal**: 1:1:1 (cubic cells) - best numerical properties
- **Acceptable**: 1:1:0.2 to 5:5:1 - common for layered reservoirs
- **Avoid**: >10:1 or <0.1:1 - causes numerical dispersion

```python
# Good aspect ratio (layered reservoir)
cell_size_x = 100  # ft
cell_size_y = 100  # ft
cell_thickness = 20  # ft

aspect_ratio = cell_thickness / cell_size_x  # 0.2 ✅

# Bad aspect ratio
cell_thickness = 5  # ft
aspect_ratio = cell_thickness / cell_size_x  # 0.05 ❌ Too small!
```

!!! warning "Thin Layers"
    Very thin layers (< 5 ft) with large areal cells (> 200 ft) create:

    - Numerical dispersion
    - Slow convergence
    - Inaccurate vertical flow

---

## Grid Orientation

### Align with Major Features

```python
# Fault-aligned grid
# If main fault strikes N-S, align grid i-direction with fault

# Dip-aligned grid
# If reservoir dips in x-direction, align grid with dip

# Well pattern-aligned grid
# For regular patterns, align grid with well lines
```

### Cartesian vs Curvilinear

**Cartesian** (recommended for most cases):

```python
model = bores.reservoir_model(
    grid_shape=(nx, ny, nz),  # Regular Cartesian grid
    # ... parameters
)
```

✅ Simple, fast, stable
❌ Doesn't follow complex geometry

**Curvilinear** (advanced, future feature):

✅ Follows faults, pinchouts
❌ Complex, slower, harder to converge

---

## Refinement Strategies

### Near-Well Refinement

Refine grid around wells for accurate well performance:

```python
# Create fine grid around producer
well_i, well_j = 10, 10

# Standard grid: 100 ft cells
# Near-well: 25 ft cells (4× refinement)

# Option 1: Manually specify fine cells
# (Future feature: local grid refinement)

# Option 2: Use smaller global grid
# Trade-off: More cells everywhere
```

**Guidelines**:

- **Radial refinement**: 3-5 cells within 1 wellbore radius
- **Peaceman correction**: Accounts for subgrid effects
- **Trade-off**: Accuracy vs cell count

### Transition Zone Refinement

Refine near fluid contacts:

```python
# Layers near OWC (oil-water contact at 5150 ft)
layers = [
    # Above OWC
    {"thickness": 40, "n_cells": 2},  # Coarse
    {"thickness": 30, "n_cells": 6},  # Fine (near contact)

    # Below OWC
    {"thickness": 30, "n_cells": 6},  # Fine (near contact)
    {"thickness": 40, "n_cells": 2},  # Coarse
]
```

---

## Upscaling

Convert fine geological model to coarse simulation grid:

### Porosity Upscaling

**Volume-weighted average**:

```python
import numpy as np

# Fine grid
fine_phi = np.array([0.18, 0.22, 0.25, 0.20])  # 4 fine cells
fine_thickness = np.array([5, 5, 5, 5])  # ft

# Coarse cell (combines 4 fine cells)
coarse_phi = np.average(fine_phi, weights=fine_thickness)
print(f"Coarse porosity: {coarse_phi:.3f}")
```

### Permeability Upscaling

**Harmonic average** (for flow in series, e.g., vertical):

```python
fine_perm = np.array([100, 50, 200, 80])  # mD
coarse_perm_harmonic = len(fine_perm) / np.sum(1.0 / fine_perm)
print(f"Harmonic mean perm: {coarse_perm_harmonic:.1f} mD")
```

**Arithmetic average** (for flow in parallel, e.g., horizontal):

```python
coarse_perm_arithmetic = np.mean(fine_perm)
print(f"Arithmetic mean perm: {coarse_perm_arithmetic:.1f} mD")
```

**Power-law average** (compromise):

```python
p = 0.5  # Power (0=harmonic, 1=arithmetic)
coarse_perm = (np.mean(fine_perm**p))**(1/p)
```

---

## Memory and Performance

### Memory Estimation

```python
# Estimate memory usage
nx, ny, nz = 100, 100, 20
n_cells = nx * ny * nz  # 200,000 cells

# Per cell (rough estimate)
bytes_per_cell = 200  # Pressure, saturations, properties, etc.

memory_mb = n_cells * bytes_per_cell / 1e6
print(f"Estimated memory: {memory_mb:.0f} MB ({memory_mb/1024:.1f} GB)")
```

**Guidelines**:

- **<100k cells**: Runs on laptop (4-8 GB RAM)
- **100k-500k**: Desktop (16-32 GB RAM)
- **>500k**: Workstation (64+ GB RAM)

### Performance Scaling

Runtime scales approximately as:

```
Time ∝ N_cells^1.3
```

Doubling cell count → 2.5× longer runtime

```python
# Coarse grid
n_coarse = 50 * 50 * 10  # 25,000 cells
time_coarse = 10  # minutes (example)

# Fine grid (2× resolution)
n_fine = 100 * 100 * 20  # 200,000 cells
time_fine = time_coarse * (n_fine / n_coarse)**1.3
print(f"Estimated fine grid time: {time_fine:.0f} minutes")
```

---

## Validation

### Grid Convergence Study

Test that results don't change with finer grid:

```python
# Run 3 grids: coarse, medium, fine
grids = [
    (10, 10, 5),    # Coarse: 500 cells
    (20, 20, 10),   # Medium: 4,000 cells
    (40, 40, 20),   # Fine: 32,000 cells
]

results = []
for grid_shape in grids:
    model = bores.reservoir_model(grid_shape=grid_shape, ...)
    # ... run simulation
    results.append(recovery_factor)

# Check convergence
print("Recovery factors:")
for grid, rf in zip(grids, results):
    print(f"  {grid}: {rf:.2%}")

# If medium ≈ fine (within 1%), medium is sufficient
```

### Checklist

✅ **Aspect ratio**: 0.1 < Δz/Δx < 10
✅ **Well spacing**: ≥3 cells between wells
✅ **Contact zones**: 2-3 cells across transition
✅ **Memory**: < 80% of available RAM
✅ **Convergence**: Medium grid within 1% of fine grid

---

## Common Pitfalls

### 1. Too Coarse

**Symptom**: Unphysical results, poor well performance

**Fix**: Refine grid, especially near wells and contacts

### 2. Too Fine

**Symptom**: Runs forever, out of memory

**Fix**: Coarsen grid, use upscaling

### 3. Bad Aspect Ratio

**Symptom**: Slow convergence, numerical dispersion

**Fix**: Balance cell dimensions (1:1:0.2 to 5:5:1)

### 4. Misaligned Grid

**Symptom**: Stair-stepped faults, poor sweep patterns

**Fix**: Align grid with major features

---

## Example: Optimal Grid Design

```python
import bores
import numpy as np

# Reservoir characteristics
reservoir_area = 2000 * 2000  # ft² (2000 ft × 2000 ft)
reservoir_thickness = 120  # ft
n_wells = 5  # 1 producer + 4 injectors

# Step 1: Horizontal resolution
# Target: 5 cells between wells, wells spaced ~500 ft
well_spacing = 500  # ft
cells_between_wells = 5
cell_size = well_spacing / cells_between_wells  # 100 ft

nx = int(2000 / cell_size)  # 20 cells
ny = int(2000 / cell_size)  # 20 cells

# Step 2: Vertical resolution
# 3 layers: shale barrier at 40 ft depth
layers = [
    {"thickness": 40, "n_cells": 4},   # 10 ft per cell (upper sand)
    {"thickness": 10, "n_cells": 2},   # 5 ft per cell (shale barrier)
    {"thickness": 70, "n_cells": 7},   # 10 ft per cell (lower sand)
]
nz = sum(layer["n_cells"] for layer in layers)  # 13 cells

# Step 3: Check aspect ratio
aspect_ratio = 10 / cell_size  # 0.1 ✅ Acceptable

# Step 4: Estimate memory
n_cells = nx * ny * nz  # 5,200 cells
memory_mb = n_cells * 200 / 1e6  # ~1 MB (very small!)

print(f"Grid design:")
print(f"  Shape: {nx} × {ny} × {nz} = {n_cells} cells")
print(f"  Cell size: {cell_size} × {cell_size} × 10 ft (avg)")
print(f"  Aspect ratio: {aspect_ratio:.2f}")
print(f"  Memory: {memory_mb:.1f} MB")
print(f"  ✅ Good design for sector model")
```

---

## Next Steps

- [Solver Selection](solver-selection.md) - Choose appropriate solver
- [Timestep Control](timestep-control.md) - Optimize timestep sizes
- [Performance Optimization](../advanced/performance-optimization.md) - Speed up simulations
