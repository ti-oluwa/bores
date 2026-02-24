# Best Practices

Learn best practices for building efficient, accurate, and maintainable reservoir simulations with BORES.

---

## Overview

These guides help you avoid common pitfalls and build better simulations.

<div class="grid cards" markdown>

-   :material-grid:{ .lg .middle } **Grid Design**

    ---

    Choose the right grid size, aspect ratio, and refinement strategy

    [:octicons-arrow-right-24: Read Guide](grid-design.md)

-   :material-cog:{ .lg .middle } **Solver Selection**

    ---

    Pick the best solver and preconditioner for your problem

    [:octicons-arrow-right-24: Read Guide](solver-selection.md)

-   :material-clock-fast:{ .lg .middle } **Timestep Control**

    ---

    Configure adaptive timestepping for stability and efficiency

    [:octicons-arrow-right-24: Read Guide](timestep-control.md)

-   :material-check-circle:{ .lg .middle } **Validation**

    ---

    Verify your results and ensure physical correctness

    [:octicons-arrow-right-24: Read Guide](validation.md)

</div>

---

## Quick Tips

### General

!!! tip "Start Simple"
    Begin with small grids (10×10×5) and short simulations (30 days). Scale up once validated.

!!! tip "Use 32-bit Precision"
    Default 32-bit is sufficient for most cases and uses half the memory:
    ```python
    bores.use_32bit_precision()  # Default
    ```

!!! warning "Check Saturation Sum"
    Always verify: `So + Sw + Sg = 1.0`
    ```python
    total = oil_sat + water_sat + gas_sat
    assert np.allclose(total, 1.0), "Saturations don't sum to 1"
    ```

### Grid Design

- **Cell aspect ratio**: Keep dx ≈ dy ≈ dz (ideally 1:1:1, max 10:1)
- **Vertical refinement**: Use more layers near wells and contacts
- **Horizontal refinement**: Refine around wells (5× local refinement typical)
- **Total cells**: Start with 1K-10K, scale to 100K-500K for production runs

### Well Configuration

- **BHP limits**: Always set realistic limits (e.g., 800 psi minimum for producers)
- **Skin factor**: Use 0-5 for damaged wells, negative for stimulated
- **Perforation**: Perforate multiple layers for stability
- **Well spacing**: Maintain at least 5 cells between wells

### Solver Settings

- **IMPES scheme**: Best for most problems (default)
- **BiCGSTAB solver**: Fast and stable for most cases
- **ILU preconditioner**: Good general-purpose choice
- **Tolerance**: `1e-6` for pressure, `1e-4` for saturation

### Timestep Control

- **Initial step**: 2-4 hours for waterfloods, 1 day for depletion
- **Max step**: 5-10 days (don't exceed well schedule intervals)
- **CFL number**: Keep < 1.0 for stability (0.8-0.9 recommended)
- **Ramp-up**: 1.1-1.2 (conservative growth)

---

## Common Pitfalls

### ❌ Don't Do This

```python
# Giant first timestep
timer = bores.Timer(initial_step_size=bores.Time(days=30))  # Too large!

# No BHP limit
control = bores.ConstantRateControl(target_rate=-1000)  # Can violate physics

# Extreme aspect ratio
cell_dimension = (500.0, 500.0)  # With 5ft thickness = 100:1 ratio!
```

### ✅ Do This Instead

```python
# Conservative initial timestep
timer = bores.Timer(initial_step_size=bores.Time(hours=4))

# Safe BHP limit
control = bores.AdaptiveBHPRateControl(
    target_rate=-1000,
    bhp_limit=800,  # Physical constraint
)

# Balanced aspect ratio
cell_dimension = (100.0, 100.0)  # With 20ft thickness ≈ 5:1 ratio
```

---

## Performance Checklist

Before running large simulations:

- [ ] Grid size reasonable (< 500K cells for workstation)
- [ ] 32-bit precision enabled
- [ ] Cached preconditioner configured
- [ ] PVT tables built (faster than correlations)
- [ ] Output frequency set (don't save every timestep)
- [ ] State streaming enabled for large runs

Example optimized config:

```python
# Cached preconditioner (reuses factorization)
precond = bores.CachedPreconditionerFactory(
    factory="ilu",
    name="cached_ilu",
    update_frequency=10,  # Rebuild every 10 steps
    recompute_threshold=0.3,
)
precond.register(override=True)

config = bores.Config(
    pressure_preconditioner="cached_ilu",
    output_frequency=10,  # Save every 10 steps
    pvt_tables=pvt_tables,  # Use tables, not correlations
)
```

---

## Validation Workflow

1. **Material balance**: Check that OOIP/OGIP is conserved
2. **Comparison**: Run against analytical solutions (1D cases)
3. **Convergence**: Verify results don't change with finer grid
4. **Physical limits**: Ensure 0 ≤ S ≤ 1, P > 0, etc.

---

## Next Steps

Dive deeper into specific topics:

- [Grid Design →](grid-design.md)
- [Solver Selection →](solver-selection.md)
- [Timestep Control →](timestep-control.md)
- [Validation →](validation.md)
