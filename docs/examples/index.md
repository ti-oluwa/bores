# Examples

Complete, working simulation examples you can run immediately.

---

## Overview

These examples demonstrate real-world simulation scenarios with full working code. Each example includes:

- **Complete source code** that runs as-is
- **Explanation** of setup and physics
- **Results** showing expected output
- **Variations** you can try

---

## Available Examples

<div class="grid cards" markdown>

-   :material-oil:{ .lg .middle } **Primary Depletion**

    ---

    Natural depletion with solution gas drive

    **Time**: 20 min | **Complexity**: ⭐

    [:octicons-arrow-right-24: View Example](primary-depletion.md)

-   :material-water-plus:{ .lg .middle } **Waterflood Pattern**

    ---

    Five-spot injection pattern with 4 producers and 1 injector

    **Time**: 30 min | **Complexity**: ⭐⭐

    [:octicons-arrow-right-24: View Example](waterflood-pattern.md)

-   :material-gas-cylinder:{ .lg .middle } **Gas Cap Expansion**

    ---

    Primary depletion with active gas cap

    **Time**: 25 min | **Complexity**: ⭐⭐

    [:octicons-arrow-right-24: View Example](gas-cap-expansion.md)

-   :material-molecule-co2:{ .lg .middle } **Miscible CO₂ EOR**

    ---

    Advanced miscible flooding with Todd-Longstaff model

    **Time**: 40 min | **Complexity**: ⭐⭐⭐

    [:octicons-arrow-right-24: View Example](miscible-eor.md)

</div>

---

## Example Structure

Each example follows this pattern:

1. **Objective** - What the simulation demonstrates
2. **Model Setup** - Grid and property configuration
3. **Well Configuration** - Injection/production setup
4. **Simulation Config** - Solver and timestep settings
5. **Running** - Execute and monitor
6. **Results** - Expected output and analysis
7. **Variations** - How to modify the example

---

## Running the Examples

### Method 1: Copy and Run

1. Copy the complete code from an example
2. Save to a `.py` file
3. Run: `python example.py`

### Method 2: Interactive (Recommended)

Use `marimo` for interactive exploration:

```bash
# Install marimo
uv add --dev marimo

# Run example interactively
marimo edit example.py
```

!!! tip "Marimo Notebooks"
    The scenarios in `scenarios/` directory are marimo notebooks. They're the same examples shown here, but fully interactive!

---

## Example Data

Examples use realistic parameters based on:

- **Sandstone reservoirs** (typical North Sea, Gulf Coast)
- **Black oil** (30-40° API)
- **Moderate depth** (6000-9000 ft)
- **Typical permeability** (10-500 mD)

Feel free to modify properties for your use case!

---

## Complexity Guide

| Level | Meaning |
|-------|---------|
| ⭐ | Beginner - Simple setup, few features |
| ⭐⭐ | Intermediate - Multiple wells, waterflooding |
| ⭐⭐⭐ | Advanced - Miscible flooding, complex controls |

---

## Next Steps

After running examples:

1. **Modify parameters** - Change rates, grid size, properties
2. **Add features** - Try different well controls, boundary conditions
3. **Analyze results** - Use the [Analysis Guide](../guides/analyzing-results.md)
4. **Build your own** - Apply concepts to your reservoir

---

**Ready to run?** Start with [Primary Depletion →](primary-depletion.md)
