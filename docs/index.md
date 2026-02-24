# BORES Framework

**3D Three-Phase Black-Oil Reservoir Modelling and Simulation Framework in Python**

<div style="text-align: center; margin: 2rem 0;">
  <img src="https://img.shields.io/pypi/v/bores-framework?color=blue" alt="PyPI">
  <img src="https://img.shields.io/pypi/pyversions/bores-framework" alt="Python">
  <img src="https://img.shields.io/github/license/ti-oluwa/bores" alt="License">
</div>

---

## What is BORES?

BORES (Black-Oil REservoir Simulator) is a **3D three-phase reservoir simulation framework** designed for educational, research, and prototyping purposes. It provides a clean, Pythonic API for building reservoir models, running simulations, and analyzing results.

!!! warning "Important Disclaimer"
    BORES is designed for **educational, research, and prototyping purposes**. It is **not production-grade software** and should not be used for critical business decisions, regulatory compliance, or field development planning. Always validate results against established commercial simulators.

---

## Why BORES?

Existing reservoir simulators are either:

- **Closed-source** (Eclipse, CMG) - expensive, limited extensibility
- **Low-level languages** (C/C++, Fortran) - steep learning curve
- **Complex APIs** (MRST, OPM) - difficult to prototype with

BORES fills this gap by providing:

- **Simple Python API** - Easy to learn, even for beginners
- **Fully 3D** - Three-phase (oil, water, gas) black-oil modeling
- **Educational Focus** - Clear code, comprehensive documentation
- **Extensible** - Build custom models and workflows
- **Fast Prototyping** - Test ideas quickly with Pythonic syntax

!!! info "Unit System"
    BORES uses **Oilfield Units** throughout: feet (ft), psi, STB, SCF, °F, etc.

---

## Quick Example

Here's a complete waterflood simulation in ~50 lines:

```python
import bores

# Define grid
grid_shape = (30, 20, 6)
cell_dimension = (100.0, 100.0)  # ft

# Build model using factory
model = bores.reservoir_model(
    grid_shape=grid_shape,
    cell_dimension=cell_dimension,
    thickness_grid=thickness_grid,
    pressure_grid=pressure_grid,
    porosity_grid=porosity_grid,
    absolute_permeability=permeability,
    # ... rock and fluid properties
)

# Define wells
injector = bores.injection_well(
    well_name="I-1",
    perforating_intervals=[((5, 5, 2), (5, 5, 4))],
    control=bores.ConstantRateControl(target_rate=500),  # STB/day
    injected_fluid=bores.InjectedFluid(phase=bores.FluidPhase.WATER),
)

producer = bores.production_well(
    well_name="P-1",
    perforating_intervals=[((25, 15, 2), (25, 15, 4))],
    control=bores.AdaptiveBHPRateControl(
        target_rate=-300,  # STB/day
        bhp_limit=800,  # psi
    ),
)

# Configure simulation
config = bores.Config(
    timer=bores.Timer(
        initial_step_size=bores.Time(hours=4),
        max_step_size=bores.Time(days=10),
        min_step_size=bores.Time(hours=1),
        simulation_time=bores.Time(days=bores.c.DAYS_PER_YEAR * 5),  # 5 years
    ),
    wells=bores.wells_(injectors=[injector], producers=[producer]),
    scheme="impes",
)

# Run simulation
run = bores.Run(model=model, config=config)
for state in run():
    print(f"Day {state.time / 86400:.1f}, Pressure: {state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

---

## Key Features

### **Model Building**

- Layered and uniform grid builders
- Structural dip support (azimuth convention)
- Saturation initialization with fluid contacts
- Anisotropic permeability

### **Wells & Controls**

- Production and injection wells
- Adaptive BHP and rate controls
- Multi-phase production
- Miscible gas injection (Todd-Longstaff model)
- Well scheduling

### **Physics**

- IMPES and explicit schemes
- PVT correlations (Standing, Vasquez-Beggs, etc.)
- PVT tables for performance
- Relative permeability models (Corey, Brooks-Corey, LET)
- Capillary pressure
- Aquifer support (Carter-Tracy)

### **Performance**

- Numba JIT compilation
- 32-bit precision option (memory/speed)
- Adaptive timestep control (CFL-based)
- Sparse solvers (BiCGSTAB, GMRES, AMG)
- Cached preconditioners

### **Storage & Analysis**

- Multiple backends (HDF5, Zarr, YAML, JSON)
- Streaming for large simulations
- Recovery factor analysis
- Production profiles
- 1D/2D/3D visualization (Plotly)

---

## Who is BORES For?

- **Students** learning reservoir engineering
- **Researchers** prototyping new methods
- **Petroleum Engineers** exploring workflows
- **Educators** teaching simulation concepts
- **Python Developers** interested in reservoir simulation

---

## Getting Started

<div class="grid cards" markdown>

- :material-clock-fast:{ .lg .middle } **Quick Start**

    ---

    Get up and running in 5 minutes

    [:octicons-arrow-right-24: Quickstart Guide](getting-started/quickstart.md)

- :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step guides from beginner to advanced

    [:octicons-arrow-right-24: Start Learning](tutorials/index.md)

- :material-book-open-variant:{ .lg .middle } **User Guide**

    ---

    Comprehensive guide to all features

    [:octicons-arrow-right-24: Read the Guide](guides/index.md)

- :material-code-braces:{ .lg .middle } **Examples**

    ---

    Complete working examples you can run

    [:octicons-arrow-right-24: View Examples](examples/index.md)

</div>

---

## Installation

```bash
# Using uv (recommended)
uv add bores-framework

# Using pip
pip install bores-framework
```

[Full installation instructions →](getting-started/installation.md)

---

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/ti-oluwa/bores/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ti-oluwa/bores/discussions)
- **Source Code**: [GitHub Repository](https://github.com/ti-oluwa/bores)

---

## License

BORES is open source software licensed under the MIT License.

---

**Ready to dive in?** Start with the [Quickstart Guide](getting-started/quickstart.md) or jump into [Tutorials](tutorials/index.md).
