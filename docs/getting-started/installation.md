# Installation

This guide walks you through installing BORES and its dependencies.

---

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB+ recommended for large models)

---

## Installation Methods

### Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is a fast Python package manager that handles dependencies efficiently.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install BORES
uv add bores-framework
```

!!! tip "Why uv?"
    `uv` is **10-100x faster** than pip and handles complex dependencies better. Perfect for scientific computing packages like BORES.

### Using `pip`

```bash
pip install bores-framework
```

### Development Installation

To install from source (for contributing or development):

```bash
# Clone the repository
git clone https://github.com/ti-oluwa/bores.git
cd bores

# Install with uv
uv sync

# Or with pip
pip install -e .
```

---

## Verify Installation

Check that BORES installed correctly:

```python
import bores

print(f"BORES version: {bores.__version__}")
print(f"Precision: {bores.get_dtype()}")
```

Expected output:
```
BORES version: 0.0.1
Precision: float32
```

---

## Dependencies

BORES automatically installs these key dependencies:

| Package | Purpose |
|---------|---------|
| **NumPy** | Array operations and numerical computing |
| **SciPy** | Sparse linear solvers |
| **Numba** | JIT compilation for performance |
| **attrs** | Immutable data structures |
| **Plotly** | Interactive 3D visualization |
| **CoolProp** | Fluid property calculations |
| **h5py/zarr** | Data storage backends |

!!! info "Platform-Specific Builds"
    On Windows/Linux x86_64, BORES uses optimized Intel MKL builds of NumPy/SciPy for better performance. macOS and ARM architectures use standard builds.

---

## Optional Dependencies

### Development Tools

For contributing to BORES:

```bash
uv sync --extra dev
```

This installs:

- **ruff**: Fast linting and formatting
- **mypy**: Type checking
- **marimo**: Interactive notebooks (for examples)

### Visualization

Plotly is included by default, but you may want:

```bash
# For exporting static images
pip install kaleido

# Already included in BORES
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'bores'`

**Solution**: Ensure BORES is installed in the active Python environment:

```bash
python -c "import sys; print(sys.executable)"
pip list | grep bores
```

### NumPy/SciPy Issues

**Problem**: Errors related to BLAS/LAPACK on Linux

**Solution**: Install system BLAS libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libblas-dev liblapack-dev

# CentOS/RHEL
sudo yum install blas-devel lapack-devel
```

### Memory Errors

**Problem**: `MemoryError` when building large models

**Solution**: Use 32-bit precision (default):

```python
import bores
bores.use_32bit_precision()  # This is the default
```

See [Performance Optimization](../advanced/performance-optimization.md) for more tips.

### CoolProp Errors

**Problem**: Errors with fluid property calculations

**Solution**: CoolProp sometimes needs manual install:

```bash
pip install --upgrade coolprop
```

---

## Platform-Specific Notes

=== "Windows"

    Windows users may need Visual C++ Redistributable:

    - Download from [Microsoft](https://aka.ms/vs/17/release/vc_redist.x64.exe)
    - Or install via Windows Update

=== "macOS"

    macOS users on Apple Silicon (M1/M2/M3):

    - BORES works natively on ARM
    - No Rosetta needed
    - Standard NumPy/SciPy (not MKL)

=== "Linux"

    Linux users benefit from optimized MKL builds:

    - Best performance on x86_64
    - Install BLAS/LAPACK for optimal results
    - WSL2 fully supported

---

## Next Steps

**Installation complete!** Now let's run your first simulation:

[:octicons-arrow-right-24: Quick Start Guide](quickstart.md)

---

## Getting Help

- **Installation issues**: [GitHub Issues](https://github.com/ti-oluwa/bores/issues)
- **General questions**: [GitHub Discussions](https://github.com/ti-oluwa/bores/discussions)
