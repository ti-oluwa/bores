# Storage and Serialization

Complete guide to saving, loading, and serializing BORES objects.

---

## Overview

BORES uses a hierarchical serialization system for persistent storage:

- **`Serializable`**: Base class for dict-based round-trip serialization
- **`StoreSerializable`**: Adds `dump()` and `load()` methods for file backends
- **`DataStore`**: Multi-backend storage (Zarr, HDF5, JSON, YAML)

This enables:

- Save/load models, configs, states, and PVT tables
- Multiple file formats (HDF5, Zarr, JSON, YAML)
- Type-safe deserialization with cattrs
- Nested object serialization
- Generic type preservation

Defined in `bores.serialization` and `bores.stores`.

---

## Serialization Hierarchy

### Serializable

Base class for objects that can be converted to/from dictionaries:

```python
import bores

class MyClass(bores.Serializable):
    def __init__(self, value: int):
        self.value = value

# Serialize to dict
obj = MyClass(value=42)
data = obj.__dump__()  # Returns dict: {"value": 42, "__type__": "MyClass"}

# Deserialize from dict
loaded = MyClass.__load__(data)
print(loaded.value)  # 42
```

**Key methods**:

- `__dump__()` → dict: Serialize to dictionary
- `__load__(data: dict)` → Self: Deserialize from dictionary

### StoreSerializable

Adds file I/O on top of `Serializable`:

```python
import bores
from pathlib import Path

# StoreSerializable objects have to_file and from_file
model = bores.reservoir_model(...)
model.to_file(Path("./model.h5"))

loaded_model = bores.ReservoirModel.from_file(Path("./model.h5"))
```

**Key methods**:

- `to_file(path: Path)` → None: Save to file (detects backend from extension)
- `from_file(path: Path)` → Self: Load from file

**Supported extensions**:

- `.h5`, `.hdf5` → HDF5Store
- `.zarr` → ZarrStore (directory)
- `.json` → JSONStore
- `.yaml`, `.yml` → YAMLStore

---

## Saving and Loading Objects

### ReservoirModel

```python
import bores
from pathlib import Path

# Create model
model = bores.reservoir_model(
    grid_shape=(20, 20, 5),
    thickness=50.0,
    porosity=0.22,
    permeability=100.0,
    initial_pressure=3000.0,
    temperature=180.0,
)

# Save to HDF5
model.to_file(Path("./model.h5"))

# Load
loaded_model = bores.ReservoirModel.from_file(Path("./model.h5"))
```

### Config

```python
# Create config
config = bores.Config(
    model=model,
    wells=wells,
    timer=timer,
    # ... other parameters
)

# Save to YAML (human-readable)
config.to_file(Path("./config.yaml"))

# Save to HDF5 (binary, faster)
config.to_file(Path("./config.h5"))

# Load
loaded_config = bores.Config.from_file(Path("./config.yaml"))
```

### PVT Tables

```python
# Create PVT table data
pvt_data = bores.build_pvt_table_data(...)

# Save to HDF5 (recommended for tables)
pvt_data.to_file(Path("./pvt.h5"))

# Load
loaded_pvt_data = bores.PVTTableData.from_file(Path("./pvt.h5"))

# Build interpolators from loaded data
pvt_tables = bores.PVTTables(pvt_table_data=loaded_pvt_data)
```

### ModelState

```python
# Get state from simulation
state = next(run())

# Save single state
state.to_file(Path("./initial_state.h5"))

# Load
loaded_state = bores.ModelState.from_file(Path("./initial_state.h5"))
```

### Run

```python
# Create run
run = bores.Run(config=config)

# Save complete run (model + config)
run.to_file(Path("./run.h5"))

# Load
loaded_run = bores.Run.from_file(Path("./run.h5"))

# Or load from separate files
run = bores.Run.from_files(
    model_path=Path("./model.h5"),
    config_path=Path("./config.yaml"),
    pvt_table_path=Path("./pvt.h5"),
)
```

---

## Storage Backends

### HDF5 (Recommended for Single Files)

Best for archiving complete runs in a single file:

```python
import bores

store = bores.HDF5Store("./simulation.h5")

# Save multiple states
states = list(run())
store.dump(states)

# Load all states
for state in store.load(bores.ModelState):
    print(f"Step {state.step}")

# Load specific indices
for state in store.load(bores.ModelState, indices=[0, 10, 20]):
    print(f"Step {state.step}")
```

**Advantages**:

- Single file (easy to transfer)
- Compression (gzip, lzf)
- Widely supported
- Metadata storage

**Limitations**:

- File locking (one writer at a time)
- Slower parallel I/O vs Zarr

### Zarr (Recommended for Large Simulations)

Best for production runs with many states:

```python
store = bores.ZarrStore("./simulation.zarr")

# Same interface as HDF5Store
store.dump(states)
for state in store.load(bores.ModelState):
    print(f"Step {state.step}")
```

**Advantages**:

- Directory-based (no file size limit)
- Fast chunked storage
- Blosc compression
- Parallel I/O capable
- Supports appending

**Use cases**:

- Large grids (> 100k cells)
- Long simulations (> 1000 states)
- Streaming workflows with `StateStream`

### JSON (Human-Readable)

Use for small data, debugging, or configuration:

```python
store = bores.JSONStore("./config.json")

# Save config (human-readable)
store.dump([config])

# Load
for cfg in store.load(bores.Config):
    print(cfg)
```

!!! warning "JSON Limitations"
    - Does **not** support appending
    - Large files (slow)
    - Not suitable for states or PVT tables

### YAML (Configuration Files)

Use for configuration files only:

```python
store = bores.YAMLStore("./config.yaml")

# Save config
store.dump([config])

# Load
for cfg in store.load(bores.Config):
    print(cfg)
```

!!! info "YAML vs JSON"
    - YAML is more human-readable (comments, no quotes)
    - JSON is stricter and faster to parse
    - Both do not support appending

---

## Advanced Storage Features

### Metadata

Attach metadata to stored entries:

```python
store = bores.ZarrStore("./simulation.zarr")

# Save with metadata
store.dump(
    data=states,
    meta=lambda s: {"step": s.step, "time_days": s.time / 86400},
)

# Load with metadata filter
for state in store.load(
    bores.ModelState,
    predicate=lambda e: e.meta.get("time_days", 0) > 100,
):
    print(f"State at {e.meta['time_days']:.1f} days")
```

### Validation

Validate data before saving or after loading:

```python
def validate_state(state: bores.ModelState) -> bores.ModelState:
    """Validate state arrays have matching shapes."""
    return bores.validate_state(state, dtype="global")

# Save with validation
store.dump(
    data=states,
    validator=validate_state,
)

# Load with validation
for state in store.load(
    bores.ModelState,
    validator=validate_state,
):
    pass  # All states validated
```

### Selective Loading

Load only specific states:

```python
# Load first, middle, and last states
indices = [0, len(states) // 2, -1]
for state in store.load(bores.ModelState, indices=indices):
    print(f"Step {state.step}")

# Load every 10th state
indices = list(range(0, store.count(), 10))
for state in store.load(bores.ModelState, indices=indices):
    print(f"Step {state.step}")

# Load states matching predicate
for state in store.load(
    bores.ModelState,
    predicate=lambda e: e.idx % 10 == 0,  # Every 10th entry
):
    print(f"Step {state.step}")
```

### Appending

Add states to existing store without rewriting:

```python
store = bores.ZarrStore("./simulation.zarr")

# Initial save
store.dump(initial_states)

# Later: append more states
for state in new_states:
    entry = store.append(state)
    print(f"Appended state {state.step} at index {entry.idx}")

# All states now in store
print(f"Total states: {store.count()}")
```

!!! warning "Append Support"
    Only Zarr and HDF5 support appending. JSON and YAML require full rewrite.

---

## Complete Workflow

### Save Simulation Setup

```python
import bores
from pathlib import Path

# Create setup directory
setup_dir = Path("./scenarios/setup")
setup_dir.mkdir(parents=True, exist_ok=True)

# Create and save model
model = bores.reservoir_model(...)
model.to_file(setup_dir / "model.h5")

# Create and save PVT tables
pvt_data = bores.build_pvt_table_data(...)
pvt_data.to_file(setup_dir / "pvt.h5")

# Create and save config
config = bores.Config(
    model=model,
    pvt_tables=bores.PVTTables(pvt_table_data=pvt_data),
    # ... other parameters
)
config.to_file(setup_dir / "config.yaml")  # Human-readable

# Create run
run = bores.Run(config=config)
run.to_file(setup_dir / "run.h5")

print(f"Setup saved to {setup_dir}")
```

### Load and Execute

```python
from pathlib import Path

# Load from files
run = bores.Run.from_files(
    model_path=Path("./scenarios/setup/model.h5"),
    config_path=Path("./scenarios/setup/config.yaml"),
    pvt_table_path=Path("./scenarios/setup/pvt.h5"),
)

# Execute with streaming
store = bores.ZarrStore("./scenarios/results/simulation.zarr")
stream = bores.StateStream(
    states=run(),
    store=store,
    batch_size=30,
    background_io=True,
)

with stream:
    stream.consume()

print("Simulation complete and saved.")
```

### Load and Analyze

```python
# Load results
store = bores.ZarrStore("./scenarios/results/simulation.zarr")
states = list(store.load(bores.ModelState))

# Analyze
analyst = bores.ModelAnalyst(states)
print(f"Oil RF: {analyst.oil_recovery_factor:.2%}")
```

---

## Serialization Details

### Type Registry

BORES maintains a type registry for deserializing nested objects:

```python
# Types are automatically registered
# Example: ReservoirModel contains FluidProperties, RockProperties, etc.
# All are deserialized correctly without manual type specification
```

### Generic Types

Generic types (e.g., `ModelState[ThreeDimensions]`) are preserved:

```python
# Save 3D state
state_3d: bores.ModelState[bores.ThreeDimensions] = ...
state_3d.to_file("state_3d.h5")

# Load preserves generic type
loaded = bores.ModelState.from_file("state_3d.h5")
# Type is ModelState[ThreeDimensions]
```

### Nested Objects

Nested objects are serialized recursively:

```python
# Config contains:
# - ReservoirModel
#   - FluidProperties
#   - RockProperties
#   - SaturationHistory
# - Wells
# - Timer
# - All nested objects

# Saving config automatically saves all nested objects
config.to_file("config.h5")

# Loading reconstructs entire hierarchy
loaded_config = bores.Config.from_file("config.h5")
```

---

## Performance Tips

### File Format Selection

| Use Case | Recommended Format | Reason |
|----------|-------------------|--------|
| Model setup | HDF5 | Single file, fast, compressed |
| Config | YAML | Human-readable, version control friendly |
| PVT tables | HDF5 | Large arrays, compression |
| Simulation results | Zarr | Many states, streaming, append |
| Small debug data | JSON | Human-readable, inspectable |

### Compression

Both HDF5 and Zarr support compression:

```python
# Zarr: Blosc compression (automatic, very fast)
store = bores.ZarrStore("./results.zarr")  # Blosc enabled by default

# HDF5: gzip compression (slower but better ratio)
store = bores.HDF5Store("./results.h5")  # gzip enabled by default
```

**Compression ratios** (typical):

- Pressure grids: 5-10× (smooth fields compress well)
- Saturation grids: 3-5×
- PVT tables: 2-3×

### Memory Management

For large datasets, stream instead of loading all at once:

```python
# Bad: Load all states into memory
states = list(store.load(bores.ModelState))  # Can use 10+ GB

# Good: Process one at a time
for state in store.load(bores.ModelState):
    analyze(state)  # Low memory footprint
```

---

## Best Practices

### Directory Structure

Organize files for clarity:

```
project/
├── scenarios/
│   ├── setup/
│   │   ├── model.h5       # Reservoir model
│   │   ├── config.yaml    # Configuration
│   │   └── pvt.h5         # PVT tables
│   └── results/
│       ├── simulation.zarr/  # Simulation states
│       └── checkpoints.zarr/ # Checkpoints
└── analysis/
    └── plots/
```

### Version Control

1. **Commit**: YAML configs, Python scripts
2. **Ignore**: HDF5 models, Zarr results (large binary files)
3. **Document**: PVT table sources, model assumptions

Example `.gitignore`:

```
# Binary data files
*.h5
*.hdf5
*.zarr/

# Keep configs
!config.yaml
```

### Backup Strategy

1. **Regular backups**: Copy Zarr results after each run
2. **Checkpoints**: Use `checkpoint_store` in `StateStream`
3. **Redundancy**: Keep setup files separate from results

---

## Troubleshooting

### Type Deserialization Errors

If you see `KeyError: 'MyClass'`:

```python
# Ensure class is imported before deserialization
import bores  # Imports register all types automatically
loaded = bores.Config.from_file("config.h5")
```

### File Corruption

If HDF5/Zarr file is corrupted:

```python
# For HDF5: Try recovery
import h5py
with h5py.File("simulation.h5", "r") as f:
    # Check if file opens
    print(list(f.keys()))

# For Zarr: Inspect directory
import zarr
z = zarr.open("simulation.zarr", mode="r")
print(z.tree())
```

### Large File Sizes

If files are too large:

1. **Check compression**: Ensure enabled
2. **Selective saving**: Use `save=lambda` in `StateStream`
3. **Reduce frequency**: Save every Nth state
4. **Lower precision**: Use `bores.use_32bit_precision()`

---

## Next Steps

- [States, Streams, and Stores](../guides/states-streams-stores.md) - State management and streaming
- [PVT Tables](pvt-tables.md) - Creating and using PVT tables
- [Running Simulations](../guides/running-simulations.md) - Configuring runs
