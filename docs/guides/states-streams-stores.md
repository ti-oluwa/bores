# States, Streams, and Stores

Complete guide to state management, streaming, and storage in BORES.

---

## Overview

BORES uses a three-layer system for managing simulation data:

- **`ModelState`** - Snapshot of reservoir model at a single timestep
- **`StateStream`** - Memory-efficient iteration over states with optional persistence
- **`DataStore`** - Multi-backend storage (Zarr, HDF5, JSON, YAML)

This design enables:

- Low memory footprint (states persisted immediately)
- Efficient I/O (batching, async writes)
- Flexible storage backends
- Resume from checkpoints
- Selective state saving

---

## Model State

### What is a ModelState?

`ModelState` is a frozen dataclass capturing the complete reservoir state at a specific timestep:

```python
import bores

# ModelState contains:
state = bores.ModelState(
    step=100,                           # Timestep index
    step_size=3600.0,                   # Timestep size (seconds)
    time=360000.0,                      # Simulation time (seconds)
    model=reservoir_model,              # ReservoirModel
    wells=wells,                        # Wells configuration
    injection=injection_rates,          # Injection rate grids
    production=production_rates,        # Production rate grids
    relative_permeabilities=relperm,    # Relative permeability grids
    relative_mobilities=mobilities,     # Relative mobility grids
    capillary_pressures=pc,             # Capillary pressure grids
    timer_state=timer_state,            # Optional timer state
)
```

### State Attributes

**Time tracking**:
- `step` - Timestep index (int)
- `step_size` - Timestep size in seconds (float)
- `time` - Cumulative simulation time in seconds (float)
- `timer_state` - Optional `TimerState` with adaptive timestep info

**Reservoir model** (`state.model`):
- `grid_shape` - Grid dimensions (nx, ny, nz)
- `thickness_grid` - Cell thicknesses (ft)
- `fluid_properties` - Pressure, saturations, PVT properties
- `rock_properties` - Porosity, permeability
- `saturation_history` - Historical saturations

**Wells** (`state.wells`):
- Production and injection wells
- Well controls and rates
- Perforation intervals

**Flow properties**:
- `injection` - Injection rate grids (ft³/day) for oil, gas, water
- `production` - Production rate grids (ft³/day)
- `relative_permeabilities` - kr_o, kr_g, kr_w grids
- `relative_mobilities` - Mobility grids (kr/μ)
- `capillary_pressures` - Pc_ow, Pc_og grids (psi)

### Accessing State Data

```python
# Timestep info
print(f"Step: {state.step}")
print(f"Time: {state.time / 86400:.1f} days")  # Convert seconds to days

# Pressure
pressure = state.model.fluid_properties.pressure_grid  # (nx, ny, nz)
avg_pressure = pressure.mean()

# Saturations (current timestep)
oil_sat = state.model.fluid_properties.saturation_history.oil_saturations[-1]
gas_sat = state.model.fluid_properties.saturation_history.gas_saturations[-1]
water_sat = state.model.fluid_properties.saturation_history.water_saturations[-1]

# Porosity and permeability
porosity = state.model.rock_properties.porosity_grid
perm_x = state.model.rock_properties.permeability.x_direction

# Production rates at well cells
oil_prod = state.production.oil  # ft³/day
gas_prod = state.production.gas
water_prod = state.production.water

# Relative permeabilities
kr_o = state.relative_permeabilities.oil
kr_g = state.relative_permeabilities.gas
kr_w = state.relative_permeabilities.water
```

### State Validation

Validate state grids have matching shapes:

```python
# Validate with optional dtype coercion
validated_state = bores.validate_state(state, dtype="float32")

# Validate using global dtype
validated_state = bores.validate_state(state, dtype="global")
```

---

## State Streaming

### Why Stream States?

Simulations generate hundreds or thousands of states. Loading all states into memory is impractical for large grids or long simulations. `StateStream` solves this by:

1. **Yielding states one at a time** (low memory)
2. **Persisting to disk immediately** (no accumulation)
3. **Batching writes** (I/O efficiency)
4. **Async I/O** (2-3x speedup when I/O is slower than simulation)
5. **Auto-saving on exit** (no lost data)

### Basic Streaming

```python
import bores
from pathlib import Path

# Create store
store = bores.ZarrStore("./results/simulation.zarr")

# Stream states from run() generator
run = bores.Run.from_files(...)
stream = bores.StateStream(
    states=run(),
    store=store,
    batch_size=30,  # Write every 30 states
    background_io=True,  # Background I/O thread
)

# Consume stream (states are saved automatically)
with stream:
    for state in stream:
        # Analyze state
        print(f"Step {state.step}: P_avg = {state.model.fluid_properties.pressure_grid.mean():.1f} psi")
```

### Stream Parameters

```python
stream = bores.StateStream(
    states=run(),                    # Generator or iterable of ModelState
    store=store,                     # Optional DataStore for persistence
    batch_size=30,                   # Number of states before flush
    validate=False,                  # Validate states before saving
    auto_save=True,                  # Auto-flush on context exit
    auto_replay=True,                # Auto-replay from store when re-iterated
    save=lambda s: s.step % 10 == 0, # Only save every 10th state
    checkpoint_interval=100,         # Checkpoint every 100 states
    checkpoint_store=checkpoint_store, # Separate store for checkpoints
    max_batch_memory_usage=50.0,     # Flush when batch > 50 MB
    background_io=True,                   # Background I/O thread
    max_queue_size=50,               # Max states in I/O queue
    io_thread_name="stream-io",      # I/O thread name (debugging)
    queue_timeout=1.0,               # Queue timeout (seconds)
)
```

!!! info "Async I/O"
    When `background_io=True`, disk writes happen in a background thread. The simulation fills a queue; the I/O worker drains it. This provides 2-3x speedup when I/O is slower than simulation.

### Selective State Saving

Save only states matching a condition:

```python
# Save every 10th state
stream = bores.StateStream(
    states=run(),
    store=store,
    save=lambda s: s.step % 10 == 0,
)

# Save states after 100 days
stream = bores.StateStream(
    states=run(),
    store=store,
    save=lambda s: s.time > 100 * 86400,
)

# Save states with high pressure
stream = bores.StateStream(
    states=run(),
    store=store,
    save=lambda s: s.model.fluid_properties.pressure_grid.mean() > 2000,
)
```

### Checkpointing

Create checkpoints for crash recovery:

```python
checkpoint_store = bores.ZarrStore("./results/checkpoints.zarr")

stream = bores.StateStream(
    states=run(),
    store=store,
    checkpoint_interval=100,  # Checkpoint every 100 states
    checkpoint_store=checkpoint_store,
)

with stream:
    stream.consume()

# Resume from last checkpoint
checkpoint_states = list(checkpoint_store.load(bores.ModelState))
last_checkpoint = checkpoint_states[-1]
# ... use last_checkpoint as initial state for new run
```

### Memory-Limited Batching

Automatically flush when batch memory exceeds limit:

```python
stream = bores.StateStream(
    states=run(),
    store=store,
    batch_size=50,  # Max 50 states per batch
    max_batch_memory_usage=100.0,  # OR max 100 MB per batch
)
```

The stream flushes when **either** condition is met.

### Stream Helpers

```python
# Get only the last state (discard others)
with stream:
    last_state = stream.last()

# Consume entire stream (don't store states in memory)
with stream:
    stream.consume()

# Progress tracking
with stream:
    for state in stream:
        progress = stream.progress()
        print(f"Yielded: {progress['yield_count']}, "
              f"Saved: {progress['saved_count']}, "
              f"Pending: {progress['batch_pending']}, "
              f"Memory: {progress['memory_usage']:.1f} MB")
```

### Replaying States

After consumption, replay from store:

```python
# Auto-replay (default)
with stream:
    for state in stream:
        pass  # First iteration: yields from run()

for state in stream:
    pass  # Second iteration: replays from store

# Manual replay
for state in stream.replay():
    print(f"Step {state.step}")

# Replay specific indices
for state in stream.replay(indices=[0, 50, 99]):
    print(f"Loaded step {state.step}")

# Replay with predicate
for state in stream.replay(predicate=lambda e: e.meta.get("step", 0) % 10 == 0):
    print(f"Loaded step {state.step}")
```

---

## Storage Backends

BORES supports multiple storage backends via the `DataStore` interface.

### Zarr Store (Recommended)

Zarr is the recommended backend for large simulations:

```python
import bores
from pathlib import Path

# Create Zarr store
store = bores.ZarrStore("./results/simulation.zarr")

# Save states
store.dump(states)

# Load all states
for state in store.load(bores.ModelState):
    print(f"Step {state.step}")

# Load specific indices
for state in store.load(bores.ModelState, indices=[0, 10, 20]):
    print(f"Step {state.step}")

# Load with predicate
for state in store.load(bores.ModelState, predicate=lambda e: e.idx < 10):
    print(f"Step {state.step}")
```

**Zarr advantages**:

- Fast chunked array storage
- Compression (Blosc)
- Supports appending
- Directory-based (no file size limits)
- Parallel I/O capable

### HDF5 Store

HDF5 is good for single-file archives:

```python
store = bores.HDF5Store("./results/simulation.h5")

# Same interface as ZarrStore
store.dump(states)
for state in store.load(bores.ModelState):
    print(f"Step {state.step}")
```

**HDF5 advantages**:

- Single file (easy to transfer)
- Widely supported
- Compression (gzip, lzf)
- Supports appending

### JSON Store

JSON is human-readable (use for small simulations):

```python
store = bores.JSONStore("./results/simulation.json")

# Save (overwrites entire file)
store.dump(states)

# Load
for state in store.load(bores.ModelState):
    print(f"Step {state.step}")
```

!!! warning "JSON Limitations"
    - Does **not** support appending (must rewrite entire file)
    - Large files (slow serialization)
    - Use only for small grids or few states

### YAML Store

YAML is human-readable (use for config, not states):

```python
store = bores.YAMLStore("./results/config.yaml")

# Save config (not typically used for states)
store.dump([config])

# Load
for item in store.load(bores.Config):
    print(item)
```

### Backend Comparison

| Backend | File Type | Append | Compression | Speed | Use Case |
|---------|-----------|--------|-------------|-------|----------|
| **Zarr** | Directory | Yes | Blosc | Fast | Large simulations (recommended) |
| **HDF5** | Single file | Yes | gzip/lzf | Fast | Archives, transfer |
| **JSON** | Single file | No | - | Slow | Small runs, debugging |
| **YAML** | Single file | No | - | Slow | Config, small data |

### Store Interface

All stores implement the same interface:

```python
# Save data (overwrites)
store.dump(
    data=states,
    validator=lambda s: bores.validate_state(s),  # Optional
    meta=lambda s: {"step": s.step},  # Optional metadata
)

# Append single item (if supported)
entry_meta = store.append(
    item=state,
    validator=lambda s: bores.validate_state(s),
    meta=lambda s: {"step": s.step},
)

# Load all
for item in store.load(bores.ModelState):
    pass

# Load specific indices
for item in store.load(bores.ModelState, indices=[0, 10, 20]):
    pass

# Load with predicate on metadata
for item in store.load(bores.ModelState, predicate=lambda e: e.meta.get("step", 0) % 10 == 0):
    pass

# Get metadata (no deserialization)
entries = store.entries()  # List[EntryMeta]
for entry in entries:
    print(f"Index {entry.idx}: {entry.meta}")

# Count entries
count = store.count()

# Max index
max_idx = store.max_index()
```

---

## Complete Workflow Example

```python
import logging
from pathlib import Path
import bores

logging.basicConfig(level=logging.INFO)
bores.use_32bit_precision()

# Load run
run = bores.Run.from_files(
    model_path=Path("./setup/model.h5"),
    config_path=Path("./setup/config.yaml"),
    pvt_table_path=Path("./setup/pvt.h5"),
)

# Setup stores
main_store = bores.ZarrStore("./results/simulation.zarr")
checkpoint_store = bores.ZarrStore("./results/checkpoints.zarr")

# Stream with checkpointing and selective saving
stream = bores.StateStream(
    states=run(),
    store=main_store,
    batch_size=30,
    background_io=True,
    save=lambda s: s.step % 5 == 0,  # Save every 5th state
    checkpoint_interval=100,
    checkpoint_store=checkpoint_store,
    max_batch_memory_usage=100.0,  # 100 MB batch limit
)

# Execute
print("Starting simulation...")
with stream:
    for state in stream:
        if state.step % 10 == 0:
            progress = stream.progress()
            print(f"Step {state.step}: "
                  f"P_avg = {state.model.fluid_properties.pressure_grid.mean():.1f} psi, "
                  f"Saved = {progress['saved_count']}, "
                  f"Memory = {progress['memory_usage']:.1f} MB")

print(f"Simulation complete. Saved {main_store.count()} states.")

# Analyze results
print("\nAnalyzing results...")
states = list(main_store.load(bores.ModelState))
analyst = bores.ModelAnalyst(states)

print(f"STOIIP: {analyst.stoiip:,.0f} STB")
print(f"Oil RF: {analyst.oil_recovery_factor:.2%}")
print(f"Cumulative oil: {analyst.No:,.0f} STB")

# Production history
print("\nProduction history:")
for step, rate in analyst.oil_production_history(interval=20):
    print(f"  Step {step}: {rate:.1f} STB/day")
```

---

## Best Practices

### Memory Efficiency

1. **Use StateStream** instead of collecting states in a list
2. **Enable async I/O** for large simulations (`background_io=True`)
3. **Batch writes** with `batch_size=30` or higher
4. **Selective saving** with `save=lambda` to reduce storage
5. **Checkpoint periodically** for crash recovery

### Storage Selection

1. **Zarr** for production runs (fast, compressed, append-capable)
2. **HDF5** for archives or single-file transfer
3. **JSON/YAML** only for debugging or very small runs

### Validation

1. **Validate before saving** with `validator=lambda s: bores.validate_state(s)`
2. **Check state consistency** after loading from store
3. **Store metadata** with `meta=lambda` for filtering

### I/O Performance

1. **Larger batches** reduce I/O overhead (try `batch_size=50`)
2. **Async I/O** prevents blocking (enable when I/O is slow)
3. **Memory limits** prevent OOM (`max_batch_memory_usage=100.0`)
4. **Compression** reduces disk usage (Zarr/HDF5 auto-compress)

---

## Next Steps

- [Analyzing Results](analyzing-results.md) - Using ModelAnalyst
- [Visualization](visualization.md) - Plotting and 3D visualization
- [Running Simulations](running-simulations.md) - Configuring runs
