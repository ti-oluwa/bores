import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def setup_grid():
    import logging
    from pathlib import Path

    import numpy as np

    import bores

    logging.basicConfig(level=logging.INFO)
    bores.use_32bit_precision()

    FT = bores.c.METERS_TO_FT

    # -------------------------------------------------------------------------
    # Grid geometry
    # Fine grid: 100 x 1 x 20, uniform cell sizes
    # DX = DY = 7.62m, DZ = 0.762m per layer (all converted to feet)
    # -------------------------------------------------------------------------
    cell_dimension = (7.62 * FT, 7.62 * FT)  # (DX, DY) in feet
    grid_shape = (100, 1, 20)

    thickness_grid = bores.uniform_grid(
        grid_shape=grid_shape,
        value=0.762 * FT,
    )

    # -------------------------------------------------------------------------
    # Pressure initialisation
    # Top of model at 0.0m, P_top = 100 psia
    # Incompressible oil: gradient = rho_o / 144 [psi/ft]
    # rho_o = 43.68 lb/ft³  →  gradient = 0.30333 psi/ft
    # -------------------------------------------------------------------------
    top_pressure = 100.0
    oil_density_lbft3 = 43.68
    pressure_gradient = oil_density_lbft3 / 144.0  # psi/ft

    dz_ft = 0.762 * FT
    layer_centre_depths = np.array([(k + 0.5) * dz_ft for k in range(20)])
    layer_pressures = top_pressure + layer_centre_depths * pressure_gradient

    pressure_grid = bores.layered_grid(
        grid_shape=grid_shape,
        layer_values=bores.array(layer_pressures),
        orientation=bores.Orientation.Z,
    )

    # Bubble point well below initial pressure — reservoir stays undersaturated
    oil_bubble_point_pressure_grid = bores.uniform_grid(
        grid_shape=grid_shape,
        value=top_pressure * 0.5,
    )

    # -------------------------------------------------------------------------
    # Rock properties
    # Porosity: uniform 0.2
    # Permeability: geostatistical field from perm_case1.dat
    #
    # File layout: 6000 values total = three sequential blocks of 2000
    #   Block 1: Kx(i,j,k)  i=1..100, j=1, k=1..20
    #   Block 2: Ky(i,j,k)
    #   Block 3: Kz(i,j,k)
    # All three blocks are identical (isotropic field stored three times).
    # Ordering within each block: i varies fastest (Eclipse/IMEX convention)
    #   → reshape as (ni=100, nj=1, nk=20) with Fortran order
    # -------------------------------------------------------------------------
    porosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.2)

    perm_raw = np.loadtxt("benchmarks/data/perm_case1.dat").ravel()
    n_cells = 100 * 1 * 20  # 2000

    kx_grid = bores.array(perm_raw[:n_cells].reshape((100, 1, 20), order="F"))
    ky_grid = bores.array(
        perm_raw[n_cells : 2 * n_cells].reshape((100, 1, 20), order="F")
    )
    kz_grid = bores.array(
        perm_raw[2 * n_cells : 3 * n_cells].reshape((100, 1, 20), order="F")
    )
    absolute_permeability = bores.RockPermeability(x=kx_grid, y=ky_grid, z=kz_grid)

    rock_compressibility = 0.0  # incompressible rock

    # -------------------------------------------------------------------------
    # Saturation endpoints
    # Initially fully oil-saturated — no connate water per problem statement
    # -------------------------------------------------------------------------
    connate_water_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    irreducible_water_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )
    residual_oil_saturation_water_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.25
    )
    residual_oil_saturation_gas_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.15
    )
    residual_gas_saturation_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.0
    )

    oil_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.999998)
    water_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.0)
    gas_saturation_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.000002)

    # -------------------------------------------------------------------------
    # Fluid properties — constant throughout run (per problem statement)
    # mu_o = 1 cP,  mu_g = 0.01 cP
    # rho_o = 43.68 lb/ft³,  rho_g = 0.0624 lb/ft³
    # Fluids are incompressible and immiscible
    # -------------------------------------------------------------------------
    oil_viscosity_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.0)  # cP

    oil_specific_gravity_grid = bores.uniform_grid(
        grid_shape=grid_shape,
        value=oil_density_lbft3 / bores.c.STANDARD_WATER_DENSITY_IMPERIAL,
    )
    oil_compressibility_grid = bores.uniform_grid(grid_shape=grid_shape, value=0.0)

    gas_viscosity_grid = bores.uniform_grid(
        grid_shape=grid_shape, value=0.01
    )  # cP
    gas_density_lbft3 = 0.0624
    gas_gravity = gas_density_lbft3 / bores.c.STANDARD_AIR_DENSITY_IMPERIAL
    gas_gravity_grid = bores.uniform_grid(grid_shape=grid_shape, value=gas_gravity)

    # -------------------------------------------------------------------------
    # Temperature — isothermal run
    # -------------------------------------------------------------------------
    temperature_grid = bores.uniform_grid(grid_shape=grid_shape, value=160.0)  # °F

    # Net-to-gross — full pay
    net_to_gross_grid = bores.uniform_grid(grid_shape=grid_shape, value=1.0)

    # -------------------------------------------------------------------------
    # PVT tables
    #
    # The benchmark specifies constant, pressure/temperature-independent fluid
    # properties throughout:
    #   μo = 1.0 cP,  μg = 0.01 cP  (given as fixed values in problem statement)
    #   ρo = 43.68 lb/ft³,  ρg = 0.0624 lb/ft³  (given as fixed values)
    #   Co = 0,  Cr = 0  (incompressible)
    #
    # We inject flat constant tables for oil and gas viscosity so the PVT
    # interpolator always returns exactly the benchmark values regardless of
    # pressure or temperature.  All other properties (Bo, Rs, Bg, Z-factor,
    # densities) are still computed from correlations, but their variation
    # is small across the narrow operating pressure range (~95–105 psia).
    # -------------------------------------------------------------------------
    pvt_pressures = bores.array([50.0, 100.0, 110.0, 500.0])
    # RectBivariateSpline used by PVTTables requires >= 2 points per axis (even for linear k=1).
    # The benchmark is isothermal at 160 F — a nearby dummy point at 161 F
    # satisfies the requirement; flat tables return the same value at both.
    pvt_temperatures = bores.array([160.0, 161.0])
    n_pvt_p = len(pvt_pressures)
    n_pvt_t = len(pvt_temperatures)

    # Constant oil viscosity: 1.0 cP — flat (n_p × n_t) table
    oil_visc_table = np.full((n_pvt_p, n_pvt_t), 1.0, dtype=np.float32)

    # Constant gas viscosity: 0.01 cP — flat (n_p × n_t) table
    gas_visc_table = np.full((n_pvt_p, n_pvt_t), 0.01, dtype=np.float32)

    # Zero oil compressibility
    oil_compressibility_table = np.full((n_pvt_p, n_pvt_t), 0.0, dtype=np.float32)

    pvt_table_data = bores.build_pvt_table_data(
        pressures=pvt_pressures,
        temperatures=pvt_temperatures,
        salinities=bores.array([0.0]),
        oil_specific_gravity=oil_density_lbft3
        / bores.c.STANDARD_WATER_DENSITY_IMPERIAL,
        gas_gravity=gas_gravity,
        # reservoir_gas="methane",
        # Constant viscosities override correlation-computed values
        oil_viscosity_table=oil_visc_table,
        gas_viscosity_table=gas_visc_table,
        oil_compressibility_table=oil_compressibility_table,
    )
    pvt_tables = bores.PVTTables(
        data=pvt_table_data, interpolation_method="linear"
    )

    # -------------------------------------------------------------------------
    # Build reservoir model
    # -------------------------------------------------------------------------
    model = bores.reservoir_model(
        grid_shape=grid_shape,
        cell_dimension=cell_dimension,
        thickness_grid=thickness_grid,
        pressure_grid=pressure_grid,
        oil_bubble_point_pressure_grid=oil_bubble_point_pressure_grid,
        absolute_permeability=absolute_permeability,
        porosity_grid=porosity_grid,
        temperature_grid=temperature_grid,
        rock_compressibility=rock_compressibility,
        oil_saturation_grid=oil_saturation_grid,
        water_saturation_grid=water_saturation_grid,
        gas_saturation_grid=gas_saturation_grid,
        oil_viscosity_grid=oil_viscosity_grid,
        oil_specific_gravity_grid=oil_specific_gravity_grid,
        oil_compressibility_grid=oil_compressibility_grid,
        gas_gravity_grid=gas_gravity_grid,
        gas_viscosity_grid=gas_viscosity_grid,
        residual_oil_saturation_water_grid=residual_oil_saturation_water_grid,
        residual_oil_saturation_gas_grid=residual_oil_saturation_gas_grid,
        irreducible_water_saturation_grid=irreducible_water_saturation_grid,
        connate_water_saturation_grid=connate_water_saturation_grid,
        residual_gas_saturation_grid=residual_gas_saturation_grid,
        net_to_gross_ratio_grid=net_to_gross_grid,
        # reservoir_gas="methane",
        dip_angle=0.0,
        dip_azimuth=0.0,
        pvt_tables=pvt_tables,
    )

    model.to_file(Path("./benchmarks/runs/spe1/setup/model.h5"))
    pvt_table_data.to_file(Path("./benchmarks/runs/spe1/setup/pvt.h5"))
    return Path, bores, np, pvt_tables


@app.cell
def setup_config(Path, bores, np, pvt_tables):
    # -------------------------------------------------------------------------
    # Relative permeability — built from the SPE1 tabulated data
    #
    # The file has columns: Sg  Krg  Kro  Pcog
    # This is a two-phase gas-oil system (no water).
    #
    # ThreePhaseRelPermTable requires:
    #   - gas_oil_table  : TwoPhaseRelPermTable indexed by *wetting phase* saturation
    #                      Wetting phase = oil  →  index by So = 1 - Sg
    #   - oil_water_table: stub (no water present, Sw=0 everywhere)
    # -------------------------------------------------------------------------
    relperm_data = np.loadtxt(
        "benchmarks/data/rel_perm_tab.txt",
        skiprows=1,  # skip header: "Sg  Krg  Kro  Pcog"
    )
    sg_tab = relperm_data[:, 0]
    krg_tab = relperm_data[:, 1]
    kro_tab = relperm_data[:, 2]

    # Convert to oil-saturation indexing and ensure monotone increasing
    # Sg = 0.0 → So = 1.0 (oil at max),  Sg = 0.85 → So = 0.15 (oil at residual)
    so_tab = 1.0 - sg_tab  # So decreases as Sg increases

    # Reverse so that the wetting phase saturation (So) is increasing
    so_asc = so_tab[::-1].copy()  # 0.15 → 1.0
    kro_asc = kro_tab[::-1].copy()  # 0.0  → 1.0
    krg_asc = krg_tab[::-1].copy()  # 1.0  → 0.0

    gas_oil_table = bores.TwoPhaseRelPermTable(
        wetting_phase=bores.FluidPhase.OIL,
        non_wetting_phase=bores.FluidPhase.GAS,
        wetting_phase_saturation=bores.array(so_asc),
        wetting_phase_relative_permeability=bores.array(kro_asc),
        non_wetting_phase_relative_permeability=bores.array(krg_asc),
    )

    # Stub oil-water table: krw = 0 everywhere (no water phase in this problem)
    oil_water_table = bores.TwoPhaseRelPermTable(
        wetting_phase=bores.FluidPhase.WATER,
        non_wetting_phase=bores.FluidPhase.OIL,
        wetting_phase_saturation=bores.array([0.0, 1.0]),
        wetting_phase_relative_permeability=bores.array([0.0, 0.0]),  # krw = 0
        non_wetting_phase_relative_permeability=bores.array(
            [1.0, 0.0]
        ),  # kro at Sw
    )
    relative_permeability_table = bores.ThreePhaseRelPermTable(
        oil_water_table=oil_water_table,
        gas_oil_table=gas_oil_table,
        mixing_rule=bores.eclipse_rule,
    )

    # -------------------------------------------------------------------------
    # Capillary pressure
    # Pcog = 0 throughout per problem statement.
    # We must supply a valid capillary pressure model to RockFluidTables, but we
    # disable its effect entirely via disable_capillary_effects=True in Config.
    # -------------------------------------------------------------------------
    capillary_pressure_table = bores.BrooksCoreyCapillaryPressureModel(
        irreducible_water_saturation=0.0,
        residual_oil_saturation_water=0.25,
        residual_oil_saturation_gas=0.15,
        residual_gas_saturation=0.0,
        oil_water_entry_pressure_water_wet=1.0,
        oil_water_pore_size_distribution_index_water_wet=2.0,
        gas_oil_entry_pressure=1.0,
        gas_oil_pore_size_distribution_index=2.0,
        wettability=bores.Wettability.WATER_WET,
    )
    rock_fluid_tables = bores.RockFluidTables(
        relative_permeability_table=relative_permeability_table,
        capillary_pressure_table=capillary_pressure_table,
    )

    # -------------------------------------------------------------------------
    # Wells
    #
    # Injector — left face (i=1), completed through all 20 layers
    # Rate: 6.97 m³/d = 43.84 STB/d (as given in problem statement)
    #
    # Producer — right face (i=100), completed through all 20 layers
    # Constant BHP limit = 95 psia (reference depth = 0.0m = top of model)
    # -------------------------------------------------------------------------
    injection_rate_stbd = 6.97 * bores.c.CUBIC_METER_TO_STB  # ~43.84 STB/d

    injector = bores.injection_well(
        well_name="INJ-1",
        perforating_intervals=[((0, 0, 0), (0, 0, 19))],
        radius=0.5,  # 1.0 ft diameter well
        control=bores.ConstantRateControl(
            target_rate=injection_rate_stbd,
            clamp=bores.InjectionClamp(),
        ),
        injected_fluid=bores.InjectedFluid(
            name="Methane",
            phase=bores.FluidPhase.GAS,
            specific_gravity=0.65,
            molecular_weight=bores.c.MOLECULAR_WEIGHT_CH4,
            is_miscible=False,
        ),
        is_active=True,
        skin_factor=0.0,
    )

    producer = bores.production_well(
        well_name="PROD-1",
        perforating_intervals=[((99, 0, 0), (99, 0, 19))],
        radius=0.5,
        control=bores.BHPControl(
            bhp=95.0,  # 95 psia BHP limit
            target_phase="gas",
            clamp=bores.ProductionClamp(),
        ),
        produced_fluids=(
            bores.ProducedFluid(
                name="Oil",
                phase=bores.FluidPhase.OIL,
                specific_gravity=43.68 / bores.c.STANDARD_WATER_DENSITY_IMPERIAL,
                molecular_weight=180.0,
            ),
            bores.ProducedFluid(
                name="Gas",
                phase=bores.FluidPhase.GAS,
                specific_gravity=0.0624 / bores.c.STANDARD_AIR_DENSITY_IMPERIAL,
                molecular_weight=bores.c.MOLECULAR_WEIGHT_CH4,
            ),
        ),
        skin_factor=0.0,
        is_active=True,
    )
    wells = bores.wells_(injectors=[injector], producers=[producer])

    # -------------------------------------------------------------------------
    # Timer
    # PV ≈ 762m * 7.62m * 15.24m * 0.2 = 1764 m³
    # 1 PVI at 6.97 m³/d ≈ 253 days.  Run 1 year to capture post-breakthrough.
    # -------------------------------------------------------------------------
    timer = bores.Timer(
        initial_step_size=bores.Time(hours=6.0),
        max_step_size=bores.Time(days=5.0),
        min_step_size=bores.Time(minutes=10.0),
        simulation_time=bores.Time(days=bores.c.DAYS_PER_YEAR),
        max_cfl_number=0.9,
        ramp_up_factor=1.2,
        backoff_factor=0.5,
        aggressive_backoff_factor=0.25,
        max_rejects=20,
    )

    # -------------------------------------------------------------------------
    # Config
    # Key settings for this problem:
    #   disable_capillary_effects=True  — Pcog = 0 per problem specification
    #   miscibility_model="immiscible"  — immiscible gas injection
    #   tight saturation/pressure change limits — low-P system, sharp gas front
    # -------------------------------------------------------------------------
    config = bores.Config(
        timer=timer,
        rock_fluid_tables=rock_fluid_tables,
        scheme="impes",
        output_frequency=1,
        max_iterations=200,
        pressure_solver="bicgstab",
        pressure_preconditioner="cached_ilu",
        log_interval=10,
        pvt_tables=pvt_tables,
        wells=wells,
        boundary_conditions=None,
        disable_capillary_effects=True,
        miscibility_model="immiscible",
        max_gas_saturation_change=0.1,
        max_oil_saturation_change=0.15,
        max_pressure_change=30.0,
    )

    config.to_file(Path("./benchmarks/runs/spe1/setup/config.yaml"))
    return


@app.cell
def run_simulation(Path, bores):
    preconditioner_factory = bores.CachedPreconditionerFactory(
        factory="ilu",
        name="cached_ilu",
        update_frequency=10,
        recompute_threshold=0.3,
    )
    preconditioner_factory.register(override=True)

    run = bores.Run.from_files(
        model_path=Path("./benchmarks/runs/spe1/setup/model.h5"),
        config_path=Path("./benchmarks/runs/spe1/setup/config.yaml"),
        pvt_table_path=Path("./benchmarks/runs/spe1/setup/pvt.h5"),
    )

    store = bores.ZarrStore(store=Path("./benchmarks/runs/spe1/results/spe1.zarr"))
    stream = bores.StateStream(
        run(),
        store=store,
        batch_size=30,
        background_io=True,
    )
    with stream:
        last_state = stream.last()

    last_state.model.to_file(Path("./benchmarks/runs/spe1/results/model.h5"))
    return


if __name__ == "__main__":
    app.run()
