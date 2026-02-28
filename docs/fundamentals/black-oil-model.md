# The Black-Oil Model

## What Is the Black-Oil Model?

The black-oil model is the most widely used fluid description in reservoir simulation, and for good reason. It strikes a balance between physical accuracy and computational efficiency that makes it suitable for the vast majority of conventional oil and gas reservoirs worldwide. If you have worked with or studied reservoir engineering, the black-oil model is almost certainly the first framework you encountered for describing how oil, water, and gas behave underground.

At its core, the black-oil model treats the hydrocarbon system as two surface components - stock-tank oil and surface gas - that can distribute themselves across three phases: an oil phase, a water phase, and a gas phase. The key simplification is that the composition of each component is fixed. Stock-tank oil is always the same oil, and surface gas is always the same gas. What changes with pressure and temperature is how much gas dissolves in the oil, not the chemical makeup of either component. This is in contrast to compositional models, which track individual hydrocarbon species (methane, ethane, propane, and heavier fractions) and allow the composition to vary throughout the reservoir.

The term "black oil" has historical roots. Early reservoir engineers observed that many crude oils appeared dark (nearly black) and had relatively simple phase behavior that could be described without detailed compositional analysis. The name stuck, even though the model applies equally well to light oils and some volatile systems. What makes a fluid system "black-oil compatible" is not its color but whether its phase behavior can be adequately described using pressure-dependent properties like formation volume factor and solution gas-oil ratio, rather than requiring a full equation-of-state calculation.

BORES is fundamentally a black-oil simulator. Every fluid property calculation, every PVT correlation, and every conservation equation in the framework is built on the black-oil assumptions. Understanding this model is therefore essential for using BORES effectively and interpreting its results correctly.

## Three Phases: Oil, Water, and Gas

In the black-oil formulation, the pore space of a reservoir rock is shared among three immiscible fluid phases. Each phase plays a distinct role in the physics of fluid flow and recovery, and each comes with its own set of properties that the simulator must track.

**Oil** is the target hydrocarbon - the fluid you are trying to recover and sell. Under reservoir conditions, oil is a compressed liquid that may contain a significant amount of dissolved natural gas. The amount of dissolved gas depends on pressure: at high pressure, the oil holds more gas in solution, and at low pressure, gas comes out of solution and forms a separate free gas phase. Oil properties like density, viscosity, and formation volume factor all depend on how much gas is dissolved, which in turn depends on pressure. This coupling between pressure and oil properties is one of the central features of the black-oil model.

**Water** is present in two forms. Connate water (also called irreducible water) is the water that was trapped in the pore spaces when the reservoir formed - it has been there for millions of years and typically cannot be mobilized. Injected water is the water you pump into the reservoir through injection wells to maintain pressure and sweep oil toward producers. In the black-oil model, water is treated as a slightly compressible liquid with properties that depend weakly on pressure and temperature. Gas can dissolve in water too, though typically in much smaller quantities than in oil.

**Gas** can exist as free gas (a separate phase in the pore space) or as dissolved gas (within the oil phase or, to a lesser extent, the water phase). Free gas appears when reservoir pressure drops below the bubble point - the pressure at which dissolved gas begins to come out of solution. In an undersaturated reservoir (pressure above bubble point), there is no free gas; all gas is dissolved in the oil. In a saturated reservoir (pressure at or below bubble point), gas has liberated from the oil and exists as its own phase. The gas phase is highly compressible, and its properties change dramatically with pressure.

!!! note "Saturation Constraint"
    The three phase saturations must always sum to unity: $S_o + S_w + S_g = 1.0$. This is a fundamental constraint in the black-oil formulation. Pore space is always 100% filled with fluid - there are no voids or partially filled pores. Even during pressure depletion, the saturations still sum to one; what changes is the pressure, not the total occupancy of the pore space.

## Solution Gas and the Gas-Oil Ratio

The solution gas-oil ratio $R_s$ is one of the most important parameters in the black-oil model. It describes how much gas is dissolved in the oil phase at a given pressure, expressed as standard cubic feet of gas per stock-tank barrel of oil (scf/STB). You can think of it as a measure of how "gassy" the oil is at a particular pressure.

The relationship between $R_s$ and pressure is straightforward in concept but critically important in practice:

$$R_s = R_s(p)$$

When reservoir pressure is above the bubble point pressure $p_b$, all of the gas that was originally dissolved in the oil remains in solution. In this **undersaturated** regime, $R_s$ stays constant at its initial value $R_{si}$ regardless of how much higher the pressure goes. The oil is not "full" of gas in any absolute sense, but there is no additional free gas available to dissolve, so $R_s$ does not increase.

When pressure drops below the bubble point, the oil can no longer hold all of its dissolved gas. Gas begins to come out of solution, forming a free gas phase. In this **saturated** regime, $R_s$ decreases with decreasing pressure. The released gas reduces the oil volume, increases oil viscosity (because the light dissolved gas was acting as a viscosity reducer), and creates a gas phase that competes with oil for flow through the pore space. This gas liberation process is one of the primary drive mechanisms in many oil reservoirs.

$$R_s(p) = \begin{cases} R_{si} & \text{if } p \geq p_b \\ R_s(p) < R_{si} & \text{if } p < p_b \end{cases}$$

!!! info "Why the Bubble Point Matters"
    The bubble point pressure $p_b$ is the single most important threshold in the black-oil model. Above it, the reservoir behaves as a single-phase (undersaturated) system with relatively simple physics. Below it, gas liberation creates a two-phase (saturated) system with more complex flow behavior, including gas-oil relative permeability effects and potentially gas coning or gas cap expansion. Getting $p_b$ right is critical for accurate simulation results.

In BORES, the solution GOR is computed from correlations (such as Standing or Vasquez-Beggs) or looked up from PVT tables that you provide. When you construct a model using `bores.reservoir_model()`, you can supply either `oil_bubble_point_pressure_grid` or `solution_gas_to_oil_ratio_grid` (or both), and the factory function will derive the missing quantity using the appropriate correlation.

## Formation Volume Factors

Formation volume factors (FVFs) are the conversion factors between reservoir conditions and surface (standard) conditions. They are essential because the volumes you measure at the surface - stock-tank barrels of oil, standard cubic feet of gas - are not the same as the volumes those fluids occupy in the reservoir at elevated pressure and temperature.

The **oil formation volume factor** $B_o$ converts reservoir oil volume to surface oil volume:

$$B_o = \frac{V_{o,\text{reservoir}}}{V_{o,\text{surface}}}$$

$B_o$ is always greater than 1.0 for live oil (oil containing dissolved gas). At reservoir conditions, the dissolved gas expands the oil, making each barrel of stock-tank oil occupy more than one barrel of space in the reservoir. A typical value might be 1.2 to 1.5 rb/STB. Above the bubble point, $B_o$ decreases slightly with increasing pressure due to liquid compression. Below the bubble point, $B_o$ decreases as gas comes out of solution and the oil shrinks.

The **gas formation volume factor** $B_g$ converts reservoir gas volume to surface gas volume:

$$B_g = \frac{V_{g,\text{reservoir}}}{V_{g,\text{surface}}}$$

$B_g$ is much less than 1.0 because gas at reservoir conditions is compressed into a much smaller volume than it occupies at the surface. Typical values range from 0.002 to 0.02 rb/scf. As pressure increases, gas compresses further, and $B_g$ decreases.

The **water formation volume factor** $B_w$ is close to 1.0 because water is nearly incompressible. Typical values are 1.0 to 1.07 rb/STB. It accounts for the slight thermal expansion and dissolved gas effects on water volume.

!!! tip "Surface vs. Reservoir Volumes"
    When you see production rates reported in "STB/day" or "Mscf/day," those are surface volumes. When working with flow equations inside the simulator, you need reservoir volumes. The formation volume factors bridge these two worlds. BORES handles the conversions internally, but understanding FVFs will help you make sense of production reports and material balance calculations.

## Viscosity

Fluid viscosity - the resistance to flow - is another pressure-dependent property that the black-oil model must track for each phase. Viscosity directly controls how fast each phase moves through the rock via Darcy's law, where flow rate is inversely proportional to viscosity.

**Oil viscosity** is the most complex of the three because it depends on pressure, temperature, and the amount of dissolved gas. There are two distinct regimes. **Dead oil** is oil with no dissolved gas. Its viscosity depends only on temperature and API gravity, and it is typically high - several centipoise to thousands of centipoise for heavy oils. **Live oil** is oil containing dissolved gas. The dissolved gas acts as a solvent, reducing viscosity significantly. A dead oil viscosity of 5 cP might drop to 1 cP with gas in solution.

Above the bubble point, oil viscosity increases with pressure because the liquid is being compressed. Below the bubble point, oil viscosity increases as gas comes out of solution - the oil loses its internal "solvent" and becomes more viscous. This creates a viscosity minimum at the bubble point, which is a distinctive feature of black-oil systems.

**Gas viscosity** is typically much lower than oil viscosity - on the order of 0.01 to 0.03 cP at reservoir conditions. It increases with pressure (unlike liquids) because gas molecules interact more at higher densities. BORES computes gas viscosity using the Lee-Gonzalez-Eakin correlation or CoolProp thermodynamic calculations.

**Water viscosity** is on the order of 0.3 to 1.0 cP at reservoir conditions, decreasing with temperature and increasing slightly with pressure and salinity. It is the most predictable of the three phase viscosities.

## Black-Oil Assumptions

The black-oil model rests on several key assumptions that define its domain of applicability. Understanding these assumptions helps you know when the model is appropriate and when you might need a different approach.

1. **Three components, three phases.** The hydrocarbon system is described by two components (stock-tank oil and surface gas) plus water. These distribute across three phases (oil, water, gas). No other components are tracked.

2. **Gas dissolves in oil (and optionally in water).** The only inter-phase mass transfer allowed is gas dissolving in or liberating from the oil phase (and a small amount in water). Oil does not evaporate into the gas phase, and oil and water do not mix.

3. **Oil and water are immiscible.** Under normal black-oil conditions, oil and water form separate, distinct phases with a clear interface at the pore scale. They do not dissolve into each other.

4. **Isothermal conditions.** Temperature is assumed constant throughout the simulation. There is no energy equation - heat conduction and convection are not modeled. Temperature may vary spatially (as an initial condition), but it does not change over time.

5. **Local thermodynamic equilibrium.** At every point in the reservoir, the fluid phases are assumed to be in thermodynamic equilibrium. Gas dissolves in oil instantly - there are no kinetic effects or time delays in mass transfer.

6. **Pressure-dependent properties.** All fluid properties ($B_o$, $B_g$, $R_s$, $\mu_o$, $\mu_g$, etc.) are functions of pressure only (at a given temperature). They do not depend on composition because composition is implicitly fixed by the two-component assumption.

!!! warning "Assumption Violations"
    Violating these assumptions does not always mean the black-oil model is useless - it means the results become less accurate. For moderately volatile oils or lean gas injection, the black-oil model may still give reasonable results with appropriate tuning. But for near-critical fluids, rich gas condensate, or thermal processes, you should use a compositional or thermal simulator instead.

## When to Use Black-Oil vs. Compositional

Choosing between a black-oil and a compositional simulator is one of the first decisions in any simulation study. The choice depends on the reservoir fluid, the recovery process, and the accuracy requirements.

**Use a black-oil model** (like BORES) when you are working with conventional oil reservoirs under primary depletion, waterflooding, or simple immiscible gas injection. The black-oil model is also appropriate for dry gas reservoirs and for screening studies where speed matters more than compositional accuracy. Most field development planning for conventional reservoirs worldwide is done with black-oil simulators.

**Use a compositional model** when the fluid phase behavior depends on composition - not just pressure. This includes gas condensate reservoirs (where liquid drops out of gas below the dewpoint), volatile oil reservoirs (where the oil contains so much dissolved gas that the black-oil two-component assumption breaks down), and rich gas injection processes (where the injected gas mixes with reservoir oil and creates intermediate-composition phases). CO2 flooding for enhanced oil recovery (EOR) also generally requires compositional modeling, though BORES offers a miscible extension using the Todd-Longstaff model that can capture first-order miscibility effects within the black-oil framework.

The computational cost difference is substantial. A compositional model with 6-10 components can be 5-20x slower than an equivalent black-oil model for the same grid size, because it must solve additional conservation equations for each component and perform flash calculations at every cell and every timestep. For large field-scale models with millions of cells, this cost difference can be the deciding factor.

## How BORES Implements Black-Oil Physics

BORES provides two pathways for computing black-oil PVT properties: **correlations** and **tables**. Both are first-class options, and you can choose whichever fits your workflow.

**PVT correlations** are empirical equations that compute fluid properties from pressure, temperature, and basic fluid characterization parameters (API gravity, gas gravity, GOR). BORES ships with industry-standard correlations including Standing (solution GOR, bubble point), Vasquez-Beggs (oil FVF, oil viscosity), Lee-Gonzalez-Eakin (gas viscosity), and several z-factor methods (Papay, Dranchuk-Abou-Kassem, Hall-Yarborough). When you build a model using `bores.reservoir_model()` and omit optional property grids, the factory function automatically fills them in using the appropriate correlations. This makes it easy to get started - you only need to provide the minimum set of inputs.

**PVT tables** (`bores.PVTTables`) let you supply tabulated property data from laboratory measurements or external PVT software. The tables are passed to `bores.reservoir_model()` via the `pvt_tables` parameter, and BORES will use table interpolation instead of correlations for any property that the tables cover. This gives you maximum control and accuracy when you have high-quality lab data.

```python
import bores

# The reservoir_model factory automatically computes PVT properties
# from correlations when they are not explicitly provided.
# For example, if you omit gas_viscosity_grid, BORES uses the
# Lee-Gonzalez-Eakin correlation internally.
model = bores.reservoir_model(
    grid_shape=(20, 20, 5),
    cell_dimension=(100.0, 100.0),
    thickness_grid=thickness,
    pressure_grid=pressure,
    temperature_grid=temperature,
    oil_viscosity_grid=oil_viscosity,
    oil_bubble_point_pressure_grid=bubble_point,
    oil_specific_gravity_grid=oil_sg,
    # ... other required grids ...
    # gas_viscosity_grid is omitted - computed from correlations
    # water_viscosity_grid is omitted - computed from correlations
    # oil_formation_volume_factor_grid is omitted - computed from correlations
)
```

During a simulation run, PVT properties are updated at each timestep as pressure changes. BORES recomputes formation volume factors, viscosities, solution GOR, and compressibilities using either the correlations or the tables you provided. This ensures that the fluid description remains physically consistent throughout the simulation, even as pressure evolves substantially from its initial value.

!!! tip "Providing vs. Computing Properties"
    You generally want to provide the properties you have high confidence in (from lab data or calibrated correlations) and let BORES compute the rest. The `bores.reservoir_model()` factory function is designed for exactly this workflow - it accepts dozens of optional property grids and fills in anything you leave out. See the [Fluid Properties](../guides/fluid-properties.md) guide for a detailed walkthrough.
