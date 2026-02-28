# What is Reservoir Simulation?

## The Big Picture

Think of reservoir simulation as weather forecasting, but underground. Instead of predicting wind patterns and rainfall across the atmosphere, you are predicting how oil, water, and gas flow through rock thousands of feet below the surface. Just as meteorologists divide the atmosphere into a 3D grid of cells and solve physics equations in each one, reservoir engineers divide a petroleum reservoir into a grid of cells and solve fluid flow equations to predict how pressure and saturation change over time.

The analogy goes deeper than you might expect. Weather models solve the Navier-Stokes equations on a grid. Reservoir simulators solve Darcy's law and mass conservation on a grid. Both are initial-boundary value problems: you start with known conditions (today's weather, or the reservoir state at discovery) and march forward in time, computing the state at each future moment from the state at the previous moment. Both struggle with the same fundamental tension between accuracy and computational cost - finer grids and smaller timesteps give better answers but take longer to run.

Where the analogy breaks down is in what you can observe. Meteorologists have satellites, weather stations, and radiosondes feeding them data in real time. Reservoir engineers have a handful of wells - tiny pinpricks in a formation that might span hundreds of square miles. Everything between those wells is uncertain. This is why reservoir simulation is as much about managing uncertainty as it is about solving equations. You build a model, calibrate it against the data you have (a process called "history matching"), and then use it to forecast the future under different development plans.

In practical terms, a reservoir simulator takes a description of the reservoir (rock properties, fluid properties, geometry, wells) and a development plan (which wells to drill, what rates to produce at, whether to inject water or gas) and predicts how the reservoir will respond over months, years, or decades. The output tells you how much oil you will recover, how quickly production will decline, when water will break through to your producers, and whether that expensive infill well is worth drilling.

## Why Model Reservoirs?

The most immediate reason to run reservoir simulations is money. A single offshore well can cost $50-100 million to drill. A full field development plan might involve billions of dollars in infrastructure. Making these investment decisions based on gut feeling or simple decline curve analysis leaves enormous value on the table - or worse, leads to catastrophic mistakes like drilling dry holes or building facilities that are either too large or too small.

Reservoir simulation lets you run "what-if" experiments that would be impossible or prohibitively expensive in the real world. What if you placed your injection wells in a line drive pattern instead of a five-spot? What if you converted that underperforming producer to a water injector? What if you delayed gas injection by two years until gas prices recover? Each of these scenarios can be modeled in hours on a computer, compared against alternatives, and ranked by net present value before a single dollar is committed.

Beyond economics, simulation is essential for regulatory compliance. Governments that grant petroleum licenses require operators to demonstrate that they are developing reservoirs responsibly and maximizing recovery. Environmental regulations may require modeling of CO2 storage behavior, aquifer contamination risk, or subsidence due to pressure depletion. In all these cases, a calibrated simulation model is the standard tool for making and defending technical arguments.

Finally, reservoir simulation serves as an integration platform. It forces you to combine data from geology (structure maps, facies models), petrophysics (porosity, permeability from logs and cores), PVT analysis (fluid properties from lab measurements), and production engineering (well completions, artificial lift) into a single consistent framework. The act of building a simulation model often reveals inconsistencies in the data that would otherwise go unnoticed.

## Types of Simulators

Not all reservoir simulators solve the same equations. The choice of simulator depends on the physics that dominate the recovery process you are modeling. There are three major families, and understanding where BORES fits will help you know when it is the right tool and when you need something else.

**Black-oil simulators** are the workhorse of the industry. They model three phases (oil, water, gas) and assume that the hydrocarbon system can be described by two components: a "stock-tank oil" component and a "surface gas" component. Gas can dissolve in oil (solution gas) and come out of solution when pressure drops below the bubble point, but the composition of each component is fixed. This simplification makes the math tractable and the simulations fast. Black-oil models are appropriate for the vast majority of conventional oil and gas reservoirs, waterflooding, and simple gas injection. BORES is a black-oil simulator.

**Compositional simulators** track the full hydrocarbon composition - methane, ethane, propane, and heavier fractions - as individual components. This is necessary when the phase behavior depends on composition, not just pressure and temperature. Common use cases include gas condensate reservoirs (where liquid drops out of gas as pressure declines), miscible gas injection (where injected gas mixes with reservoir oil to create a single phase), and volatile oil reservoirs (where the oil contains so much dissolved gas that the black-oil assumption breaks down). Compositional simulation is significantly more expensive than black-oil, often 5-20x slower for the same grid size.

**Thermal simulators** add energy balance equations to the flow equations, tracking temperature changes throughout the reservoir. These are essential for heavy oil and bitumen recovery processes like steam-assisted gravity drainage (SAGD), cyclic steam stimulation (CSS), and in-situ combustion. When you inject steam at 300 degrees Celsius into a reservoir containing oil with the viscosity of peanut butter, the temperature distribution controls everything - and a black-oil or compositional model that assumes isothermal conditions will give you nonsense.

There are also specialized simulators for chemical flooding (polymer, surfactant, alkaline), geomechanical coupling (compaction, subsidence, fracture propagation), and dual-porosity/dual-permeability models for naturally fractured reservoirs. In practice, many commercial simulators blur these boundaries by offering optional modules that extend a base black-oil engine.

## Basic Physics: Darcy's Law and Conservation

The physics of reservoir simulation rests on two pillars: Darcy's law, which describes how fluids move through porous rock, and conservation of mass, which ensures that fluid is neither created nor destroyed. Everything else - PVT correlations, relative permeability curves, well models - is supporting detail that feeds into these two fundamental principles.

**Darcy's law** states that the flow rate of a fluid through a porous medium is proportional to the pressure gradient and inversely proportional to the fluid's viscosity:

$$q = -\frac{kA}{\mu}\frac{\partial P}{\partial x}$$

where $q$ is the volumetric flow rate, $k$ is the permeability of the rock (a measure of how easily fluid can flow through it), $A$ is the cross-sectional area perpendicular to flow, $\mu$ is the dynamic viscosity of the fluid, and $\partial P / \partial x$ is the pressure gradient in the flow direction. The negative sign indicates that fluid flows from high pressure to low pressure - down the pressure gradient.

If you have a programming background, you can think of Darcy's law as the "flux function" - it tells you how much stuff moves through each face of a grid cell per unit time, given the pressure difference between neighboring cells. Permeability $k$ is like the conductance of a wire in an electrical circuit: higher permeability means less resistance to flow. Viscosity $\mu$ acts like additional resistance - thick, viscous oil flows more slowly than thin, watery fluid under the same pressure gradient.

For **multiphase flow** (oil, water, and gas flowing simultaneously), Darcy's law is extended with relative permeability, which accounts for the fact that each phase interferes with the others:

$$q_\alpha = -\frac{k \cdot k_{r\alpha}}{\mu_\alpha} A \frac{\partial P_\alpha}{\partial x}$$

Here $\alpha$ denotes the phase (oil, water, or gas), $k_{r\alpha}$ is the relative permeability of phase $\alpha$ (a dimensionless number between 0 and 1 that depends on saturation), and $P_\alpha$ is the phase pressure (which may differ between phases due to capillary pressure). When oil saturation is high and water saturation is low, $k_{ro}$ is large and $k_{rw}$ is small, so oil flows easily and water barely moves. As water injection increases water saturation, the situation reverses.

**Conservation of mass** is the other half of the puzzle. For each phase, the rate of mass accumulation in a cell must equal the net inflow minus the net outflow, plus any source/sink terms (wells). In differential form:

$$\frac{\partial}{\partial t}(\phi \rho_\alpha S_\alpha) + \nabla \cdot (\rho_\alpha \vec{q}_\alpha) = Q_\alpha$$

where $\phi$ is porosity (the fraction of rock volume that is pore space), $\rho_\alpha$ is the density of phase $\alpha$, $S_\alpha$ is the saturation (fraction of pore space occupied by that phase), and $Q_\alpha$ represents source/sink terms from wells. The saturations must always sum to one: $S_o + S_w + S_g = 1$.

## Grid-Based Discretization

You cannot solve Darcy's law and the conservation equations analytically for a real reservoir - the geometry is too complex, the properties vary spatially, and there is no closed-form solution. Instead, you discretize the reservoir into a grid of cells and solve a system of algebraic equations that approximate the continuous differential equations.

The basic idea is to divide the reservoir into a large number of small rectangular blocks (cells). Each cell has its own rock properties (porosity, permeability), fluid properties (saturation, pressure, viscosity), and geometry (dimensions, depth). At each timestep, you compute the flow between neighboring cells using Darcy's law and update the pressure and saturation in each cell based on the net inflow and outflow. The finer your grid, the more accurately you capture spatial variations - but the more equations you have to solve and the slower the simulation runs.

BORES uses a **finite difference** approach on structured Cartesian grids. "Structured" means the cells are arranged in a regular i-j-k pattern (think of a 3D spreadsheet), and "Cartesian" means the cells are rectangular blocks aligned with the coordinate axes. This is the simplest and most common grid type in reservoir simulation. The alternative - unstructured grids with tetrahedral or polyhedral cells - can conform better to complex geology but are much harder to implement and debug.

The spacing between cell centers is where the numerical approximation enters. BORES computes the flow between cell $(i,j,k)$ and cell $(i+1,j,k)$ using the pressure difference between those two cells, divided by the distance between their centers, multiplied by a "transmissibility" that encodes the geometric and rock property information at that interface. This transmissibility calculation is done once at the start (and updated when properties change) so that the inner loop of the solver only needs to multiply transmissibility by pressure difference.

For Python developers, the grid is stored as 3D NumPy arrays where each element corresponds to a cell. Pressure, saturation, porosity, permeability - these are all arrays of shape `(nx, ny, nz)`. Operations on these arrays are vectorized, meaning you can update all cells simultaneously rather than looping through them one at a time. This is essential for performance, especially when combined with Numba JIT compilation for the most critical numerical kernels.

## Time-Stepping

Reservoir simulation advances through time in discrete steps. At each step, the simulator computes the pressure and saturation distributions at the new time level from the known state at the current time level. The choice of how to do this - how large the timestep is, and whether to treat the equations implicitly or explicitly - has profound implications for accuracy, stability, and speed.

**Explicit methods** compute the new state entirely from the old state. They are simple and cheap per step, but they are conditionally stable: if your timestep is too large relative to your grid spacing and flow velocities, the solution will oscillate wildly and blow up. The stability limit is governed by the **CFL condition** (Courant-Friedrichs-Lewy), which essentially says that information should not travel more than one cell width per timestep.

**Implicit methods** compute the new state by solving a system of equations that couples the old state and the new state together. They are unconditionally stable - you can take arbitrarily large timesteps without blowing up - but each step requires solving a large sparse linear system, which is expensive. The trade-off is fewer, more expensive steps versus many cheap steps.

**IMPES** (Implicit Pressure, Explicit Saturation), which is the scheme BORES uses, is a compromise. The pressure equation is solved implicitly, giving unconditional stability for the pressure field. The saturation equation is then updated explicitly using the newly computed pressure field. This means pressure can change dramatically in a single step without instability, but the saturation update is still subject to CFL-like constraints. BORES handles this by monitoring the saturation change per step and reducing the timestep if it gets too large.

The IMPES approach is well-suited for problems where pressure changes are large and fast (like near wells) but saturation fronts move relatively slowly (like a waterflood advancing through a reservoir). This covers the majority of practical reservoir simulation scenarios. For problems where saturation changes are very sharp - like gas coning near a well - IMPES may require very small timesteps, and a fully implicit method might be more efficient. But for the typical field-scale simulation, IMPES offers an excellent balance of speed and robustness.

## Where BORES Fits

BORES is a **3D three-phase black-oil simulator** that uses the **IMPES** scheme on **structured Cartesian grids**. In the taxonomy of reservoir simulators, this places it firmly in the "general-purpose black-oil" category - the most commonly used type of simulator in the petroleum industry.

What makes BORES different from commercial simulators like Eclipse, CMG IMEX, or tNavigator is not the physics (which are standard) but the interface. BORES is a Python library, not a standalone application with a proprietary input format. You define your reservoir model, configure your simulation, and analyze results using Python code - the same language you use for data science, machine learning, and automation. This means you can integrate reservoir simulation into larger workflows, parametrize models programmatically, and leverage the entire Python scientific ecosystem.

BORES is appropriate for conventional oil and gas reservoirs under primary depletion, waterflooding, and immiscible or miscible gas injection. It supports features like faults, fractures, aquifer models (Carter-Tracy), multiple well types, well schedules, and various boundary conditions. It is not the right tool for thermal recovery, compositional gas condensate modeling, or chemical flooding - those require specialized physics that are outside the black-oil framework.

For Python developers coming to reservoir simulation for the first time, BORES is designed to be the tool where you learn by doing. The API is designed to be readable, the error messages are informative, and you can inspect every intermediate result as a NumPy array. For reservoir engineers who want to move beyond the limitations of commercial GUI-based tools, BORES gives you full programmatic control over every aspect of the simulation.
