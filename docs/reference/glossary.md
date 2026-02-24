# Glossary

Definitions of reservoir engineering and BORES-specific terms.

---

## A

**Absolute Permeability**
: The measure of rock's ability to transmit fluids when 100% saturated with a single phase. Measured in millidarcies (mD). Independent of fluid type.

**Adaptive Timestep**
: Automatic adjustment of timestep size during simulation based on convergence and CFL stability. Increases on success, decreases on failure.

**Anisotropic Permeability**
: Permeability that varies by direction (kx, ky, kz). Typical in layered reservoirs where kv << kh.

**API Gravity**
: Measure of oil density. Higher API = lighter oil. Related to specific gravity: API = (141.5/SG) - 131.5.

**Aquifer**
: Water-bearing formation adjacent to reservoir. Provides pressure support through influx. Can be edge drive or bottom drive.

---

## B

**BHP (Bottom Hole Pressure)**
: Pressure at the bottom of the wellbore, in the perforated interval. Key constraint for well controls.

**Black-Oil Model**
: Simplified compositional model with three phases (oil, water, gas) and two components. Assumes gas dissolves in oil but not vice versa.

**Brooks-Corey Model**
: Empirical correlation for relative permeability and capillary pressure as power-law functions of saturation.

**Bubble Point Pressure (Pb)**
: Pressure at which first gas bubble forms when pressure declines. Below Pb, free gas evolves from oil.

**Buckley-Leverett**
: Analytical solution for immiscible displacement. Describes fractional flow and saturation front movement.

---

## C

**Capillary Pressure (Pc)**
: Pressure difference across interface between two immiscible fluids. Caused by surface tension and pore geometry.

**Carter-Tracy Model**
: Analytical aquifer model based on van Everdingen-Hurst solution. Provides time-dependent water influx.

**CFL Number (Courant Number)**
: Dimensionless number expressing ratio of physical wave speed to numerical grid speed. Must be < 1 for stability.

**Connate Water**
: Water present in reservoir at initial conditions. Immobile and does not flow.

**Corey Exponent**
: Exponent in Corey-type relative permeability correlations. Typical range 1.5-4.0.

---

## D

**Darcy's Law**
: Fundamental equation relating flow rate to pressure gradient, permeability, and viscosity: q = -(kA/μ)(dP/dx).

**Depletion**
: Production strategy relying solely on natural reservoir energy (pressure decline). No fluid injection.

**Depth Grid**
: 3D array containing cell-center depths. Calculated from thickness grid using cumulative sum.

**Dip Angle**
: Angle of reservoir tilt from horizontal. Affects fluid distribution and flow patterns.

**Dip Azimuth**
: Direction of reservoir dip, measured clockwise from North (0° = North, 90° = East, 180° = South, 270° = West).

---

## E

**Eclipse Rule**
: Mixing rule for three-phase relative permeability. Interpolates between two-phase curves based on saturations.

**EOR (Enhanced Oil Recovery)**
: Tertiary recovery techniques (chemical, thermal, miscible) to recover oil beyond primary and secondary methods.

---

## F

**Formation Volume Factor (B)**
: Ratio of volume at reservoir conditions to volume at standard conditions. Bo for oil, Bw for water, Bg for gas.

**Fractional Flow**
: Fraction of total flow rate contributed by a specific phase. fw = qw / (qo + qw + qg).

---

## G

**Gas Cap**
: Free gas accumulation at top of reservoir. Provides pressure support during production.

**Gas-Oil Contact (GOC)**
: Depth at which gas-oil interface occurs. Above GOC = gas, below = oil (with transition zone).

**Gas-Oil Ratio (GOR)**
: Ratio of gas production to oil production. Solution GOR (Rs) = dissolved gas in oil.

---

## H

**Heterogeneous**
: Non-uniform property distribution. Opposite of homogeneous. Real reservoirs are always heterogeneous.

**Hydrocarbon Pore Volume (HCPV)**
: Pore volume occupied by oil and gas (excludes connate water). HCPV = PV × (1 - Swc).

---

## I

**IMPES**
: IMplicit Pressure, Explicit Saturation. Numerical scheme solving pressure implicitly, saturations explicitly.

**Irreducible Water Saturation (Swi)**
: Minimum water saturation below which water does not flow. Bound by capillary forces.

---

## K

**kr (Relative Permeability)**
: Reduction in effective permeability due to presence of other phases. Dimensionless, 0 to 1.

---

## L

**Layered Grid**
: Grid where property varies by layer (z-direction). Created with `bores.layered_grid()`.

**LET Model**
: Three-parameter correlation for relative permeability. More flexible than Corey.

---

## M

**Material Balance**
: Equation relating production to reservoir pressure change. Used for validation and reserves estimation.

**Miscible Displacement**
: Injection where injected fluid and reservoir fluid mix at all proportions. Higher recovery than immiscible.

**Minimum Miscibility Pressure (MMP)**
: Pressure above which injected gas becomes fully miscible with oil. Critical for miscible EOR.

**Mobility (λ)**
: Ratio of relative permeability to viscosity. λ = kr/μ. High mobility = easy to flow.

**Mobility Ratio (M)**
: Ratio of displacing fluid mobility to displaced fluid mobility. M > 1 = unfavorable (fingering).

---

## N

**Net-to-Gross (NTG)**
: Fraction of gross thickness that is productive reservoir rock (excludes shales). Typical 0.5-0.9.

**Numba**
: Python JIT compiler. BORES uses Numba to accelerate numerical operations.

---

## O

**Oil-Water Contact (OWC)**
: Depth at which oil-water interface occurs. Above OWC = oil, below = water (with transition zone).

**OOIP (Original Oil In Place)**
: Total oil volume in reservoir at initial conditions. OOIP = 7758 × A × h × φ × (1-Swc) / Bo.

---

## P

**Peaceman Model**
: Well model relating grid-block pressure to bottom-hole pressure through effective wellbore radius.

**Perforating Interval**
: Depth range where wellbore is perforated for production/injection. Specified as cell indices in BORES.

**Porosity (φ)**
: Fraction of rock volume that is pore space. Typical sandstone: 0.15-0.30, carbonate: 0.05-0.25.

**Preconditioner**
: Matrix transformation that improves solver convergence. ILU and AMG are common types.

**PVT (Pressure-Volume-Temperature)**
: Properties describing fluid behavior with pressure/temperature changes. Bo, μo, Rs, etc.

---

## R

**Recovery Factor (RF)**
: Fraction of OOIP recovered. Primary: 5-30%, waterflooding: 30-50%, EOR: 40-70%.

**Relative Permeability (kr)**
: See **kr** above.

**Reservoir Model**
: In BORES, the `ReservoirModel` object containing all grid properties (pressure, saturation, porosity, etc.).

**Residual Saturation**
: Remaining saturation of a phase that cannot be displaced. Sor (oil), Sgr (gas), Swr (water).

---

## S

**Saturation (S)**
: Fraction of pore volume occupied by a phase. So + Sw + Sg = 1.0 always.

**Skin Factor (s)**
: Dimensionless measure of well damage (s > 0) or stimulation (s < 0). Affects productivity.

**Solution GOR (Rs)**
: Volume of gas dissolved in oil at reservoir conditions. SCF/STB. Increases with pressure.

**Stock Tank Barrel (STB)**
: Oil volume at standard conditions (60°F, 14.7 psi). 1 STB = 42 gallons.

**Supersaturated**
: Condition where pressure is below bubble point. Free gas phase exists.

---

## T

**Todd-Longstaff Model**
: Empirical model for miscible displacement. Blends fully-mixed and segregated flow assumptions using omega parameter.

**Transition Zone**
: Region where saturation varies gradually (e.g., between oil and water). Thickness depends on capillary pressure.

**Transmissibility (T)**
: Flow conductivity between grid cells. T = kA/Δx. Modified by faults/fractures.

---

## U

**Undersaturated**
: Condition where pressure is above bubble point. All gas dissolved in oil, no free gas.

**Uniform Grid**
: Grid where all cells have same property value. Created with `bores.uniform_grid()`.

---

## V

**Van Genuchten Model**
: Alternative to Brooks-Corey for capillary pressure and relative permeability.

**Viscosity (μ)**
: Resistance to flow. Higher viscosity = harder to flow. Measured in centipoise (cP).

---

## W

**Waterflood**
: Secondary recovery method injecting water to displace oil toward producers.

**Water Cut**
: Fraction of produced fluid that is water. Increases over time in waterfloods.

**Wettability**
: Preference of rock surface for one fluid over another. Water-wet (most common) or oil-wet.

**Well Control**
: Operating constraint for well. Can be rate control, BHP control, or adaptive.

**Well Schedule**
: Time-dependent changes to well controls. E.g., increase rate after 1 year.

---

## Z

**Z-Factor**
: Gas compressibility factor. Accounts for non-ideal gas behavior. Used in gas PVT.

---

## BORES-Specific Terms

**Config**
: Configuration object controlling simulation behavior (solvers, tolerances, scheme, output).

**Grid Shape**
: Tuple (nx, ny, nz) defining number of cells in each direction.

**ModelState**
: Snapshot of reservoir state at a specific timestep. Contains updated model, wells, rates.

**Rock-Fluid Tables**
: Object containing relative permeability and capillary pressure models.

**StateStream**
: Iterator for simulation results with persistence to storage backends.

**Timer**
: Adaptive timestep controller with CFL-based stability and ramping logic.

---

## See Also

- [Core Concepts](../getting-started/core-concepts.md) - Framework fundamentals
- [API Reference](api.md) - Function signatures and parameters
- [User Guide](../guides/index.md) - Detailed usage documentation
