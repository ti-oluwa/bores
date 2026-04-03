import typing

import attrs

from bores.capillary_pressures import CapillaryPressureTable
from bores.relperm import RelativePermeabilityTable
from bores.serialization import Serializable
from bores.types import CapillaryPressures, RelativePermeabilities

__all__ = ["RockFluidTables"]


@typing.final
@attrs.frozen
class RockFluidTables(Serializable):
    """
    Tables defining rock-fluid interactions in the reservoir.

    Made up of a relative permeability table and an optional capillary pressure table. The relative
    permeability table is required, while the capillary pressure table is optional
    (but required if `config.disable_capillary_effects=False`).
    """

    relative_permeability_table: RelativePermeabilityTable
    """Callable that evaluates the relative permeability curves based on fluid saturations."""
    capillary_pressure_table: typing.Optional[CapillaryPressureTable] = None
    """Optional callable that evaluates the capillary pressure curves based on fluid saturations. This is required if `config.disable_capillary_effects=False`"""

    def get_relative_permeabilities(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> RelativePermeabilities:
        """
        Evaluates the relative permeability curves based on the provided fluid saturations and
        any additional parameters required by the specific relative permeability model.

        :param water_saturation: The saturation of water in the reservoir (between 0 and 1).
        :param oil_saturation: The saturation of oil in the reservoir (between 0 and 1).
        :param gas_saturation: The saturation of gas in the reservoir (between 0 and 1).
        :param kwargs: Additional parameters required by the specific relative permeability model
            (e.g., irreducible saturations, residual saturations, etc.).
        :return: A `RelativePermeabilities` object containing the relative permeabilities for water, oil, and gas
        """
        return self.relative_permeability_table.get_relative_permeabilities(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )

    def get_capillary_pressures(
        self,
        water_saturation: float,
        oil_saturation: float,
        gas_saturation: float,
        **kwargs: typing.Any,
    ) -> CapillaryPressures:
        """
        Evaluates the capillary pressure curves based on the provided fluid saturations and
        any additional parameters required by the specific capillary pressure model.

        :param water_saturation: The saturation of water in the reservoir (between 0 and 1).
        :param oil_saturation: The saturation of oil in the reservoir (between 0 and 1).
        :param gas_saturation: The saturation of gas in the reservoir (between 0 and 1).
        :param kwargs: Additional parameters required by the specific capillary pressure model
            (e.g., entry pressure, pore size distribution index, etc.).
        :return: A `CapillaryPressures` object containing the capillary pressures for water-oil and gas-oil interfaces
        """
        if self.capillary_pressure_table is None:
            raise ValueError("Capillary pressure table is not defined.")
        return self.capillary_pressure_table.get_capillary_pressures(
            water_saturation=water_saturation,
            oil_saturation=oil_saturation,
            gas_saturation=gas_saturation,
            **kwargs,
        )
