"""Tests for visualization base module (property registry, colorbars, etc.)."""

import pytest

from bores.visualization.base import (
    ColorbarConfig,
    ColorbarPresets,
    ColorScheme,
    PropertyMeta,
    PropertyRegistry,
    property_registry,
)


class TestColorScheme:
    """Tests for ColorScheme enum."""

    def test_color_scheme_values(self):
        """Test that color schemes have correct values."""
        assert ColorScheme.VIRIDIS.value == "viridis"
        assert ColorScheme.PLASMA.value == "plasma"
        assert ColorScheme.INFERNO.value == "inferno"

    def test_color_scheme_str(self):
        """Test string conversion of color schemes."""
        assert str(ColorScheme.VIRIDIS) == "viridis"


class TestPropertyMeta:
    """Tests for PropertyMeta dataclass."""

    def test_create_property_meta(self):
        """Test creating PropertyMeta instance."""
        meta = PropertyMeta(
            name="pressure_grid",
            display_name="Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
        )
        assert meta.name == "pressure_grid"
        assert meta.display_name == "Pressure"
        assert meta.unit == "psi"
        assert not meta.log_scale

    def test_property_meta_with_log_scale(self):
        """Test PropertyMeta with log scale."""
        meta = PropertyMeta(
            name="viscosity",
            display_name="Viscosity",
            unit="cP",
            color_scheme=ColorScheme.INFERNO,
            log_scale=True,
        )
        assert meta.log_scale

    def test_property_meta_with_min_max(self):
        """Test PropertyMeta with min/max clipping."""
        meta = PropertyMeta(
            name="saturation",
            display_name="Saturation",
            unit="fraction",
            color_scheme=ColorScheme.CIVIDIS,
            min_val=0.0,
            max_val=1.0,
        )
        assert meta.min_val == 0.0
        assert meta.max_val == 1.0

    def test_property_meta_with_aliases(self):
        """Test PropertyMeta with aliases."""
        meta = PropertyMeta(
            name="pressure",
            display_name="Pressure",
            unit="psi",
            color_scheme=ColorScheme.VIRIDIS,
            aliases=["p", "pres"],
        )
        assert "p" in meta.aliases
        assert "pres" in meta.aliases


class TestPropertyRegistry:
    """Tests for PropertyRegistry class."""

    def test_registry_has_properties(self):
        """Test that registry contains expected properties."""
        assert "oil_pressure" in property_registry
        assert "water_saturation" in property_registry
        assert "gas_saturation" in property_registry

    def test_registry_get_property(self):
        """Test getting property from registry."""
        meta = property_registry.get("oil_pressure")
        assert meta.display_name == "Oil Pressure"
        assert meta.unit == "psi"

    def test_registry_get_with_alias(self):
        """Test getting property using alias."""
        meta = property_registry.get("pressure")
        assert meta.display_name == "Oil Pressure"

    def test_registry_get_unknown_property_raises(self):
        """Test that getting unknown property raises ValueError."""
        with pytest.raises(ValueError, match="Unknown property"):
            property_registry.get("nonexistent_property")

    def test_registry_contains(self):
        """Test __contains__ method."""
        assert "oil_pressure" in property_registry
        assert "pressure" in property_registry  # alias
        assert "nonexistent" not in property_registry

    def test_registry_iteration(self):
        """Test iterating over registry."""
        properties = list(property_registry)
        assert "oil_pressure" in properties
        assert "water_saturation" in properties

    def test_registry_properties_list(self):
        """Test getting list of all properties."""
        props = property_registry.properties
        assert isinstance(props, list)
        assert len(props) > 0

    def test_registry_count(self):
        """Test registry count."""
        assert property_registry.count > 0

    def test_registry_clean_name(self):
        """Test name cleaning."""
        clean = PropertyRegistry._clean_name("Oil-Pressure ")
        assert clean == "oil_pressure"

    def test_registry_setitem(self):
        """Test adding new property to registry."""
        registry = PropertyRegistry()
        new_meta = PropertyMeta(
            name="custom",
            display_name="Custom Property",
            unit="units",
            color_scheme=ColorScheme.VIRIDIS,
        )
        registry["custom_property"] = new_meta
        assert "custom_property" in registry
        retrieved = registry.get("custom_property")
        assert retrieved.display_name == "Custom Property"


class TestColorbarConfig:
    """Tests for ColorbarConfig dataclass."""

    def test_create_colorbar_config(self):
        """Test creating ColorbarConfig instance."""
        config = ColorbarConfig(
            colorscale="viridis",
            cmin=0.0,
            cmax=1.0,
        )
        assert config.colorscale == "viridis"
        assert config.cmin == 0.0
        assert config.cmax == 1.0

    def test_colorbar_config_to_plotly_dict(self):
        """Test converting ColorbarConfig to Plotly dict."""
        config = ColorbarConfig(
            colorscale="plasma",
            reversescale=True,
            cmin=0.0,
            cmax=100.0,
            title="Pressure (psi)",
            tickformat=".1f",
        )
        plotly_dict = config.to_plotly_dict()

        assert plotly_dict["colorscale"] == "plasma"
        assert plotly_dict["reversescale"] is True
        assert plotly_dict["cmin"] == 0.0
        assert plotly_dict["cmax"] == 100.0
        assert plotly_dict["title"] == "Pressure (psi)"
        assert plotly_dict["tickformat"] == ".1f"

    def test_colorbar_config_minimal(self):
        """Test ColorbarConfig with minimal parameters."""
        config = ColorbarConfig(colorscale="viridis")
        plotly_dict = config.to_plotly_dict()

        assert plotly_dict["colorscale"] == "viridis"
        assert "reversescale" not in plotly_dict
        assert "cmin" not in plotly_dict


class TestColorbarPresets:
    """Tests for ColorbarPresets class."""

    def test_saturation_preset(self):
        """Test saturation preset."""
        preset = ColorbarPresets.SATURATION
        assert preset.cmin == 0.0
        assert preset.cmax == 1.0
        assert preset.tickformat == ".2f"

    def test_oil_saturation_preset(self):
        """Test oil saturation preset."""
        preset = ColorbarPresets.OIL_SATURATION
        assert preset.colorscale == "Cividis"
        assert preset.cmin == 0.0
        assert preset.cmax == 1.0

    def test_pressure_preset(self):
        """Test pressure preset."""
        preset = ColorbarPresets.PRESSURE
        assert preset.colorscale == "Viridis"
        assert preset.tickformat == ".0f"

    def test_viscosity_preset(self):
        """Test viscosity preset (log scale)."""
        preset = ColorbarPresets.VISCOSITY
        assert preset.colorscale == "Inferno"
        assert preset.tickformat == ".2e"

    def test_get_for_property_found(self):
        """Test get_for_property with known property."""
        preset = ColorbarPresets.get_for_property("oil_saturation")
        assert preset is not None
        assert preset == ColorbarPresets.OIL_SATURATION

    def test_get_for_property_not_found(self):
        """Test get_for_property with unknown property."""
        preset = ColorbarPresets.get_for_property("unknown_property")
        assert preset is None

    def test_get_for_property_case_insensitive(self):
        """Test get_for_property is case-insensitive."""
        preset1 = ColorbarPresets.get_for_property("oil_saturation")
        preset2 = ColorbarPresets.get_for_property("OIL_SATURATION")
        assert preset1 == preset2

    def test_diverging_presets(self):
        """Test diverging colorbar presets."""
        preset = ColorbarPresets.DIVERGING
        assert preset.colorscale == "RdBu"

        preset_balanced = ColorbarPresets.DIVERGING_BALANCED
        assert preset_balanced.colorscale == "Balance"

    def test_depth_preset(self):
        """Test depth preset."""
        preset = ColorbarPresets.DEPTH
        assert preset.colorscale == "Earth"
        assert preset.reversescale is True

    def test_preset_to_plotly_dict(self):
        """Test converting preset to Plotly dict."""
        preset = ColorbarPresets.OIL_SATURATION
        plotly_dict = preset.to_plotly_dict()

        assert "colorscale" in plotly_dict
        assert "cmin" in plotly_dict
        assert "cmax" in plotly_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
