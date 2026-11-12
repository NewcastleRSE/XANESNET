# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Unit tests for the generic ``Registry`` used by every XANESNET extension point."""

import pytest

from xanesnet.utils.registry import Registry


class _Widget:
    def __init__(self, size: int = 1, label: str = "w") -> None:
        self.size = size
        self.label = label


@pytest.fixture
def registry() -> Registry[type[_Widget]]:
    """Case-insensitive registry, matching how production registries are configured."""
    return Registry("Widget", normalize_key=str.lower)


###############################################################################
############################### REGISTRATION ##################################
###############################################################################


def test_register_returns_the_decorated_object_unchanged(registry):
    assert registry.register("a")(_Widget) is _Widget


def test_registered_object_is_retrievable_by_identity(registry):
    registry.register("a")(_Widget)
    assert registry.get("a") is _Widget


def test_duplicate_registration_is_rejected_and_keeps_the_first_entry(registry):
    class Other(_Widget):
        pass

    registry.register("a")(_Widget)
    with pytest.raises(KeyError, match="Widget 'a' already registered"):
        registry.register("a")(Other)
    assert registry.get("a") is _Widget


def test_duplicate_detection_uses_the_normalized_key(registry):
    registry.register("Name")(_Widget)
    with pytest.raises(KeyError, match="already registered"):
        registry.register("NAME")(_Widget)


def test_one_object_may_be_registered_under_several_aliases(registry):
    registry.register("none")(registry.register("no")(_Widget))
    assert registry.get("none") is registry.get("no") is _Widget


###############################################################################
################################### LOOKUP ####################################
###############################################################################


def test_lookup_of_unknown_key_raises_with_item_name_and_key(registry):
    with pytest.raises(KeyError, match="Widget 'missing' not found in registry"):
        registry.get("missing")


@pytest.mark.parametrize("key", ["mixed", "MIXED", "Mixed"])
def test_lookup_is_case_insensitive_when_normalizer_is_configured(registry, key):
    registry.register("Mixed")(_Widget)
    assert registry.get(key) is _Widget


def test_lookup_is_case_sensitive_without_a_normalizer():
    plain: Registry[type[_Widget]] = Registry("Widget")
    plain.register("Mixed")(_Widget)
    with pytest.raises(KeyError):
        plain.get("mixed")


def test_list_returns_normalized_keys_in_registration_order(registry):
    registry.register("Zeta")(_Widget)
    registry.register("alpha")(_Widget)
    assert registry.list() == ["zeta", "alpha"]


def test_list_returns_a_copy(registry):
    registry.register("a")(_Widget)
    registry.list().append("injected")
    assert registry.list() == ["a"]


def test_empty_registry_lists_nothing(registry):
    assert registry.list() == []


###############################################################################
################################### CREATION ##################################
###############################################################################


def test_create_forwards_keyword_arguments_to_the_registered_factory(registry):
    registry.register("a")(_Widget)
    widget = registry.create("A", size=5, label="x")
    assert isinstance(widget, _Widget)
    assert (widget.size, widget.label) == (5, "x")


def test_create_with_unknown_key_raises_key_error(registry):
    with pytest.raises(KeyError, match="not found"):
        registry.create("missing")


def test_create_supports_plain_callables(registry):
    registry.register("double")(lambda value: value * 2)
    assert registry.create("double", value=4) == 8


###############################################################################
############################ CUSTOM (TUPLE) KEYS ##############################
###############################################################################


def test_custom_key_normalization_and_formatting():
    registry: Registry[type[_Widget], tuple[str, str]] = Registry(
        "Pair",
        normalize_key=lambda key: (key[0].lower(), key[1].lower()),
        format_key=lambda key: f"for {key[0]}, {key[1]}",
    )
    registry.register(("Descriptor", "MLP"))(_Widget)

    assert registry.get(("descriptor", "mlp")) is _Widget
    assert registry.list() == [("descriptor", "mlp")]
    with pytest.raises(KeyError, match="Pair for a, b not found in registry"):
        registry.get(("A", "B"))
