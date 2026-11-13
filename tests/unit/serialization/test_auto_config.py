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

"""Unit tests for automatic ("auto") config resolution and its streaming spectral statistics.

The tests plug a tiny in-memory dataset, batch processor and resolvers into the public registries under unique
``unit_test_*`` keys, so the resolution logic is exercised without depending on any real model or encoding.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.batchprocessors.forward.base import ForwardBatchProcessor
from xanesnet.encodings import SpectraEncoding
from xanesnet.serialization.auto_config import (
    resolve_auto_encoding_config,
    resolve_auto_model_config,
)
from xanesnet.serialization.auto_config import statistics as statistics_module
from xanesnet.serialization.auto_config.helpers import (
    format_value,
    requested_auto_fields,
)
from xanesnet.serialization.auto_config.registries import (
    EncodingAutoResolver,
    ModelAutoResolver,
)
from xanesnet.serialization.auto_config.statistics import (
    SpectralStatistics,
    SpectralStatisticsCollector,
    collect_spectral_statistics,
)
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError

DATASET_TYPE = "unit_test_dataset"
MODEL_TYPE = "unit_test_model"
UNRESOLVED_MODEL_TYPE = "unit_test_unresolved_model"
ENCODING_TYPE = "unit_test_encoding"
UNRESOLVED_ENCODING_TYPE = "unit_test_unresolved_encoding"

N_SAMPLES, N_FEATURES, N_POINTS = 6, 4, 5

###############################################################################
################################# TEST DOUBLES ################################
###############################################################################


class FakeDataset:
    """Dataset of ``N_SAMPLES`` samples; sample ``i`` has spectrum ``i + arange(N_POINTS)`` and element 26 or 29."""

    dataset_type = DATASET_TYPE
    train_subset = None

    def __len__(self) -> int:
        return N_SAMPLES

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.full((N_FEATURES,), float(index)),
            "y": index + torch.arange(N_POINTS, dtype=torch.float32),
            "element": torch.tensor(26 if index % 2 == 0 else 29),
        }

    @staticmethod
    def collate_fn(samples: list[dict[str, torch.Tensor]]) -> SimpleNamespace:
        return SimpleNamespace(**{key: torch.stack([s[key] for s in samples]) for key in samples[0]})

    @staticmethod
    def get_dataloader() -> type[DataLoader]:
        return DataLoader


class FakeBatchProcessor(ForwardBatchProcessor):
    def input_preparation(self, batch: Any) -> dict[str, Any]:
        return {"x": batch.x}

    def target_preparation(self, batch: Any) -> torch.Tensor:
        return batch.y

    def element_preparation(self, batch: Any) -> torch.Tensor:
        return batch.element

    def sample_id_preparation(self, batch: Any) -> np.ndarray:
        return np.array([""] * len(batch.x))


for _model_type in (MODEL_TYPE, UNRESOLVED_MODEL_TYPE):
    BatchProcessorRegistry.register((DATASET_TYPE, _model_type))(FakeBatchProcessor)

RESOLVER_CALLS: list[Any] = []


@ModelAutoResolver.register(MODEL_TYPE)
def _resolve_model(inputs, target, dataset):
    RESOLVER_CALLS.append((inputs, target, dataset))
    return {"in_size": inputs["x"].shape[-1], "out_size": target.shape[-1]}


@ModelAutoResolver.register(UNRESOLVED_MODEL_TYPE)
def _resolve_nothing(inputs, target, dataset):
    return {}


@EncodingAutoResolver.register(ENCODING_TYPE)
def _resolve_encoding(item, statistics):
    return {"num_points": statistics.overall.num_points, "scale": statistics.overall.global_maximum}


@EncodingAutoResolver.register(UNRESOLVED_ENCODING_TYPE)
def _resolve_nothing_for_encoding(item, statistics):
    return {}


@pytest.fixture(autouse=True)
def _single_process_statistics(monkeypatch):
    """Collect statistics in-process: worker processes would only slow these tests down."""
    monkeypatch.setattr(statistics_module, "STATISTICS_NUM_WORKERS", 0)
    RESOLVER_CALLS.clear()


@pytest.fixture
def dataset() -> FakeDataset:
    return FakeDataset()


def _train_config(model: dict[str, Any], encodings: list[dict[str, Any]]) -> Config:
    return Config({"model": model, "encodings": encodings})


###############################################################################
################################### HELPERS ###################################
###############################################################################


def test_requested_auto_fields_finds_case_insensitive_tokens_at_the_top_level_only():
    section = {"in_size": "auto", "out_size": "AUTO", "hidden": 8, "name": "autonomous", "nested": {"x": "auto"}}
    assert requested_auto_fields(section) == {"in_size", "out_size"}


def test_requested_auto_fields_of_a_concrete_section_is_empty():
    assert requested_auto_fields({"a": 1, "b": "x"}) == set()


@pytest.mark.parametrize(
    "value, expected",
    [
        (3, "3"),
        ("text", "text"),
        ([1, 2, 3], "[1, 2, 3]"),
        ({"a": 1, "b": [2]}, "{a=1, b=[2]}"),
    ],
)
def test_format_value_prints_small_values_in_full(value, expected):
    assert format_value(value) == expected


def test_long_lists_are_truncated_with_their_length():
    text = format_value(list(range(20)))
    assert text == "[0, 1, 2, ...]  (len=20)"


def test_long_nested_lists_report_rows_and_inner_length():
    text = format_value([[0, 0]] * 10)
    assert "rows=10" in text and "inner_len=2" in text


def test_ragged_nested_lists_report_all_inner_lengths():
    text = format_value([[0] * (1 + i % 2) for i in range(10)])
    assert "inner_lens=[1, 2]" in text


def test_large_dicts_are_truncated_with_their_key_count():
    text = format_value({f"k{i}": i for i in range(6)})
    assert text == "{k0=0, k1=1, k2=2, ...  (keys=6)}"


###############################################################################
########################### STREAMING STATISTICS ##############################
###############################################################################


def test_statistics_match_the_full_batch_computation_regardless_of_chunking():
    data = torch.randn(50, 7, generator=torch.Generator().manual_seed(0))
    chunked = SpectralStatistics()
    for start in range(0, 50, 13):
        chunked.update(data[start : start + 13])

    torch.testing.assert_close(chunked.mean, data.double().mean(0))
    torch.testing.assert_close(chunked.std, data.double().std(0, unbiased=False))
    torch.testing.assert_close(chunked.minimum, data.double().amin(0))
    torch.testing.assert_close(chunked.maximum, data.double().amax(0))
    assert chunked.num_points == 7


def test_global_statistics_aggregate_over_all_points():
    stats = SpectralStatistics()
    stats.update(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))

    assert stats.global_mean == pytest.approx(4.0)
    assert stats.global_std == pytest.approx(np.std([1, 3, 5, 7]))
    assert stats.global_minimum == 1.0
    assert stats.global_maximum == 7.0


def test_constant_spectra_have_zero_standard_deviation():
    stats = SpectralStatistics()
    stats.update(torch.full((4, 3), 2.5))
    assert torch.count_nonzero(stats.std) == 0
    assert stats.global_std == 0.0


def test_collector_without_elements_only_tracks_the_overall_statistics():
    collector = SpectralStatisticsCollector()
    collector.update(torch.ones(3, 2))
    assert collector.overall.num_points == 2
    assert collector.sorted_elements() == []


def test_collector_routes_rows_to_their_element_bucket():
    collector = SpectralStatisticsCollector()
    spectra = torch.tensor([[1.0], [10.0], [3.0], [30.0]])
    collector.update(spectra, torch.tensor([26, 29, 26, 29]))
    collector.update(torch.tensor([[5.0]]), torch.tensor([26]))

    assert collector.sorted_elements() == [26, 29]
    assert collector.element_stats(26).mean.item() == pytest.approx(3.0)
    assert collector.element_stats(29).mean.item() == pytest.approx(20.0)
    assert collector.overall.mean.item() == pytest.approx(49.0 / 5)


def test_collector_reports_unknown_elements():
    collector = SpectralStatisticsCollector()
    collector.update(torch.ones(1, 2), torch.tensor([26]))
    with pytest.raises(ConfigError, match="Element 8 not found"):
        collector.element_stats(8)


def test_collect_spectral_statistics_streams_the_whole_dataset(dataset):
    collector = collect_spectral_statistics(dataset, FakeBatchProcessor())

    expected = torch.stack([dataset[i]["y"] for i in range(N_SAMPLES)]).double()
    torch.testing.assert_close(collector.overall.mean, expected.mean(0))
    assert collector.sorted_elements() == [26, 29]
    assert collector.element_stats(26).mean[0].item() == pytest.approx((0 + 2 + 4) / 3)


def test_collect_spectral_statistics_uses_the_training_subset_when_present(dataset):
    dataset.train_subset = [dataset[0], dataset[1]]
    collector = collect_spectral_statistics(dataset, FakeBatchProcessor())
    assert collector.overall.mean[0].item() == pytest.approx(0.5)


###############################################################################
############################ MODEL AUTO RESOLUTION ############################
###############################################################################


def test_model_auto_fields_are_replaced_by_resolved_values(dataset):
    config = _train_config({"model_type": MODEL_TYPE, "in_size": "auto", "out_size": "AUTO", "hidden": 8}, [])

    resolved = resolve_auto_model_config(config, dataset, encoding=None)  # type: ignore[arg-type]

    assert resolved.section("model").as_dict() == {
        "model_type": MODEL_TYPE,
        "in_size": N_FEATURES,
        "out_size": N_POINTS,
        "hidden": 8,
    }


def test_model_resolution_does_not_mutate_the_input_config(dataset):
    config = _train_config({"model_type": MODEL_TYPE, "in_size": "auto"}, [])
    resolve_auto_model_config(config, dataset, encoding=None)  # type: ignore[arg-type]
    assert config.section("model").get_str("in_size") == "auto"


def test_resolver_sees_the_first_sample_as_representative(dataset):
    resolve_auto_model_config(
        _train_config({"model_type": MODEL_TYPE, "in_size": "auto"}, []), dataset, None  # type: ignore[arg-type]
    )

    inputs, target, seen_dataset = RESOLVER_CALLS[0]
    assert inputs["x"].shape == (1, N_FEATURES)
    assert target.shape == (1, N_POINTS)
    assert seen_dataset is dataset


def test_resolved_dimensions_reflect_the_encoding(dataset):
    class Doubling(SpectraEncoding):
        def __init__(self) -> None:
            super().__init__("doubling")

        def output_size(self, input_size: int) -> int:
            return 2 * input_size

        def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
            return torch.cat([targets, targets], dim=-1)

        def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
            first, second = predictions.chunk(2, dim=-1)
            return (first + second) / 2

        @property
        def signature(self) -> list[Config]:
            return [Config({"encoding_type": "doubling"})]

    config = _train_config({"model_type": MODEL_TYPE, "out_size": "auto"}, [])

    resolved = resolve_auto_model_config(config, dataset, encoding=Doubling())

    assert resolved.section("model").get_int("out_size") == 2 * N_POINTS


def test_model_without_auto_fields_skips_resolution_entirely(dataset):
    config = _train_config({"model_type": MODEL_TYPE, "in_size": 3}, [])
    resolved = resolve_auto_model_config(config, dataset, encoding=None)  # type: ignore[arg-type]
    assert resolved.as_dict() == config.as_dict()
    assert RESOLVER_CALLS == []


def test_auto_fields_for_a_model_without_resolver_are_rejected(dataset):
    BatchProcessorRegistry.register((DATASET_TYPE, "unit_test_no_resolver_model"))(FakeBatchProcessor)
    config = _train_config({"model_type": "unit_test_no_resolver_model", "in_size": "auto"}, [])
    with pytest.raises(ConfigError, match="No automatic model config resolver registered"):
        resolve_auto_model_config(config, dataset, encoding=None)  # type: ignore[arg-type]


def test_resolver_that_misses_a_requested_field_is_reported(dataset):
    config = _train_config({"model_type": UNRESOLVED_MODEL_TYPE, "in_size": "auto"}, [])
    with pytest.raises(ConfigError, match="did not produce model.in_size"):
        resolve_auto_model_config(config, dataset, encoding=None)  # type: ignore[arg-type]


###############################################################################
########################## ENCODING AUTO RESOLUTION ###########################
###############################################################################


def test_encoding_auto_fields_are_resolved_from_dataset_statistics(dataset):
    encodings = [{"encoding_type": ENCODING_TYPE, "num_points": "auto", "scale": "auto", "fixed": 1}]

    resolved = resolve_auto_encoding_config(_train_config({"model_type": MODEL_TYPE}, encodings), dataset)

    assert resolved.get_config_list("encodings")[0].as_dict() == {
        "encoding_type": ENCODING_TYPE,
        "num_points": N_POINTS,
        "scale": float(N_SAMPLES - 1 + N_POINTS - 1),
        "fixed": 1,
    }


def test_encoding_resolution_does_not_mutate_the_input_config(dataset):
    encodings = [{"encoding_type": ENCODING_TYPE, "num_points": "auto"}]
    config = _train_config({"model_type": MODEL_TYPE}, encodings)
    resolve_auto_encoding_config(config, dataset)
    assert config.get_config_list("encodings")[0].get_str("num_points") == "auto"


def test_nested_concat_encodings_are_resolved_recursively(dataset):
    encodings = [
        {
            "encoding_type": "concat",
            "encodings": [
                {"encoding_type": ENCODING_TYPE, "num_points": "auto"},
                {"encoding_type": "identity"},
            ],
        }
    ]

    resolved = resolve_auto_encoding_config(_train_config({"model_type": MODEL_TYPE}, encodings), dataset)

    nested = resolved.as_dict()["encodings"][0]["encodings"]
    assert nested[0]["num_points"] == N_POINTS
    assert nested[1] == {"encoding_type": "identity"}


def test_encodings_without_auto_fields_do_not_touch_the_dataset():
    class ExplodingDataset:
        dataset_type = DATASET_TYPE

        def __getattr__(self, name):
            raise AssertionError(f"dataset.{name} must not be accessed when nothing is 'auto'")

    config = _train_config({"model_type": MODEL_TYPE}, [{"encoding_type": ENCODING_TYPE, "num_points": 5}])
    resolved = resolve_auto_encoding_config(config, ExplodingDataset())  # type: ignore[arg-type]
    assert resolved.as_dict() == config.as_dict()


def test_auto_fields_for_an_encoding_without_resolver_are_rejected(dataset):
    encodings = [{"encoding_type": "unit_test_no_resolver_encoding", "num_points": "auto"}]
    with pytest.raises(ConfigError, match="No automatic encoding config resolver registered"):
        resolve_auto_encoding_config(_train_config({"model_type": MODEL_TYPE}, encodings), dataset)


def test_encoding_resolver_that_misses_a_field_is_reported(dataset):
    encodings = [{"encoding_type": UNRESOLVED_ENCODING_TYPE, "num_points": "auto"}]
    with pytest.raises(ConfigError, match="did not produce num_points"):
        resolve_auto_encoding_config(_train_config({"model_type": MODEL_TYPE}, encodings), dataset)


def test_batch_processor_registry_is_consulted_with_the_dataset_and_model_type(dataset):
    """An unregistered dataset/model pair surfaces as the registry's ``KeyError``."""
    encodings = [{"encoding_type": ENCODING_TYPE, "num_points": "auto"}]
    with pytest.raises(KeyError, match="not found"):
        resolve_auto_encoding_config(_train_config({"model_type": "unit_test_unregistered"}, encodings), dataset)
