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

"""Unit tests for the Lightning worker entry point (``train_worker``) and its launch-context validation."""

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from xanesnet import core_lightning
from xanesnet.serialization.config import Config
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.lightning import LIGHTNING_CONTEXT_ENV, LIGHTNING_LAUNCH_ENVIRONMENT

from . import fakes


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "1")


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the optional Lightning import; the pipeline itself runs against the registered fakes."""
    monkeypatch.setattr(core_lightning, "_register_lightning_trainer", lambda: None)
    monkeypatch.setattr(core_lightning, "reset_peak_memory_stats", lambda device: fakes.EVENTS.append(("reset", device)))


def make_config() -> Config:
    return Config(
        {
            "seed": 3,
            "device": "cpu",
            "datasource": {"datasource_type": fakes.DATASOURCE_TYPE},
            "dataset": {"dataset_type": fakes.DATASET_TYPE, "root": "dataset_root"},
            "strategy": {"strategy_type": fakes.STRATEGY_TYPE},
            "trainer": {"trainer_type": "basic"},
            "model": {"model_type": "fake"},
        }
    )


def write_context(directory: Path, **overrides) -> Path:
    context = {
        "config": make_config().as_dict(),
        "dry_run": False,
        "model_path": None,
        "restart_training_state": False,
    }
    context.update(overrides)
    path = directory / "lightning_context.json"
    path.write_text(json.dumps(context), encoding="utf-8")
    return path


###############################################################################
############################## CONTEXT VALIDATION #############################
###############################################################################


def test_missing_context_file_is_reported(tmp_path, worker_env):
    with pytest.raises(ConfigError, match="Cannot load Lightning context"):
        core_lightning.train_worker(tmp_path / "missing.json")


def test_malformed_context_file_is_reported(tmp_path, worker_env):
    path = tmp_path / "lightning_context.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigError, match="Cannot load Lightning context"):
        core_lightning.train_worker(path)


@pytest.mark.parametrize("payload", ["[]", '"text"', "3", "null"])
def test_non_object_context_is_rejected(tmp_path, worker_env, payload):
    path = tmp_path / "lightning_context.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain an object"):
        core_lightning.train_worker(path)


@pytest.mark.parametrize("rank", [None, "", "first", "1.5"])
def test_worker_requires_an_integer_local_rank(tmp_path, monkeypatch, rank):
    if rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", rank)

    with pytest.raises(ConfigError, match="integer LOCAL_RANK"):
        core_lightning.train_worker(write_context(tmp_path))


@pytest.mark.parametrize("rank", ["0", "-1"])
def test_non_worker_processes_are_rejected(tmp_path, monkeypatch, rank):
    monkeypatch.setenv("LOCAL_RANK", rank)

    with pytest.raises(ConfigError, match="does not identify this process as a worker"):
        core_lightning.train_worker(write_context(tmp_path))


def test_context_problems_are_reported_before_anything_is_built(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")

    with pytest.raises(ConfigError):
        core_lightning.train_worker(write_context(tmp_path))

    assert fakes.EVENTS == []
    assert not (tmp_path / "workers").exists()


def test_missing_lightning_dependency_gives_an_actionable_message(tmp_path, worker_env, monkeypatch):
    def not_installed(name: str):
        raise ModuleNotFoundError("No module named 'lightning'", name="lightning")

    monkeypatch.setattr(core_lightning.importlib, "import_module", not_installed)

    with pytest.raises(ConfigError, match=r"xanesnet\[lightning\]"):
        core_lightning.train_worker(write_context(tmp_path))


def test_unrelated_import_errors_are_not_masked(tmp_path, worker_env, monkeypatch):
    def broken(name: str):
        raise ModuleNotFoundError("No module named 'numpyy'", name="numpyy")

    monkeypatch.setattr(core_lightning.importlib, "import_module", broken)

    with pytest.raises(ModuleNotFoundError, match="numpyy"):
        core_lightning.train_worker(write_context(tmp_path))


###############################################################################
################################## WORKER RUN #################################
###############################################################################


def test_worker_builds_the_prepared_run_and_trains(tmp_path, worker_env, stubbed):
    core_lightning.train_worker(write_context(tmp_path))

    names = fakes.event_names()
    assert names.index("strategy") < names.index("setup_models") < names.index("setup_trainers")
    assert names[-1] == "run_training"
    assert "checkpointer_save" not in names, "only the initial process exports results"


def test_worker_reuses_the_prepared_dataset_and_splits(tmp_path, worker_env, stubbed):
    core_lightning.train_worker(write_context(tmp_path))

    dataset_kwargs = fakes.payload("dataset")
    assert dataset_kwargs["skip_prepare"] is True
    assert dataset_kwargs["split_indexfile"] == str(tmp_path / "split_indices.json")


def test_worker_logs_to_a_per_rank_directory(tmp_path, worker_env, stubbed):
    core_lightning.train_worker(write_context(tmp_path))

    assert (tmp_path / "workers" / "rank_1").is_dir()


def test_worker_does_not_enable_tensorboard(tmp_path, worker_env, stubbed):
    core_lightning.train_worker(write_context(tmp_path))

    assert fakes.payload("strategy")["tensorboard_dir"] is None


@pytest.mark.parametrize("dry_run", [True, False])
def test_dry_run_flag_controls_memory_statistics_reset(tmp_path, worker_env, stubbed, dry_run):
    core_lightning.train_worker(write_context(tmp_path, dry_run=dry_run))

    assert ("reset" in fakes.event_names()) is dry_run


def test_worker_restores_checkpoint_weights_and_training_state(tmp_path, worker_env, stubbed, monkeypatch):
    from xanesnet.serialization.checkpoints import Checkpoint

    checkpoint = Checkpoint(
        signature=Config(fakes.SIGNATURE),
        model_states=[{"weight": 1}],
        training_state={
            "optimizer_states": [{"lr": 0.1}],
            "epoch_scheduler_states": [{"epoch": 4}],
            "warmup_scheduler_states": [None],
        },
    )
    monkeypatch.setattr(core_lightning.Checkpoint, "load", classmethod(lambda cls, path: checkpoint))

    core_lightning.train_worker(write_context(tmp_path, model_path="model.pth"))

    assert fakes.payload("set_state_dicts") == [{"weight": 1}]
    assert fakes.payload("load_optimizer") == {"lr": 0.1}
    assert fakes.payload("load_epoch_scheduler") == {"epoch": 4}


def test_restart_training_state_skips_optimizer_and_scheduler_restore(tmp_path, worker_env, stubbed, monkeypatch):
    from xanesnet.serialization.checkpoints import Checkpoint

    checkpoint = Checkpoint(
        signature=Config(fakes.SIGNATURE),
        model_states=[{"weight": 1}],
        training_state={
            "optimizer_states": [{"lr": 0.1}],
            "epoch_scheduler_states": [{"epoch": 4}],
            "warmup_scheduler_states": [None],
        },
    )
    monkeypatch.setattr(core_lightning.Checkpoint, "load", classmethod(lambda cls, path: checkpoint))

    core_lightning.train_worker(write_context(tmp_path, model_path="model.pth", restart_training_state=True))

    assert "set_state_dicts" in fakes.event_names()
    assert not any(name.startswith("load_") for name in fakes.event_names())


def test_environment_is_not_modified_by_the_worker(tmp_path, worker_env, stubbed):
    before = dict(os.environ)

    core_lightning.train_worker(write_context(tmp_path))

    assert dict(os.environ) == before



###############################################################################
################################ INITIAL PROCESS ##############################
###############################################################################


class TestTrainInitial:
    """``train_initial`` prepares the run, publishes the worker context while training, and exports results."""

    @pytest.fixture(autouse=True)
    def _prepare(self, tmp_path, monkeypatch, stubbed):
        (tmp_path / "models").mkdir()
        monkeypatch.setattr(core_lightning, "resolve_auto_model_config", lambda config, dataset, encoding: config)
        monkeypatch.setattr(core_lightning, "_save_results", lambda *params: fakes.EVENTS.append(("save_results", None)))
        for name in (*LIGHTNING_LAUNCH_ENVIRONMENT, LIGHTNING_CONTEXT_ENV):
            monkeypatch.delenv(name, raising=False)

    @pytest.fixture
    def args(self) -> Namespace:
        return Namespace(tensorboard=False, dry_run=False, restart_training_state=False)

    @pytest.fixture
    def published(self, monkeypatch) -> dict:
        """What a Lightning worker would see while ``run_training`` executes."""
        seen: dict = {}
        original = fakes.FakeStrategy.run_training

        def run_training(strategy):
            path = os.environ.get(LIGHTNING_CONTEXT_ENV)
            seen["path"] = path
            seen["context"] = json.loads(Path(path).read_text()) if path else None
            return original(strategy)

        monkeypatch.setattr(fakes.FakeStrategy, "run_training", run_training)
        return seen

    def test_context_is_published_only_while_training(self, tmp_path, args, published):
        core_lightning.train_initial(make_config(), args, tmp_path, None)

        assert published["path"] == str(tmp_path / "lightning_context.json")
        assert LIGHTNING_CONTEXT_ENV not in os.environ

    def test_context_carries_everything_a_worker_needs(self, tmp_path, args, published):
        args.dry_run = True
        args.restart_training_state = True

        core_lightning.train_initial(make_config(), args, tmp_path, None)

        context = published["context"]
        assert context["dry_run"] is True
        assert context["restart_training_state"] is True
        assert context["model_path"] is None
        assert context["config"]["seed"] == 3
        assert context["config"]["dataset"]["root"] == str((Path.cwd() / "dataset_root").resolve())

    def test_published_context_is_accepted_by_a_worker(self, tmp_path, args, published, monkeypatch):
        core_lightning.train_initial(make_config(), args, tmp_path, None)
        fakes.EVENTS.clear()

        monkeypatch.setenv("LOCAL_RANK", "1")
        core_lightning.train_worker(tmp_path / "lightning_context.json")

        assert fakes.event_names()[-1] == "run_training"

    def test_launch_environment_is_restored_after_training(self, tmp_path, args, monkeypatch):
        monkeypatch.setenv("MASTER_PORT", "1234")
        monkeypatch.setattr(
            fakes.FakeStrategy, "run_training", lambda strategy: os.environ.update(MASTER_PORT="9999", RANK="0") or []
        )

        core_lightning.train_initial(make_config(), args, tmp_path, None)

        assert os.environ["MASTER_PORT"] == "1234"
        assert "RANK" not in os.environ

    def test_environment_is_restored_when_training_fails(self, tmp_path, args, monkeypatch):
        def failing(strategy):
            raise RuntimeError("training crashed")

        monkeypatch.setattr(fakes.FakeStrategy, "run_training", failing)

        with pytest.raises(RuntimeError, match="training crashed"):
            core_lightning.train_initial(make_config(), args, tmp_path, None)

        assert LIGHTNING_CONTEXT_ENV not in os.environ
        assert not list(tmp_path.glob("*.tmp")), "no half-written context file is left behind"

    def test_initial_process_prepares_and_exports(self, tmp_path, args):
        core_lightning.train_initial(make_config(), args, tmp_path, None)

        names = fakes.event_names()
        assert names.index("setup_trainers") < names.index("run_training") < names.index("save_results")
        assert (tmp_path / "resolved_train_config.yaml").is_file()
        assert (tmp_path / "split_indices.json").is_file()
        assert (tmp_path / "models" / "signature.yaml").is_file()

    def test_dataset_root_is_made_absolute_for_workers(self, tmp_path, args):
        core_lightning.train_initial(make_config(), args, tmp_path, None)

        assert Path(fakes.payload("dataset")["root"]).is_absolute()

    def test_checkpoint_state_is_restored_in_the_initial_process(self, tmp_path, args):
        from xanesnet.serialization.checkpoints import Checkpoint

        checkpoint = Checkpoint(
            signature=Config(fakes.SIGNATURE),
            model_states=[{"weight": 1}],
            training_state={
                "optimizer_states": [{"lr": 0.1}],
                "epoch_scheduler_states": [{"epoch": 4}],
                "warmup_scheduler_states": [None],
            },
        )

        core_lightning.train_initial(make_config(), args, tmp_path, checkpoint)

        assert fakes.payload("set_state_dicts") == [{"weight": 1}]
        assert fakes.payload("load_optimizer") == {"lr": 0.1}

    def test_missing_lightning_dependency_stops_before_building_anything(self, tmp_path, args, monkeypatch):
        def missing():
            raise ConfigError("Lightning training requires `python -m pip install 'xanesnet[lightning]'`.")

        monkeypatch.setattr(core_lightning, "_register_lightning_trainer", missing)

        with pytest.raises(ConfigError, match="Lightning"):
            core_lightning.train_initial(make_config(), args, tmp_path, None)

        assert fakes.EVENTS == []
