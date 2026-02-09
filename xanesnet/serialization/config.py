"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.filesystem import copy_file

from .defaults import (
    DATASET_DEFAULT,
    DATASET_REQUIRED,
    DATASOURCE_DEFAULT,
    DATASOURCE_REQUIRED,
    INFERENCER_DEFAULTS,
    INFERENCER_REQUIRED,
    MODEL_DEFAULTS,
    MODEL_REQUIRED,
    STRATEGY_DEFAULTS,
    STRATEGY_REQUIRED,
    TRAINER_DEFAULTS,
    TRAINER_REQUIRED,
)

###############################################################################
##################################### RAW #####################################
###############################################################################

# Type alias for raw config data loaded from YAML files.
ConfigRaw = dict[str, Any]


def load_raw_config(file_path: str | Path) -> ConfigRaw:
    """
    Load a YAML config file and return the raw data as a dictionary.
    """
    file_path = Path(file_path)

    if not file_path.exists() or not file_path.is_file():
        raise ConfigError(f"Config file does not exist: {file_path}")

    with open(file_path, "r") as f:
        try:
            data: ConfigRaw = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise ConfigError(f"Config file must contain a top-level dictionary. Found: {type(data)}")
            return data
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config file: {e}") from e


def save_raw_config(config: ConfigRaw, file_path: str | Path) -> Path:
    """
    Save a raw config dictionary to a YAML file.
    """
    file_path = Path(file_path)

    if not str(file_path).endswith((".yaml", ".yml")):
        raise ConfigError(f"Config file must have a .yaml or .yml extension: {file_path}")

    if not file_path.parent.exists() or not file_path.parent.is_dir():
        raise ConfigError(f"Directory for config file does not exist: {file_path.parent}")

    if file_path.exists():
        raise ConfigError(f"Config file already exists: {file_path}")

    with open(file_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

        return file_path


def copy_raw_config(
    file_path: str | Path,
    dst_dir: str | Path,
    new_name: str | None = None,
) -> Path:
    """
    Copy a raw config file to a new location.
    """
    return copy_file(file_path, dst_dir, new_name, allowed_suffixes={".yaml", ".yml"})


def merge_raw_configs(a: ConfigRaw, b: ConfigRaw) -> ConfigRaw:
    """
    Merge two raw configuration dictionaries strictly.
    """
    merged = a.copy()

    for k, v in b.items():
        if k in merged:
            if isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_raw_configs(merged[k], v)
            elif merged[k] != v:
                raise ConfigError(f"Conflict for key '{k}': {merged[k]} != {v}")
        else:
            merged[k] = v

    return merged


###############################################################################
#################################### SAFE #####################################
###############################################################################


class Config:
    """
    A class for accessing configuration values.
    Type-safe.
    """

    def __init__(self, data: ConfigRaw) -> None:
        self._data: ConfigRaw = self._normalize_raw(data)

    # Getters for config values with type checking

    def section(self, section: str) -> "Config":
        value = self._get_typed(section, dict)
        return Config(value)

    def optional_section(self, section: str) -> "Config | None":
        value = self._get_typed(section, dict, optional=True)
        return Config(value) if value is not None else None

    def get_str(self, key: str) -> str:
        return self._get_typed(key, str)

    def get_optional_str(self, key: str) -> str | None:
        return self._get_typed(key, str, optional=True)

    def get_int(self, key: str) -> int:
        return self._get_typed(key, int)

    def get_optional_int(self, key: str) -> int | None:
        return self._get_typed(key, int, optional=True)

    def get_float(self, key: str) -> float:
        return self._get_typed(key, float)

    def get_optional_float(self, key: str) -> float | None:
        return self._get_typed(key, float, optional=True)

    def get_bool(self, key: str) -> bool:
        return self._get_typed(key, bool)

    def get_optional_bool(self, key: str) -> bool | None:
        return self._get_typed(key, bool, optional=True)

    def _get_typed(self, key: str, expected_type: type, optional: bool = False) -> Any:
        value = self._data.get(key, None)
        if value is None:
            if optional:
                return None
            else:
                raise ConfigError(f"Key '{key}' is missing.")
        if not isinstance(value, expected_type):
            raise ConfigError(f"Key '{key}' is not of type {expected_type.__name__}.")
        return value

    def get(self, key: str) -> Any:
        """
        Not type-safe. Not recommended.
        """
        if key not in self._data:
            raise ConfigError(f"Key '{key}' is missing.")
        return self._data.get(key)

    # Other functions

    def as_dict(self) -> ConfigRaw:
        """
        Return a deep copy of the config as a raw dictionary.
        """
        return copy.deepcopy(self._data)

    def as_kwargs(self) -> dict[str, Any]:
        """
        Convert the config into a dictionary suitable for function/class calls.
        - Nested dictionaries are converted into new Config objects.
        - Lists of dictionaries are converted to lists of Config objects recursively.
        - All returned data is a deep copy; modifying it will not affect the original config.
        """

        def convert(value: Any) -> Any:
            if isinstance(value, dict):
                return Config(copy.deepcopy(value))
            elif isinstance(value, list):
                return [convert(v) for v in copy.deepcopy(value)]
            else:
                return value

        return {key: convert(val) for key, val in self._data.items()}

    def save(self, file_path: str | Path) -> Path:
        """
        Save the config to a YAML file.
        """
        return save_raw_config(self._data, file_path)

    def update(self, other: "Config") -> None:
        """
        Update this config with values from another config.
        """
        normalized_other = self._normalize_raw(other.as_dict())
        self._data = merge_raw_configs(self._data, normalized_other)

    def update_with_dict(self, other: ConfigRaw) -> None:
        """
        Update this config with values from a raw config dictionary.
        """
        normalized_other = self._normalize_raw(other)
        self._data = merge_raw_configs(self._data, normalized_other)

    @staticmethod
    def _normalize_raw(value: ConfigRaw) -> ConfigRaw:
        if isinstance(value, Config):
            return Config._normalize_raw(value._data)
        if isinstance(value, dict):
            return {k: Config._normalize_raw(v) for k, v in value.items()}
        if isinstance(value, list):
            return [Config._normalize_raw(v) for v in value]
        return value


###############################################################################
################################# VALIDATION ##################################
###############################################################################

# TODO writing better config validation
# TODO config validation per pipeline command (train, infer, analyze)


def validate_config(config: ConfigRaw) -> Config:
    logging.info("Validating the raw input config file...")

    # Global settings
    _validate_global(config, "seed", None)
    _validate_global(config, "device", "cpu")

    # Sections
    _validate_section(config, "datasource", "datasource_type", DATASOURCE_REQUIRED)
    _validate_section(config, "dataset", "dataset_type", DATASET_REQUIRED)
    _validate_section(config, "model", "model_type", MODEL_REQUIRED)
    _validate_mutually_exclusive(config, "trainer", "inferencer")
    if config.get("trainer", None):
        _validate_section(config, "trainer", "trainer_type", TRAINER_REQUIRED)
    if config.get("inferencer", None):
        _validate_section(config, "inferencer", "inferencer_type", INFERENCER_REQUIRED)
    _validate_section(config, "strategy", "strategy_type", STRATEGY_REQUIRED)

    # Section defaults
    _assign_defaults(config, "datasource", "datasource_type", DATASOURCE_DEFAULT)
    _assign_defaults(config, "dataset", "dataset_type", DATASET_DEFAULT)
    _assign_defaults(config, "model", "model_type", MODEL_DEFAULTS)
    if config.get("trainer", None):
        _assign_defaults(config, "trainer", "trainer_type", TRAINER_DEFAULTS)
    if config.get("inferencer", None):
        _assign_defaults(config, "inferencer", "inferencer_type", INFERENCER_DEFAULTS)
    _assign_defaults(config, "strategy", "strategy_type", STRATEGY_DEFAULTS)

    logging.info("Config: OK")

    return Config(config)


def _validate_global(
    config: ConfigRaw,
    key: str,
    default: Any = None,
    required: bool = False,
) -> None:
    value = config.get(key, None)
    if value is None:
        if required:
            raise ConfigError(f"Missing required key '{key}' in config.")

        else:
            logging.warning(f"Missing optional key '{key}' in config. Using default: '{default}'.")
            config[key] = default


def _validate_section(
    config: ConfigRaw,
    section: str,
    type_key: str,
    required_by_type: dict[str, list[str]],
    required: bool = True,
) -> None:
    section_config = config.get(section, None)
    if section_config is None:
        if required:
            raise ConfigError(f"Missing required section '{section}' in config.")
        else:
            logging.info(f"Optional section '{section}' not present, skipping.")
            return

    assert section_config is not None
    section_type = section_config.get(type_key)
    if not section_type:
        raise ConfigError(f"'{section}.{type_key}' must be specified.")

    if section_type not in required_by_type:
        raise ConfigError(f"Unknown {section} type: {section_type!r}")

    for key_path in required_by_type[section_type]:
        current = section_config
        for i, part in enumerate(key_path.split(".")):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                raise ConfigError(f"Missing required key '{key_path}' in section '{section_type}'.")


def _validate_mutually_exclusive(config: ConfigRaw, section_a: str, section_b: str) -> None:
    section_a_config = config.get(section_a, None)
    section_b_config = config.get(section_b, None)

    if not section_a_config and not section_b_config:
        raise ConfigError(f"Missing either section '{section_a}' or section '{section_b}'.")

    if section_a_config and section_b_config:
        raise ConfigError(f"Section '{section_a}' and section '{section_b}' are mutually exclusive.")


def _assign_defaults(
    config: ConfigRaw,
    section: str,
    type_key: str,
    defaults_by_type: dict[str, ConfigRaw],
) -> None:
    section_config = config[section]
    section_type = section_config[type_key]

    defaults = defaults_by_type.get(section_type, {})

    for key_path, default_value in defaults.items():
        parts = key_path.split(".")
        current = section_config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        last_key = parts[-1]
        # assign default if key is missing or value is None
        if last_key not in current or current[last_key] is None:
            current[last_key] = default_value
            logging.warning(f"Assigning default for '{section}.{key_path}' (type={section_type}): {default_value}.")
