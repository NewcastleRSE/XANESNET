# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Configuration loading, validation, and type-safe access for XANESNET."""

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
    """Load a YAML configuration file.

    Args:
        file_path: Path to the ``.yaml`` or ``.yml`` file to load.

    Returns:
        The parsed configuration as a ``ConfigRaw`` dictionary.

    Raises:
        ConfigError: If the file does not exist, is not a file, is not a valid
            YAML document, or does not contain a top-level mapping.
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
    """Write a raw configuration dictionary to a YAML file.

    Args:
        config: The configuration data to serialize.
        file_path: Destination path. Must end with ``.yaml`` or ``.yml``.

    Returns:
        The resolved ``Path`` to the written file.

    Raises:
        ConfigError: If the extension is wrong, the parent directory does not
            exist, or the file already exists.
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
    """Copy a YAML configuration file to a destination directory.

    Args:
        file_path: Source file path.
        dst_dir: Directory to copy the file into.
        new_name: Optional new filename (without directory). If ``None``, the
            original filename is used.

    Returns:
        The ``Path`` to the copied file.
    """
    return copy_file(file_path, dst_dir, new_name, allowed_suffixes={".yaml", ".yml"})


def merge_raw_configs(a: ConfigRaw, b: ConfigRaw) -> ConfigRaw:
    """Recursively merge two raw configuration dictionaries.

    Args:
        a: Base configuration dictionary.
        b: Overlay configuration dictionary.

    Returns:
        A new ``ConfigRaw`` with all keys from both dicts.  Nested dicts are
        merged recursively.

    Raises:
        ConfigError: If the same key exists in both dicts with conflicting
            non-dict values.
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
    """Validated, type-safe view over a raw YAML configuration.

    Args:
        data: A ``ConfigRaw`` dictionary (usually the result of
            ``load_raw_config``).
    """

    def __init__(self, data: ConfigRaw) -> None:
        """Store a normalized raw configuration mapping.

        Args:
            data: Raw configuration dictionary, possibly containing nested
                ``Config`` instances that will be unwrapped.
        """
        self._data: ConfigRaw = self._normalize_raw(data)

    # Getters for config values with type checking

    def section(self, section: str) -> "Config":
        """Return a sub-section as a ``Config``.

        Args:
            section: Key of the sub-dictionary to wrap.

        Returns:
            A ``Config`` wrapping the section's dictionary.

        Raises:
            ConfigError: If the key is missing or is not a dictionary.
        """
        value = self._get_typed(section, dict)
        return Config(value)

    def optional_section(self, section: str) -> "Config | None":
        """Return a sub-section as a ``Config``, or ``None`` if absent.

        Args:
            section: Key of the sub-dictionary to wrap.

        Returns:
            A ``Config`` wrapping the section's dictionary, or ``None``.

        Raises:
            ConfigError: If the key is present but is not a dictionary.
        """
        value = self._get_typed(section, dict, optional=True)
        return Config(value) if value is not None else None

    def get_str(self, key: str) -> str:
        """Return the value at ``key`` as a ``str``.

        Args:
            key: Config key to look up.

        Returns:
            The string value.

        Raises:
            ConfigError: If the key is missing or the value is not a ``str``.
        """
        return self._get_typed(key, str)

    def get_optional_str(self, key: str) -> str | None:
        """Return the value at ``key`` as a ``str``, or ``None`` if absent.

        Args:
            key: Config key to look up.

        Returns:
            The string value, or ``None``.

        Raises:
            ConfigError: If the key is present but the value is not a ``str``.
        """
        return self._get_typed(key, str, optional=True)

    def get_int(self, key: str) -> int:
        """Return the value at ``key`` as an ``int``.

        Args:
            key: Config key to look up.

        Returns:
            The integer value.

        Raises:
            ConfigError: If the key is missing or the value is not an ``int``.
        """
        return self._get_typed(key, int)

    def get_optional_int(self, key: str) -> int | None:
        """Return the value at ``key`` as an ``int``, or ``None`` if absent.

        Args:
            key: Config key to look up.

        Returns:
            The integer value, or ``None``.

        Raises:
            ConfigError: If the key is present but the value is not an ``int``.
        """
        return self._get_typed(key, int, optional=True)

    def get_float(self, key: str) -> float:
        """Return the value at ``key`` as a ``float``.

        Args:
            key: Config key to look up.

        Returns:
            The float value.

        Raises:
            ConfigError: If the key is missing or the value is not a ``float``.
        """
        return self._get_typed(key, float)

    def get_optional_float(self, key: str) -> float | None:
        """Return the value at ``key`` as a ``float``, or ``None`` if absent.

        Args:
            key: Config key to look up.

        Returns:
            The float value, or ``None``.

        Raises:
            ConfigError: If the key is present but the value is not a ``float``.
        """
        return self._get_typed(key, float, optional=True)

    def get_bool(self, key: str) -> bool:
        """Return the value at ``key`` as a ``bool``.

        Args:
            key: Config key to look up.

        Returns:
            The boolean value.

        Raises:
            ConfigError: If the key is missing or the value is not a ``bool``.
        """
        return self._get_typed(key, bool)

    def get_optional_bool(self, key: str) -> bool | None:
        """Return the value at ``key`` as a ``bool``, or ``None`` if absent.

        Args:
            key: Config key to look up.

        Returns:
            The boolean value, or ``None``.

        Raises:
            ConfigError: If the key is present but the value is not a ``bool``.
        """
        return self._get_typed(key, bool, optional=True)

    def get_config_list(self, key: str) -> list["Config"]:
        """Return a list of ``Config`` objects from a list of dicts.

        Args:
            key: Config key that maps to a list of dicts.

        Returns:
            List of ``Config`` objects, one per dict entry.

        Raises:
            ConfigError: If the key is missing, the value is not a list, or
                any element is not a dictionary.
        """
        value = self._get_typed(key, list)
        if not all(isinstance(v, dict) for v in value):
            raise ConfigError(f"Key '{key}' must be a list of dictionaries.")
        return [Config(v) for v in value]

    def _get_typed(self, key: str, expected_type: type, optional: bool = False) -> Any:
        """Retrieve a value from the config data with optional type enforcement.

        Args:
            key: Config key to look up.
            expected_type: The ``type`` the value must be an instance of.
            optional: If ``True``, return ``None`` when the key is absent.

        Returns:
            The typed value, or ``None`` when ``optional=True`` and the key is missing.

        Raises:
            ConfigError: If the key is missing (and ``optional=False``) or the
                value is not an instance of ``expected_type``.
        """
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
        """Return the raw value for ``key``.

        Note:
            Not type-safe. Prefer the typed ``get_*`` accessors where possible.

        Args:
            key: Config key to look up.

        Returns:
            The raw value associated with ``key``.

        Raises:
            ConfigError: If the key is missing.
        """
        if key not in self._data:
            raise ConfigError(f"Key '{key}' is missing.")
        return self._data.get(key)

    # Other functions

    def as_dict(self) -> ConfigRaw:
        """Return a deep copy of the underlying raw config dictionary.

        Returns:
            A ``ConfigRaw`` deep-copy of the internal data.
        """
        return copy.deepcopy(self._data)

    def as_kwargs(self) -> dict[str, Any]:
        """Convert the config into keyword arguments suitable for a class or function call.

        Nested dicts are wrapped in ``Config``; lists of dicts become lists of
        ``Config``; all other values are returned as deep copies.

        Returns:
            Dict mapping config keys to converted values.
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
        """Save this config to a YAML file.

        Args:
            file_path: Destination path. Must end with ``.yaml`` or ``.yml``.

        Returns:
            The resolved ``Path`` to the written file.

        Raises:
            ConfigError: Propagated from ``save_raw_config``.
        """
        return save_raw_config(self._data, file_path)

    def update(self, other: "Config") -> None:
        """Merge another ``Config`` into this one.

        Args:
            other: The ``Config`` whose values are overlaid onto this one.

        Raises:
            ConfigError: If any key has conflicting non-dict values.
        """
        normalized_other = self._normalize_raw(other.as_dict())
        self._data = merge_raw_configs(self._data, normalized_other)

    def update_with_dict(self, other: ConfigRaw) -> None:
        """Merge a raw dictionary into this config.

        Args:
            other: Raw config dictionary whose values are overlaid onto this one.

        Raises:
            ConfigError: If any key has conflicting non-dict values.
        """
        normalized_other = self._normalize_raw(other)
        self._data = merge_raw_configs(self._data, normalized_other)

    @staticmethod
    def _normalize_raw(value: ConfigRaw) -> ConfigRaw:
        """Recursively unwrap any nested ``Config`` objects to plain dicts.

        Args:
            value: A raw config value, possibly containing nested ``Config``
                instances.

        Returns:
            The same structure with all ``Config`` instances replaced by their
            underlying raw dictionaries.
        """
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


def validate_config_train(config: ConfigRaw) -> Config:
    """Validate a config dict for a training run.

    Args:
        config: Raw configuration dictionary.

    Returns:
        A validated ``Config`` object.
    """
    # TODO writing better train validation
    return validate_config(config)


def validate_config_infer(config: ConfigRaw) -> Config:
    """Validate a config dict for an inference run.

    Args:
        config: Raw configuration dictionary.

    Returns:
        A validated ``Config`` object.
    """
    # TODO writing better infer validation
    return validate_config(config)


def validate_config_analyze(config: ConfigRaw) -> Config:
    """Validate a config dict for an analysis run.

    Args:
        config: Raw configuration dictionary.

    Returns:
        A ``Config`` object (minimal validation applied).
    """
    # TODO writing better analyze validation
    return Config(config)


def validate_config(config: ConfigRaw) -> Config:
    """Run full config validation and fill in defaults.

    Args:
        config: Raw configuration dictionary to validate and mutate in-place.

    Returns:
        A ``Config`` object wrapping the validated and defaulted data.

    Raises:
        ConfigError: If any required key is missing, a type key is absent, or
            mutually exclusive sections coexist.
    """
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
    """Validate or assign a top-level global config key.

    Args:
        config: The raw config dictionary (mutated in-place when setting defaults).
        key: The top-level key to check.
        default: Default value assigned when the key is absent and not required.
        required: If ``True``, raise ``ConfigError`` when the key is absent.

    Raises:
        ConfigError: If ``required=True`` and the key is missing.
    """
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
    """Validate a config section against a registry of required keys.

    Args:
        config: The raw config dictionary.
        section: The section name (top-level key whose value is a sub-dict).
        type_key: Key inside the section that identifies the component type.
        required_by_type: Mapping from component type name to the list of
            required dot-separated key paths.
        required: If ``True`` (default), raise ``ConfigError`` when the section
            is absent.

    Raises:
        ConfigError: If the section is missing (and required), the type key is
            absent, the type is unknown, or any required key path is absent.
    """
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
    """Ensure exactly one of two mutually exclusive sections is present.

    Args:
        config: The raw config dictionary.
        section_a: Name of the first section.
        section_b: Name of the second section.

    Raises:
        ConfigError: If neither or both sections are present.
    """
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
    """Fill in default values for a config section based on its component type.

    Missing keys (or keys set to ``None``) are populated from the defaults
    registry.  The operation is applied in-place on ``config``.

    Args:
        config: The raw config dictionary (mutated in-place).
        section: The section name to process.
        type_key: Key inside the section that identifies the component type.
        defaults_by_type: Mapping from component type name to a flat dict of
            dot-separated key paths and their default values.
    """
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
