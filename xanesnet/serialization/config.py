# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

"""Configuration loading, validation, and type-safe access for XANESNET."""

import copy
from pathlib import Path
from typing import Any

import yaml

from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.filesystem import copy_file

from ._types import ConfigRaw

###############################################################################
##################################### RAW #####################################
###############################################################################


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
        The ``Path`` to the written file.

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

    Raises:
        ConfigError: If the source file is missing, has an unsupported suffix,
            or cannot be copied to ``dst_dir``.
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
            data: Raw configuration dictionary to wrap.
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
            """Convert one raw config value into a constructor-friendly value.

            Args:
                value: Raw config value to convert.

            Returns:
                Nested dictionaries as ``Config`` objects, lists converted
                recursively, and scalar values unchanged.
            """
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
            The ``Path`` to the written file.

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
    def _normalize_raw(value: Any) -> Any:
        """Recursively unwrap any nested ``Config`` objects to plain dicts.

        Args:
            value: Raw config value, possibly containing nested ``Config``
                instances, dictionaries, or lists.

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
