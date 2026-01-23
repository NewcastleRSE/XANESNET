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

import logging
from pathlib import Path
from typing import Any

import yaml

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


def save_dict_as_yaml(config: dict[str, Any], dst_dir: str | Path, name: str) -> Path:
    """
    Save a dictionary as a YAML file in the destination folder.
    """
    if not isinstance(dst_dir, Path):
        dst_dir = Path(dst_dir)

    if not dst_dir.exists() or not dst_dir.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {dst_dir}")

    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"

    dst_file = dst_dir / name

    with open(dst_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return dst_file


def merge_configs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two dictionaries strictly.
    """
    merged = a.copy()

    for k, v in b.items():
        if k in merged:
            if isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_configs(merged[k], v)
            elif merged[k] != v:
                raise ValueError(f"Conflict for key '{k}': {merged[k]} != {v}")
        else:
            merged[k] = v

    return merged


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    logging.info("Validating the input config file...")

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

    return config


def _validate_global(
    config: dict[str, Any],
    key: str,
    default: Any = None,
    required: bool = False,
) -> None:
    value = config.get(key, None)
    if value is None:
        if required:
            logging.critical(f"Missing required key '{key}' in config.")
        else:
            logging.warning(f"Missing optional key '{key}' in config. Using default: '{default}'.")
            config[key] = default


def _validate_section(
    config: dict[str, Any],
    section: str,
    type_key: str,
    required_by_type: dict[str, list[str]],
    required: bool = True,
) -> None:
    section_config = config.get(section, None)
    if section_config is None:
        if required:
            logging.critical(f"Missing required section '{section}' in config.")
        else:
            logging.info(f"Optional section '{section}' not present, skipping.")
            return

    assert section_config is not None
    section_type = section_config.get(type_key)
    if not section_type:
        logging.critical(f"'{section}.{type_key}' must be specified.")

    if section_type not in required_by_type:
        logging.critical(f"Unknown {section} type: {section_type!r}")

    for key_path in required_by_type[section_type]:
        current = section_config
        for i, part in enumerate(key_path.split(".")):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                logging.critical(f"Missing required key '{key_path}' in section '{section_type}'.")


def _validate_mutually_exclusive(config: dict[str, Any], section_a: str, section_b: str) -> None:
    section_a_config = config.get(section_a, None)
    section_b_config = config.get(section_b, None)

    if not section_a_config and not section_b_config:
        logging.critical(f"Missing either section '{section_a}' or section '{section_b}'.")
    if section_a_config and section_b_config:
        logging.critical(f"Sectopm '{section_a}' and section '{section_b}' are mutually exclusive.")


def _assign_defaults(
    config: dict[str, Any],
    section: str,
    type_key: str,
    defaults_by_type: dict[str, dict[str, Any]],
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
