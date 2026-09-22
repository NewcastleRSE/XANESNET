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
#   ...

"""JSON Schema loading, default materialization, and config validation."""

import copy
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable
from typing import Any
from urllib.parse import unquote

import yaml
from jsonschema import exceptions as jsonschema_exceptions
from jsonschema.validators import Draft202012Validator

from xanesnet.utils.exceptions import ConfigError

from ._types import ConfigMode, ConfigRaw
from .runtime_contracts import validate_runtime_contracts

SchemaRaw = dict[str, Any]

SCHEMA_ENTRY_POINTS: dict[ConfigMode, str] = {
    "train": "train.schema.yaml",
    "infer": "infer_runtime.schema.yaml",
    "analyze": "analyze.schema.yaml",
}


def load_config_schema(mode: ConfigMode) -> SchemaRaw:
    """Load a resolved JSON Schema for a config validation mode.

    The returned schema is deep-copied from an internal cache so callers may
    inspect or mutate it without affecting future validation calls.

    Args:
        mode: Config mode to load. Supported values are ``"train"``,
            ``"infer"``, and ``"analyze"``.

    Returns:
        Resolved Draft 2020-12 JSON Schema for ``mode``.

    Raises:
        ConfigError: If the schema file is missing, invalid YAML, or not a
            valid JSON Schema document.
    """
    return copy.deepcopy(_load_config_schema_cached(mode))


def validate_config_schema(config: ConfigRaw, mode: ConfigMode) -> ConfigRaw:
    """Validate and default a raw configuration dictionary.

    Validation uses the packaged schemas in ``xanesnet/schemas``. Schema
    ``default`` annotations are materialized before validation, including the
    selected branch of ``oneOf`` and ``anyOf`` schemas. Registry-backed runtime
    contracts that are cumbersome to maintain in JSON Schema are checked after
    schema validation. On success, ``config`` is updated in place to preserve
    the historic validation contract.

    Args:
        config: Raw configuration dictionary to validate and update.
        mode: Config mode to validate against.

    Returns:
        Validated and defaulted raw configuration dictionary.

    Raises:
        ConfigError: If the config does not satisfy the schema for ``mode``.
    """
    schema = load_config_schema(mode)
    normalized = copy.deepcopy(config)
    _materialize_defaults(normalized, schema)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(normalized), key=_validation_error_sort_key)
    if errors:
        raise ConfigError(_format_validation_error(mode, errors[0]))

    validate_runtime_contracts(normalized, mode)

    config.clear()
    config.update(copy.deepcopy(normalized))
    return normalized


@lru_cache(maxsize=None)
def _load_config_schema_cached(mode: ConfigMode) -> SchemaRaw:
    """Load and cache a resolved JSON Schema for ``mode``.

    Args:
        mode: Config validation mode to load.

    Returns:
        Resolved Draft 2020-12 schema for ``mode``.

    Raises:
        ConfigError: If ``mode`` is unknown, the entry-point schema is missing,
            or the resolved schema is not a valid Draft 2020-12 schema.
    """
    try:
        entry_point = SCHEMA_ENTRY_POINTS[mode]
    except KeyError as exc:
        valid_modes = ", ".join(sorted(SCHEMA_ENTRY_POINTS))
        raise ConfigError(f"Unknown config validation mode: {mode!r}. Expected one of: {valid_modes}.") from exc

    raw_schemas = _load_raw_schemas()
    if entry_point not in raw_schemas:
        raise ConfigError(f"Missing config schema entry point: {entry_point}")

    schema = _resolve_schema_node(raw_schemas[entry_point], entry_point, raw_schemas)
    try:
        Draft202012Validator.check_schema(schema)
    except jsonschema_exceptions.SchemaError as exc:
        raise ConfigError(f"Invalid JSON Schema entry point '{entry_point}': {exc.message}") from exc

    return schema


@lru_cache(maxsize=1)
def _load_raw_schemas() -> dict[str, SchemaRaw]:
    """Load all packaged YAML schema files by relative POSIX path.

    Returns:
        Mapping from schema path relative to ``xanesnet/schemas`` to parsed
        schema dictionaries.

    Raises:
        ConfigError: If the packaged schema directory is missing, a schema file
            cannot be parsed as YAML, or a schema file does not contain a
            top-level dictionary.
    """
    schemas_root = resources.files("xanesnet") / "schemas"
    if not schemas_root.is_dir():
        raise ConfigError("Packaged schema directory 'xanesnet/schemas' was not found.")

    raw_schemas: dict[str, SchemaRaw] = {}
    for relative_path, schema_resource in _iter_schema_files(schemas_root):
        try:
            parsed = yaml.safe_load(schema_resource.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Error parsing schema file '{relative_path}': {exc}") from exc
        if not isinstance(parsed, dict):
            raise ConfigError(f"Schema file '{relative_path}' must contain a top-level dictionary.")
        raw_schemas[relative_path] = parsed

    return raw_schemas


def _iter_schema_files(directory: Traversable, prefix: str = "") -> list[tuple[str, Traversable]]:
    """Return schema resources under ``directory`` with POSIX relative paths.

    Args:
        directory: Package resource directory to scan recursively.
        prefix: Relative path prefix accumulated during recursion.

    Returns:
        List of ``(relative_path, resource)`` tuples for ``*.schema.yaml``
        files.
    """
    schema_files: list[tuple[str, Traversable]] = []
    for child in sorted(directory.iterdir(), key=lambda resource: resource.name):
        child_path = f"{prefix}/{child.name}" if prefix else child.name
        if child.is_dir():
            schema_files.extend(_iter_schema_files(child, child_path))
        elif child.name.endswith(".schema.yaml"):
            schema_files.append((child_path, child))
    return schema_files


def _resolve_schema_node(
    value: Any,
    current_path: str,
    raw_schemas: dict[str, SchemaRaw],
    seen_refs: frozenset[str] = frozenset(),
) -> Any:
    """Resolve local ``$ref`` values in a schema node.

    Args:
        value: Schema node to resolve.
        current_path: Relative schema path containing ``value``.
        raw_schemas: Mapping of all loaded raw schemas by relative path.
        seen_refs: Reference keys already visited in the current resolution
            chain.

    Returns:
        Deep-copied schema node with local references resolved and schema
        bookkeeping keys removed.

    Raises:
        ConfigError: If a local reference cannot be resolved or a circular
            reference is detected.
    """
    if isinstance(value, list):
        return [_resolve_schema_node(item, current_path, raw_schemas, seen_refs) for item in value]

    if not isinstance(value, dict):
        return copy.deepcopy(value)

    ref = value.get("$ref")
    if isinstance(ref, str):
        file_part, _, pointer_part = ref.partition("#")
        target_path = _normalize_relative_path(current_path, file_part) if file_part else current_path
        ref_key = f"{target_path}#{pointer_part}"
        if ref_key in seen_refs:
            raise ConfigError(f"Circular schema reference detected while resolving {ref!r} from {current_path}.")

        try:
            target_schema = raw_schemas[target_path]
        except KeyError as exc:
            raise ConfigError(f"Cannot resolve schema reference {ref!r} from {current_path}.") from exc

        target = _read_json_pointer(target_schema, pointer_part, target_path)
        resolved_target = _resolve_schema_node(target, target_path, raw_schemas, seen_refs | {ref_key})
        siblings = {key: subvalue for key, subvalue in value.items() if key != "$ref"}
        if not siblings:
            return resolved_target

        resolved_siblings = _resolve_schema_node(siblings, current_path, raw_schemas, seen_refs)
        return _merge_schema_objects(resolved_target, resolved_siblings)

    return {
        key: _resolve_schema_node(subvalue, current_path, raw_schemas, seen_refs)
        for key, subvalue in value.items()
        if key not in {"$schema", "$id"}
    }


def _normalize_relative_path(from_path: str, target_path: str) -> str:
    """Resolve a relative schema path from another schema file path.

    Args:
        from_path: Relative path of the schema containing the reference.
        target_path: Relative or absolute path fragment from a ``$ref``.

    Returns:
        Normalized POSIX-style path relative to ``xanesnet/schemas``.
    """
    if target_path.startswith("/"):
        return target_path.removeprefix("/")

    parts = from_path.split("/")[:-1]
    for part in target_path.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def _read_json_pointer(schema: Any, pointer: str, schema_path: str) -> Any:
    """Read a JSON Pointer fragment from ``schema``.

    Args:
        schema: Schema document or node to traverse.
        pointer: JSON Pointer fragment without the leading ``#``.
        schema_path: Relative schema path used in error messages.

    Returns:
        Referenced schema node.

    Raises:
        ConfigError: If ``pointer`` cannot be resolved in ``schema``.
    """
    pointer = unquote(pointer)
    if pointer in {"", "/"}:
        return schema

    current = schema
    for raw_part in pointer.removeprefix("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        try:
            current = current[int(part)] if isinstance(current, list) else current[part]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ConfigError(f"Cannot resolve JSON pointer '#{pointer}' in schema '{schema_path}'.") from exc

    return current


def _merge_schema_objects(base: Any, override: Any) -> Any:
    """Merge a resolved ``$ref`` target with sibling schema keywords.

    Args:
        base: Resolved target of a ``$ref``.
        override: Sibling schema keywords from the referencing node.

    Returns:
        Merged schema dictionary when both inputs are dictionaries; otherwise a
        deep copy of ``override``.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return copy.deepcopy(override)

    merged = copy.deepcopy(base)
    merged.update(copy.deepcopy(override))
    return merged


def _materialize_defaults(instance: Any, schema: Any) -> Any:
    """Recursively apply JSON Schema ``default`` annotations to ``instance``.

    Args:
        instance: Config value being defaulted.
        schema: Schema node that describes ``instance``.

    Returns:
        ``instance`` with defaults applied. Explicit ``None`` values are
        preserved.
    """
    if not isinstance(schema, dict):
        return instance

    if instance is None:
        return instance

    if isinstance(instance, dict):
        _materialize_object_defaults(instance, schema)
    elif isinstance(instance, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                instance[index] = _materialize_defaults(item, item_schema)

    return instance


def _materialize_object_defaults(instance: dict[str, Any], schema: SchemaRaw) -> None:
    """Apply defaults from object, ``allOf``, ``oneOf``, and ``anyOf`` schemas.

    Args:
        instance: Mutable config object to update in place.
        schema: Object schema whose defaults should be applied.
    """
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if not isinstance(property_schema, dict):
                continue
            if key not in instance:
                if "default" in property_schema:
                    instance[key] = copy.deepcopy(property_schema["default"])
                else:
                    continue
            instance[key] = _materialize_defaults(instance[key], property_schema)

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for subschema in all_of:
            _materialize_defaults(instance, subschema)

    for keyword in ("oneOf", "anyOf"):
        options = schema.get(keyword)
        if isinstance(options, list):
            _materialize_matching_branch_defaults(instance, options)


def _materialize_matching_branch_defaults(instance: dict[str, Any], options: list[Any]) -> None:
    """Apply defaults from the first branch that validates after defaulting.

    If no branch matches, ``instance`` is left unchanged. The subsequent full
    schema validation pass reports the branch mismatch with normal JSON Schema
    error context.

    Args:
        instance: Mutable config object to update in place when a branch
            matches.
        options: ``oneOf`` or ``anyOf`` branch schemas.
    """
    for option in options:
        if not isinstance(option, dict):
            continue
        candidate = copy.deepcopy(instance)
        _materialize_defaults(candidate, option)
        if Draft202012Validator(option).is_valid(candidate):
            instance.clear()
            instance.update(candidate)
            return


def _validation_error_sort_key(error: jsonschema_exceptions.ValidationError) -> tuple[int, list[str], str]:
    """Sort validation errors for deterministic, useful reporting.

    Args:
        error: JSON Schema validation error.

    Returns:
        Tuple that prioritizes shallower config paths, then path text, then the
        validation message.
    """
    return (len(error.absolute_path), [str(part) for part in error.absolute_path], error.message)


def _format_validation_error(mode: ConfigMode, error: jsonschema_exceptions.ValidationError) -> str:
    """Format a JSON Schema validation error as a user-facing message.

    Args:
        mode: Config validation mode that produced ``error``.
        error: Top-level validation error to format.

    Returns:
        Single-line message suitable for ``ConfigError``.
    """
    path = _format_instance_path(error.absolute_path)
    message = f"Invalid {mode} config at {path}: {_shorten(error.message)}"

    context_errors = _collect_context_errors(error)
    if context_errors:
        details = []
        seen_details: set[str] = set()
        for context_error in context_errors:
            detail = f"{_format_instance_path(context_error.absolute_path)}: {_shorten(context_error.message)}"
            if detail not in seen_details:
                details.append(detail)
                seen_details.add(detail)
            if len(details) == 6:
                break
        if details:
            message += " Details: " + "; ".join(details)

    return message


def _collect_context_errors(
    error: jsonschema_exceptions.ValidationError,
) -> list[jsonschema_exceptions.ValidationError]:
    """Collect leaf context errors from a nested validation error.

    Args:
        error: Validation error whose nested ``context`` entries should be
            flattened.

    Returns:
        Sorted list of leaf validation errors.
    """
    if not error.context:
        return []

    collected: list[jsonschema_exceptions.ValidationError] = []
    for context_error in error.context:
        nested_context = _collect_context_errors(context_error)
        if nested_context:
            collected.extend(nested_context)
        else:
            collected.append(context_error)

    return sorted(collected, key=_validation_error_sort_key)


def _format_instance_path(path: Any) -> str:
    """Format a JSON Schema error path as a dotted config path.

    Args:
        path: Iterable path from ``ValidationError.absolute_path``.

    Returns:
        Human-readable path beginning with ``"config"``.
    """
    formatted = "config"
    for part in path:
        if isinstance(part, int):
            formatted += f"[{part}]"
        else:
            formatted += f".{part}"
    return formatted


def _shorten(value: str, limit: int = 220) -> str:
    """Return a compact one-line validation message.

    Args:
        value: Message text to normalize.
        limit: Maximum output length.

    Returns:
        Whitespace-normalized message, truncated with an ellipsis if needed.
    """
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."
