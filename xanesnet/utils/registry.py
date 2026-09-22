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

"""Generic keyed registries for XANESNET extension points."""

from collections.abc import Callable, Hashable
from typing import Any, Generic, TypeVar, cast

from typing_extensions import TypeVar as TypeVarWithDefault

_T = TypeVar("_T")
_K = TypeVarWithDefault("_K", bound=Hashable, default=str)
_RegisteredT = TypeVar("_RegisteredT")


class Registry(Generic[_T, _K]):
    """Store registered objects under normalized keys.

    ``T`` is the exact object type stored in the registry. ``K`` is the key
    type and defaults to ``str`` for ordinary name-based registries.

    Args:
        item_name: Human-readable item name used in error messages.
        normalize_key: Optional function that canonicalizes keys before lookup
            and registration.
        format_key: Optional function that formats normalized keys for error
            messages.
    """

    def __init__(
        self,
        item_name: str,
        *,
        normalize_key: Callable[[_K], _K] | None = None,
        format_key: Callable[[_K], str] | None = None,
    ) -> None:
        """Initialize an empty registry."""
        self._item_name = item_name
        self._normalize_key = normalize_key
        self._format_key = format_key
        self._registry: dict[_K, _T] = {}

    def register(self, key: _K) -> Callable[[_RegisteredT], _RegisteredT]:
        """Register an object under ``key``.

        Args:
            key: Registry key. The key is normalized before storage when this
                registry was configured with ``normalize_key``.

        Returns:
            Decorator that registers and returns the object unchanged.

        Raises:
            KeyError: If ``key`` is already registered.
        """
        normalized_key = self._normalize(key)

        def decorator(value: _RegisteredT) -> _RegisteredT:
            """Register and return the decorated object unchanged."""
            if normalized_key in self._registry:
                raise KeyError(f"{self._item_name} {self._format(normalized_key)} already registered")
            self._registry[normalized_key] = cast(_T, value)
            return value

        return decorator

    def get(self, key: _K) -> _T:
        """Return the object registered under ``key`` unchanged.

        Args:
            key: Registry key. The key is normalized before lookup when this
                registry was configured with ``normalize_key``.

        Returns:
            The registered object exactly as it was stored.

        Raises:
            KeyError: If no object is registered under ``key``.
        """
        normalized_key = self._normalize(key)
        if normalized_key not in self._registry:
            raise KeyError(f"{self._item_name} {self._format(normalized_key)} not found in registry")
        return self._registry[normalized_key]

    def create(self, registry_key: _K, **kwargs: Any) -> Any:
        """Instantiate or call the registered object for ``registry_key``.

        Args:
            registry_key: Registry key to resolve.
            **kwargs: Keyword arguments forwarded to the registered callable.

        Returns:
            The result of ``get(registry_key)(**kwargs)``.
        """
        factory = cast(Callable[..., Any], self.get(registry_key))
        return factory(**kwargs)

    def list(self) -> list[_K]:
        """Return all registered keys in insertion order.

        Returns:
            Registry keys after normalization.
        """
        return list(self._registry.keys())

    def _normalize(self, key: _K) -> _K:
        """Return ``key`` in the registry's canonical form."""
        if self._normalize_key is None:
            return key
        return self._normalize_key(key)

    def _format(self, key: _K) -> str:
        """Return a user-facing representation of ``key``."""
        if self._format_key is not None:
            return self._format_key(key)
        return f"'{key}'"
