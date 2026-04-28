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

"""Prompt helpers for interactive XANESNET CLI confirmations."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_GLOBAL_AUTO_YES = False
_CONTEXT_AUTO_YES: ContextVar[bool | None] = ContextVar("xanesnet_auto_yes", default=None)


def set_auto_yes(enabled: bool) -> None:
    """Set the global auto-yes mode for confirmation prompts.

    Args:
        enabled: Whether confirmation prompts should be automatically accepted.
    """
    global _GLOBAL_AUTO_YES
    _GLOBAL_AUTO_YES = enabled


def is_auto_yes_enabled() -> bool:
    """Return whether confirmation prompts are auto-accepted in this context.

    Returns:
        ``True`` if either the context-scoped or global auto-yes mode is active.
    """
    context_auto_yes = _CONTEXT_AUTO_YES.get()
    if context_auto_yes is not None:
        return context_auto_yes
    return _GLOBAL_AUTO_YES


@contextmanager
def auto_yes(enabled: bool = True) -> Iterator[None]:
    """Temporarily set auto-yes mode for the current execution context.

    Args:
        enabled: Whether prompts in the current context should be automatically
            accepted.

    Yields:
        ``None`` while the context-scoped auto-yes mode is active.
    """
    token = _CONTEXT_AUTO_YES.set(enabled)
    try:
        yield
    finally:
        _CONTEXT_AUTO_YES.reset(token)


def confirm_yes_no(message: str, default_yes: bool = True) -> bool:
    """Ask a yes/no confirmation question.

    When auto-yes mode is active, the prompt is skipped and accepted with an
    info log entry. Otherwise, the user is prompted interactively and an empty
    response selects the configured default.

    Args:
        message: Confirmation question to display, without the ``[Y/n]`` or
            ``[y/N]`` suffix.
        default_yes: Whether an empty response should be treated as yes.

    Returns:
        ``True`` when the answer is yes, otherwise ``False``.
    """
    if is_auto_yes_enabled():
        logging.info(f"Auto-yes active; auto-accepted confirmation prompt: {message}")
        return True

    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    response = input(f"{message}{suffix}").strip().lower()
    if response == "":
        return default_yes
    return response in ("y", "yes")
