"""Outbound-network guard for installations that must work offline.

With ``AUTOBANNER_OFFLINE=1`` the server refuses every outbound connection that is
not to the loopback interface (or a Unix socket). The product needs none: planning,
text shaping, quality checks, learning and exports run locally. The guard turns that
promise into an error the operator would see in the logs, instead of a silent
dependency on the network.
"""

from __future__ import annotations

import ipaddress
import logging
import socket

logger = logging.getLogger("autobanner.api.offline")

_LOCAL_HOSTS = {"localhost", "localhost.localdomain", "ip6-localhost", ""}
_installed = False
_originals: dict[str, object] = {}


class OfflineError(OSError):
    """Raised for an outbound connection while AUTOBANNER_OFFLINE is set."""


def _is_local(address: object) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    if not isinstance(address, tuple) or not address:
        return False
    host = str(address[0]).strip("[]").split("%")[0]
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def install_offline_guard() -> None:
    """Patch socket connects so anything non-local raises ``OfflineError``."""
    global _installed  # noqa: PLW0603 - process-wide guard
    if _installed:
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address, *args, **kwargs):
        if not _is_local(address):
            logger.error("outbound connection refused (AUTOBANNER_OFFLINE): %r", address)
            raise OfflineError(f"outbound network disabled (AUTOBANNER_OFFLINE): {address!r}")
        return original_connect(self, address, *args, **kwargs)

    def guarded_connect_ex(self: socket.socket, address, *args, **kwargs):
        if not _is_local(address):
            logger.error("outbound connection refused (AUTOBANNER_OFFLINE): %r", address)
            raise OfflineError(f"outbound network disabled (AUTOBANNER_OFFLINE): {address!r}")
        return original_connect_ex(self, address, *args, **kwargs)

    _originals["connect"] = original_connect
    _originals["connect_ex"] = original_connect_ex
    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    _installed = True
    logger.warning("AUTOBANNER_OFFLINE: outbound network connections are refused")


def uninstall_offline_guard() -> None:
    """Restore the original socket methods (tests)."""
    global _installed  # noqa: PLW0603
    if not _installed:
        return
    socket.socket.connect = _originals["connect"]  # type: ignore[method-assign]
    socket.socket.connect_ex = _originals["connect_ex"]  # type: ignore[method-assign]
    _installed = False


def offline_guard_installed() -> bool:
    return _installed
