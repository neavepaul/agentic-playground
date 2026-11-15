"""Who may reach Kip over the network: the names he answers to, and which pages may act.

Kip serves phones on the home network, at the Pi's name (kip-pi, kip-pi.local) or its address
(192.168.x.x). A DNS-rebinding page needs its own domain name in the Host header, so any other name
is refused; an IP address or this machine's own name cannot be an attacker's domain. A page may act
(POST, the event stream) only when it was served by Kip itself: its Origin names the same host.
"""
import ipaddress
import socket
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocketClose


def _host_only(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    if host.startswith("["):  # [::1]:8000
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def own_names(configured: str) -> set[str]:
    machine = socket.gethostname().lower()
    names = {_host_only(name) for name in configured.split(",") if name.strip()}
    return names | {machine, f"{machine}.local", f"{machine}.lan"}


def host_allowed(host: str | None, names: set[str]) -> bool:
    name = _host_only(host or "")
    if name in names:
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def origin_allowed(origin: str | None, host: str | None, origins: set[str]) -> bool:
    """No Origin (not a browser page), one of the configured origins, or Kip's own page."""
    if origin is None:
        return True
    origin = origin.strip().rstrip("/")
    if origin in origins:
        return True
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and bool(host) and parts.netloc.lower() == host.strip().lower()


class OwnHostsOnly:
    """ASGI middleware: refuse HTTP and WebSocket requests addressed to any other host name."""

    def __init__(self, app, names: set[str]) -> None:
        self.app, self.names = app, names

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and not host_allowed(Headers(scope=scope).get("host"), self.names):
            if scope["type"] == "http":
                await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)
            else:
                await WebSocketClose(code=1008)(scope, receive, send)
            return
        await self.app(scope, receive, send)
