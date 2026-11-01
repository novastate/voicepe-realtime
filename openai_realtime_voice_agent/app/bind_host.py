"""Loopback defaults for the listeners behind Comms."""
import ipaddress
import logging
import os

logger = logging.getLogger(__name__)
_warned_variables: set[str] = set()


def is_loopback_host(host: str | None) -> bool:
    """Recognize literal loopback IPs, including IPv4-mapped IPv6."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback


def resolve_bind_host(env_name: str, default: str = "127.0.0.1") -> str:
    """Keep explicit binds, warning once per process per exposed variable."""
    host = os.environ.get(env_name, default)
    if env_name in os.environ and not is_loopback_host(host) and env_name not in _warned_variables:
        _warned_variables.add(env_name)
        logger.warning("%s=%s: Comms is no longer the only door", env_name, host)
    return host
