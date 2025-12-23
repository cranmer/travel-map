"""Map style renderers."""

from .base import BaseRenderer
from .pins import PinsRenderer
from .arcs import ArcsRenderer
from .indiana import IndianaJonesRenderer

STYLES = {
    "pins": PinsRenderer,
    "arcs": ArcsRenderer,
    "indiana_jones": IndianaJonesRenderer,
}

__all__ = ["BaseRenderer", "PinsRenderer", "ArcsRenderer", "IndianaJonesRenderer", "STYLES"]
