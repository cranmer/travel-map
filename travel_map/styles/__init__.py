"""Map style renderers."""

from .base import BaseRenderer
from .pins import PinsRenderer
from .arcs import ArcsRenderer
from .indiana import IndianaJonesRenderer
from .worldline import WorldlineRenderer

STYLES = {
    "pins": PinsRenderer,
    "arcs": ArcsRenderer,
    "indiana_jones": IndianaJonesRenderer,
    "worldline": WorldlineRenderer,
}

__all__ = ["BaseRenderer", "PinsRenderer", "ArcsRenderer", "IndianaJonesRenderer", "WorldlineRenderer", "STYLES"]
