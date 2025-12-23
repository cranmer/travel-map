"""Base renderer abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..config import TravelConfig


class BaseRenderer(ABC):
    """Abstract base class for map renderers."""

    def __init__(self, config: TravelConfig):
        """Initialize the renderer with a configuration."""
        self.config = config

    @abstractmethod
    def render_interactive(self) -> str:
        """Render an interactive HTML map.

        Returns:
            HTML string of the interactive map.
        """
        pass

    @abstractmethod
    def render_static(self, width: int = 1200, height: int = 800) -> "Image":
        """Render a static image of the map.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            PIL Image object.
        """
        pass

    def render(self, output_path: Optional[str | Path] = None) -> str | Path:
        """Render the map based on config output type.

        Args:
            output_path: Optional path to save the output.

        Returns:
            HTML string (interactive) or path to saved image (static).
        """
        if self.config.output == "interactive":
            html = self.render_interactive()
            if output_path:
                output_path = Path(output_path)
                if not output_path.suffix:
                    output_path = output_path.with_suffix(".html")
                output_path.write_text(html)
                return output_path
            return html
        else:
            img = self.render_static()
            if output_path:
                output_path = Path(output_path)
                if not output_path.suffix:
                    output_path = output_path.with_suffix(".png")
                img.save(output_path)
                return output_path
            # Return as bytes if no path specified
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    def _get_bounds(self) -> tuple[float, float, float, float]:
        """Get map bounds from locations.

        Returns:
            Tuple of (min_lat, max_lat, min_lon, max_lon).
        """
        lats = [loc.lat for loc in self.config.locations]
        lons = [loc.lon for loc in self.config.locations]

        padding = 2.0  # degrees of padding
        return (
            min(lats) - padding,
            max(lats) + padding,
            min(lons) - padding,
            max(lons) + padding,
        )

    def _get_center(self) -> tuple[float, float]:
        """Get center point of all locations.

        Returns:
            Tuple of (lat, lon) for the center.
        """
        lats = [loc.lat for loc in self.config.locations]
        lons = [loc.lon for loc in self.config.locations]
        return (sum(lats) / len(lats), sum(lons) / len(lons))

    def _format_date(self, location) -> str:
        """Format a location's date for display."""
        if location.date and self.config.show_dates:
            return location.date.strftime(self.config.date_format)
        return ""
