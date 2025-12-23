"""Pins style renderer - clean modern map with location markers."""

import folium
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import io
import numpy as np

from .base import BaseRenderer
from ..config import TravelConfig


class PinsRenderer(BaseRenderer):
    """Render maps with simple pin markers at each location."""

    def __init__(self, config: TravelConfig):
        super().__init__(config)
        self.marker_color = "#e74c3c"  # Red markers

    def render_interactive(self) -> str:
        """Render an interactive Folium map with markers."""
        center = self._get_center()
        m = folium.Map(
            location=center,
            zoom_start=4,
            tiles="OpenStreetMap",
        )

        # Fit bounds to show all locations
        bounds = self._get_bounds()
        m.fit_bounds([[bounds[0], bounds[2]], [bounds[1], bounds[3]]])

        # Add markers for each location
        for loc in self.config.locations:
            # Build popup content
            popup_content = f"<b>{loc.name}</b>"
            if loc.label:
                popup_content += f"<br>{loc.label}"
            date_str = self._format_date(loc)
            if date_str:
                popup_content += f"<br><i>{date_str}</i>"

            # Build tooltip (shown on hover)
            tooltip = loc.name
            if date_str:
                tooltip += f" ({date_str})"

            folium.Marker(
                location=[loc.lat, loc.lon],
                popup=folium.Popup(popup_content, max_width=200),
                tooltip=tooltip,
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m)

        # Add home marker if routes_from_home is enabled
        if self.config.routes_from_home:
            home = self.config.get_home()
            folium.Marker(
                location=[home.lat, home.lon],
                popup=folium.Popup(f"<b>{home.name}</b><br>Home", max_width=200),
                tooltip=f"{home.name} (Home)",
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(m)

        # Add title
        title_html = f'''
        <div style="position: fixed;
                    top: 10px; left: 50px;
                    z-index: 1000;
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                    font-family: Arial, sans-serif;
                    font-size: 16px;
                    font-weight: bold;">
            {self.config.title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        return m._repr_html_()

    def render_static(self, width: int = 1200, height: int = 800) -> Image.Image:
        """Render a static matplotlib/cartopy map with markers."""
        # Calculate figure size in inches (assuming 100 dpi)
        fig_width = width / 100
        fig_height = height / 100

        bounds = self._get_bounds()
        center_lon = (bounds[2] + bounds[3]) / 2

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=center_lon))

        # Set extent
        ax.set_extent([bounds[2], bounds[3], bounds[0], bounds[1]], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#d4e6f1")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.add_feature(cfeature.LAKES, facecolor="#d4e6f1", alpha=0.5)

        # Plot markers
        for loc in self.config.locations:
            ax.plot(
                loc.lon, loc.lat,
                marker="o",
                color=self.marker_color,
                markersize=12,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

            # Add label
            label = loc.name
            date_str = self._format_date(loc)
            if date_str:
                label += f"\n{date_str}"

            ax.text(
                loc.lon, loc.lat + 0.5,
                label,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                zorder=6,
            )

        # Add title
        ax.set_title(self.config.title, fontsize=16, fontweight="bold", pad=10)

        # Convert to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).copy()
