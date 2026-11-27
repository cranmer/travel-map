"""Arcs style renderer - pins with curved connection lines."""

import folium
from folium.plugins import AntPath, MarkerCluster
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
import io
import html
import numpy as np

from .base import BaseRenderer
from ..config import TravelConfig


class ArcsRenderer(BaseRenderer):
    """Render maps with pin markers and curved arcs connecting locations."""

    def __init__(self, config: TravelConfig):
        super().__init__(config)
        self.marker_color = "#3498db"  # Blue markers
        self.arc_color = "#e74c3c"  # Red arcs

    def _interpolate_great_circle(
        self, lat1: float, lon1: float, lat2: float, lon2: float, num_points: int = 50
    ) -> list[tuple[float, float]]:
        """Interpolate points along a great circle arc."""
        # Convert to radians
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

        # Calculate great circle distance
        d = np.arccos(
            np.clip(
                np.sin(lat1_r) * np.sin(lat2_r) +
                np.cos(lat1_r) * np.cos(lat2_r) * np.cos(lon2_r - lon1_r),
                -1, 1
            )
        )

        if d < 1e-10:
            return [(lat1, lon1), (lat2, lon2)]

        points = []
        for i in range(num_points + 1):
            f = i / num_points
            A = np.sin((1 - f) * d) / np.sin(d)
            B = np.sin(f * d) / np.sin(d)

            x = A * np.cos(lat1_r) * np.cos(lon1_r) + B * np.cos(lat2_r) * np.cos(lon2_r)
            y = A * np.cos(lat1_r) * np.sin(lon1_r) + B * np.cos(lat2_r) * np.sin(lon2_r)
            z = A * np.sin(lat1_r) + B * np.sin(lat2_r)

            lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
            lon = np.degrees(np.arctan2(y, x))
            points.append((lat, lon))

        return points

    def _should_go_westward(self, from_lon: float, to_lon: float) -> bool:
        """Determine if arc should go westward (across Pacific) or eastward.

        From home (Madison ~-89°), go westward to destinations east of Nepal (~85°E).
        This creates the classic "around the world" look for Asia/Pacific destinations.
        """
        # Nepal is around 85°E - destinations east of this go westward
        nepal_longitude = 85.0

        # Only apply westward routing if starting from western hemisphere
        # and destination is in eastern Asia/Pacific
        if from_lon < 0 and to_lon > nepal_longitude:
            return True
        return False

    def _unwrap_longitudes(
        self, points: list[tuple[float, float]], force_westward: bool = False
    ) -> list[tuple[float, float]]:
        """Unwrap longitude values to make them continuous across the dateline.

        Instead of splitting arcs at ±180°, this shifts longitude values to be
        continuous (e.g., going from 170° to 190° instead of 170° to -170°).
        Leaflet handles extended longitude values correctly.

        If force_westward is True, ensures the arc goes west (negative direction).
        """
        if len(points) < 2:
            return points

        unwrapped = [points[0]]
        offset = 0.0

        for i in range(1, len(points)):
            prev_lon = unwrapped[i - 1][1]
            curr_lat = points[i][0]
            curr_lon = points[i][1] + offset

            # Check for dateline crossing (jump > 180°)
            delta = curr_lon - prev_lon
            if delta > 180:
                offset -= 360
                curr_lon -= 360
            elif delta < -180:
                offset += 360
                curr_lon += 360

            unwrapped.append((curr_lat, curr_lon))

        # If we should go westward but ended up going east, shift everything
        if force_westward and len(unwrapped) > 1:
            start_lon = unwrapped[0][1]
            end_lon = unwrapped[-1][1]
            # If we went eastward (end > start) but should go west, shift by -360
            if end_lon > start_lon:
                unwrapped = [(lat, lon - 360) for lat, lon in unwrapped]

        return unwrapped

    # Leaflet.markercluster icon: a pill listing the stop numbers it contains,
    # with a native hover title listing each visit.
    CLUSTER_ICON_JS = """
    function(cluster) {
        var markers = cluster.getAllChildMarkers();
        markers.sort(function(a, b) { return a.options.number - b.options.number; });
        var text = markers.map(function(m) { return m.options.number; }).join(',');
        var title = markers.map(function(m) { return m.options.visit_text; }).join('\\n')
            .replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
        var w = Math.max(22, 7 * text.length + 12);
        return L.divIcon({
            html: '<div title="' + title + '" style="box-sizing: border-box; width: ' + w + 'px; height: 22px; '
                + 'border-radius: 11px; border: 2px solid %(color)s; background: rgba(52, 152, 219, 0.8); '
                + 'color: white; font: bold 10px/18px sans-serif; text-align: center; white-space: nowrap;">'
                + text + '</div>',
            className: 'empty',
            iconSize: [w, 22],
            iconAnchor: [w / 2, 11]
        });
    }
    """

    def render_interactive(self) -> str:
        """Render an interactive Folium map with markers and animated arcs."""
        routes = self.config.get_routes()

        # First pass: calculate longitude offsets for all locations
        lon_offset_by_name = {}
        for from_loc, to_loc in routes:
            from_offset = lon_offset_by_name.get(from_loc.name, 0)
            effective_from_lon = from_loc.lon + from_offset
            force_westward = self._should_go_westward(effective_from_lon, to_loc.lon)

            arc_points = self._interpolate_great_circle(
                from_loc.lat, from_loc.lon,
                to_loc.lat, to_loc.lon,
            )
            arc_points = self._unwrap_longitudes(arc_points, force_westward=force_westward)

            if abs(from_offset) > 1:
                arc_points = [(lat, lon + from_offset) for lat, lon in arc_points]

            end_lon = arc_points[-1][1]
            to_offset = end_lon - to_loc.lon
            if abs(to_offset) > 1:
                lon_offset_by_name[to_loc.name] = to_offset

        # Calculate bounds using effective (shifted) longitudes
        lats = [loc.lat for loc in self.config.locations]
        effective_lons = []
        for loc in self.config.locations:
            effective_lon = loc.lon
            if loc.name in lon_offset_by_name:
                effective_lon = loc.lon + lon_offset_by_name[loc.name]
            effective_lons.append(effective_lon)

        # Add home location for bounds calculation
        home = self.config.get_home()
        lats.append(home.lat)
        effective_lons.append(home.lon)

        padding = 5
        bounds = [
            min(lats) - padding,
            max(lats) + padding,
            min(effective_lons) - padding,
            max(effective_lons) + padding,
        ]

        # Center on home location (Madison)
        center = [home.lat, home.lon]
        m = folium.Map(
            location=center,
            zoom_start=4,
            # Esri Light Gray Canvas: similar look to CartoDB Positron, no API key
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
            max_zoom=16,
        )

        m.fit_bounds([[bounds[0], bounds[2]], [bounds[1], bounds[3]]])

        # Draw arcs first (so markers appear on top)
        for from_loc, to_loc in routes:
            # Get the offset for from_loc (for multi-leg trips crossing dateline)
            from_offset = lon_offset_by_name.get(from_loc.name, 0)

            # Determine if this arc should go westward (across Pacific)
            effective_from_lon = from_loc.lon + from_offset
            force_westward = self._should_go_westward(effective_from_lon, to_loc.lon)

            arc_points = self._interpolate_great_circle(
                from_loc.lat, from_loc.lon,
                to_loc.lat, to_loc.lon,
            )

            # Unwrap longitudes to make continuous across dateline
            arc_points = self._unwrap_longitudes(arc_points, force_westward=force_westward)

            # If we have a from_offset, apply it to all points
            # (the interpolation normalizes longitudes, so we need to shift them back)
            if abs(from_offset) > 1:
                arc_points = [(lat, lon + from_offset) for lat, lon in arc_points]

            # Calculate the total offset for the destination
            end_lon = arc_points[-1][1]
            to_offset = end_lon - to_loc.lon
            if abs(to_offset) > 1:  # Only track significant offsets
                lon_offset_by_name[to_loc.name] = to_offset

            # Convert to [lat, lon] format for folium
            arc_coords = [[p[0], p[1]] for p in arc_points]

            # Animated ant path
            AntPath(
                locations=arc_coords,
                color=self.arc_color,
                weight=3,
                opacity=0.8,
                delay=1000,
                dash_array=[10, 20],
                pulse_color="#ffffff",
            ).add_to(m)

        # Numbered markers go in a cluster group: markers that overlap on screen
        # merge into one pill (e.g. "7,15") and split apart as you zoom in.
        cluster = MarkerCluster(
            icon_create_function=self.CLUSTER_ICON_JS % {"color": self.marker_color},
            options={
                "maxClusterRadius": 25,  # px; roughly where two 22px markers start to overlap
                "showCoverageOnHover": False,
                "spiderfyOnMaxZoom": True,
                "zoomToBoundsOnClick": True,
            },
        ).add_to(m)

        # Add markers at their effective positions (shifted if crossing dateline)
        for i, loc in enumerate(self.config.locations):
            number = str(i + 1)
            date_str = self._format_date(loc)
            name = html.escape(loc.name)

            visit_text = f"{number}. {loc.name}" + (f" ({date_str})" if date_str else "")
            popup_content = f"<b>{name}</b>"
            if loc.label:
                popup_content += f"<br>{html.escape(loc.label)}"
            if date_str:
                popup_content += f"<br><i>{date_str}</i>"

            effective_lon = loc.lon
            if loc.name in lon_offset_by_name:
                effective_lon = loc.lon + lon_offset_by_name[loc.name]

            # A single interactive DivIcon carries the number, tooltip and popup,
            # so hovering anywhere on the circle works.
            label_html = (
                f'<div style="box-sizing: border-box; width: 22px; height: 22px; '
                f'border-radius: 11px; border: 2px solid {self.marker_color}; '
                f'background: rgba(52, 152, 219, 0.8); color: white; font: bold 10px/18px sans-serif; '
                f'text-align: center; white-space: nowrap;">{number}</div>'
            )
            folium.Marker(
                location=[loc.lat, effective_lon],
                icon=folium.DivIcon(html=label_html, icon_size=(22, 22), icon_anchor=(11, 11)),
                tooltip=folium.Tooltip(html.escape(visit_text)),
                popup=folium.Popup(popup_content, max_width=250),
                number=i + 1,
                visit_text=visit_text,
            ).add_to(cluster)

        # Add home marker if routes_from_home is enabled
        if self.config.routes_from_home:
            home = self.config.get_home()
            folium.Marker(
                location=[home.lat, home.lon],
                popup=folium.Popup(f"<b>{home.name}</b><br>Home", max_width=200),
                tooltip=f"{home.name} (Home)",
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(m)

        # Add title if specified
        if self.config.title:
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
        """Render a static map with markers and curved arcs."""
        fig_width = width / 100
        fig_height = height / 100

        bounds = self._get_bounds()
        center_lon = (bounds[2] + bounds[3]) / 2

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=center_lon))

        ax.set_extent([bounds[2], bounds[3], bounds[0], bounds[1]], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5")
        ax.add_feature(cfeature.OCEAN, facecolor="#e8f4f8")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")

        # Draw arcs
        routes = self.config.get_routes()
        for from_loc, to_loc in routes:
            arc_points = self._interpolate_great_circle(
                from_loc.lat, from_loc.lon,
                to_loc.lat, to_loc.lon,
            )
            lats = [p[0] for p in arc_points]
            lons = [p[1] for p in arc_points]

            ax.plot(
                lons, lats,
                color=self.arc_color,
                linewidth=2.5,
                transform=ccrs.PlateCarree(),
                zorder=4,
                alpha=0.8,
            )

        # Plot markers
        for i, loc in enumerate(self.config.locations):
            ax.plot(
                loc.lon, loc.lat,
                marker="o",
                color=self.marker_color,
                markersize=15,
                transform=ccrs.PlateCarree(),
                zorder=5,
                markeredgecolor="white",
                markeredgewidth=2,
            )

            # Add number inside marker
            ax.text(
                loc.lon, loc.lat,
                str(i + 1),
                transform=ccrs.PlateCarree(),
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                zorder=6,
            )

            # Add label
            label = loc.name
            date_str = self._format_date(loc)
            if date_str:
                label += f"\n{date_str}"

            ax.text(
                loc.lon, loc.lat + 1.0,
                label,
                transform=ccrs.PlateCarree(),
                fontsize=9,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                zorder=6,
            )

        ax.set_title(self.config.title, fontsize=16, fontweight="bold", pad=10)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf).copy()
