"""Worldline style renderer - 3D spacetime visualization with time as vertical axis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import io
from datetime import datetime, timedelta

from .base import BaseRenderer
from ..config import TravelConfig


class WorldlineRenderer(BaseRenderer):
    """Render maps as 3D worldline visualization with time on vertical axis."""

    def __init__(self, config: TravelConfig, map_opacity: float = 0.5, base_map_opacity: float = 0.9):
        super().__init__(config)
        self.path_color = "#e74c3c"  # Red for worldlines
        self.marker_color = "#3498db"  # Blue for location markers
        self.home_color = "#2ecc71"  # Green for home
        self.home_line_color = "#27ae60"  # Darker green for home vertical line
        self.map_opacity = map_opacity  # Opacity for trip-level map surfaces
        self.base_map_opacity = base_map_opacity  # Opacity for bottom base map
        self._map_image_cache = {}  # Cache rendered map images

    def _render_map_image(
        self,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        resolution: int = 200
    ) -> np.ndarray:
        """Render a map image using cartopy Natural Earth features and return as RGB array.

        Returns array of shape (resolution, resolution, 3) with RGB values 0-255.
        """
        cache_key = (lon_range, lat_range, resolution)
        if cache_key in self._map_image_cache:
            return self._map_image_cache[cache_key]

        # Calculate appropriate zoom level based on extent
        lon_span = lon_range[1] - lon_range[0]
        zoom_level = max(1, min(8, int(8 - np.log2(lon_span / 45))))

        # Create figure - size determines output resolution
        dpi = 100
        fig_size = resolution / dpi
        fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set extent
        ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

        # Use cartopy's built-in Natural Earth features for reliable, detailed maps
        # Colors chosen to match CartoDB Positron style
        ax.add_feature(cfeature.OCEAN, facecolor='#aad3df', zorder=0)
        ax.add_feature(cfeature.LAND, facecolor='#f5f3e5', zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#999999', zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#cccccc', linestyle='-', zorder=2)
        ax.add_feature(cfeature.LAKES, facecolor='#aad3df', zorder=1)
        ax.add_feature(cfeature.RIVERS, edgecolor='#aad3df', linewidth=0.2, zorder=1)

        # Add more detail with states/provinces for higher zoom
        if zoom_level >= 4:
            states = cfeature.NaturalEarthFeature(
                'cultural', 'admin_1_states_provinces_lines', '50m',
                edgecolor='#dddddd', facecolor='none', linewidth=0.15
            )
            ax.add_feature(states, zorder=2)

        # Remove axes and margins
        ax.set_frame_on(False)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Render to image buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                    transparent=False, facecolor='white')
        buf.seek(0)
        plt.close(fig)

        # Load and resize to exact resolution
        img = Image.open(buf)
        img = img.convert('RGB')
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img_array = np.array(img)

        self._map_image_cache[cache_key] = img_array
        return img_array

    def _normalize_time(self, locations: list) -> tuple[list[float], datetime, datetime]:
        """Convert dates to normalized time values (0-1 range).

        Returns: (normalized_times, min_date, max_date)
        """
        dated_locs = [loc for loc in locations if loc.date]
        if not dated_locs:
            return [0.5] * len(locations), None, None

        dates = [loc.date for loc in dated_locs]
        min_date = min(dates)
        max_date = max(dates)

        # Add padding so points aren't at exact top/bottom
        date_range = (max_date - min_date).days or 1

        normalized = []
        for loc in locations:
            if loc.date:
                days_from_start = (loc.date - min_date).days
                normalized.append(days_from_start / date_range)
            else:
                normalized.append(0.5)

        return normalized, min_date, max_date

    def _create_worldline_trace(
        self,
        from_loc,
        to_loc,
        from_time: float,
        to_time: float,
        num_points: int = 50
    ) -> tuple[list, list, list]:
        """Create a 3D path between two locations through time.

        Returns (lons, lats, times) for the path.
        """
        # Interpolate great circle path
        arc_points = self._interpolate_great_circle(
            from_loc.lat, from_loc.lon,
            to_loc.lat, to_loc.lon,
            num_points
        )

        lats = [p[0] for p in arc_points]
        lons = [p[1] for p in arc_points]

        # Linear interpolation for time
        times = np.linspace(from_time, to_time, num_points + 1).tolist()

        return lons, lats, times

    def _interpolate_great_circle(
        self, lat1: float, lon1: float, lat2: float, lon2: float, num_points: int = 50
    ) -> list[tuple[float, float]]:
        """Interpolate points along a great circle arc."""
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

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

    def _create_map_surface(
        self,
        time_level: float,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        resolution: int = 200,
        opacity: float = None
    ) -> go.Surface:
        """Create a semi-transparent surface with real map imagery at a time level."""
        if opacity is None:
            opacity = self.map_opacity

        # Render actual map using cartopy with web tiles
        map_img = self._render_map_image(lon_range, lat_range, resolution)

        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Create a flat surface at the given time level
        z_grid = np.full_like(lon_grid, time_level)

        # Flip vertically because image y-axis is inverted relative to lat
        map_img = np.flipud(map_img)

        # Pack RGB into a single value for surfacecolor (0-1 range)
        # We'll create a colorscale that maps these packed values back to RGB
        r = map_img[:, :, 0].astype(float) / 255.0
        g = map_img[:, :, 1].astype(float) / 255.0
        b = map_img[:, :, 2].astype(float) / 255.0

        # Use grayscale-ish encoding that preserves color variation
        # Weight by luminance but keep some color info
        colors = 0.299 * r + 0.587 * g + 0.114 * b

        # Build a colorscale from sampled image colors
        # Sample colors at regular intervals to build the colorscale
        num_stops = 64
        colorscale = []
        for i in range(num_stops):
            val = i / (num_stops - 1)
            # Find pixels with this approximate value
            mask = np.abs(colors - val) < (1.0 / num_stops)
            if np.any(mask):
                # Average the RGB values for pixels in this range
                avg_r = int(np.mean(map_img[:, :, 0][mask]))
                avg_g = int(np.mean(map_img[:, :, 1][mask]))
                avg_b = int(np.mean(map_img[:, :, 2][mask]))
            else:
                # Interpolate gray
                avg_r = avg_g = avg_b = int(val * 255)
            colorscale.append([val, f'rgba({avg_r}, {avg_g}, {avg_b}, {opacity})'])

        return go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=z_grid,
            surfacecolor=colors,
            colorscale=colorscale,
            showscale=False,
            hoverinfo='skip',
            opacity=opacity,
        )

    def _is_land(self, lat: float, lon: float) -> bool:
        """Very rough approximation of whether a point is on land."""
        # This is a simplified heuristic - could be replaced with actual coastline data
        # North America
        if -170 < lon < -50 and 25 < lat < 75:
            if lon > -130 or lat > 45:
                return True
        # South America
        if -85 < lon < -30 and -60 < lat < 15:
            return True
        # Europe
        if -10 < lon < 60 and 35 < lat < 75:
            return True
        # Africa
        if -20 < lon < 55 and -40 < lat < 40:
            return True
        # Asia
        if 60 < lon < 150 and 10 < lat < 75:
            return True
        # Australia
        if 110 < lon < 160 and -45 < lat < -10:
            return True
        return False

    def render_interactive(self) -> str:
        """Render an interactive 3D worldline visualization."""
        locations = self.config.locations
        normalized_times, min_date, max_date = self._normalize_time(locations)

        # Create location name to time mapping
        loc_times = {loc.name: t for loc, t in zip(locations, normalized_times)}

        # Get bounds
        lats = [loc.lat for loc in locations]
        lons = [loc.lon for loc in locations]

        if self.config.routes_from_home:
            home = self.config.get_home()
            lats.append(home.lat)
            lons.append(home.lon)

        padding = 10
        lon_range = (min(lons) - padding, max(lons) + padding)
        lat_range = (min(lats) - padding, max(lats) + padding)

        fig = go.Figure()

        # Add base map at bottom of cube (slightly below 0 to be visible)
        base_surface = self._create_map_surface(
            -0.02, lon_range, lat_range, resolution=300, opacity=self.base_map_opacity
        )
        fig.add_trace(base_surface)

        # Get unique trip time levels from the actual location dates
        trip_times = sorted(set(normalized_times))
        # Filter out times very close to 0 (already have base map there)
        trip_times = [t for t in trip_times if t > 0.05]

        # Add semi-transparent map surfaces at each trip's time level
        for t in trip_times:
            surface = self._create_map_surface(
                t, lon_range, lat_range, resolution=200, opacity=self.map_opacity
            )
            fig.add_trace(surface)

        # Add vertical line at home location (time axis)
        home = self.config.get_home()
        fig.add_trace(go.Scatter3d(
            x=[home.lon, home.lon],
            y=[home.lat, home.lat],
            z=[-0.02, 1.02],
            mode='lines',
            line=dict(color=self.home_line_color, width=8, dash='dot'),
            name='Home Timeline',
            hoverinfo='name',
        ))

        # Add worldline paths (outbound and return)
        routes = self.config.get_routes()
        for from_loc, to_loc in routes:
            from_time = loc_times.get(from_loc.name, 0.5)
            to_time = loc_times.get(to_loc.name, 0.5)

            # For home, use the destination's time minus a small offset
            if from_loc.name == home.name:
                from_time = max(0, to_time - 0.02)

            # Outbound arc
            lons_path, lats_path, times_path = self._create_worldline_trace(
                from_loc, to_loc, from_time, to_time
            )

            fig.add_trace(go.Scatter3d(
                x=lons_path,
                y=lats_path,
                z=times_path,
                mode='lines',
                line=dict(color=self.path_color, width=4),
                name=f"{from_loc.name} → {to_loc.name}",
                hoverinfo='name',
            ))

        # Add return arcs from each location back to home (same day)
        if self.config.routes_from_home:
            for loc, loc_time in zip(locations, normalized_times):
                # Create return arc at the same time level (horizontal arc back to home)
                return_time = loc_time + 0.01  # Slightly after arrival
                lons_path, lats_path, times_path = self._create_worldline_trace(
                    loc, home, loc_time, return_time
                )
                fig.add_trace(go.Scatter3d(
                    x=lons_path,
                    y=lats_path,
                    z=times_path,
                    mode='lines',
                    line=dict(color=self.path_color, width=3, dash='dash'),
                    name=f"{loc.name} → Home",
                    hoverinfo='name',
                ))

        # Add location markers
        marker_lons = [loc.lon for loc in locations]
        marker_lats = [loc.lat for loc in locations]
        marker_times = normalized_times
        marker_names = [loc.name for loc in locations]
        marker_dates = [
            self._format_date(loc) if loc.date else ""
            for loc in locations
        ]

        hover_text = [
            f"{name}<br>{date}" if date else name
            for name, date in zip(marker_names, marker_dates)
        ]

        fig.add_trace(go.Scatter3d(
            x=marker_lons,
            y=marker_lats,
            z=marker_times,
            mode='markers+text',
            marker=dict(
                size=10,
                color=self.marker_color,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=marker_names,
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertext=hover_text,
            hoverinfo='text',
            name='Locations',
        ))

        # Add home marker if routes_from_home
        if self.config.routes_from_home:
            home = self.config.get_home()
            fig.add_trace(go.Scatter3d(
                x=[home.lon],
                y=[home.lat],
                z=[0],  # Home at bottom
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=self.home_color,
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                text=[home.name],
                textposition='top center',
                textfont=dict(size=10, color='black'),
                name='Home',
                hoverinfo='text',
                hovertext=f"{home.name} (Home)",
            ))

        # Configure layout
        title = self.config.title if self.config.title else "Travel Worldline"

        # Create time axis labels
        if min_date and max_date:
            tickvals = [0, 0.25, 0.5, 0.75, 1.0]
            date_range = (max_date - min_date).days
            ticktext = [
                (min_date + timedelta(days=int(t * date_range))).strftime('%b %Y')
                for t in tickvals
            ]
        else:
            tickvals = [0, 0.5, 1.0]
            ticktext = ['Start', 'Middle', 'End']

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            scene=dict(
                xaxis=dict(
                    title='Longitude',
                    range=lon_range,
                    backgroundcolor='rgba(230, 230, 250, 0.5)',
                    gridcolor='rgba(0, 0, 0, 0.1)',
                ),
                yaxis=dict(
                    title='Latitude',
                    range=lat_range,
                    backgroundcolor='rgba(230, 230, 250, 0.5)',
                    gridcolor='rgba(0, 0, 0, 0.1)',
                ),
                zaxis=dict(
                    title='Time',
                    range=[-0.05, 1.05],
                    tickvals=tickvals,
                    ticktext=ticktext,
                    backgroundcolor='rgba(230, 230, 250, 0.5)',
                    gridcolor='rgba(0, 0, 0, 0.1)',
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1.5, z=1),
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='rgba(245, 240, 230, 1)',
        )

        # Generate HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=True)
        return html

    def render_static(self, width: int = 1200, height: int = 800) -> Image.Image:
        """Render a static image of the 3D worldline visualization."""
        # Generate the plotly figure
        locations = self.config.locations
        normalized_times, min_date, max_date = self._normalize_time(locations)

        loc_times = {loc.name: t for loc, t in zip(locations, normalized_times)}

        lats = [loc.lat for loc in locations]
        lons = [loc.lon for loc in locations]

        if self.config.routes_from_home:
            home = self.config.get_home()
            lats.append(home.lat)
            lons.append(home.lon)

        padding = 10
        lon_range = (min(lons) - padding, max(lons) + padding)
        lat_range = (min(lats) - padding, max(lats) + padding)

        fig = go.Figure()

        # Add base map at bottom of cube (slightly below 0 to be visible)
        base_surface = self._create_map_surface(
            -0.02, lon_range, lat_range, resolution=300, opacity=self.base_map_opacity
        )
        fig.add_trace(base_surface)

        # Get unique trip time levels from the actual location dates
        trip_times = sorted(set(normalized_times))
        # Filter out times very close to 0 (already have base map there)
        trip_times = [t for t in trip_times if t > 0.05]

        # Add semi-transparent map surfaces at each trip's time level
        for t in trip_times:
            surface = self._create_map_surface(
                t, lon_range, lat_range, resolution=200, opacity=self.map_opacity
            )
            fig.add_trace(surface)

        # Add vertical line at home location (time axis)
        home = self.config.get_home()
        fig.add_trace(go.Scatter3d(
            x=[home.lon, home.lon],
            y=[home.lat, home.lat],
            z=[-0.02, 1.02],
            mode='lines',
            line=dict(color=self.home_line_color, width=8, dash='dot'),
            showlegend=False,
        ))

        # Add worldline paths (outbound)
        routes = self.config.get_routes()
        for from_loc, to_loc in routes:
            from_time = loc_times.get(from_loc.name, 0.5)
            to_time = loc_times.get(to_loc.name, 0.5)

            if from_loc.name == home.name:
                from_time = max(0, to_time - 0.02)

            lons_path, lats_path, times_path = self._create_worldline_trace(
                from_loc, to_loc, from_time, to_time
            )

            fig.add_trace(go.Scatter3d(
                x=lons_path,
                y=lats_path,
                z=times_path,
                mode='lines',
                line=dict(color=self.path_color, width=5),
                showlegend=False,
            ))

        # Add return arcs from each location back to home (same day)
        if self.config.routes_from_home:
            for loc, loc_time in zip(locations, normalized_times):
                return_time = loc_time + 0.01
                lons_path, lats_path, times_path = self._create_worldline_trace(
                    loc, home, loc_time, return_time
                )
                fig.add_trace(go.Scatter3d(
                    x=lons_path,
                    y=lats_path,
                    z=times_path,
                    mode='lines',
                    line=dict(color=self.path_color, width=4, dash='dash'),
                    showlegend=False,
                ))

        # Add markers
        fig.add_trace(go.Scatter3d(
            x=[loc.lon for loc in locations],
            y=[loc.lat for loc in locations],
            z=normalized_times,
            mode='markers+text',
            marker=dict(size=8, color=self.marker_color),
            text=[loc.name for loc in locations],
            textposition='top center',
            textfont=dict(size=9),
            showlegend=False,
        ))

        # Add home
        if self.config.routes_from_home:
            home = self.config.get_home()
            fig.add_trace(go.Scatter3d(
                x=[home.lon],
                y=[home.lat],
                z=[0],
                mode='markers+text',
                marker=dict(size=10, color=self.home_color, symbol='diamond'),
                text=[home.name],
                textposition='top center',
                showlegend=False,
            ))

        # Create time labels
        if min_date and max_date:
            tickvals = [0, 0.33, 0.66, 1.0]
            date_range = (max_date - min_date).days
            ticktext = [
                (min_date + timedelta(days=int(t * date_range))).strftime('%b %Y')
                for t in tickvals
            ]
        else:
            tickvals = [0, 0.5, 1.0]
            ticktext = ['Start', 'Middle', 'End']

        title = self.config.title if self.config.title else "Travel Worldline"

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=18)),
            scene=dict(
                xaxis=dict(title='Longitude', range=lon_range),
                yaxis=dict(title='Latitude', range=lat_range),
                zaxis=dict(
                    title='Time',
                    range=[-0.05, 1.05],
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.0),
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1.5, z=1),
            ),
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=50, b=0),
            paper_bgcolor='rgba(245, 240, 230, 1)',
        )

        # Convert to image
        img_bytes = fig.to_image(format='png', width=width, height=height)
        img = Image.open(io.BytesIO(img_bytes))

        return img
