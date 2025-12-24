"""Worldline style renderer - 3D spacetime visualization with time as vertical axis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import io
from datetime import datetime, timedelta

from .base import BaseRenderer
from ..config import TravelConfig


class WorldlineRenderer(BaseRenderer):
    """Render maps as 3D worldline visualization with time on vertical axis."""

    def __init__(self, config: TravelConfig):
        super().__init__(config)
        self.path_color = "#e74c3c"  # Red for worldlines
        self.marker_color = "#3498db"  # Blue for location markers
        self.home_color = "#2ecc71"  # Green for home

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
        resolution: int = 50
    ) -> go.Surface:
        """Create a semi-transparent surface representing the map at a time level."""
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Create a flat surface at the given time level
        z_grid = np.full_like(lon_grid, time_level)

        # Simple land/ocean coloring based on rough continental outlines
        # This is a simplified version - could be enhanced with actual coastline data
        colors = np.zeros((resolution, resolution))

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Very rough approximation of land masses
                if self._is_land(lat, lon):
                    colors[i, j] = 0.7  # Land
                else:
                    colors[i, j] = 0.3  # Ocean

        return go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=z_grid,
            surfacecolor=colors,
            colorscale=[
                [0, 'rgba(100, 149, 237, 0.3)'],  # Ocean - light blue, transparent
                [1, 'rgba(144, 238, 144, 0.3)']   # Land - light green, transparent
            ],
            showscale=False,
            hoverinfo='skip',
            opacity=0.5,
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

        # Add semi-transparent map surfaces at different time levels
        time_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in time_levels:
            surface = self._create_map_surface(t, lon_range, lat_range, resolution=30)
            fig.add_trace(surface)

        # Add worldline paths
        routes = self.config.get_routes()
        for from_loc, to_loc in routes:
            from_time = loc_times.get(from_loc.name, 0.5)
            to_time = loc_times.get(to_loc.name, 0.5)

            # For home, use the destination's time minus a small offset
            if from_loc.name == self.config.get_home().name:
                from_time = max(0, to_time - 0.02)

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

        # Add map surfaces
        time_levels = [0.0, 0.33, 0.66, 1.0]
        for t in time_levels:
            surface = self._create_map_surface(t, lon_range, lat_range, resolution=40)
            fig.add_trace(surface)

        # Add worldline paths
        routes = self.config.get_routes()
        for from_loc, to_loc in routes:
            from_time = loc_times.get(from_loc.name, 0.5)
            to_time = loc_times.get(to_loc.name, 0.5)

            if from_loc.name == self.config.get_home().name:
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
