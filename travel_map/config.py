"""YAML configuration parsing and validation."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Location:
    """A single location on the map."""

    name: str
    lat: float
    lon: float
    date: Optional[datetime] = None
    label: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Location":
        """Create a Location from a dictionary."""
        date = None
        if "date" in data and data["date"]:
            if isinstance(data["date"], datetime):
                date = data["date"]
            else:
                date = datetime.fromisoformat(str(data["date"]))

        return cls(
            name=data["name"],
            lat=float(data["lat"]),
            lon=float(data["lon"]),
            date=date,
            label=data.get("label"),
        )


@dataclass
class Route:
    """A route between two locations."""

    from_location: str
    to_location: str

    @classmethod
    def from_dict(cls, data: dict) -> "Route":
        """Create a Route from a dictionary."""
        return cls(
            from_location=data["from"],
            to_location=data["to"],
        )


@dataclass
class TravelConfig:
    """Complete travel map configuration."""

    title: str
    locations: list[Location]
    style: str = "pins"
    output: str = "interactive"
    show_dates: bool = True
    date_format: str = "%b %d"
    routes: list[Route] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_styles = ["pins", "arcs", "indiana_jones"]
        if self.style not in valid_styles:
            raise ValueError(f"Invalid style '{self.style}'. Must be one of: {valid_styles}")

        valid_outputs = ["static", "interactive"]
        if self.output not in valid_outputs:
            raise ValueError(f"Invalid output '{self.output}'. Must be one of: {valid_outputs}")

        if not self.locations:
            raise ValueError("At least one location is required")

    def get_routes(self) -> list[tuple[Location, Location]]:
        """Get routes as location pairs. Auto-generates from dates if not specified."""
        location_map = {loc.name: loc for loc in self.locations}

        if self.routes:
            # Use explicit routes
            result = []
            for route in self.routes:
                from_loc = location_map.get(route.from_location)
                to_loc = location_map.get(route.to_location)
                if from_loc and to_loc:
                    result.append((from_loc, to_loc))
            return result

        # Auto-generate routes from chronological order
        dated_locations = [loc for loc in self.locations if loc.date]
        if len(dated_locations) < 2:
            return []

        sorted_locations = sorted(dated_locations, key=lambda x: x.date)
        return [(sorted_locations[i], sorted_locations[i + 1])
                for i in range(len(sorted_locations) - 1)]

    @classmethod
    def from_dict(cls, data: dict) -> "TravelConfig":
        """Create a TravelConfig from a dictionary."""
        locations = [Location.from_dict(loc) for loc in data.get("locations", [])]
        routes = [Route.from_dict(r) for r in data.get("routes", [])]

        return cls(
            title=data.get("title", "My Travel Map"),
            locations=locations,
            style=data.get("style", "pins"),
            output=data.get("output", "interactive"),
            show_dates=data.get("show_dates", True),
            date_format=data.get("date_format", "%b %d"),
            routes=routes,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TravelConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)
