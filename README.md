# Indiana Jones Travel Map

Generate beautiful travel maps from YAML configuration with multiple visual styles.

## Installation

```bash
pip install -e .
```

## Usage

### Generate a map from YAML config

```bash
# Generate with settings from config file
travel-map generate examples/sample_trip.yml

# Override style or output format
travel-map generate trip.yml --style indiana_jones --format static

# Specify output path
travel-map generate trip.yml -o my_map.html
```

### Preview in browser

```bash
travel-map preview examples/sample_trip.yml
```

### Generate config from geotagged photos

```bash
travel-map from-photos ./vacation_photos -o trip.yml --title "Summer Vacation"
```

## Styles

- **pins** - Clean modern map with location markers
- **arcs** - Pins with curved connection lines (Facebook checkin style)
- **indiana_jones** - Vintage map with red dotted flight paths

## YAML Configuration

```yaml
title: "My Adventure"
style: indiana_jones  # pins | arcs | indiana_jones
output: interactive   # static | interactive
show_dates: true
date_format: "%b %d, %Y"

locations:
  - name: "New York"
    lat: 40.7128
    lon: -74.0060
    date: "2024-01-01"
    label: "Departure"

  - name: "London"
    lat: 51.5074
    lon: -0.1278
    date: "2024-01-03"

# Optional explicit routes (auto-generated from dates if omitted)
routes:
  - from: "New York"
    to: "London"
```

## Requirements

- Python 3.9+
- folium (interactive maps)
- matplotlib + cartopy (static maps)
- Pillow (image processing)
- PyYAML (config parsing)
- click (CLI)
