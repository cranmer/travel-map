"""Command-line interface for travel-map."""

import webbrowser
from pathlib import Path
import tempfile

import click

from .config import TravelConfig
from .styles import STYLES


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Generate beautiful travel maps from YAML configuration."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Output file path. Extension determines format (.html or .png/.svg).",
)
@click.option(
    "-s", "--style",
    type=click.Choice(["pins", "arcs", "indiana_jones"]),
    help="Override the style from config.",
)
@click.option(
    "-f", "--format",
    "output_format",
    type=click.Choice(["static", "interactive"]),
    help="Override the output format from config.",
)
@click.option(
    "--width",
    type=int,
    default=1200,
    help="Width for static output (default: 1200).",
)
@click.option(
    "--height",
    type=int,
    default=800,
    help="Height for static output (default: 800).",
)
def generate(config_file, output, style, output_format, width, height):
    """Generate a map from a YAML configuration file."""
    # Load config
    config = TravelConfig.from_yaml(config_file)

    # Apply overrides
    if style:
        config.style = style
    if output_format:
        config.output = output_format

    # Determine output path
    if not output:
        config_path = Path(config_file)
        ext = ".html" if config.output == "interactive" else ".png"
        output = config_path.with_suffix(ext)

    # Get renderer
    renderer_class = STYLES.get(config.style)
    if not renderer_class:
        raise click.ClickException(f"Unknown style: {config.style}")

    renderer = renderer_class(config)

    # Generate map
    click.echo(f"Generating {config.style} map ({config.output})...")

    if config.output == "interactive":
        html = renderer.render_interactive()
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".html")
        output_path.write_text(html)
        click.echo(f"Saved to: {output_path}")
    else:
        img = renderer.render_static(width=width, height=height)
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        img.save(output_path)
        click.echo(f"Saved to: {output_path}")


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "-s", "--style",
    type=click.Choice(["pins", "arcs", "indiana_jones"]),
    help="Override the style from config.",
)
def preview(config_file, style):
    """Preview a map in your default browser."""
    config = TravelConfig.from_yaml(config_file)

    if style:
        config.style = style

    # Force interactive for preview
    config.output = "interactive"

    renderer_class = STYLES.get(config.style)
    if not renderer_class:
        raise click.ClickException(f"Unknown style: {config.style}")

    renderer = renderer_class(config)

    click.echo(f"Generating {config.style} preview...")
    html = renderer.render_interactive()

    # Save to temp file and open
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        temp_path = f.name

    click.echo(f"Opening preview in browser...")
    webbrowser.open(f"file://{temp_path}")


@cli.command()
@click.argument("photo_dir", type=click.Path(exists=True))
@click.option(
    "-o", "--output",
    type=click.Path(),
    default="trip.yml",
    help="Output YAML file (default: trip.yml).",
)
@click.option(
    "-t", "--title",
    default="My Trip",
    help="Title for the trip (default: 'My Trip').",
)
@click.option(
    "-s", "--style",
    type=click.Choice(["pins", "arcs", "indiana_jones"]),
    default="arcs",
    help="Default style for the config (default: arcs).",
)
def from_photos(photo_dir, output, title, style):
    """Generate a YAML config from geotagged photos."""
    from .photos import extract_locations_from_photos

    click.echo(f"Scanning photos in {photo_dir}...")

    locations = extract_locations_from_photos(photo_dir)

    if not locations:
        raise click.ClickException("No geotagged photos found!")

    click.echo(f"Found {len(locations)} geotagged photos.")

    # Generate YAML
    import yaml

    config_data = {
        "title": title,
        "style": style,
        "output": "interactive",
        "show_dates": True,
        "date_format": "%b %d, %Y",
        "locations": locations,
    }

    output_path = Path(output)
    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Config saved to: {output_path}")
    click.echo("You can edit this file to customize location names and add labels.")


if __name__ == "__main__":
    cli()
