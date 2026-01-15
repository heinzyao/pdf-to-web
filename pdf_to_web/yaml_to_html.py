"""YAML to HTML conversion module.

This module renders YAML data to HTML using Jinja2 templates,
maintaining the same structure as ppt-to-web for consistency.
"""

import json
import shutil
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


def _create_html_env() -> Environment:
    """Create Jinja2 environment with template directory."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    # Add tojson filter for data serialization
    env.filters['tojson'] = lambda x: json.dumps(x, ensure_ascii=False)
    return env


def yaml_to_html(
    yaml_path: str,
    html_output_dir: str,
    template_name: str = "index.html",
    output_filename: str | None = None,
) -> str:
    """Convert YAML data to HTML.

    Args:
        yaml_path: Path to the input YAML file.
        html_output_dir: Directory to save the HTML file.
        template_name: Name of the Jinja2 template to use.
        output_filename: Optional custom output filename.

    Returns:
        Path to the created HTML file.
    """
    yaml_file = Path(yaml_path)
    html_dir = Path(html_output_dir)
    html_dir.mkdir(parents=True, exist_ok=True)

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    env = _create_html_env()
    template = env.get_template(template_name)

    html_content = template.render(data=data)

    if output_filename:
        html_output_path = html_dir / output_filename
    else:
        html_output_path = html_dir / f"{data['title']}.html"

    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Copy media files to output directory
    media_dir = yaml_file.parent / "media"
    if media_dir.exists():
        output_media_dir = html_dir / "media"
        output_media_dir.mkdir(exist_ok=True)

        for media_file in media_dir.iterdir():
            if media_file.is_file():
                dst = output_media_dir / media_file.name
                if media_file != dst:
                    shutil.copy2(media_file, dst)

    return str(html_output_path)
