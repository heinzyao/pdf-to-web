"""Command-line interface for pdf-to-web."""

import click

from pdf_to_web import pdf_to_yaml, yaml_to_html


@click.group()
def cli():
    """Convert PDF documents to professional web pages."""
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="./output", help="Output directory for YAML files"
)
def convert(pdf_path: str, output: str):
    """Convert PDF to YAML format."""
    yaml_path = pdf_to_yaml(pdf_path, output)
    click.echo(f"YAML file created: {yaml_path}")


@cli.command()
@click.argument("yaml_path", type=click.Path(exists=True))
@click.option(
    "--output", "-o", default="./output", help="Output directory for HTML files"
)
@click.option("--template", "-t", default="index.html", help="HTML template to use")
def build(yaml_path: str, output: str, template: str):
    """Convert YAML to HTML web page."""
    html_path = yaml_to_html(yaml_path, output, template)
    click.echo(f"HTML file created: {html_path}")


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="./output", help="Output directory")
@click.option("--template", "-t", default="index.html", help="HTML template to use")
def run(pdf_path: str, output: str, template: str):
    """Convert PDF to HTML in one step."""
    click.echo(f"Converting {pdf_path} to YAML...")
    yaml_path = pdf_to_yaml(pdf_path, output)
    click.echo(f"YAML file created: {yaml_path}")

    click.echo("Converting YAML to HTML...")
    html_path = yaml_to_html(yaml_path, output, template)
    click.echo(f"HTML file created: {html_path}")

    click.echo("\nConversion complete!")
    click.echo(f"Open {html_path} in your browser to view the result.")


if __name__ == "__main__":
    cli()
