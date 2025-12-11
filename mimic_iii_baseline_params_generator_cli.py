from __future__ import annotations

import itertools
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

app = typer.Typer(help="CLI to generate MIMIC-III SDG experiment YAML configs.")
console = Console()


def generate_filename(
    generator: str,
    data_seed: int,
    experiment_seed: int,
) -> str:
    """
    Generate filename according to:
    mimic_iii_baseline_dseedDATASEED_synthcity_GENERATOR_mseedEXPERIMENTSEED.yaml
    """
    return (
        f"mimic_iii_baseline_dseed{data_seed}_"
        f"synthcity_{generator}_mseed{experiment_seed}.yaml"
    )


def instantiate_template(
    template: str,
    generator: str,
    data_seed: int,
    experiment_seed: int,
) -> str:
    """
    Replace placeholders in the template with concrete values.
    Assumes placeholders:
      - GENERATOR
      - DATASEED
      - EXPERIMENTSEED
    """
    content = template
    content = content.replace("GENERATOR", generator)
    content = content.replace("DATASEED", str(data_seed))
    content = content.replace("EXPERIMENTSEED", str(experiment_seed))
    return content


@app.command("generate")
def generate_configs(
    template: Path = typer.Option(
        ...,
        "--template",
        "-t",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the YAML template file.",
    ),
    output_dir: Path = typer.Option(
        Path("./configs"),
        "--output-dir",
        "-o",
        help="Directory where generated YAML files will be written.",
    ),
    generator: List[str] = typer.Option(
        ...,
        "--generator",
        "-g",
        help="Generator model type(s) to substitute for GENERATOR (e.g. rtvae).",
    ),
    data_seed: List[int] = typer.Option(
        ...,
        "--data-seed",
        "-d",
        help="Data seed(s) to substitute for DATASEED.",
    ),
    experiment_seed: List[int] = typer.Option(
        ...,
        "--experiment-seed",
        "-e",
        help="Experiment seed(s) to substitute for EXPERIMENTSEED.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="If set, do not write files, just show what would be generated.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="If set, overwrite existing files with the same name.",
    ),
) -> None:
    """
    Generate YAML configuration files from a template for all combinations
    of GENERATOR x DATASEED x EXPERIMENTSEED.
    """
    console.print(
        Panel.fit(
            f"[bold]Generating configs from template[/bold]\n[dim]{template}[/dim]",
            border_style="cyan",
        )
    )

    # Read template
    template_text = template.read_text(encoding="utf-8")

    # Prepare output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(generator, data_seed, experiment_seed))

    if not combos:
        console.print("[red]No combinations provided; nothing to do.[/red]")
        raise typer.Exit(code=1)

    generated_files = []

    for gen, dseed, eseed in track(
        combos,
        description="Generating YAML configs...",
        console=console,
    ):
        filename = generate_filename(gen, dseed, eseed)
        full_path = output_dir / filename

        yaml_content = instantiate_template(
            template=template_text,
            generator=gen,
            data_seed=dseed,
            experiment_seed=eseed,
        )

        if dry_run:
            # Just record what would happen
            generated_files.append((gen, dseed, eseed, str(full_path), "DRY-RUN"))
            continue

        if full_path.exists() and not overwrite:
            status = "SKIPPED (exists)"
        else:
            full_path.write_text(yaml_content, encoding="utf-8")
            status = "WRITTEN"

        generated_files.append((gen, dseed, eseed, str(full_path), status))

    # Summary table
    table = Table(title="Generated configuration files", show_lines=True)
    table.add_column("GENERATOR", style="cyan", no_wrap=True)
    table.add_column("DATASEED", style="magenta", justify="right")
    table.add_column("EXPERIMENTSEED", style="magenta", justify="right")
    table.add_column("Path", style="green")
    table.add_column("Status", style="yellow")

    for gen, dseed, eseed, path, status in generated_files:
        table.add_row(str(gen), str(dseed), str(eseed), path, status)

    console.print(table)

    if dry_run:
        console.print(
            "[bold yellow]Dry run completed.[/bold yellow] "
            "No files were written. Use --dry-run=false (default) to write files."
        )
    else:
        console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    app()
