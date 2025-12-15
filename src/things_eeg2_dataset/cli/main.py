import datetime
import http.server
import logging
import os
import socket
import socketserver
import subprocess
import webbrowser
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated

import typer
from rich import print  # noqa: A004
from rich.panel import Panel

from things_eeg2_dataset import __version__
from things_eeg2_dataset.cli.logger import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="things-eeg2",
    help="Unified CLI for THINGS-EEG2 processing & dataloader tools.",
)

DEFAULT_SUBJECTS = list(range(1, 11))
DEFAULT_MODELS: list[str] = []
DEFAULT_PROJECT_DIR = Path.home() / "things_eeg2"


def get_local_ip() -> str:
    """
    Best-effort attempt to get the local network IP address.
    Uses a dummy UDP connection to determine the primary interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # We don't actually send data, so 10.255.255.255 (unroutable) works fine.
        # We just want the OS to tell us which interface it would use.
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        # Fallback if no network is available
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def _init_pipeline(  # noqa: PLR0913
    project_dir: Path,
    subjects: list[int],
    force: bool,
    dry_run: bool,
    skip_download: bool,
    skip_preprocessing: bool,
    create_embeddings: bool,
    skip_merging: bool,
    device: str = "cuda:0",
    sfreq: int = 250,
    models: list[str] | None = None,
) -> "ThingsEEGPipeline":  # type: ignore # noqa: F821
    from things_eeg2_dataset.processing.pipeline import (  # noqa: PLC0415
        PipelineConfig,
        ThingsEEGPipeline,
    )

    models = models or []

    config = PipelineConfig(
        project_dir=project_dir,
        subjects=subjects,
        models=models,
        sfreq=sfreq,
        device=device,
        overwrite=force,
        dry_run=dry_run,
        skip_download=skip_download,
        skip_processing=skip_preprocessing,
        create_embeddings=create_embeddings,
        skip_merging=skip_merging,
    )

    return ThingsEEGPipeline(config)


def version_callback(value: bool) -> None:
    if value:
        print(f"{__package__.split('.')[0]} version: [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show the application's version and exit.",
            callback=version_callback,
        ),
    ] = None,
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v, -vv, -vvv).",
    ),
) -> None:
    setup_logging(verbosity=verbose)


@app.command(name="download")
def download(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to download."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing downloaded data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
) -> None:
    """
    Download the THINGS-EEG2 raw dataset.
    """

    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        force=overwrite,
        dry_run=dry_run,
        skip_download=False,
        skip_preprocessing=True,  # Not needed for downloading
        create_embeddings=False,  # Not needed for downloading
        skip_merging=True,  # Not needed for downloading
    )
    pipeline.step_download_data()


@app.command(name="preprocess")
def preprocess(  # noqa: PLR0913
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing processed data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Enable pyinstrument profiling for the preprocessing step.",
    ),
    open_report: bool = typer.Option(
        True, "--open/--no-open", help="Open profiling report automatically."
    ),
) -> None:
    """
    Preprocess the THINGS-EEG2 raw dataset.
    """
    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        models=[],  # Not needed for preprocessing
        sfreq=sfreq,
        device="cuda:0",  # Not needed for preprocessing
        force=force,
        dry_run=dry_run,
        skip_download=True,  # Not needed for preprocessing
        skip_preprocessing=False,
        create_embeddings=False,  # Not needed for preprocessing
        skip_merging=True,  # Not needed for preprocessing
    )

    if profile:
        from pyinstrument import Profiler  # noqa: PLC0415

        profile_dir = project_dir / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_preprocessing.html"
        output_path = profile_dir / filename

        with Profiler() as profiler:
            pipeline.step_process_eeg()
        # Include timestamp in filename to avoid overwriting
        output_path.write_text(profiler.output_html())

        print(f"\n[green]Profiling report saved to:[/green] {output_path}")

        is_headless = os.environ.get("DISPLAY") is None

        if open_report and not is_headless:
            print("[yellow]Opening report in browser...[/yellow]")
            webbrowser.open(output_path.as_uri())
        elif open_report and is_headless:
            print("[yellow]Headless environment detected. Skipping auto-open.[/yellow]")
            print(
                "Run [bold cyan]things-eeg2 view-profile --serve[/bold cyan] to view it remotely."
            )

        return

    pipeline.step_process_eeg()


@app.command(name="view-profile")
def view_profile(
    project_dir: Path = typer.Option(DEFAULT_PROJECT_DIR, "--project-dir"),
    list_all: bool = typer.Option(
        False, "--list", help="List all available reports without opening."
    ),
    serve: bool = typer.Option(
        False,
        "--serve",
        help="Host the reports on a local web server (good for remote SSH).",
    ),
    port: int = typer.Option(8000, "--port", help="Port to use for the web server."),
) -> None:
    """
    Open the most recent profiling report in your web browser.
    """
    profile_dir = project_dir / "profiling"

    if not profile_dir.exists():
        print(f"[red]No profiling directory found at {profile_dir}[/red]")
        raise typer.Exit(code=1)

    # Get all html files sorted by modification time (newest first)
    reports = sorted(profile_dir.glob("*.html"), key=os.path.getmtime, reverse=True)

    if not reports:
        print("[yellow]No profiling reports found.[/yellow]")
        raise typer.Exit()

    if serve:
        os.chdir(profile_dir)  # Change cwd to serve files relative to here

        # Create a simple handler that lists directory
        Handler = http.server.SimpleHTTPRequestHandler
        local_ip = get_local_ip()
        localhost_url = f"http://localhost:{port}"
        network_url = f"http://{local_ip}:{port}"
        print(
            Panel(
                f"[bold]Serving Profiling Reports[/bold]\n"
                f"[dim]Directory: {profile_dir}[/dim]\n\n"
                f"🏠 [bold green]Local:[/bold green]   {localhost_url}\n"
                f"📡 [bold cyan]Network:[/bold cyan] {network_url}\n\n"
                f"[yellow]Press Ctrl+C to stop.[/yellow]",
                title="Web Server Running",
                border_style="green",
                expand=False,
            )
        )

        # Bind to "0.0.0.0" to allow external connections (important!)
        try:
            with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:  # noqa: S104
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[yellow]Server stopped.[/yellow]")
        return

    if list_all:
        print(f"[bold]Found {len(reports)} reports in {profile_dir}:[/bold]")
        for r in reports:
            print(
                f" - {r.name} ({datetime.datetime.fromtimestamp(r.stat().st_mtime, datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')})"
            )
        return

    # Open the newest one
    latest_report = reports[0]
    print(f"[green]Opening latest report:[/green] {latest_report.name}")
    webbrowser.open(latest_report.as_uri())


@app.command(name="embed")
def embed(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    models: list[str] = typer.Option(
        ..., "--models", help="List of models to generate embeddings for."
    ),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing embeddings."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
) -> None:
    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=[],  # Not needed for embedding
        models=models,
        sfreq=250,  # Not needed for embedding
        device=device,
        force=force,
        dry_run=dry_run,
        skip_download=True,  # Not needed for embedding
        skip_preprocessing=True,  # Not needed for embedding
        create_embeddings=True,
        skip_merging=True,  # Not needed for embedding
    )
    pipeline.step_generate_embeddings()


@app.command(name="pipeline")
def pipeline(  # noqa: PLR0913
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project."
    ),
    subjects: list[int] = typer.Option(
        DEFAULT_SUBJECTS, "--subjects", help="List of subject numbers to process."
    ),
    models: list[str] = typer.Option(
        DEFAULT_MODELS, "--models", help="List of models to use."
    ),
    sfreq: int = typer.Option(250, "--sfreq", help="Downsampling frequency in Hz."),
    device: str = typer.Option(
        "cuda:0", "--device", help="Device for model inference."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing processed data."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Don't write data to disk."),
    skip_download: bool = typer.Option(False, "--skip-download"),
    skip_preprocessing: bool = typer.Option(False, "--skip-preprocessing"),
    create_embeddings: bool = typer.Option(False, "--create-embeddings"),
    skip_merging: bool = typer.Option(False, "--skip-merging"),
) -> None:
    """
    Run the full THINGS-EEG2 raw processing pipeline.
    """
    pipeline = _init_pipeline(
        project_dir=project_dir,
        subjects=subjects,
        models=models,
        sfreq=sfreq,
        device=device,
        force=force,
        dry_run=dry_run,
        skip_download=skip_download,
        skip_preprocessing=skip_preprocessing,
        create_embeddings=create_embeddings,
        skip_merging=skip_merging,
    )
    pipeline.run()
    raise typer.Exit(code=0)


@app.command(name="info")
def info(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project root."
    ),
    subject: int = typer.Option(..., "--subject", help="Subject number (e.g., 1)."),
    session: int = typer.Option(..., "--session", help="Session number (e.g., 1)."),
    data_index: int = typer.Option(
        ...,
        "--data-index",
        help="0-based index of the numpy array element you want information about.",
    ),
    partition: str = typer.Option(
        "training", "--partition", help="Partition ('training' or 'test')."
    ),
) -> None:
    """
    Load and display information for a specific sample based on metadata.
    """

    from things_eeg2_dataset.dataloader.sample_info import (  # noqa: PLC0415
        get_info_for_sample,
    )

    info = get_info_for_sample(
        project_dir=project_dir,
        subject=subject,
        session=session,
        data_idx=data_index,
        partition=partition,
    )

    print(info)


def _run_streamlit(app_path: Path, project_dir: Path) -> None:
    if not app_path.exists():
        print(f"[bold red]Error:[/bold red] Could not find Streamlit app at {app_path}")
        raise typer.Exit(code=1)

    welcome_msg = (
        f"[bold]Project Directory:[/bold] {project_dir}\n"
        "[yellow]Press Ctrl+C to stop the server[/yellow]"
    )
    print(Panel(welcome_msg, title="🚀 THINGS EEG2 EXPLORER", border_style="green"))

    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--",
        f"--project-dir={project_dir}",
    ]

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except KeyboardInterrupt:
        print("\n[green]Explorer stopped.[/green]")
    except subprocess.CalledProcessError:
        print("\n[bold red]Streamlit crashed.[/bold red]")
        raise typer.Exit(code=1) from None


@app.command(name="show")
def show(
    project_dir: Path = typer.Option(
        DEFAULT_PROJECT_DIR, "--project-dir", help="Path to project root."
    ),
) -> None:
    """
    Visualize EEG data for a specific sample.
    """

    try:
        traversable = files("things_eeg2_dataset.visualization").joinpath("app.py")
        with as_file(traversable) as app_path:
            _run_streamlit(app_path, project_dir)
    except (ImportError, TypeError):
        package_dir = Path(__file__).parent.parent.resolve()
        app_path = package_dir / "visualization" / "app.py"
        _run_streamlit(app_path, project_dir)

    if not app_path.exists():
        print(f"[bold red]Error:[/bold red] Could not find Streamlit app at {app_path}")
        raise typer.Exit(code=1)

    welcome_msg = (
        f"[bold]Project Directory:[/bold] {project_dir}\n"
        "[yellow]Press Ctrl+C to stop the server[/yellow]"
    )
    print(Panel(welcome_msg, title="🚀 THINGS EEG2 EXPLORER", border_style="green"))

    # Disable usage stats to hide the "Collecting usage stats..." message
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        "streamlit",
        "run",
        str(app_path),
        "--",
        f"--project-dir={project_dir}",
    ]
    try:
        subprocess.run(cmd, check=True)  # noqa: S603
    except KeyboardInterrupt:
        print("\n[green]Explorer stopped.[/green]")
    except subprocess.CalledProcessError:
        print("\n[bold red]Streamlit crashed.[/bold red]")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
