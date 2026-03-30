"""Console script for audio_xai_fragility."""

import typer
from rich.console import Console

from audio_xai_fragility import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for audio_xai_fragility."""
    console.print("Replace this message by putting your code into audio_xai_fragility.cli.master")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
