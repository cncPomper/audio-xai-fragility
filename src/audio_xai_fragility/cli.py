"""Console script for audio_xai_fragility."""

import typer
from rich.console import Console

from audio_xai_fragility import h, utils
from audio_xai_fragility.metrics.peaq import peaq

app = typer.Typer(help="audio_xai_fragility cli")
console = Console()


@app.command()
def main() -> None:
    """Console script for audio_xai_fragility."""
    console.print("Replace this message by putting your code into audio_xai_fragility.cli.master")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


@app.command()
def hello(name: str = "world") -> None:
    """Print 'Hello, world!' to the console."""
    h.hello(name)


@app.command()
def peaq_demo(sr: int = 16000) -> None:
    """Run a demo of the PEAQ metric."""
    peaq_result = peaq(
        reference=[0.0, 0.5, 0.0, -0.5] * 1000,
        test=[0.0, 0.4, 0.1, -0.4] * 1000,
        sample_rate=sr,
    )
    console.print(f"PEAQ Result: {peaq_result}")


if __name__ == "__main__":
    app()
