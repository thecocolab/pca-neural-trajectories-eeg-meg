"""Execute one tutorial notebook headlessly and save the executed copy.

The executed notebook is written to ``outputs/notebooks/`` whether or not the
run succeeded, so a failing cell's traceback is preserved for inspection.

Examples
--------
python scripts/run_one_notebook.py tutorial_eegbci_decoding.ipynb
python scripts/run_one_notebook.py tutorial_eegbci_decoding.ipynb --timeout 7200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook",
        help="Filename under tutorials/, e.g. tutorial_eegbci_main.ipynb",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "notebooks")
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell timeout in seconds; -1 disables it.",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = REPO / "tutorials" / args.notebook
    destination = args.output_dir / args.notebook

    print(f"=== executing {args.notebook} ===", flush=True)
    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO)}},
    )
    try:
        client.execute()
        status = "OK"
    except CellExecutionError as error:
        status = f"FAILED: {error}"
    finally:
        nbformat.write(notebook, destination)
    print(f"=== {args.notebook}: {status[:2000]} ===", flush=True)
    print(f"executed copy -> {destination}", flush=True)
    if status != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
