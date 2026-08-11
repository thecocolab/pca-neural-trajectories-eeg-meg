from pathlib import Path

import nbformat
import pytest


@pytest.mark.parametrize("path", sorted(Path("tutorials").glob("*.ipynb")))
def test_tutorial_is_valid_and_cleared(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []
            compile(cell.source, f"{path}:cell-{index}", "exec")
