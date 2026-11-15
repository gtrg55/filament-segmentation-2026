#!/usr/bin/env python3
"""Execute the code cells of a notebook in one namespace and store stdout and figures as nbformat-4 outputs.

Jupyter is not a dependency of the repository, so this minimal executor stands in for `nbconvert --execute`.
`--set NAME=VALUE` overrides a top-level assignment (e.g. `LIMIT=3`) for the run only; the stored source cells stay
unchanged. Absolute paths of the machine are replaced by `.` in the captured output.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import re
import traceback
from pathlib import Path


def apply_overrides(source: str, overrides: dict[str, str]) -> str:
    for name, value in overrides.items():
        source = re.sub(rf"^{re.escape(name)}\s*=.*$", f"{name} = {value}", source, flags=re.M)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="directory whose absolute path is hidden in outputs")
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.set)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: list[bytes] = []

    def capture_show(*_args, **_kwargs) -> None:
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", dpi=90, bbox_inches="tight")
            figures.append(buffer.getvalue())
        plt.close("all")

    plt.show = capture_show
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__main__"}
    hidden = str(args.root.resolve())
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        source = apply_overrides("".join(cell["source"]), overrides)
        stream, outputs = io.StringIO(), []
        figures.clear()
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(source, f"<cell {count}>", "exec"), namespace)
        except Exception:  # noqa: BLE001 - the traceback is stored as the cell output
            stream.write(traceback.format_exc())
        text = stream.getvalue().replace(hidden, ".")
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)})
        for png in figures:
            outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(png).decode("ascii"), "text/plain": ["<Figure>"]}})
        cell["outputs"], cell["execution_count"] = outputs, count
    args.out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"executed {count} code cells -> {args.out}")


if __name__ == "__main__":
    main()
