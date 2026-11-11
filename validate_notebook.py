# -*- coding: utf-8 -*-
"""Validate the generated notebook: JSON structure + python syntax of code cells."""
import json
import re
import sys
from pathlib import Path

nb_path = Path(r"E:\machine learing\project4\vessel_segmentation_kaggle.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

print("nbformat:", nb["nbformat"], nb["nbformat_minor"])
print("cells:", len(nb["cells"]))

MAGIC = re.compile(r"^\s*[!%]")

errors = []
for i, cell in enumerate(nb["cells"]):
    ct = cell["cell_type"]
    src = cell["source"]
    if isinstance(src, list):
        src = "".join(src)
    if ct == "code":
        lines = []
        for line in src.split("\n"):
            lines.append("pass" if MAGIC.match(line) else line)
        code = "\n".join(lines)
        try:
            compile(code, f"<cell {i}>", "exec")
        except SyntaxError as e:
            errors.append((i, e))
            print(f"[FAIL] cell {i}: {e}")
    elif ct == "markdown":
        if not src.strip():
            errors.append((i, "empty markdown"))

print()
if errors:
    print(f"{len(errors)} problem(s)")
    sys.exit(1)
print("OK: all cells valid")
