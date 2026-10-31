"""Placeholder filling of the README from the claim ledger."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CODE_ROOT / "scripts" / "fill_readme.py"

CLAIMS = {
    "a": {"value": 0.66666, "unit": "fraction", "ci": {"lo": 0.5, "hi": 0.75}, "n": 9, "count": 6},
    "b": {"value": 3, "unit": "count"},
    "s": {"value": 0.000234, "unit": "seconds"},
    "m": {"value": 2_064_789_504, "unit": "bytes"},
    "t": {"value": "cap_settle", "unit": "text"},
    "e": {"value": None, "unit": "fraction"},
}


@pytest.fixture(scope="module")
def fill_readme_module():
    spec = importlib.util.spec_from_file_location("fill_readme", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fill_text_formats_every_field_without_escaping(fill_readme_module) -> None:
    text = "x <<a>> <<a:ci>> <<a:n>> <<a:count>> <<b>> <<s>> <<m>> <<t>> <<e>> <<b:ci>>"
    filled, placeholders = fill_readme_module.fill_text(text, CLAIMS)
    assert filled == "x 66.7 [50.0, 75.0] 9 6 3 0.000234 2.06 cap_settle n/a n/a"
    assert len(placeholders) == 10
    assert fill_readme_module.fill_text("plain text", CLAIMS) == ("plain text", [])


def test_fill_text_asserts_each_placeholder_exactly_once(fill_readme_module) -> None:
    with pytest.raises(AssertionError):
        fill_readme_module.fill_text("<<a>> and <<a>>", CLAIMS)
    with pytest.raises(AssertionError):
        fill_readme_module.fill_text("<<a:ci>> then <<a:ci>>", CLAIMS)
    with pytest.raises(KeyError):
        fill_readme_module.fill_text("<<a>> <<missing>>", CLAIMS)
    with pytest.raises(AssertionError):
        fill_readme_module.fill_text("<<a>> <<not a claim>>", CLAIMS)


def test_fill_readme_writes_the_output_and_reports_placeholders(fill_readme_module, tmp_path: Path) -> None:
    template = tmp_path / "README.template.md"
    template.write_text("# Title\n\n| x | <<a>> <<a:ci>> | <<b>> |\n", encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(CLAIMS), encoding="utf-8")
    out = tmp_path / "README.md"
    placeholders = fill_readme_module.fill_readme(template, ledger, out)
    assert sorted(placeholders) == ["<<a:ci>>", "<<a>>", "<<b>>"]
    assert out.read_text(encoding="utf-8") == "# Title\n\n| x | 66.7 [50.0, 75.0] | 3 |\n"
    with pytest.raises(FileNotFoundError):
        fill_readme_module.fill_readme(template, tmp_path / "absent.json", out)


def test_fill_readme_script_fills_the_shipped_template(tmp_path: Path) -> None:
    ledger = CODE_ROOT / "results" / "ledger.json"
    if not ledger.is_file():
        pytest.skip("no claim ledger under results/")
    out = tmp_path / "README.md"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--out", str(out)], capture_output=True, text=True, cwd=str(CODE_ROOT))
    assert proc.returncode == 0, proc.stderr
    text = out.read_text(encoding="utf-8")
    assert "<<" not in text and ">>" not in text
    assert text.startswith("# Tasdiq")
    shipped = CODE_ROOT / "README.md"
    if shipped.is_file():
        assert shipped.read_text(encoding="utf-8") == text
