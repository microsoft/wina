#!/usr/bin/env python3
"""Generate and optionally run the Lean axiom audit from catalog.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def catalog_declarations(catalog_path: Path) -> list[str]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    declarations: list[str] = []
    for item in catalog["items"]:
        for declaration in item.get("lean", {}).get("declarations", []):
            if declaration in declarations:
                raise ValueError(f"duplicate catalog declaration: {declaration}")
            declarations.append(declaration)
    if not declarations:
        raise ValueError("catalog contains no Lean declarations")
    return declarations


def generated_source(module: str, declarations: list[str]) -> str:
    lines = [
        "-- This file is generated from catalog.json by generate_catalog_axioms.py.",
        "-- Do not edit it by hand.",
        f"import {module}",
        "",
    ]
    lines.extend(f"#print axioms {declaration}" for declaration in declarations)
    return "\n".join(lines) + "\n"


def parse_audit(output: str, declarations: list[str]) -> list[str]:
    findings: dict[str, set[str]] = {}
    depends_pattern = re.compile(
        r"^'(.+?)' depends on axioms: \[(.*?)\]$", re.MULTILINE | re.DOTALL
    )
    none_pattern = re.compile(
        r"^'(.+?)' does not depend on any axioms$", re.MULTILINE
    )
    for match in depends_pattern.finditer(output):
        findings[match.group(1)] = {
            " ".join(axiom.split())
            for axiom in match.group(2).split(",")
            if axiom.strip()
        }
    for match in none_pattern.finditer(output):
        findings[match.group(1)] = set()

    errors: list[str] = []
    expected = set(declarations)
    missing = expected - findings.keys()
    extra = findings.keys() - expected
    if missing:
        errors.append("missing audit output: " + ", ".join(sorted(missing)))
    if extra:
        errors.append("unexpected audit output: " + ", ".join(sorted(extra)))
    for declaration in declarations:
        unsupported = findings.get(declaration, set()) - ALLOWED_AXIOMS
        if unsupported:
            errors.append(
                f"{declaration} uses unsupported axioms: "
                + ", ".join(sorted(unsupported))
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="umbrella Lean module")
    parser.add_argument("--catalog", type=Path, default=Path("catalog.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("CatalogAxioms.lean")
    )
    parser.add_argument(
        "--receipt", type=Path, default=Path("catalog-axiom-audit.txt")
    )
    parser.add_argument(
        "--run", action="store_true", help="run lake env lean and validate axioms"
    )
    args = parser.parse_args()

    declarations = catalog_declarations(args.catalog)
    source = generated_source(args.module, declarations)
    args.output.write_text(source, encoding="utf-8", newline="\n")

    if not args.run:
        print(f"generated {args.output} for {len(declarations)} declarations")
        return 0

    command = ["lake", "env", "lean", str(args.output)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    combined = completed.stdout + completed.stderr
    receipt_header = (
        f"command: {' '.join(command)}\n"
        f"catalog declarations: {len(declarations)}\n"
        f"allowed axioms: {', '.join(sorted(ALLOWED_AXIOMS))}\n"
        f"exit code: {completed.returncode}\n\n"
    )
    args.receipt.write_text(receipt_header + combined, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        print(combined, file=sys.stderr)
        return completed.returncode

    errors = parse_audit(combined, declarations)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(
        f"audited {len(declarations)} declarations; no axioms outside the allowed set"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
