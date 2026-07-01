from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_COMPARE_FIELDS = [
    "evidence_standard_met",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
    "risk_flags",
]


def load_solution_module(solution_path: Path):
    spec = importlib.util.spec_from_file_location("solution_main", str(solution_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load solution module from {solution_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize(s: str) -> str:
    return (s or "").strip().lower()


def compare_risk_flags(a: str, b: str) -> bool:
    if not a and not b:
        return True
    set_a = {p.strip() for p in (a or "").split(";") if p.strip()}
    set_b = {p.strip() for p in (b or "").split(";") if p.strip()}
    return set_a == set_b


def compare_field(pred: str, expected: str, field: str) -> bool:
    if field == "risk_flags":
        return compare_risk_flags(pred, expected)
    return normalize(pred) == normalize(expected)


def run_evaluation(
    sample_claims_path: Path,
    solution_mod,
    history_path: Path,
    requirements: Dict[tuple[str, str], int],
) -> Tuple[str, List[str]]:
    # load user history
    history = {}
    if hasattr(solution_mod, "load_user_history"):
        history = solution_mod.load_user_history(history_path)

    rows: List[Dict[str, str]] = []
    with sample_claims_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    field_matches = {f: 0 for f in REQUIRED_COMPARE_FIELDS}
    overall_exact = 0
    mismatches: List[str] = []

    for row in rows:
        pred = solution_mod.predict_claim(
            row, sample_claims_path.parent.parent, history, requirements
        )
        expected = row
        all_match = True
        for field in REQUIRED_COMPARE_FIELDS:
            pred_val = getattr(pred, field)
            exp_val = expected.get(field, "")
            ok = compare_field(pred_val, exp_val, field)
            if ok:
                field_matches[field] += 1
            else:
                all_match = False
        if all_match:
            overall_exact += 1
        else:
            mismatches.append(
                f"user_id={row.get('user_id','?')}: predicted={ {f: getattr(pred,f) for f in REQUIRED_COMPARE_FIELDS} } expected={ {f: expected.get(f,'') for f in REQUIRED_COMPARE_FIELDS} }"
            )

    report_lines: List[str] = []
    report_lines.append("# Evaluation Report")
    report_lines.append("")
    report_lines.append(f"Total examples: {total}")
    report_lines.append(
        f"Overall exact-match (all compared fields): {overall_exact} ({overall_exact/total:.2%})"
        if total
        else "Overall exact-match: 0"
    )
    report_lines.append("")
    report_lines.append("Field-level accuracy:")
    for f in REQUIRED_COMPARE_FIELDS:
        cnt = field_matches[f]
        pct = f"{(cnt/total):.2%}" if total else "0%"
        report_lines.append(f"- {f}: {cnt}/{total} ({pct})")
    report_lines.append("")
    report_lines.append("Top mismatches (up to 20):")
    for m in mismatches[:20]:
        report_lines.append(f"- {m}")

    return "\n".join(report_lines), mismatches


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sample_claims = repo_root / "dataset" / "sample_claims.csv"
    history = repo_root / "dataset" / "user_history.csv"
    solution_path = repo_root / "code" / "main.py"

    print(f"Loading solution from: {solution_path}")
    sol = load_solution_module(solution_path)

    # Load evidence requirements
    req_path = sample_claims.parent / "evidence_requirements.csv"
    requirements: Dict[tuple[str, str], int] = {}
    if hasattr(sol, "load_evidence_requirements"):
        requirements = sol.load_evidence_requirements(req_path)

    report_text, mismatches = run_evaluation(sample_claims, sol, history, requirements)

    out_dir = repo_root / "code" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "evaluation_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Wrote evaluation report to {report_path}")


if __name__ == "__main__":
    main()
