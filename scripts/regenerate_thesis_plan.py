"""Regenerate the auto-generated regions of THESIS_PLAN.tex from THESIS_PLAN_DATA.yaml.

Only the content between matched marker pairs is replaced:

    % AUTO-GEN START: <region_name>
    ...everything in here is replaced...
    % AUTO-GEN END: <region_name>

Existing markers in THESIS_PLAN.tex (must already be present):
  - milestones        : the longtable of milestone rows
  - weeks             : the week-by-week subsections
  - version_log       : the bullet list at the bottom
  - title_date        : the \\date{...} line in the preamble
  - period_line       : the "Period:" + supervisor line right after \\maketitle

Usage:
    python backend/scripts/regenerate_thesis_plan.py

Optional:
    python backend/scripts/regenerate_thesis_plan.py --bump-version "note about what changed"
        appends a new entry to version_log with today's date and bumps document_version.
"""

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
TEX_PATH = ROOT / "THESIS_PLAN.tex"
YAML_PATH = ROOT / "THESIS_PLAN_DATA.yaml"

STATUS_MACRO = {"done": "\\sdone", "wip": "\\swip", "pending": "\\spending"}


def render_milestones(milestones: list[dict]) -> str:
    lines = []
    for m in milestones:
        macro = STATUS_MACRO.get(m["status"], "\\spending")
        result = f"{{}} {m['result']}" if m.get("result") else ""
        lines.append(
            f"{m['id']} & {m['title']} & {m['target']} & {macro}{result}\\\\"
        )
    return "\n".join(lines)


def render_weeks(weeks: list[dict]) -> str:
    out = []
    out.append("\\subsection{Past, current, and future weeks}")
    out.append("")
    for w in weeks:
        title = w["title"]
        out.append(f"\\subsubsection*{{Week {w['n']} ({w['range']}): {title}}}")
        out.append("\\begin{itemize}")
        for b in w["bullets"]:
            out.append(f"    \\item {b}")
        out.append("\\end{itemize}")
        out.append("")
    return "\n".join(out)


def render_version_log(entries: list[dict]) -> str:
    out = ["\\begin{itemize}"]
    for e in entries:
        out.append(f"    \\item \\textbf{{v{e['version']} ({e['date']}):}} {e['note']}")
    out.append("\\end{itemize}")
    return "\n".join(out)


def render_title_date(version: int, date: str) -> str:
    return f"\\date{{Document version {version} --- {date} \\emph{{(living document)}}}}"


def render_period_line() -> str:
    # Static for now but kept as a region in case period or supervisor changes
    return ("\\noindent\\textbf{Period:} 7 April 2026 \\textendash{} 30 June 2026 (12 weeks)"
            "\\hfill\\textbf{Supervisor:} \\emph{to fill in}")


def replace_region(tex: str, region: str, new_body: str) -> str:
    """Replace content between the AUTO-GEN markers for `region` with new_body.
    Returns modified tex. Raises ValueError if markers are missing."""
    start_marker = f"% AUTO-GEN START: {region}"
    end_marker = f"% AUTO-GEN END: {region}"
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    if not pattern.search(tex):
        raise ValueError(
            f"Markers for region '{region}' not found in {TEX_PATH.name}.\n"
            f"Make sure both '{start_marker}' and '{end_marker}' exist."
        )
    replacement = f"{start_marker}\n{new_body}\n{end_marker}"
    # Use a callable so backslashes in LaTeX (\\, \section, etc.) are not interpreted
    # as regex back-references. lambda receives the match and returns the literal string.
    return pattern.sub(lambda _: replacement, tex, count=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bump-version", metavar="NOTE",
                   help="Append a new entry to version_log with today's date, "
                        "bump document_version by 1, then regenerate.")
    args = p.parse_args()

    if not TEX_PATH.exists():
        print(f"Missing {TEX_PATH}", file=sys.stderr); sys.exit(1)
    if not YAML_PATH.exists():
        print(f"Missing {YAML_PATH}", file=sys.stderr); sys.exit(1)

    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))

    if args.bump_version:
        new_v = int(data.get("document_version", 1)) + 1
        today = _dt.date.today().strftime("%-d %B %Y") if sys.platform != "win32" \
                else _dt.date.today().strftime("%#d %B %Y")
        data["document_version"] = new_v
        data["document_date"] = today
        data["version_log"].append({
            "version": new_v,
            "date": today,
            "note": args.bump_version,
        })
        YAML_PATH.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                             encoding="utf-8")
        print(f"Bumped document_version to {new_v} ({today}) and updated YAML.")

    tex = TEX_PATH.read_text(encoding="utf-8")
    tex = replace_region(tex, "title_date",
                         render_title_date(data["document_version"], data["document_date"]))
    tex = replace_region(tex, "period_line", render_period_line())
    tex = replace_region(tex, "milestones", render_milestones(data["milestones"]))
    tex = replace_region(tex, "weeks", render_weeks(data["weeks"]))
    tex = replace_region(tex, "version_log", render_version_log(data["version_log"]))
    TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"Regenerated {TEX_PATH.name} from {YAML_PATH.name}.")
    print(f"  document_version: {data['document_version']}  date: {data['document_date']}")
    print(f"  milestones: {len(data['milestones'])}  weeks: {len(data['weeks'])}  "
          f"version_log entries: {len(data['version_log'])}")


if __name__ == "__main__":
    main()
