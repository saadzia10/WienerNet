#!/usr/bin/env python3
"""PostToolUse hook: when plotting code is created or edited, remind Claude to
apply the `plot-conventions` skill and verify the figure conforms.

Fires on Write / Edit / MultiEdit / NotebookEdit. Reads the hook JSON on stdin,
inspects the target file + the written content, and — only if the content looks
like plotting code — emits `additionalContext` pointing at the skill. Stays
silent (no context) otherwise, so it never interferes with non-plotting edits.
"""
import json
import re
import sys

# Signals that a file is plotting code. Kept broad but specific enough to avoid
# firing on incidental mentions.
PLOT_PATTERNS = re.compile(
    r"""(
        import\s+matplotlib
      | from\s+matplotlib
      | matplotlib\.pyplot
      | import\s+seaborn
      | import\s+plotly
      | \bplt\.
      | \bsns\.
      | \.savefig\s*\(
      | \bsavefig\b
      | plt\.subplots\s*\(
      | \.set_xlabel\s*\(
      | \.set_ylabel\s*\(
    )""",
    re.VERBOSE,
)

REMINDER = (
    "This edit touches plotting code. Before finishing, apply the `plot-conventions` "
    "skill (Skill tool → plot-conventions) and make the code conform: no titles/suptitles "
    "or baked-in panel letters; unit-bearing axis labels; one standalone file per plot "
    "(no merged multi-panel images); vector PDF/SVG output (PNG only for heavy raster at "
    "≥300 DPI); bbox_inches='tight'; a consistent colorblind-safe palette; descriptive "
    "groupable filenames; and a printed + written `<name>.txt` draft-caption sidecar per "
    "figure. Then render the figure and eyeball it."
)


def collect_text(tool_input: dict) -> str:
    """Pull whatever textual payload this Write/Edit variant carries."""
    parts = []
    for key in ("content", "new_string", "new_str"):
        v = tool_input.get(key)
        if isinstance(v, str):
            parts.append(v)
    # MultiEdit: list of {old_string, new_string}
    for edit in tool_input.get("edits", []) or []:
        if isinstance(edit, dict):
            v = edit.get("new_string") or edit.get("new_str")
            if isinstance(v, str):
                parts.append(v)
    # NotebookEdit
    v = tool_input.get("new_source")
    if isinstance(v, str):
        parts.append(v)
    return "\n".join(parts)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # never block on a parse error

    tool_input = data.get("tool_input", {}) or {}
    path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")

    # Only Python-ish / notebook targets.
    if not re.search(r"\.(py|ipynb)$", path):
        return 0

    text = collect_text(tool_input)
    if not text or not PLOT_PATTERNS.search(text):
        return 0

    out = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": REMINDER,
        }
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
