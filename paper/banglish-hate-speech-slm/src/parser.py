"""Response parser — converts raw model output into (label, reasoning).

Faithful to the reference notebook's `parse_response`:

    if text.startswith("```") and text.endswith("```"):
        text = "\\n".join(text.splitlines()[1:-1]).strip()
    if " - " in text:
        label, reason = text.split(" - ", 1)
    else:
        label, reason = text, ""

We extend it to:
  - Tolerate a leading bullet/quote/numbering.
  - Map the label string to one of the seven canonical labels (fuzzy).
  - Return a structured ParsedResponse so callers can distinguish
    'valid label' from 'invalid label' from 'parse error'.
  - Recover from a literal `<Chosen Label>` placeholder by scanning
    the full response for the first canonical label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .utils import normalise_label


@dataclass
class ParsedResponse:
    label: str            # canonical label, or "" if invalid
    reasoning: str        # free text, may be ""
    raw: str              # the original raw response, trimmed
    parse_status: str     # 'ok' | 'invalid_label' | 'parse_error' | 'empty'

    @property
    def is_valid(self) -> bool:
        return self.parse_status == "ok"


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return t


def parse_response(raw: str) -> ParsedResponse:
    """Parse a model response into (label, reasoning).

    Mirrors the reference notebook:
        1. Strip ``` fences if present.
        2. Split on " - " → (label, reason); else label=whole, reason="".

    Extensions for SLM output:
        - Tolerate the literal placeholder `<Chosen Label>` and recover
          the actual category from the reasoning paragraph.
        - Tolerate Markdown reasoning headers (`**Reasoning**: ...`).
    """
    if raw is None:
        return ParsedResponse("", "", "", "empty")

    text = _strip_code_fence(raw)
    if not text:
        return ParsedResponse("", text, text, "empty")

    # Prefer the first non-empty line for label extraction.
    first_line = next((ln for ln in text.splitlines() if ln.strip()), text)
    candidate = first_line

    if " - " in candidate:
        label_str, reason = candidate.split(" - ", 1)
    else:
        # Fallback: look for the first " - " anywhere.
        m = re.search(r" - ", candidate)
        if m:
            label_str = candidate[: m.start()]
            reason = candidate[m.end() :]
        else:
            label_str, reason = candidate, ""

    label_str = label_str.strip().strip('"').strip("'").strip("`")
    reason = reason.strip().strip('"').strip("'").strip("`")
    # Drop leading bullets / numbering / Markdown bold from the label.
    label_str = re.sub(r"^[\s\-\*\d\.\(\)]+", "", label_str)
    label_str = re.sub(r"\*\*\s*|\s*\*\*", "", label_str)  # strip **Markdown** wrappers

    canon = normalise_label(label_str)

    # Recovery path A: model emitted the literal placeholder.
    if canon is None and re.search(r"<\s*Chosen\s*Label\s*>", label_str, re.IGNORECASE):
        full_lower = text.lower()
        from .utils import _LABEL_TOKEN_MAP
        for key, c in _LABEL_TOKEN_MAP.items():
            if key in full_lower:
                canon = c
                if " - " in text:
                    _, reason_full = text.split(" - ", 1)
                else:
                    reason_full = text
                reason = re.split(r"\n\s*\n", reason_full.strip(), maxsplit=1)[0].strip()
                break

    # Recovery path B: model gave a Markdown conclusion like
    #   **Conclusion**: This text is classified as **Not Hate Speech**.
    # Find the first canonical label that follows a colon/bold/etc.
    if canon is None:
        from .utils import _LABEL_TOKEN_MAP
        # Try every canonical label in order of decreasing length, case-insensitive.
        candidates_sorted = sorted(_LABEL_TOKEN_MAP.items(), key=lambda kv: -len(kv[0]))
        for key, c in candidates_sorted:
            # Look for the label as a standalone token (surrounded by non-word chars).
            pat = r"(?:^|[\s\*\-\:\.\(\)\,\;])" + re.escape(key) + r"(?:$|[\s\*\-\:\.\(\)\,\;\.]+)"
            if re.search(pat, text, re.IGNORECASE):
                canon = c
                # Use the sentence that contains the match as reasoning.
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    # Take the paragraph containing this match.
                    paragraphs = re.split(r"\n\s*\n", text)
                    for para in paragraphs:
                        if re.search(pat, para, re.IGNORECASE):
                            reason = para.strip()[:300]
                            break
                else:
                    reason = ""
                break

    if canon is None:
        return ParsedResponse("", reason, text, "invalid_label")
    return ParsedResponse(canon, reason, text, "ok")