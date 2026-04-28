"""
LaTeX -> Markdown + BibTeX exporter for revised_manuscript_2.tex.

Scope: handles the constructs actually used in this paper (verified by inspection).
Anything unrecognized is left as raw LaTeX and logged to unrecognized.log so the
next agent (or human reviewer) can spot-fix it.

Inputs (read-only):
  ../../MANUSCRIPT/revised_manuscript_2.tex
  ../../MANUSCRIPT/bibliography_2.bib

Outputs (written under ../):
  ../text.md
  ../references.bib
  ../_build/unrecognized.log
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import OrderedDict

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent
ROOT = EXPORT.parent

TEX_PATH = ROOT / "MANUSCRIPT" / "revised_manuscript_2.tex"
BIB_PATH = ROOT / "MANUSCRIPT" / "bibliography_2.bib"

OUT_MD = EXPORT / "text.md"
OUT_BIB = EXPORT / "references.bib"
OUT_LOG = HERE / "unrecognized.log"


# ---------- Bibliography helpers ----------

# Known alias map: cited key -> equivalent .bib key (when keys differ but entries
# describe the same publication). Discovered during inspection.
BIB_ALIASES = {
    "Zhang2021_DATE": ["Zhang2021_DATE", "HPC"],
    "ZHANG2024DR": ["ZHANG2024DR", "Zhang2024", "Zhang2024_SusComp"],
    "Zhang2025": ["Zhang2025", "Zhang2023"],
    "Zheng2020": ["Zheng2020", "zheng2020mitigating"],
    "Takci2025": ["Takci2025", "takci2025data"],
}


def parse_bibfile(text: str) -> dict[str, str]:
    """Return {key: full BibTeX entry string} from a .bib file."""
    entries: dict[str, str] = {}
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "@":
            # find entry type and key
            m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[i:])
            if not m:
                i += 1
                continue
            key = m.group(2)
            # find matching closing brace at top level
            j = i + m.end()
            depth = 1
            while j < n and depth > 0:
                c = text[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                j += 1
            entries[key] = text[i:j]
            i = j
        else:
            i += 1
    return entries


def synthesize_bibtex_from_bibitem(key: str, body: str) -> str:
    """Best-effort BibTeX from a free-form \\bibitem body."""
    body = re.sub(r"\\newblock\b", " ", body)
    body = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\path\s*\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\url\s*\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\urlprefix", "", body)
    body = re.sub(r"\\newline", " ", body)
    body = re.sub(r"\\textit\s*\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\\textbf\s*\{([^{}]*)\}", r"\1", body)
    body = re.sub(r"\{\\([a-zA-Z]+)\}", r"\1", body)
    body = re.sub(r"~", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    # year — last 4-digit group in parentheses
    year = ""
    ym = re.findall(r"\((\d{4})\)", body)
    if ym:
        year = ym[-1]
    # doi
    doi = ""
    dm = re.search(r"doi:\s*([^\s,}]+)", body, flags=re.I)
    if dm:
        doi = dm.group(1).rstrip(".,")
    # url (first http(s)://)
    url = ""
    um = re.search(r"https?://[^\s,}]+", body)
    if um:
        url = um.group(0).rstrip(".,")

    out = [f"@misc{{{key},"]
    out.append(f"  note         = {{{body}}},")
    if year:
        out.append(f"  year         = {{{year}}},")
    if doi:
        out.append(f"  doi          = {{{doi}}},")
    if url:
        out.append(f"  url          = {{{url}}},")
    out.append("}")
    out.insert(1, "  % source: \\bibitem (free-form parsed -- review fields)")
    return "\n".join(out)


def parse_bibitems(thebib_block: str) -> dict[str, str]:
    """Return {key: free-form body} from the inline \\bibitem block."""
    bibitems: dict[str, str] = {}
    parts = re.split(r"\\bibitem\s*\{([^{}]+)\}", thebib_block)
    # parts[0] = preamble, then key, body, key, body, ...
    for i in range(1, len(parts), 2):
        key = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        bibitems[key] = body
    return bibitems


def build_references_bib(cited_keys: list[str], bibitems: dict[str, str],
                         bib_entries: dict[str, str]) -> str:
    """Emit BibTeX for the cited keys, preferring structured .bib entries."""
    out_chunks: list[str] = []
    out_chunks.append("% References for thesis-export of revised_manuscript_2.tex.\n"
                      "% 54 entries, one per citation key actually used in the paper.\n"
                      "% Entries marked '% source: \\bibitem (free-form parsed)' were\n"
                      "% synthesized from the inline thebibliography because the .bib\n"
                      "% file did not contain a matching key. Review those for accuracy.\n")
    for key in sorted(cited_keys, key=str.lower):
        # prefer exact match
        chosen = None
        if key in bib_entries:
            chosen = bib_entries[key]
            chosen = re.sub(r"@(\w+)\s*\{\s*[^,\s]+\s*,",
                            lambda m: f"@{m.group(1)}{{{key},", chosen, count=1)
        else:
            # try aliases
            for canonical, aliases in BIB_ALIASES.items():
                if key in aliases:
                    for alt in aliases:
                        if alt in bib_entries:
                            entry = bib_entries[alt]
                            entry = re.sub(r"@(\w+)\s*\{\s*[^,\s]+\s*,",
                                           lambda m: f"@{m.group(1)}{{{key},",
                                           entry, count=1)
                            chosen = entry
                            break
                if chosen:
                    break
        if chosen is None and key in bibitems:
            chosen = synthesize_bibtex_from_bibitem(key, bibitems[key])
        if chosen is None:
            chosen = (f"@misc{{{key},\n"
                      f"  % source: not found in either .bib or thebibliography\n"
                      f"  note = {{Missing entry; please supply.}},\n"
                      f"}}")
        out_chunks.append(chosen.strip())
    return "\n\n".join(out_chunks) + "\n"


# ---------- LaTeX -> Markdown ----------

UNRECOGNIZED: list[str] = []


def _log(msg: str) -> None:
    UNRECOGNIZED.append(msg)


def strip_comments(text: str) -> str:
    """Remove % comments, but not \\% escapes."""
    out_lines = []
    for line in text.splitlines():
        # find unescaped %
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                i += 2
                continue
            if line[i] == "%":
                line = line[:i]
                break
            i += 1
        out_lines.append(line)
    return "\n".join(out_lines)


def find_balanced(text: str, start: int) -> int:
    """Given text[start] == '{', return index of matching '}' (exclusive)."""
    assert text[start] == "{"
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def grab_brace_arg(text: str, after: int) -> tuple[str, int]:
    """Skip whitespace then read {...}; return (inner, end_index)."""
    i = after
    while i < len(text) and text[i] in " \t\n":
        i += 1
    if i >= len(text) or text[i] != "{":
        return "", after
    end = find_balanced(text, i)
    return text[i + 1:end - 1], end


def grab_optional_arg(text: str, after: int) -> tuple[str, int]:
    """Skip whitespace then read [...] if present; return (inner, end_index)."""
    i = after
    while i < len(text) and text[i] in " \t\n":
        i += 1
    if i >= len(text) or text[i] != "[":
        return "", after
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1
        j += 1
    return "", after


# ---------- Environment extractors ----------


def extract_environments(text: str) -> list[tuple]:
    """Return list of (kind, start, end, body, extras) — for figure/table/equation/etc.

    Just used to reason about positions; not for the conversion order. The actual
    conversion is a single pass through `convert_body`.
    """
    return []  # not used directly; kept for future extension


def render_figure_block(env_body: str) -> str:
    """Render a figure / subfigure-bearing figure block."""
    # find subfigures first
    subs = []
    pos = 0
    while True:
        m = re.search(r"\\begin\{subfigure\}", env_body[pos:])
        if not m:
            break
        s = pos + m.start()
        # match \end{subfigure}
        e = env_body.find(r"\end{subfigure}", s)
        if e == -1:
            break
        sub_body = env_body[s:e]
        # extract image
        img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^{}]+)\}", sub_body)
        # extract caption
        cap = re.search(r"\\caption(?:\[[^\]]*\])?\s*\{([^{}]*)\}", sub_body)
        # extract label
        lbl = re.search(r"\\label\s*\{([^{}]+)\}", sub_body)
        if img:
            subs.append({
                "file": img.group(1).strip(),
                "caption": (cap.group(1).strip() if cap else ""),
                "label": (lbl.group(1).strip() if lbl else ""),
            })
        pos = e + len(r"\end{subfigure}")

    # parent caption / label
    # careful: parent caption sits OUTSIDE subfigure environments, so strip subfigures first
    parent_text = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "",
                         env_body, flags=re.DOTALL)
    parent_cap = re.search(
        r"\\caption(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", parent_text)
    parent_lbl = re.search(r"\\label\s*\{([^{}]+)\}", parent_text)

    if subs:
        out = []
        if parent_lbl:
            out.append(f'<a id="{parent_lbl.group(1).strip()}"></a>\n')
        for idx, sf in enumerate(subs):
            letter = chr(ord("a") + idx)
            anchor = f'<a id="{sf["label"]}"></a>' if sf["label"] else ""
            out.append(f"{anchor}\n![]({'figures/' + sf['file']})\n*({letter})*\n")
        if parent_cap:
            cap_clean = clean_inline(parent_cap.group(1))
            out.append(f"*Figure: {cap_clean}*\n")
        return "\n".join(out)

    # single-image figure
    img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\s*\{([^{}]+)\}", env_body)
    if not img:
        _log(f"figure with no \\includegraphics: {env_body[:120]}")
        return env_body
    fname = img.group(1).strip()
    cap_text = clean_inline(parent_cap.group(1)) if parent_cap else ""
    label = parent_lbl.group(1).strip() if parent_lbl else ""
    out = []
    if label:
        out.append(f'<a id="{label}"></a>')
    out.append(f"![{cap_text}](figures/{fname})")
    if cap_text:
        out.append(f"*Figure: {cap_text}*")
    return "\n".join(out) + "\n"


def render_table_block(env_body: str) -> str:
    """Try to render a table as a Markdown table; fall back to fenced LaTeX."""
    cap_match = re.search(
        r"\\caption(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", env_body)
    lbl_match = re.search(r"\\label\s*\{([^{}]+)\}", env_body)
    caption = clean_inline(cap_match.group(1)) if cap_match else ""
    label = lbl_match.group(1).strip() if lbl_match else ""

    # find inner tabular*/tabular block
    tab_match = re.search(r"\\begin\{tabular\*?\}", env_body)
    if not tab_match:
        return _fence_latex_block(env_body, caption, label)
    end_marker = (r"\end{tabular*}"
                  if env_body[tab_match.start():tab_match.end()].endswith("*}")
                  else r"\end{tabular}")
    end_idx = env_body.find(end_marker, tab_match.end())
    if end_idx == -1:
        return _fence_latex_block(env_body, caption, label)

    tab_body = env_body[tab_match.end():end_idx]
    # detect complexity that we won't cleanly render in MD
    complex_signals = (r"\multicolumn", r"\multirow", r"\cmidrule",
                       r"\diagbox", r"\resizebox", r"tabular*")
    if any(s in env_body[tab_match.start():end_idx] for s in complex_signals):
        return _fence_latex_block(env_body, caption, label)

    # consume the tabular preamble: skip {colspec} immediately after \begin{tabular}
    # tab_body currently starts AFTER \begin{tabular...}, so the first '{' is the colspec
    p = 0
    while p < len(tab_body) and tab_body[p] in " \t\n":
        p += 1
    if p < len(tab_body) and tab_body[p] == "{":
        end_brace = find_balanced(tab_body, p)
        colspec = tab_body[p + 1:end_brace - 1]
        tab_body = tab_body[end_brace:]
    else:
        colspec = ""

    # remove rule commands
    tab_body = re.sub(r"\\(toprule|midrule|bottomrule|hline)\b", "", tab_body)

    # split rows on \\
    rows_raw = re.split(r"\\\\", tab_body)
    rows: list[list[str]] = []
    for r in rows_raw:
        r = r.strip()
        if not r:
            continue
        cells = [clean_inline(c.strip()) for c in r.split("&")]
        rows.append(cells)
    if not rows:
        return _fence_latex_block(env_body, caption, label)

    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    out: list[str] = []
    if label:
        out.append(f'<a id="{label}"></a>')
    if caption:
        out.append(f"*Table: {caption}*")
    out.append("")

    header = rows[0]
    body_rows = rows[1:]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * ncols) + "|")
    for r in body_rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return "\n".join(out)


def _fence_latex_block(env_body: str, caption: str, label: str) -> str:
    out = []
    if label:
        out.append(f'<a id="{label}"></a>')
    if caption:
        out.append(f"*Table: {caption}*")
    out.append("")
    out.append("```latex")
    out.append(env_body.strip())
    out.append("```")
    return "\n".join(out) + "\n"


def render_equation_block(env_kind: str, env_body: str) -> str:
    """Render equation / numcases / align. Preserves \\text{} inside math."""
    labels = re.findall(r"\\label\s*\{([^{}]+)\}", env_body)
    eq_text = re.sub(r"\\label\s*\{[^{}]+\}", "", env_body).strip()

    out = []
    for lab in labels:
        out.append(f'<a id="{lab}"></a>')
    if labels:
        out.append("")
    else:
        out.append("")

    if env_kind == "numcases":
        # numcases form: \begin{numcases}{LHS} row1 & cond1 \\ row2 & cond2 ... \end{numcases}
        # Extract LHS using balanced brace matching (LHS may contain \text{...}).
        if eq_text.startswith("{"):
            end = find_balanced(eq_text, 0)
            lhs = eq_text[1:end - 1].strip()
            body = eq_text[end:]
        else:
            lhs = ""
            body = eq_text
        out.append("$$")
        if lhs:
            out.append(lhs)
        out.append(r"\begin{cases}")
        rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
        for r in rows:
            r = re.sub(r"\\quad\s*", " ", r)
            # Keep \text{...} annotations on the right side; cases envs accept them.
            out.append(r + r" \\")
        out.append(r"\end{cases}")
        out.append("$$")
    elif env_kind in ("align", "align*"):
        out.append("$$")
        out.append(r"\begin{aligned}")
        out.append(eq_text)
        out.append(r"\end{aligned}")
        out.append("$$")
    else:
        out.append("$$")
        out.append(eq_text)
        out.append("$$")
    return "\n".join(out) + "\n"


# ---------- Inline cleaner ----------

INLINE_PASSTHROUGH = {
    r"\sep": ", ",
    r"\texteuro": "€",
    r"\noindent": "",
    r"\centering": "",
    r"\raggedbottom": "",
    r"\raggedright": "",
    r"\smallskip": "",
    r"\medskip": "",
    r"\bigskip": "",
}

UNICODE_TRANS = {
    r"\`{a}": "à", r"\'{a}": "á", r"\`{e}": "è", r"\'{e}": "é",
    r"\`{o}": "ò", r"\'{o}": "ó", r'\"{u}': "ü", r"\'{i}": "í",
    r'\"{o}': "ö", r"\~{n}": "ñ", r'\"{a}': "ä",
}


def clean_inline(s: str) -> str:
    """Convert inline LaTeX to inline Markdown / plain text. Math kept verbatim."""
    if not s:
        return ""

    # protect inline math: $...$
    math_blobs: list[str] = []

    def _stash_math(m):
        math_blobs.append(m.group(0))
        return f"\x00MATH{len(math_blobs)-1}\x00"

    s = re.sub(r"\$[^$]*\$", _stash_math, s)

    # accents (do simple, common ones)
    for k, v in UNICODE_TRANS.items():
        s = s.replace(k, v)

    # citations: \cite{a,b} -> [@a; @b]
    def _cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        return "[" + "; ".join(f"@{k}" for k in keys if k) + "]"
    s = re.sub(r"\\cite\s*\{([^{}]*)\}", _cite, s)

    # \ref{lab} -> [lab](#lab)  ; \eqref{lab} -> ([lab](#lab))
    s = re.sub(r"\\eqref\s*\{([^{}]+)\}", r"([\1](#\1))", s)
    s = re.sub(r"\\ref\s*\{([^{}]+)\}", r"[\1](#\1)", s)
    s = re.sub(r"\\label\s*\{([^{}]+)\}", "", s)  # drop stray labels

    # \href{url}{text} -> [text](url)
    s = re.sub(r"\\href\s*\{([^{}]*)\}\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
               lambda m: f"[{m.group(2)}]({m.group(1)})", s)
    # \url{...} -> <...>
    s = re.sub(r"\\url\s*\{([^{}]+)\}", r"<\1>", s)
    # \path{...}, \urlprefix
    s = re.sub(r"\\path\s*\{([^{}]+)\}", r"\1", s)
    s = s.replace(r"\urlprefix", "")
    s = s.replace(r"\newblock", " ")
    s = s.replace(r"\newline", " ")

    # \textbf{x}, \textit{x}, \emph{x}, \texttt{x}, \textsc{x}
    s = re.sub(r"\\textbf\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"**\1**", s)
    s = re.sub(r"\\textit\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"*\1*", s)
    s = re.sub(r"\\emph\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"*\1*", s)
    s = re.sub(r"\\texttt\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"`\1`", s)
    s = re.sub(r"\\textsc\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)

    # \footnote{...}: keep inline, mark as note (rare/none in this paper)
    s = re.sub(r"\\footnote\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
               r" *(footnote: \1)*", s)

    # passthroughs
    for k, v in INLINE_PASSTHROUGH.items():
        s = s.replace(k, v)

    # whitespace and special chars
    s = s.replace("~", " ")  # NBSP - keep as regular space for prose
    s = s.replace(r"\&", "&")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\#", "#")
    s = s.replace(r"\$", "$")
    s = s.replace(r"\_", "_")
    s = s.replace(r"\{", "{")
    s = s.replace(r"\}", "}")
    s = s.replace(r"\,", " ")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\:", " ")
    s = s.replace(r"\!", "")
    # double-backslash line breaks in prose -> hard newline (do this BEFORE
    # the single-backslash strip so we don't decompose \\ into \)
    s = re.sub(r"\\\\(\s|$)", r"  \n", s)
    # single \  (backslash + whitespace) is forced inter-word space or stray
    s = re.sub(r"(?<!\\)\\(?=\s)", "", s)

    # restore math
    def _restore(m):
        idx = int(m.group(1))
        return math_blobs[idx]
    s = re.sub(r"\x00MATH(\d+)\x00", _restore, s)

    # collapse runs of spaces (but preserve newlines)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


# ---------- Block-level walker ----------

ENVS_OF_INTEREST = {
    "figure": render_figure_block,
    "figure*": render_figure_block,
    "table": render_table_block,
    "table*": render_table_block,
    "equation": lambda b: render_equation_block("equation", b),
    "equation*": lambda b: render_equation_block("equation*", b),
    "align": lambda b: render_equation_block("align", b),
    "align*": lambda b: render_equation_block("align*", b),
    "numcases": lambda b: render_equation_block("numcases", b),
    "subnumcases": lambda b: render_equation_block("numcases", b),
}


def render_itemize(body: str, ordered: bool = False) -> str:
    items = re.split(r"\\item\b", body)
    out = []
    for i, raw in enumerate(items):
        raw = raw.strip()
        if not raw:
            continue
        prefix = f"{i}." if ordered else "-"
        out.append(f"{prefix} {clean_inline(raw)}")
    return "\n".join(out) + "\n"


def render_nomenclature_tabular(body: str) -> str:
    """Convert a 2-column tabular that acts as a definition list."""
    body = re.sub(r"\\(toprule|midrule|bottomrule|hline)\b", "", body)
    # strip the colspec at start
    p = 0
    while p < len(body) and body[p] in " \t\n":
        p += 1
    if p < len(body) and body[p] == "{":
        e = find_balanced(body, p)
        body = body[e:]
    rows_raw = re.split(r"\\\\", body)
    out = ["| Symbol | Description |", "|---|---|"]
    for r in rows_raw:
        r = r.strip()
        if not r:
            continue
        cells = r.split("&", 1)
        if len(cells) != 2:
            continue
        sym = clean_inline(cells[0].strip())
        desc = clean_inline(cells[1].strip())
        out.append(f"| {sym} | {desc} |")
    return "\n".join(out) + "\n"


def find_env(text: str, start: int):
    """Find the next \\begin{name}...\\end{name} after start. Returns
    (name, optional_arg, body_start, body_end, env_end)."""
    m = re.search(r"\\begin\{([a-zA-Z*]+)\}", text[start:])
    if not m:
        return None
    name = m.group(1)
    abs_begin = start + m.start()
    body_start = start + m.end()
    # consume optional arg [...]
    opt, body_start = grab_optional_arg(text, body_start)
    end_marker = f"\\end{{{name}}}"
    # need to handle nesting for the SAME env name
    depth = 1
    i = body_start
    pat_b = re.compile(r"\\begin\{" + re.escape(name) + r"\}")
    pat_e = re.compile(r"\\end\{" + re.escape(name) + r"\}")
    while i < len(text):
        nb = pat_b.search(text, i)
        ne = pat_e.search(text, i)
        if ne is None:
            return None
        if nb is not None and nb.start() < ne.start():
            depth += 1
            i = nb.end()
        else:
            depth -= 1
            if depth == 0:
                body_end = ne.start()
                env_end = ne.end()
                return (name, opt, abs_begin, body_start, body_end, env_end)
            i = ne.end()
    return None


def convert_body(text: str) -> str:
    """Walk the document body, replacing environments and section commands."""
    out: list[str] = []
    pos = 0
    n = len(text)

    section_re = re.compile(
        r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}"
    )
    section_levels = {
        "section": "## ",
        "subsection": "### ",
        "subsubsection": "#### ",
        "paragraph": "##### ",
        "subparagraph": "###### ",
    }

    while pos < n:
        # find next env or section
        env_match = re.search(r"\\begin\{([a-zA-Z*]+)\}", text[pos:])
        sec_match = section_re.search(text[pos:])
        # special inline-math-equation \[...\]
        disp_eq_match = re.search(r"\\\[", text[pos:])

        candidates = []
        if env_match:
            candidates.append(("env", pos + env_match.start()))
        if sec_match:
            candidates.append(("sec", pos + sec_match.start()))
        if disp_eq_match:
            candidates.append(("dispeq", pos + disp_eq_match.start()))

        if not candidates:
            # rest is prose
            chunk = text[pos:]
            out.append(render_prose(chunk))
            break

        candidates.sort(key=lambda x: x[1])
        kind, idx = candidates[0]
        if idx > pos:
            out.append(render_prose(text[pos:idx]))

        if kind == "sec":
            sm = section_re.match(text, idx)
            level_cmd = sm.group(1)
            title = clean_inline(sm.group(2))
            prefix = section_levels[level_cmd]
            out.append("\n" + prefix + title + "\n")
            pos = sm.end()
            continue

        if kind == "dispeq":
            # \[ ... \]
            close = text.find(r"\]", idx)
            if close == -1:
                pos = idx + 2
                continue
            eq = text[idx + 2:close].strip()
            out.append("\n$$\n" + eq + "\n$$\n")
            pos = close + 2
            continue

        # env
        env = find_env(text, idx)
        if env is None:
            _log(f"unmatched \\begin near pos {idx}: {text[idx:idx+80]}")
            pos = idx + len(r"\begin")
            continue
        name, opt, b0, body_start, body_end, env_end = env
        body = text[body_start:body_end]

        if name in ENVS_OF_INTEREST:
            out.append("\n" + ENVS_OF_INTEREST[name](body) + "\n")
        elif name == "itemize":
            out.append("\n" + render_itemize(body, ordered=False) + "\n")
        elif name == "enumerate":
            out.append("\n" + render_itemize(body, ordered=True) + "\n")
        elif name == "abstract":
            out.append("\n## Abstract\n\n" + render_prose(body) + "\n")
        elif name == "keyword":
            kws = re.split(r"\\sep|,", body)
            kws = [clean_inline(k).strip() for k in kws if clean_inline(k).strip()]
            out.append("\n**Keywords:** " + ", ".join(kws) + "\n")
        elif name == "frontmatter":
            out.append(convert_body(body))
        elif name == "document":
            out.append(convert_body(body))
        elif name == "tabular":
            # bare tabular outside table env (used in nomenclature)
            out.append("\n" + render_nomenclature_tabular(body) + "\n")
        elif name == "thebibliography":
            # we drop this and emit a pointer
            out.append("\n*References: see `references.bib`.*\n")
        elif name in ("linenumbers",):
            out.append(convert_body(body))
        else:
            _log(f"unhandled env: {name} at pos {idx}: {body[:120]}")
            out.append("\n```latex\n" + text[b0:env_end] + "\n```\n")
        pos = env_end

    return "".join(out)


def render_prose(text: str) -> str:
    """Handle frontmatter macros and ordinary prose."""
    # Strip preamble noise
    text = re.sub(r"\\documentclass[^\n]*", "", text)
    text = re.sub(r"\\usepackage(?:\[[^\]]*\])?\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\usetikzlibrary\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\captionsetup\s*\[[^\]]*\]\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\renewcommand\s*\{[^{}]*\}\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\setlength\s*\{[^{}]*\}\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\journal\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\vspace\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\hspace\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\bigskip|\\medskip|\\smallskip", "", text)
    text = re.sub(r"\\noindent", "", text)
    text = re.sub(r"\\centering", "", text)
    text = re.sub(r"\\maketitle", "", text)

    # Title -> H1
    title_m = re.search(r"\\title\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text)
    title = clean_inline(title_m.group(1)) if title_m else ""

    # Authors: \author[1]{Name}\ead{email}
    authors = []
    for am in re.finditer(
            r"\\author(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text):
        authors.append(clean_inline(am.group(1)))

    # Affiliations: \affiliation[N]{...} - we render as a list of "Affiliation N: ..."
    affils = []
    for af in re.finditer(
            r"\\affiliation(?:\[(\d+)\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text):
        idx = af.group(1) or "?"
        body = af.group(2)
        # extract nested key=value entries
        org = re.search(r"organization\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)
        city = re.search(r"city\s*=\s*\{([^{}]*)\}", body)
        country = re.search(r"country\s*=\s*\{([^{}]*)\}", body)
        parts = []
        if org:
            parts.append(clean_inline(org.group(1)))
        if city:
            parts.append(clean_inline(city.group(1)))
        if country:
            parts.append(clean_inline(country.group(1)))
        affils.append(f"[{idx}] " + ", ".join(parts))

    out_parts: list[str] = []
    if title:
        out_parts.append(f"# {title}\n")
    if authors:
        out_parts.append("**Authors:** " + ", ".join(authors) + "\n")
    if affils:
        out_parts.append("**Affiliations:**\n\n" +
                         "\n".join(f"- {a}" for a in affils) + "\n")

    # remove the structural macros now that they're rendered
    text = re.sub(r"\\title\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    text = re.sub(
        r"\\author(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)
    text = re.sub(r"\\ead\s*\{[^{}]*\}", "", text)
    text = re.sub(
        r"\\affiliation(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\}", "", text)

    # standard inline cleaning
    text = clean_inline(text)
    # collapse blank-line runs to single blank
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text.strip():
        out_parts.append(text)
    return "\n".join(out_parts)


# ---------- Driver ----------


def main() -> int:
    raw = TEX_PATH.read_text(encoding="utf-8")
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    bib_entries = parse_bibfile(bib_text)

    # 1. strip line-comments
    body = strip_comments(raw)

    # 2. split off the inline thebibliography for separate handling
    bib_match = re.search(r"\\begin\{thebibliography\}", body)
    if bib_match:
        bib_end = body.find(r"\end{thebibliography}", bib_match.end())
        thebib_block = body[bib_match.end():bib_end]
        body_main = body[:bib_match.start()] + body[bib_end + len(
            r"\end{thebibliography}"):]
    else:
        thebib_block = ""
        body_main = body

    bibitems = parse_bibitems(thebib_block) if thebib_block else {}

    # 3. collect cited keys from main body
    cited: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\\cite\s*\{([^{}]+)\}", body_main):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                cited.append(k)

    # 4. produce text.md
    md = convert_body(body_main)
    # tighten whitespace
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip() + "\n"

    OUT_MD.write_text(md, encoding="utf-8")

    # 5. produce references.bib
    references_text = build_references_bib(cited, bibitems, bib_entries)
    OUT_BIB.write_text(references_text, encoding="utf-8")

    # 6. unrecognized log
    if UNRECOGNIZED:
        OUT_LOG.write_text("\n".join(UNRECOGNIZED) + "\n", encoding="utf-8")
    else:
        OUT_LOG.write_text("(no unrecognized constructs)\n", encoding="utf-8")

    print(f"wrote {OUT_MD} ({len(md)} chars)")
    print(f"wrote {OUT_BIB} ({len(references_text)} chars, {len(cited)} keys)")
    print(f"wrote {OUT_LOG} ({len(UNRECOGNIZED)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
