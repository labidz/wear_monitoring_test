"""Build the reference list in CIRP Annals style, in order of first citation (reads main.aux and refs.bib).

    [n] Surname I, Surname I (Year) Title. Journal Volume(Issue):first-last.
Writes manuscript/refs_cirp.tex (a thebibliography environment usable with natbib's numeric citations).
"""
_REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import re

ROOT = _REPO + "/manuscript"
ACCESSED = "accessed 1 November 2026"
JOURNAL = {"CIRP Ann": "CIRP Annals"}


def parse_bib(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{(\w+),(.*?)\}\s*$", text, flags=re.M):
        kind, key, body = m.group(1).lower(), m.group(2), m.group(3)
        fields = dict(re.findall(r"(\w+)=\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body))
        fields["_kind"] = kind
        out[key] = fields
    return out


def authors(a):
    names = [x.strip() for x in a.split(" and ")]
    res = []
    for n in names:
        if n.startswith("{") and n.endswith("}"):
            res.append(n[1:-1])
            continue
        if "," in n:
            last, first = [x.strip() for x in n.split(",", 1)]
            ini = "".join(p[0] for p in re.split(r"[\s.\-]+", first.replace("{", "").replace("}", "")) if p)
            res.append(f"{last} {ini}")
        else:
            res.append(n)
    return ", ".join(res)


def pages(p):
    p = p.replace("--", "-")
    if "-" in p:
        a, b = p.split("-", 1)
        if a.isdigit() and b.isdigit() and len(b) < len(a):
            b = a[: len(a) - len(b)] + b
        return f"{a}--{b}"
    return p


def entry(f):
    s = f"{authors(f['author'])} ({f['year']}) {f['title']}."
    k = f["_kind"]
    if k == "article":
        j = JOURNAL.get(f["journal"], f["journal"])
        s += f" {j} {f.get('volume', '')}"
        if f.get("number"):
            s += f"({f['number'].replace('--', '-')})"
        if f.get("pages"):
            s += f":{pages(f['pages'])}"
        s += "."
    elif k == "inproceedings":
        s += f" {f['booktitle']}"
        if f.get("volume"):
            s += f" {f['volume']}"
        s += "."
    else:
        s += f" {f.get('howpublished', '')}."
    if f.get("doi"):
        s += f" \\url{{https://doi.org/{f['doi']}}}."
    elif f["_kind"] == "misc":
        s += f" ({ACCESSED})."
    return s


def main():
    bib = parse_bib(open(f"{ROOT}/refs.bib").read())
    order = []
    for m in re.finditer(r"\\citation\{([^}]*)\}", open(f"{ROOT}/main.aux").read()):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    missing = [k for k in order if k not in bib]
    if missing:
        raise SystemExit(f"missing in refs.bib: {missing}")
    lines = [r"\begin{thebibliography}{" + str(len(order)) + "}"]
    lines += [rf"\bibitem{{{k}}} {entry(bib[k])}" for k in order]
    lines.append(r"\end{thebibliography}")
    open(f"{ROOT}/refs_cirp.tex", "w").write("\n".join(lines) + "\n")
    unused = sorted(set(bib) - set(order))
    print(len(order), "references cited; not cited:", unused)


if __name__ == "__main__":
    main()
