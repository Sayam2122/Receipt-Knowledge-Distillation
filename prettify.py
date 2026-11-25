"""Display-only: restore spaces that the OCR engine dropped (the scored value is compared space-insensitively).
Uses wordninja if installed (pip install wordninja), after splitting off common Malaysian company suffixes."""
import re

_SUFFIX = re.compile(r"(?<=[A-Za-z])(SDN\.?BHD\.?|SDN|BHD|ENTERPRISE|TRADING|RESTAURANT|PLT|BERHAD)(?=[^a-z]|$)", re.I)


def _split_chunk(chunk):
    try:
        import wordninja
    except ImportError:
        return chunk
    toks = wordninja.split(chunk)
    out = []
    for t in toks:                       # glue runs of 1-2 letter fragments back together (O JC -> OJC)
        if out and len(t) <= 2 and len(out[-1]) <= 3 and not out[-1].isdigit() and not t.isdigit():
            out[-1] += t
        else:
            out.append(t)
    return " ".join(out)


def prettify(text):
    """Only touches strings that look unspaced (long, no spaces, mostly letters)."""
    t = (text or "").strip()
    if len(t) < 10 or " " in t:
        return t
    t = _SUFFIX.sub(lambda m: " " + ("SDN BHD" if re.sub(r"[.\s]", "", m.group(1)).upper() == "SDNBHD" else m.group(1)) + " ", t)
    parts = [p for p in re.split(r"\s+", t) if p]
    return " ".join(p if len(p) < 8 or not p.isalnum() else _split_chunk(p) for p in parts)
