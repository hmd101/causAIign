"""
Overlays engine for RW17 domains.

Applies YAML-defined overlays to create cloned domain variants with appended or
overridden content for introductions, variable detailed texts, and explanations.

Supported directives (per overlay entry):
- domain: <base_domain>
- new_name: <clone_domain_name>
- introduction_append: <str>
- introduction_append_from: [<other_domain>, ...]
- variables:
    X|Y|Z:
      detailed_append: <str>
      detailed_append_from: [<other_domain>, ...]
      detailed_filler_words: <int>
      explanations_append: {p_p: str, p_m: str, m_p: str, m_m: str}
      explanations_append_from: [<other_domain>, ...]
      explanations_filler_words: <int>

Notes:
- Explanations are only consumed from variables X and Y by verbalization. If an overlay
  targets Z.explanations*, we mirror the resulting updates onto X and Y.
- "filler_words" appends neutral text of the requested word count and ensures a final period.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, MutableMapping


EXPL_KEYS = ("p_p", "p_m", "m_p", "m_m")


def _ensure_trailing_period(text: str) -> str:
    text = (text or "").rstrip()
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    return text + "."


def _neutral_filler(n_words: int) -> str:
    if n_words <= 0:
        return ""
    # Construct neutral filler from a lorem-ipsum style vocabulary (no domain content).
    vocab = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
        "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
        "magna", "aliqua", "ut", "enim", "ad", "minim", "veniam", "quis", "nostrud",
        "exercitation", "ullamco", "laboris", "nisi", "ut", "aliquip", "ex", "ea", "commodo",
        "consequat", "duis", "aute", "irure", "dolor", "in", "reprehenderit", "in", "voluptate",
        "velit", "esse", "cillum", "dolore", "eu", "fugiat", "nulla", "pariatur", "excepteur",
        "sint", "occaecat", "cupidatat", "non", "proident", "sunt", "in", "culpa", "qui",
        "officia", "deserunt", "mollit", "anim", "id", "est", "laborum",
    ]
    words: List[str] = []
    i = 0
    while len(words) < n_words:
        words.append(vocab[i % len(vocab)])
        i += 1
    filler = " ".join(words)
    # Sentence-case and ensure trailing period.
    if filler:
        filler = filler[0].upper() + filler[1:]
    return _ensure_trailing_period(filler)


def _append_text(base: str, addition: str) -> str:
    if not addition:
        return base
    base = base or ""
    if base:
        sep = " " if not base.endswith("\n") else ""
        out = base + sep + addition.strip()
    else:
        out = addition.strip()
    return _ensure_trailing_period(out)


def _append_detailed_from(
    target: MutableMapping[str, Any], source: Mapping[str, Any], var_key: str
) -> None:
    tvar = target["variables"].get(var_key, {})
    svar = source["variables"].get(var_key, {})
    target_text = tvar.get("detailed", "")
    source_text = svar.get("detailed", "")
    tvar["detailed"] = _append_text(target_text, source_text)
    target["variables"][var_key] = tvar


def _append_explanations_from_pair(
    tvar: MutableMapping[str, Any], svar: Mapping[str, Any]
) -> None:
    # Ensure explanations dict present on target.
    texpl = dict(tvar.get("explanations", {}))
    sexpl = svar.get("explanations", {}) or {}
    for k in EXPL_KEYS:
        if k in sexpl:
            texpl[k] = _append_text(texpl.get(k, ""), sexpl[k])
    if texpl:
        tvar["explanations"] = texpl


def _append_explanations_from(
    target: MutableMapping[str, Any], source: Mapping[str, Any], var_key: str
) -> None:
    # For X/Y: append from same var in source.
    if var_key in ("X", "Y"):
        tvar = target["variables"].get(var_key, {})
        svar = source["variables"].get(var_key, {})
        _append_explanations_from_pair(tvar, svar)
        target["variables"][var_key] = tvar
        return
    # For Z: mirror append into X and Y using source X and Y.
    if var_key == "Z":
        for k in ("X", "Y"):
            tvar = target["variables"].get(k, {})
            svar = source["variables"].get(k, {})
            _append_explanations_from_pair(tvar, svar)
            target["variables"][k] = tvar


def _apply_detailed_filler(target: MutableMapping[str, Any], var_key: str, n_words: int) -> None:
    tvar = target["variables"].get(var_key, {})
    tvar["detailed"] = _append_text(tvar.get("detailed", ""), _neutral_filler(n_words))
    target["variables"][var_key] = tvar


def _apply_explanations_filler(target: MutableMapping[str, Any], var_key: str, n_words: int) -> None:
    filler = _neutral_filler(n_words)
    def append_to_var(vkey: str) -> None:
        tvar = target["variables"].get(vkey, {})
        texpl = dict(tvar.get("explanations", {}))
        for k in EXPL_KEYS:
            texpl[k] = _append_text(texpl.get(k, ""), filler)
        tvar["explanations"] = texpl
        target["variables"][vkey] = tvar

    if var_key in ("X", "Y"):
        append_to_var(var_key)
    elif var_key == "Z":
        # Mirror onto X and Y
        append_to_var("X")
        append_to_var("Y")


def create_overloaded_domains(
    base_domains: Mapping[str, Mapping[str, Any]], overlays: List[Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Apply overlays to base RW17 domain components and return a new mapping
    including cloned domains under their new_name keys.

    base_domains: mapping of domain_name -> domain_component dict
    overlays: list of overlay directive dicts (parsed from YAML)
    """
    result: Dict[str, Dict[str, Any]] = {k: deepcopy(dict(v)) for k, v in base_domains.items()}

    for ov in overlays or []:
        try:
            base = ov.get("domain") or ov.get("domain_name")
            new_name = ov.get("new_name") or (f"{base}_overloaded" if base else None)
            if not base or not new_name:
                continue
            if base not in base_domains:
                # Unknown base domain; skip
                continue
            clone: Dict[str, Any] = deepcopy(dict(base_domains[base]))

            # Introduction appends
            intro = clone.get("introduction", "")
            if isinstance(ov.get("introduction_append"), str):
                intro = _append_text(intro, ov["introduction_append"])
            for s in ov.get("introduction_append_from", []) or []:
                if s in base_domains:
                    intro = _append_text(intro, base_domains[s].get("introduction", ""))
            if intro:
                clone["introduction"] = _ensure_trailing_period(intro)

            # Variable-level operations
            vops = ov.get("variables") or {}
            if isinstance(vops, dict):
                for vkey, instr in vops.items():
                    if vkey not in ("X", "Y", "Z") or not isinstance(instr, dict):
                        continue
                    # detailed_append literal
                    if isinstance(instr.get("detailed_append"), str):
                        tvar = clone["variables"].get(vkey, {})
                        tvar["detailed"] = _append_text(tvar.get("detailed", ""), instr["detailed_append"])
                        clone["variables"][vkey] = tvar
                    # detailed_append_from
                    for s in instr.get("detailed_append_from", []) or []:
                        if s in base_domains:
                            _append_detailed_from(clone, base_domains[s], vkey)
                    # detailed_filler_words
                    if isinstance(instr.get("detailed_filler_words"), int):
                        _apply_detailed_filler(clone, vkey, int(instr["detailed_filler_words"]))

                    # explanations_append literal dict
                    if isinstance(instr.get("explanations_append"), dict):
                        # Append provided keys
                        if vkey in ("X", "Y"):
                            tvar = clone["variables"].get(vkey, {})
                            texpl = dict(tvar.get("explanations", {}))
                            for k, text in instr["explanations_append"].items():
                                if k in EXPL_KEYS and isinstance(text, str):
                                    texpl[k] = _append_text(texpl.get(k, ""), text)
                            tvar["explanations"] = texpl
                            clone["variables"][vkey] = tvar
                        elif vkey == "Z":
                            # Mirror onto X and Y
                            for dest in ("X", "Y"):
                                tvar = clone["variables"].get(dest, {})
                                texpl = dict(tvar.get("explanations", {}))
                                for k, text in instr["explanations_append"].items():
                                    if k in EXPL_KEYS and isinstance(text, str):
                                        texpl[k] = _append_text(texpl.get(k, ""), text)
                                tvar["explanations"] = texpl
                                clone["variables"][dest] = tvar

                    # explanations_append_from
                    for s in instr.get("explanations_append_from", []) or []:
                        if s in base_domains:
                            _append_explanations_from(clone, base_domains[s], vkey)

                    # explanations_filler_words
                    if isinstance(instr.get("explanations_filler_words"), int):
                        _apply_explanations_filler(clone, vkey, int(instr["explanations_filler_words"]))

            # Finalize clone metadata
            clone["domain_name"] = new_name

            # Register clone
            result[new_name] = clone
        except Exception:
            # Best-effort: skip faulty overlay entry
            continue

    return result
