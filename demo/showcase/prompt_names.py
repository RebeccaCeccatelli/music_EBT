"""Descriptive names for showcase prompts, from the music itself (the MIDI has no titles).

    name = describe(original_midi)   # e.g. "Moody jazz combo with flute"

Mood from an estimated mode (Krumhansl-Schmuckler key profiles on duration-weighted pitch classes)
and note density; ensemble from the General MIDI programs; lead = the busiest melodic instrument.
Heuristic labels for the reader, not analysis results.
"""

from collections import Counter

import symusic

MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

LEAD = {0: "piano", 1: "piano", 2: "piano", 3: "piano", 4: "electric piano", 5: "electric piano",
        6: "harpsichord", 11: "vibraphone", 24: "nylon guitar", 25: "acoustic guitar", 26: "jazz guitar",
        27: "clean guitar", 28: "muted guitar", 29: "overdriven guitar", 30: "distorted guitar",
        40: "violin", 41: "viola", 42: "cello", 46: "harp", 48: "strings", 49: "strings", 52: "choir",
        56: "trumpet", 57: "trombone", 60: "horn", 61: "brass", 64: "sax", 65: "sax", 66: "sax", 67: "sax",
        68: "oboe", 70: "bassoon", 71: "clarinet", 73: "flute", 74: "recorder", 75: "pan flute",
        105: "banjo", 108: "kalimba", 112: "bells", 114: "steel drums"}
FAMILY_LEAD = ["piano", "keys", "organ", "guitar", None, "strings", "strings", "brass", "sax", "flute",
               "synth lead", "synth pad", "synth", "world instruments", None, None]


def _lead_name(program):
    return LEAD.get(program) or FAMILY_LEAD[program // 8] or "melody"


def _corr(x, y):
    mx, my = sum(x) / 12, sum(y) / 12
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else 0.0


def features(path):
    score = symusic.Score(str(path)).to("second")
    pitched, drums, onsets = [], 0, set()
    for t in score.tracks:
        for n in t.notes:
            onsets.add(round(n.time, 2))
            if t.is_drum:
                drums += 1
            else:
                pitched.append((t.program, n.pitch, n.duration))
    end = max((n.time + n.duration for t in score.tracks for n in t.notes), default=1.0)
    pc = [0.0] * 12
    for _, p, d in pitched:
        pc[p % 12] += d
    best = max(((_corr(pc, prof[-k:] + prof[:-k]), mode) for prof, mode in ((MAJOR, "major"), (MINOR, "minor"))
                for k in range(12)), default=(0, "major"))
    return {"programs": Counter(p for p, _, _ in pitched), "pitched": len(pitched), "drums": drums,
            "density": len(onsets) / max(end, 1e-6), "mode": best[1],
            "mean_pitch": {pr: sum(p for q, p, _ in pitched if q == pr) / c
                           for pr, c in Counter(p for p, _, _ in pitched).items()}}


MOOD = {("major", "slow"): "Easygoing", ("major", "mid"): "Bright", ("major", "fast"): "Upbeat",
        ("minor", "slow"): "Wistful", ("minor", "mid"): "Moody", ("minor", "fast"): "Restless"}


def describe(path):
    f = features(path)
    total = f["pitched"] + f["drums"]
    progs = f["programs"]
    if not total:
        return "Silence"
    share = lambda pred: sum(c for p, c in progs.items() if pred(p)) / max(f["pitched"], 1)
    has_drums = f["drums"] > 0.08 * total
    pace = "fast" if f["density"] > 6.5 else "slow" if f["density"] < 3 else "mid"
    mood = MOOD[(f["mode"], pace)]
    if f["pitched"] < 0.05 * total:
        return {"slow": "Laid-back", "mid": "Steady", "fast": "Driving"}[pace] + " drum groove"

    # melodic instruments (not bass 32-39), busiest first, one name per instrument
    names = []
    for c, p in sorted(((c, p) for p, c in progs.items() if not 32 <= p <= 39), reverse=True):
        n = _lead_name(p)
        if n not in names and c >= 0.08 * f["pitched"]:
            names.append(n)
    lead = names[0] if names else "bass"
    if len(names) > 1 and "guitar" in names[0] and "guitar" in names[1]:
        names = ["guitar"] + names[2:]   # two kinds of guitar: just "guitar"
    pair = " & ".join(names[:2]) if names else "bass"

    if share(lambda p: p <= 7) > 0.85 and not has_drums:
        return f"{mood} solo piano"
    if share(lambda p: 40 <= p <= 79) > 0.6 and not has_drums:
        return f"{mood} {pair} ensemble"
    if share(lambda p: p in (29, 30)) > 0.15:
        return f"{mood} rock band" + (f" with {lead}" if "guitar" not in lead else "")
    if share(lambda p: p == 26 or 64 <= p <= 67 or 56 <= p <= 58) > 0.2 and progs.get(32, 0):
        return f"{mood} {lead} combo" if lead in ("jazz guitar", "sax", "trumpet") else f"{mood} jazz combo with {lead}"
    if share(lambda p: 80 <= p <= 103) > 0.3:
        return f"{mood} synth groove"
    return f"{mood} {pair} " + ("groove" if has_drums else "duet" if len(names) == 2 else "piece")
