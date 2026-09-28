# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyannote.metrics>=4"]
# ///
"""Score `hark.replay` dumps: diarization error and cross-slot duplicate words.

    uv run scripts/diar_eval.py RUN_DIR [RUN_DIR…] [--ref REF] [--json OUT]

REF is an RTTM (exact speaker timings, e.g. AMI's word-aligned RTTMs) or a Zoom
VTT (caption-level timings, reliable names); by default it is looked up from the
run's meta.json audio path (<stem>.rttm, <stem>.transcript.vtt). The hypothesis is
the ASR mask itself: slot k is active in an 80 ms frame when its mean diarizer
probability over that frame exceeds the session threshold.

Per word emitted by a slot, the scorer records whether that slot's mask was open
around it and, given a reference, whether the slot's mapped reference speaker was
speaking. A duplicate is a word also emitted by another slot within ±DUP_WINDOW s.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

FRAME = 0.08
DUP_WINDOW = 1.5
LOOKBACK = 0.48  # an RNN-T word is emitted up to this long after its audio starts


def activity(probs, threshold=0.5, factor=8):
    padded = np.pad(probs, [(0, -len(probs) % factor), (0, 0)])
    return padded.reshape(-1, factor, probs.shape[1]).mean(1) > threshold


def gate(probs, threshold=0.5, overlap=None, factor=8):
    """hark.masking.MaskPolicy on raw 10 ms probabilities."""
    padded = np.pad(probs, [(0, -len(probs) % factor), (0, 0)])
    p = padded.reshape(-1, factor, probs.shape[1]).mean(1)
    active = p > threshold
    if overlap is not None:
        active &= (p == p.max(1, keepdims=True)) | (p >= overlap)
    return active


def annotation(act, offset=0.0):
    ann = Annotation()
    for k in range(act.shape[1]):
        edges = np.flatnonzero(np.diff(np.r_[0, act[:, k].astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ann[Segment(offset + a * FRAME, offset + b * FRAME), f"_{k}"] = f"speaker_{k}"
    return ann


def words(tokens):
    out = {}
    for t in sorted(tokens, key=lambda t: (t["speaker"], t["start"])):
        ws = out.setdefault(t["speaker"], [])
        if not ws or t["text"].startswith(" ") or not t["text"].strip():
            ws.append(dict(speaker=t["speaker"], start=t["start"], end=t["end"], text=t["text"]))
        else:
            ws[-1]["end"], ws[-1]["text"] = t["end"], ws[-1]["text"] + t["text"]
    flat = [w | {"norm": re.sub(r"[^\w']", "", w["text"].lower())} for ws in out.values() for w in ws]
    return [w for w in flat if w["norm"]]


def read_rttm(path, start, duration):
    ann = Annotation()
    for i, line in enumerate(Path(path).read_text().splitlines()):
        f = line.split()
        a, b = float(f[3]) - start, float(f[3]) + float(f[4]) - start
        if b > 0 and a < duration:
            ann[Segment(max(a, 0), min(b, duration)), i] = f[7]
    return ann


def read_vtt(path, start, duration):
    ann, texts = Annotation(), []
    stamp = lambda s: sum(float(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
    for i, block in enumerate(Path(path).read_text().split("\n\n")):
        m = re.search(r"([\d:.]+) --> ([\d:.]+)\n([^:\n]+): ", block)
        if not m:
            continue
        a, b = stamp(m[1]) - start, stamp(m[2]) - start
        if b > 0 and a < duration:
            ann[Segment(max(a, 0), min(b, duration)), i] = m[3].strip()
    return ann


def ref_words(path, start, duration):
    """Reference words (speaker, start, end, norm): AMI word XML beside an RTTM, or VTT cue text."""
    path = Path(path)
    out = []
    if path.suffix == ".rttm":
        meeting = path.stem
        names = dict(re.findall(rf'nite:id="{meeting}_\d" channel="\d" nxt_agent="(\w)"[^>]*global_name="(\w+)"',
                                (path.parent / "corpusResources/meetings.xml").read_text()))
        for xml in sorted(path.parent.glob(f"words/{meeting}.*.words.xml")):
            agent = xml.name.split(".")[1]
            for a, b, text in re.findall(r'<w [^>]*starttime="([\d.]+)" endtime="([\d.]+)"(?![^>]*punc)[^>]*>([^<]*)</w>',
                                         xml.read_text(encoding="latin-1")):
                out.append((names[agent], float(a) - start, float(b) - start, text))
    else:
        stamp = lambda s: sum(float(x) * m for x, m in zip(s.split(":"), (3600, 60, 1)))
        for block in path.read_text().split("\n\n"):
            m = re.search(r"([\d:.]+) --> ([\d:.]+)\n([^:\n]+): (.*)", block, re.S)
            if not m:
                continue
            a, b, toks = stamp(m[1]) - start, stamp(m[2]) - start, m[4].split()
            out += [(m[3].strip(), a + (b - a) * i / len(toks), a + (b - a) * (i + 1) / len(toks), t)
                    for i, t in enumerate(toks)]
    out = [(spk, a, b, re.sub(r"[^\w']", "", t.lower())) for spk, a, b, t in out]
    return [r for r in out if r[3] and r[2] > 0 and r[1] < duration]


def attribute(ws, rws, ref, act, mapping, window=1.5, pad=0.16):
    """Who really said each emitted word, and what the emitting slot's mask was doing then.

    foreign: the word was said (per reference) by someone other than the slot's mapped speaker.
    For its source audio span: leak = the slot's mask was closed throughout; fa = open while
    the slot's own speaker was silent; overlap = open while the slot's own speaker also talked.
    """
    by_norm = {}
    for r in rws:
        by_norm.setdefault(r[3], []).append(r)
    for w in ws:
        cands = [r for r in by_norm.get(w["norm"], []) if r[1] - window <= w["start"] <= r[2] + window]
        own = mapping.get(w["speaker"])
        w["truth"] = "unmatched" if not cands else "own" if any(r[0] == own for r in cands) else "foreign"
        if w["truth"] != "foreign":
            continue
        src = min(cands, key=lambda r: abs(r[1] - w["start"]))
        k = int(w["speaker"].rsplit("_", 1)[1])
        a, b = int(max(src[1] - pad, 0) / FRAME), int((src[2] + pad) / FRAME) + 1
        seg = Segment(src[1], max(src[2], src[1] + 0.01))
        own_talking = own in ref.labels() and ref.crop(seg).label_timeline(own).duration() > 0
        w["source"] = src[0]
        w["cause"] = "leak" if not act[a:b, k].any() else "overlap" if own_talking else "fa"
        w["source_open_frac"] = float(act[a:b, k].mean())
    slot_of = {v: k for k, v in mapping.items()}
    hyp = {}
    for w in ws:
        hyp.setdefault(w["norm"], []).append(w)
    near = lambda r, w: r[1] - window <= w["start"] <= r[2] + window
    heard = [any(near(r, w) for w in hyp.get(r[3], [])) for r in rws]
    attributed = [any(near(r, w) and w["speaker"] == slot_of.get(r[0]) for w in hyp.get(r[3], [])) for r in rws]
    foreign = [w for w in ws if w["truth"] == "foreign"]
    recall = {"recall_any": np.mean(heard) if rws else None, "recall_attributed": np.mean(attributed) if rws else None}
    matched = [w for w in ws if w["truth"] != "unmatched"]
    causes = Counter(w["cause"] for w in foreign)
    return recall | {"matched": len(matched), "foreign_rate": len(foreign) / max(len(matched), 1),
            "foreign_dup": np.mean([w["dup"] for w in foreign]) if foreign else None,
            "foreign_cause": {c: round(n / len(foreign), 3) for c, n in causes.items()} if foreign else {},
            "dup_foreign": np.mean([w["truth"] == "foreign" for w in ws if w["dup"] and w["truth"] != "unmatched"])
            if any(w["dup"] for w in ws) else None}


def default_ref(meta):
    audio = Path(meta["audio"])
    stem = audio.name.split(".")[0]
    for cand in (audio.with_name(f"{stem}.rttm"), audio.with_name(f"{audio.stem}.transcript.vtt")):
        if cand.exists():
            return cand
    return None


def score(run, ref_path=None):
    run = Path(run)
    meta = json.loads((run / "meta.json").read_text())
    probs = np.load(run / "probs.npy")
    tokens = [json.loads(l) for l in (run / "tokens.jsonl").read_text().splitlines()]
    act = activity(probs)  # the diarizer's own decision, scored for DER
    mask = gate(probs, **(meta.get("mask") or {}))  # the frames each ASR stream heard
    dur = meta["duration"]
    hyp = annotation(act)
    ws = words(tokens)

    # duplicates: same normalized word in another slot within ±DUP_WINDOW
    for w in ws:
        w["dup"] = any(v["norm"] == w["norm"] and v["speaker"] != w["speaker"]
                       and abs(v["start"] - w["start"]) <= DUP_WINDOW for v in ws)
        k = int(w["speaker"].rsplit("_", 1)[1])
        a, b = int(max(w["start"] - LOOKBACK, 0) / FRAME), int(w["end"] / FRAME) + 1
        w["mask_open"] = bool(mask[a:b, k].any())
        w["n_active"] = int(mask[a:b].any(0).sum())
    words_by_slot = Counter(w["speaker"] for w in ws)
    out = {"run": run.name, "duration": round(dur, 1), "slots_used": len(words_by_slot),
           "words": len(ws), "dup_rate": np.mean([w["dup"] for w in ws]) if ws else 0.0,
           "words_by_slot": dict(words_by_slot)}
    out["dup_mask_open"] = np.mean([w["mask_open"] for w in ws if w["dup"]]) if any(w["dup"] for w in ws) else None
    out["frames_multi"] = float((mask.sum(1) > 1).mean())
    out["frames_speech"] = float((mask.sum(1) > 0).mean())

    ref_path = ref_path or default_ref(meta)
    if ref_path:
        ref_path = Path(ref_path)
        ref = (read_rttm if ref_path.suffix == ".rttm" else read_vtt)(ref_path, meta["start"], dur)
        uem = Timeline([Segment(0, dur)])
        for collar in (0.0, 0.25):
            der = DiarizationErrorRate(collar=collar, skip_overlap=False)
            d = der(ref, hyp, uem=uem, detailed=True)
            tot = d["total"]
            out[f"der@{collar}"] = {"der": round(d["diarization error rate"], 4)} | {
                k.split()[0]: round(d[k] / tot, 4) for k in ("missed detection", "false alarm", "confusion")}
        mapping = DiarizationErrorRate().optimal_mapping(ref, hyp)  # hyp label -> ref label
        out["mapping"] = dict(mapping)
        out |= attribute(ws, ref_words(ref_path, meta["start"], dur), ref, mask, mapping)
        out["ref"] = str(ref_path)
    (run / "words.jsonl").write_text("".join(json.dumps(w) + "\n" for w in ws))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    results = [score(r, args.ref) for r in args.runs]
    for r in results:
        print(json.dumps(r))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
