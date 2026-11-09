"""
Paired bootstrap CIs for the unguided music-quality comparison
(docs/thesis_findings/2026-10-06_unguided_music_quality.md).

Every model continued the same validation songs, and each song's real
continuation is the reference. Songs are resampled jointly (the same draw for
every model and for the reference), the mean OA over the 8 quality metrics is
recomputed per model, and the difference to EBT gives a paired 95% CI.
Resampling with replacement duplicates pieces, which biases KDE overlap down
(the per-model CIs sit below the point estimates), so the main test is a paired
permutation test: per song, swap the EBT and baseline continuation with
probability 1/2 and recompute the OA difference (two-sided p).
Only songs that every model and the reference could score are used, so the
point estimates can differ slightly from the finding's per-model sets.

EBT's samples are numbered 000..099 in generation order; the baselines are named
by dataset index. They are matched through identical prompt tokens.
Reference: the baselines' real continuations (prompt + 256 tokens; GPT-2's and
Llama's are identical and checked). EBT's folder stores a differently cut ground
truth (228 vs 320 tokens on REMI), so it is not used: every model is compared
with the same reference.

Usage:
    python eval/unguided_quality_ci.py --root <music_quality dir> --tok remi --out ci_remi.json
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.music_quality import METRICS, overlapping_area

MODELS = ['ebt', 'gpt2', 'llama']


def read_scores(path: Path) -> dict:
    """sample id (the number after 'sample_') -> {metric: value}."""
    import csv
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            sid = int(re.match(r'sample_(\d+)_', r['file']).group(1))
            out[sid] = {m: float(r[m]) if r[m] != '' else np.nan for m in METRICS}
    return out


def ebt_song_ids(root: Path, tok: str) -> dict:
    """EBT file number -> dataset index, via identical prompts in the gpt2 run."""
    by_prompt = {}
    for f in glob.glob(str(root / f'gen_gpt2_{tok}' / 'tokens' / '*_prompt.json')):
        idx = int(re.search(r'sample_(\d+)_prompt', f).group(1))
        by_prompt[tuple(json.load(open(f)))] = idx
    gens = [json.loads(l) for l in open(root / f'gen_ebt_{tok}' / 'tokens' / 'generations.jsonl')]
    return {g['sample_idx']: by_prompt[tuple(g['prompt_tokens'])] for g in gens}


def mean_oa(gen: dict, ref: dict, songs: list) -> float:
    return float(np.nanmean([overlapping_area(np.array([gen[s][m] for s in songs]),
                                              np.array([ref[s][m] for s in songs])) for m in METRICS]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--tok', required=True, help="remi | ant-at-full-ar")
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--n_perm', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--models', default=','.join(MODELS),
                    help="gen_<model>_<tok> folders to compare, EBT first (e.g. "
                         "ebt,gpt2,llama,gpt2-ebtrecipe,llama-ebtrecipe)")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    models = args.models.split(',')
    assert models[0] == 'ebt', 'differences are taken against the first model, EBT'

    ebt_map = ebt_song_ids(root, args.tok)
    gen = {}
    for m in models:
        g = read_scores(root / f'gen_{m}_{args.tok}' / 'scores_generated.csv')
        gen[m] = {ebt_map[k]: v for k, v in g.items()} if m == 'ebt' else g
    gt = {m: read_scores(root / f'gen_{m}_{args.tok}' / 'scores_ground_truth.csv')
          for m in ['gpt2', 'llama'] + [m for m in models if m not in ('ebt', 'gpt2', 'llama')]}
    shared_gt = set(gt['gpt2']) & set(gt['llama'])
    for other in gt:  # same song, same real continuation → identical scores
        for s in set(gt['gpt2']) & set(gt[other]):
            assert np.allclose([gt['gpt2'][s][k] for k in METRICS], [gt[other][s][k] for k in METRICS],
                               equal_nan=True), (other, s)
    ref = {k: v for g in gt.values() for k, v in g.items()}
    songs = sorted(set.intersection(set(ref), *(set(gen[m]) for m in models)))
    print(f"{args.tok}: {len(songs)} songs scored by every model and the reference "
          f"(per model: {', '.join(f'{m} {len(gen[m])}' for m in models)}); GT agrees on {len(shared_gt)} shared songs")

    point = {m: mean_oa(gen[m], ref, songs) for m in models}
    rng = np.random.default_rng(args.seed)
    boots = {m: [] for m in models}
    for _ in range(args.n_boot):
        draw = [songs[i] for i in rng.integers(0, len(songs), len(songs))]
        for m in models:
            boots[m].append(mean_oa(gen[m], ref, draw))
    boots = {m: np.array(v) for m, v in boots.items()}
    res = {'tok': args.tok, 'n_songs': len(songs), 'n_boot': args.n_boot, 'n_perm': args.n_perm,
           'models': {}, 'diff_vs_ebt': {}}
    for m in models:
        res['models'][m] = {'mean_oa': point[m], 'ci95': np.percentile(boots[m], [2.5, 97.5]).tolist()}
        print(f"  {m:6s} mean OA {point[m]:.3f} [{res['models'][m]['ci95'][0]:.3f}, {res['models'][m]['ci95'][1]:.3f}]")
    for m in models[1:]:
        d = boots[m] - boots['ebt']
        r = {'diff': point[m] - point['ebt'], 'ci95': np.percentile(d, [2.5, 97.5]).tolist(),
             'frac_boot_le_0': float((d <= 0).mean())}
        null = []
        for _ in range(args.n_perm):
            swap = rng.random(len(songs)) < 0.5
            a = {s: gen['ebt'][s] if w else gen[m][s] for s, w in zip(songs, swap)}
            b = {s: gen[m][s] if w else gen['ebt'][s] for s, w in zip(songs, swap)}
            null.append(mean_oa(a, ref, songs) - mean_oa(b, ref, songs))
        r['p_perm'] = float((np.sum(np.abs(null) >= abs(r['diff']) - 1e-12) + 1) / (args.n_perm + 1))
        res['diff_vs_ebt'][m] = r
        print(f"  {m} − ebt: {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"permutation p = {r['p_perm']:.4f}")
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"→ {args.out}")


if __name__ == '__main__':
    main()
