"""
Playable listening grids for the AR guidance sweeps (attribute_control/
ar_guidance_sweep.py), logged to wandb in the same style as the EBT sweeps'
`listening_grid` tables, so GPT-2 / Llama guidance can be judged by ear next
to EBT's.

The AR sweeps only save MIDI. This renders a curated subset to WAV with the
same synth the EBT sweeps use (demo/convert_midi_simple.simple_synth): for
each attribute, a few prompts × {weakest, middle, strongest} strength ×
{push down, push up} at the largest target offset, plus the prompt's unguided
draw and its real continuation. Every clip is prompt + continuation.

One wandb run per sweep directory (= model × method), one table per
attribute: `listening_grid/<attribute>`.

Usage:
    python eval/ar_listening_grid.py <ar_out_dir> [<ar_out_dir> ...] \\
        --prompts 1326,4579,9938,13781 --wav_dir <scratch dir>
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demo'))

from convert_midi_simple import simple_synth

GUIDED_ORDER = {'down': 0, 'up': 1}


def pick_rows(rows: list[dict], prompts: set[int], offset: str) -> list[dict]:
    """Ground truth, the first unguided draw, and guided samples at the
    weakest / middle / strongest strength with target offset ±`offset` SD."""
    strengths = sorted({r['lambda'] for r in rows if r.get('target_delta') is not None})
    chosen = sorted({strengths[0], strengths[len(strengths) // 2], strengths[-1]})
    picked = []
    for r in rows:
        if r['prompt_id'] not in prompts:
            continue
        sid = r.get('sample_id') or ''
        if r.get('target_delta') is None:
            if str(r['condition']).startswith(('ground truth', 'baseline')):
                picked.append(r)
        elif r['lambda'] in chosen and (sid.endswith(f'_d-{offset}') or sid.endswith(f'_d+{offset}')):
            picked.append(r)

    def key(r):
        if r.get('target_delta') is None:
            return (r['prompt_id'], 0 if str(r['condition']).startswith('ground') else 1, 0, 0)
        return (r['prompt_id'], 2, r['lambda'], GUIDED_ORDER['up' if r['target_delta'] > 0 else 'down'])
    return sorted(picked, key=key)


def find_midi(midi_root: Path, sample_id: str) -> Path | None:
    for kind in ('generated', 'ground_truth'):
        hits = list(midi_root.rglob(f'{sample_id}_{kind}.mid'))
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sweep_dirs', nargs='+')
    ap.add_argument('--prompts', default='1326,4579,9938,13781')
    ap.add_argument('--offset', default='2', help='target offset in SD (as in sample ids, e.g. 2 or 0.5)')
    ap.add_argument('--wav_dir', required=True, help='where rendered WAVs are cached')
    ap.add_argument('--wandb_project', default='mus_symb_attr_control')
    args = ap.parse_args()

    import wandb
    prompts = {int(p) for p in args.prompts.split(',')}
    for sweep_dir in map(Path, args.sweep_dirs):
        config = json.loads(next(sweep_dir.glob('*.config.json')).read_text())
        name = f"{config.get('model_name', sweep_dir.name.split('_')[0])}"
        run_name = f"listen-ar-{sweep_dir.name}"
        run = wandb.init(project=args.wandb_project, name=run_name, job_type='ar_listening_grid',
                         config={'sweep_dir': str(sweep_dir), 'prompts': sorted(prompts),
                                 'offset_sd': args.offset, **{k: config.get(k) for k in
                                                              ('model_name', 'method', 'model_checkpoint')}},
                         reinit=True)
        wav_root = Path(args.wav_dir) / sweep_dir.name
        n_clips = 0
        for table_path in sorted(glob.glob(str(sweep_dir / '*.table.json'))):
            t = json.loads(Path(table_path).read_text())
            rows = [dict(zip(t['columns'], r)) for r in t['data']]
            if not rows:
                continue
            attr = rows[0]['attribute']
            cols = ['prompt_id', 'condition', 'strength', 'push', 'target', 'achieved_value',
                    'baseline_value', 'audio', 'sample_id']
            data = []
            for r in pick_rows(rows, prompts, args.offset):
                midi = find_midi(sweep_dir / 'midi', r['sample_id'])
                if midi is None:
                    continue
                wav = wav_root / attr / (midi.stem + '.wav')
                if not wav.exists():
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        simple_synth(str(midi), str(wav))
                    except Exception as e:
                        print(f"  render failed {midi.name}: {e}")
                        continue
                push = None if r.get('target_delta') is None else ('up' if r['target_delta'] > 0 else 'down')
                data.append([r['prompt_id'], r['condition'], r.get('lambda'), push, r.get('target'),
                             r.get('achieved_value'), r.get('baseline_value'),
                             wandb.Audio(str(wav), caption=f"p{r['prompt_id']} {r['condition']} {push or ''}"),
                             r['sample_id']])
            wandb.log({f'listening_grid/{attr}': wandb.Table(columns=cols, data=data)})
            n_clips += len(data)
            print(f"{run_name}: {attr}: {len(data)} clips")
        print(f"→ {run.url}  ({n_clips} clips, model {name})")
        run.finish()


if __name__ == '__main__':
    main()
