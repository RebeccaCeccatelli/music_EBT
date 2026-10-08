"""
Download the `listening_grid` table an EBT guidance sweep
(attribute_control/listen_density_sweep.py) logs to wandb, and save it as
<out_dir>/<attribute>_<tokenizer>.table.json — the {columns, data} layout and
file naming that attribute_control/aggregate_guidance_sweeps.py and
eval/score_sweep_quality.py read (the attribute is taken from the filename).

The run is found from the sweep's SLURM log (the "View run at .../runs/<id>"
line wandb prints), or given directly with --run_id.

Usage:
    python eval/fetch_listening_table.py --slurm_log logs/slurm_25261488.out \\
        --attribute velocity --tokenizer remi --out_dir <dir>
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

ENTITY = "rceccatelli-eth-z-rich"
PROJECT = "mus_symb_attr_control"


def run_id_from_log(path: Path) -> str:
    m = re.findall(r"/runs/([a-z0-9]{8})", path.read_text(errors='replace'))
    if not m:
        raise SystemExit(f"no wandb run URL found in {path}")
    return m[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--slurm_log', type=Path)
    src.add_argument('--run_id')
    ap.add_argument('--attribute', required=True)
    ap.add_argument('--tokenizer', required=True, help='filename tag, e.g. remi or ant')
    ap.add_argument('--out_dir', type=Path, required=True)
    ap.add_argument('--project', default=PROJECT)
    args = ap.parse_args()

    import wandb
    run_id = args.run_id or run_id_from_log(args.slurm_log)
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{args.project}/{run_id}")
    arts = [a for a in run.logged_artifacts() if a.type == 'run_table' and 'listening_grid' in a.name]
    if not arts:
        raise SystemExit(f"run {run_id} ({run.state}) has no listening_grid table yet")
    art = arts[-1]
    # Download only the table JSON: the artifact also holds every sample's
    # audio (~1.4 GB for a full sweep).
    entry = next(k for k in art.manifest.entries if k.endswith('.table.json'))
    with tempfile.TemporaryDirectory() as tmp:
        table = json.loads(Path(art.get_entry(entry).download(root=tmp)).read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.attribute}_{args.tokenizer}.table.json"
    out.write_text(json.dumps({'columns': table['columns'], 'data': table['data']}))
    print(f"{run.name} ({run_id}): {len(table['data'])} rows, "
          f"sample_id column: {'sample_id' in table['columns']} → {out}")


if __name__ == '__main__':
    main()
