#!/usr/bin/env python3
"""Fill the interactive demo's song cache (demo/app.py, _SONG_CACHE_DIR) ahead of time.

For every named song x tokenizer: the time map, the full-song render (MP3 +
piano roll) and the preview at the song's default start point. With the cache
filled, a song change in the demo is a file lookup instead of a ~10 s
FluidSynth render. Resumable: anything already cached is skipped by the app's
own cache lookups.

    python demo/precompute_song_cache.py --workers 16          # all songs
    python demo/precompute_song_cache.py --limit 3 --workers 1 # quick check

Run with the demo env (music_EBT_demo: gradio + lame). CPU only.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/mus/symbolic"))
os.environ.setdefault("DEMO_NUM_THREADS", "1")  # one FluidSynth per worker; no thread pools fighting


def _one(job):
    import shutil
    import tempfile
    tok, song = job
    from demo import app
    t0 = time.time()
    # The app renders into tempfile.mkdtemp() dirs (full-song WAVs are ~30 MB);
    # the cache keeps only the MP3s, so each task's temp files go afterwards.
    tmp = tempfile.mkdtemp(prefix="songcache_", dir=os.environ.get("SONG_CACHE_TMP"))
    tempfile.tempdir = tmp
    try:
        app._get_time_map(tok, song)
        full = app._get_full_song_render(tok, song)
        slider_max, default_offset, _ = app.max_start_offset(tok, song)
        audio, status = app.preview_prompt(tok, song, min(default_offset, slider_max))
        ok = bool(full) and bool(audio) and str(audio).endswith(".mp3")
        return tok, song, ok, time.time() - t0, "" if ok else f"full={full} preview={status[:120]}"
    except Exception as e:
        return tok, song, False, time.time() - t0, repr(e)
    finally:
        tempfile.tempdir = None
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N songs (testing)")
    args = ap.parse_args()

    from demo import app
    songs = [s for s in app._SONG_CHOICES if s != app.RANDOM_SONG_LABEL]
    if args.limit:
        songs = songs[:args.limit]
    jobs = [(tok, s) for s in songs for tok in app.TOKENIZER_CHOICES]
    print(f"{len(jobs)} song x tokenizer pairs -> {app._SONG_CACHE_DIR}", flush=True)

    failed, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, (tok, song, ok, dt, err) in enumerate(pool.imap_unordered(_one, jobs), 1):
            if not ok:
                failed.append((tok, song, err))
            print(f"[{i}/{len(jobs)}] {'ok  ' if ok else 'FAIL'} {dt:5.1f}s {tok} | {song} {err}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed", flush=True)
    for f in failed:
        print("FAILED", *f, flush=True)


if __name__ == "__main__":
    main()
