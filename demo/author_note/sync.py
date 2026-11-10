"""Copy author-note.js inline into pages, between <!--author-note--> and <!--/author-note--> markers.

    python3 demo/author_note/sync.py <page.html> [<page.html> ...]

A page without the markers gets them (with the script) just before </body>. Pages then only need a trigger:
any element with a data-author-note attribute, e.g. <button type="button" data-author-note></button>.
The interactive demo reads author-note.js directly (gr.Blocks head), so it needs no sync.
"""
import re
import sys
from pathlib import Path

JS = (Path(__file__).resolve().parent / "author-note.js").read_text()
BLOCK = "<!--author-note-->\n<script>\n" + JS.rstrip() + "\n</script>\n<!--/author-note-->"


def sync(path: Path) -> str:
    s = path.read_text()
    pat = re.compile(r"<!--author-note-->.*?<!--/author-note-->", re.S)
    if pat.search(s):
        new = pat.sub(lambda _: BLOCK, s)
    else:
        assert s.count("</body>") == 1, f"{path}: no single </body>"
        new = s.replace("</body>", BLOCK + "\n</body>")
    if new != s:
        path.write_text(new)
        return "updated"
    return "unchanged"


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, sync(Path(p)))
