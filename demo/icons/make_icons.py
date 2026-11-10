"""Browser-tab icons for the public sites (showcase, Listening Archive, Listening Study).

Writes <name>.svg and <name>.png (64 px, for browsers without SVG favicons) and prints nothing else.
`head_tags(name)` returns the <link> tags with both inlined as data URIs, so pages in any sub-folder
can use them without a relative path. Run: python demo/icons/make_icons.py
"""
import base64
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
BG = '<rect width="64" height="64" rx="14" fill="{c}"/>'
ICONS = {
    # showcase: two beamed eighth notes
    "showcase": ("#6d5ce8", '<g fill="#fff"><path d="M23 17l23-6v6l-23 6z"/><rect x="23" y="17" width="4" height="25"/>'
                 '<rect x="42" y="11" width="4" height="25"/><ellipse cx="19.5" cy="43" rx="7.5" ry="5.6" transform="rotate(-20 19.5 43)"/>'
                 '<ellipse cx="38.5" cy="37" rx="7.5" ry="5.6" transform="rotate(-20 38.5 37)"/></g>'),
    # archive: a record
    "archive": ("#2f5bd3", '<circle cx="32" cy="32" r="21" fill="#fff"/><circle cx="32" cy="32" r="15" fill="none" stroke="#2f5bd3" '
                'stroke-width="1.4" opacity=".55"/><circle cx="32" cy="32" r="8" fill="#2f5bd3"/><circle cx="32" cy="32" r="2.6" fill="#fff"/>'),
    # study (survey): headphones
    "study": ("#1d8a5c", '<path d="M15 38v-5a17 17 0 0 1 34 0v5" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/>'
              '<rect x="11" y="34" width="11" height="17" rx="4" fill="#fff"/><rect x="42" y="34" width="11" height="17" rx="4" fill="#fff"/>'),
}


def svg(name):
    c, body = ICONS[name]
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">{BG.format(c=c)}{body}</svg>'


def head_tags(name):
    s = "data:image/svg+xml," + urllib.parse.quote(svg(name))
    p = "data:image/png;base64," + base64.b64encode((HERE / f"{name}.png").read_bytes()).decode()
    return (f'<link rel="icon" type="image/svg+xml" href="{s}">'
            f'<link rel="icon" type="image/png" sizes="64x64" href="{p}">'
            f'<link rel="apple-touch-icon" href="{p}">')


if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 64, "height": 64})
        for n in ICONS:
            (HERE / f"{n}.svg").write_text(svg(n))
            pg.set_content(f'<body style="margin:0;background:transparent">{svg(n)}</body>')
            pg.locator("svg").screenshot(path=str(HERE / f"{n}.png"), omit_background=True)
        b.close()
