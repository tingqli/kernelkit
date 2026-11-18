"""``python -m kernelkit plot``: tuned tables as interactive charts in a browser.

    python -m kernelkit plot a8w4_moe_tuned.csv [more.csv ...]    # serve; open the printed URL
    python -m kernelkit plot axpy_tune:TUNING                      # the table of a Tuning
    python -m kernelkit plot axpy_tuned.csv --out axpy.html        # one self-contained file

The page is ``kernelkit/plot.html`` (Plotly, from its CDN). The rows with the same key values
(the columns before the ``~<dim>`` bucket column) form a group; per group the bucket goes on x,
``us`` on y, one line per config, and a strip shows the fastest config of every bucket range
(what the dispatch runs there). Clicking a point shows its row and every config at that point.

The server (standard library) binds 127.0.0.1 and serves the page at / and each table at
/<its file name>, nothing else. It reads a table again on every request and the page re-reads
it every 2 s (``live``), so a table being tuned fills in as it goes. Without the server, open
plot.html in a browser, then a local CSV (or drop it on the page), or ``plot.html?csv=<url>``.
"""

import argparse
import email.utils
import http.server
import json
import os
import socket
import sys
import urllib.parse
import webbrowser

__all__ = ["page", "embed", "serve", "main"]

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot.html")
EMBED = "/*KK_EMBED*/"  # where the page takes embedded tables


def page():
    """The viewer page (HTML)."""
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


def embed(tables):
    """The page with ``tables`` ([(name, csv text)]) inside it: one self-contained file."""
    # JSON with "<" escaped: nothing in a table can end the <script> it sits in
    data = json.dumps([{"name": n, "text": t} for n, t in tables]).replace("<", "\\u003c")
    html = page()
    if html.count(EMBED) != 1:
        raise RuntimeError(f"{PAGE}: expected one {EMBED}")
    return html.replace(EMBED, f"window.KK_EMBEDDED = {data};")


def _names(paths):
    """{URL name: absolute path}: the tables' file names, made unique."""
    out = {}
    for p in paths:
        stem, ext = os.path.splitext(os.path.basename(p))
        name, k = stem + ext, 2
        while name in out:
            name, k = f"{stem}_{k}{ext}", k + 1
        out[name] = os.path.abspath(p)
    return out


def serve(paths, host="127.0.0.1", port=8765):
    """A server of the page at / and of each table at /<its file name> (read on every
    request), nothing else; not started (``serve_forever``). It takes ``port`` or, if that
    is busy, one of the next 20 (``port=0``: any free one). Returns (server, the page's URL)."""
    files = _names(paths)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server's name)
            path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
            mtime = None
            try:
                if path in ("/", "/index.html"):
                    body, ctype = page().encode(), "text/html; charset=utf-8"
                elif path[1:] in files:
                    with open(files[path[1:]], "rb") as fh:
                        body = fh.read()
                    ctype, mtime = "text/csv; charset=utf-8", os.path.getmtime(files[path[1:]])
                else:
                    self.send_error(404)
                    return
            except OSError as e:
                self.send_error(404, e.strerror)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if mtime is not None:
                self.send_header("Last-Modified", email.utils.formatdate(mtime, usegmt=True))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # no line per request
            pass

    ports = [port] if port == 0 else range(port, port + 21)
    for i, p in enumerate(ports):
        try:
            server = http.server.ThreadingHTTPServer((host, p), Handler)
            break
        except OSError:
            if i == len(ports) - 1:
                raise
    shown = socket.gethostname() if host in ("", "0.0.0.0") else host
    query = urllib.parse.urlencode([("csv", n) for n in files] + [("live", "1")])
    return server, f"http://{shown}:{server.server_address[1]}/?{query}"


def _table(spec):
    """A table's path: a file, else ``<module>:<Tuning name>`` (the tuning's table)."""
    if os.path.exists(spec) or ":" not in spec:
        return spec
    from .tune import resolve  # imports torch and the op's module
    t = resolve(spec)
    if not t.table:
        raise SystemExit(f"{spec}: the tuning has no table")
    return t.table


def parser(prog="python -m kernelkit plot"):
    p = argparse.ArgumentParser(prog=prog, description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tables", nargs="+", metavar="TABLE",
                   help="tuned tables: CSV files, or <module>:<Tuning name> (its table)")
    p.add_argument("--out", metavar="FILE.html",
                   help="write one self-contained HTML file (the tables inside) instead of serving")
    p.add_argument("--host", default="127.0.0.1",
                   help="the address to serve on (default 127.0.0.1: this machine only)")
    p.add_argument("--port", type=int, default=8765, help="the first port to try (0: any free one)")
    p.add_argument("--open", action="store_true", help="also open the page in a browser here")
    return p


def main(argv=None, prog="python -m kernelkit plot"):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["plot"]:
        argv = argv[1:]
    a = parser(prog).parse_args(argv)
    paths = [_table(s) for s in a.tables]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"no such table: {missing}")
    if a.out:
        tables = []
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                tables.append((os.path.basename(p), fh.read()))
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(embed(tables))
        print(f"wrote {a.out} ({len(tables)} table{'s' * (len(tables) > 1)} inside): open it in a browser")
        return 0
    server, url = serve(paths, a.host, a.port)
    print(f"kernelkit plot: {url}")
    for name, path in _names(paths).items():
        print(f"  /{name} <- {path}")
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  serving on {a.host}: whoever reaches this machine can read these tables")
    print("  (each table is read again on every request; Ctrl-C to stop)", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
