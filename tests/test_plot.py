"""CPU tests of ``python -m kernelkit plot``: the self-contained page and the server."""

import base64
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request

import pytest

from kernelkit import plot

CSV = ("dtype,~n,config,us,err\n"
       "bf16,1000,tiled b1024,6.0,2.2e-03\n"
       "bf16,1000,torch,6.1,2.2e-03\n"
       "bf16,4096,torch,5.9,2.0e-03\n")


def test_page_has_one_placeholder():
    assert plot.page().count(plot.EMBED) == 1


def test_plotly_integrity_is_the_cdn_files():
    """A wrong integrity hash makes the browser refuse Plotly: no chart (network; skipped offline)."""
    url, want = re.search(r'src="(https://cdn\.plot\.ly/[^"]+)"\s+integrity="sha384-([^"]+)"', plot.page()).groups()
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            body = r.read()
    except OSError as e:
        pytest.skip(f"cannot fetch {url}: {e}")
    assert base64.b64encode(hashlib.sha384(body).digest()).decode() == want


def test_embed_keeps_the_tables_inside_their_script():
    evil = CSV + 'bf16,8192,"</script><script>alert(1)</script>",5.0,0\n'
    html = plot.embed([("a.csv", evil), ("b.csv", CSV)])
    assert html.count("</script>") == plot.page().count("</script>")
    data = re.search(r"window\.KK_EMBEDDED = (.*?);</script>", html, re.S).group(1)
    assert json.loads(data) == [{"name": "a.csv", "text": evil}, {"name": "b.csv", "text": CSV}]


def test_out_writes_one_file(tmp_path, capsys):
    t = tmp_path / "axpy_tuned.csv"
    t.write_text(CSV)
    out = tmp_path / "axpy.html"
    assert plot.main(["plot", str(t), "--out", str(out)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert '"name": "axpy_tuned.csv"' in out.read_text()
    with pytest.raises(SystemExit, match="no such table"):
        plot.main([str(tmp_path / "missing.csv")])


def test_server_serves_the_page_and_the_tables_only(tmp_path):
    a = tmp_path / "x" / "t_tuned.csv"
    b = tmp_path / "y" / "t_tuned.csv"  # the same file name: served as t_tuned_2.csv
    for p in (a, b):
        p.parent.mkdir()
        p.write_text(CSV)
    server, url = plot.serve([str(a), str(b)], "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.read().decode(), r.headers

    try:
        assert url == f"{base}/?csv=t_tuned.csv&csv=t_tuned_2.csv&live=1"
        page, headers = get("/?csv=t_tuned.csv")
        assert plot.EMBED in page and headers["Cache-Control"] == "no-store"
        text, headers = get("/t_tuned.csv")
        assert text == CSV and headers["Content-Type"].startswith("text/csv") and headers["Last-Modified"]
        b.write_text(CSV + "bf16,8192,torch,7.0,1e-3\n")  # read again on every request
        assert get("/t_tuned_2.csv")[0].endswith("7.0,1e-3\n")
        for bad in ("/plot.py", "/x/t_tuned.csv", "/../t_tuned.csv", "/%2e%2e/t_tuned.csv", "/etc/passwd"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(bad)
            assert e.value.code == 404, bad
    finally:
        server.shutdown()
        server.server_close()
