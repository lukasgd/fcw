"""E2E: the FirecREST ``view`` byte-range read that ``fcw job logs --follow`` relies on.

``_follow_stream`` (src/fcw/commands/job.py) drains a growing log in ``READ_CHUNK_BYTES`` ranged
``view`` reads: ``view(offset, size)`` returns exactly ``[offset, offset+size)``. This verifies
that one contract against a live system — chunked ranged reads reassemble a file byte-for-byte,
and a read past EOF returns empty (so the drain terminates). pyfirecrest's ``view()`` wrapper omits
``offset``/``size``, so we call the endpoint directly.

(The ~5 MB output cap of ``tail``/``head``/``view`` and the flakiness of large ``view`` reads are
recorded in the ``READ_CHUNK_BYTES`` comment in job.py — not re-characterized here, since the
follow loop depends only on the small ranged read.)

Run: ``pytest tests/e2e/test_e2e_view_range.py --run-e2e -v -s``.
"""

import pytest

CHUNK = 1024 * 1024  # matches READ_CHUNK_BYTES; small reads are reliable on clariden
FILE_SIZE = 2 * CHUNK + 500_000  # spans several chunks plus a partial final window


def _make_ascii(size: int) -> bytes:
    """Deterministic, line-numbered ASCII of exactly ``size`` bytes (64-byte lines)."""
    out = bytearray()
    n = 0
    while len(out) < size:
        line = f"L{n:08d} ".encode()
        line += b"." * (64 - len(line) - 1) + b"\n"
        out += line
        n += 1
    return bytes(out[:size])


def _view_range(client, system, path, *, offset, size) -> bytes:
    """Ranged view read [offset, offset+size). pyfirecrest's view() omits offset/size, so we
    hit the endpoint directly via the client's request helpers."""
    resp = client._get_request(
        endpoint=f"/filesystem/{system}/ops/view",
        params={"path": path, "offset": offset, "size": size})
    out = client._check_response(resp, 200)["output"]
    return (out or "").encode("utf-8")


@pytest.fixture
def uploaded(client, system, account, remote_workdir, tmp_path):
    """Upload a deterministic file and yield (content, remote_path); remove it afterwards."""
    content = _make_ascii(FILE_SIZE)
    name = "viewrange.txt"
    local = tmp_path / name
    local.write_bytes(content)
    client.mkdir(system_name=system, path=remote_workdir, create_parents=True)
    client.upload(system_name=system, local_file=str(local),
                  directory=remote_workdir, filename=name,
                  account=account, blocking=True, transfer_method="s3")
    path = f"{remote_workdir}/{name}"
    try:
        yield content, path
    finally:
        try:
            client.rm(system, path)
        except Exception:
            pass


class TestViewRangeFollow:
    def test_chunked_ranged_reads_reassemble_file(self, uploaded, client, system):
        """Draining a file in CHUNK-sized ranged view reads reconstructs it exactly.

        This is precisely what ``_follow_stream`` does: read ``[offset, offset+n)``, advance by the
        bytes returned, repeat. Byte-exact reassembly proves no skip, no overlap, no drift.
        """
        content, path = uploaded
        size = len(content)
        data = bytearray()
        offset = 0
        while offset < size:
            chunk = _view_range(client, system, path,
                                offset=offset, size=min(CHUNK, size - offset))
            assert chunk, f"ranged view returned no bytes at offset {offset}/{size}"
            data += chunk
            offset += len(chunk)
        assert bytes(data) == content

    def test_read_past_eof_is_empty(self, uploaded, client, system):
        """A read starting past EOF returns nothing, so the drain loop terminates cleanly."""
        content, path = uploaded
        assert _view_range(client, system, path, offset=len(content) + 1024, size=CHUNK) == b""
