"""Complete episodes of partially downloaded Memory Maze training archives -> a training subset.

Google Drive refuses most requests for the official training archives (download quota), so each
train-part*.zip holds only its first 0.5-0.7 GB. A zip stores every member behind a local header,
so the members before the cut are complete; their bytes are inflated and checked against the
CRC of the header. Output: DEST/train/*.npz and DEST/eval.zip -> the official evaluation archive,
the layout probe_mm.py reads.

usage: python scripts/data/salvage_memory_maze.py [SRC] [DEST]
"""
import pathlib
import struct
import sys
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "memory_maze_9x9_official"
DEST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "memory_maze_9x9_official_subset"


def members(path):
    """(name, method, crc, compressed bytes) of the complete members of a possibly truncated zip."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        pos = 0
        while True:
            f.seek(pos)
            head = f.read(30)
            if len(head) < 30 or head[:4] != b"PK\x03\x04":
                return
            _, _, flag, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack("<IHHHHHIIIHH", head)
            name, extra = f.read(nlen).decode(), f.read(xlen)
            if 0xFFFFFFFF in (csize, usize):  # zip64: sizes in the extra field
                i = 0
                while i < len(extra):
                    hid, hlen = struct.unpack("<HH", extra[i:i + 4])
                    if hid == 1:
                        usize, csize = struct.unpack("<QQ", extra[i + 4:i + 20])
                        break
                    i += 4 + hlen
            assert not flag & 8, f"{path}: sizes in a data descriptor"
            end = pos + 30 + nlen + xlen + csize
            if end > size:
                return
            yield name, method, crc, f.read(csize)
            pos = end


def main():
    (DEST / "train").mkdir(parents=True, exist_ok=True)
    total = 0
    for part in sorted(SRC.glob("train-part*.zip")):
        n = 0
        for name, method, crc, data in members(part):
            out = DEST / "train" / pathlib.Path(name).name
            if not out.exists():
                raw = zlib.decompress(data, -15) if method == 8 else data
                assert zlib.crc32(raw) == crc, f"{part}:{name} CRC mismatch"
                out.with_suffix(".tmp").write_bytes(raw)
                out.with_suffix(".tmp").rename(out)
            n += 1
        print(f"{part.name}: {n} episodes")
        total += n
    link = DEST / "eval.zip"
    if not link.exists():
        link.symlink_to(SRC / "eval.zip")
    print(f"{total} training episodes in {DEST / 'train'}")


if __name__ == "__main__":
    main()
