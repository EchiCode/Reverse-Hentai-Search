"""Split failed gallery IDs and retry them with scrap.py.

Run this file directly. Existing chunks are reused; the source log is preserved.
Run while main.py is stopped.
"""

from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
FAILED_LOG = ROOT / "logging" / "failed.txt"
FAILED_FOLDER = ROOT / "logging" / "failed_chunks"


def _read_ids(path):
    ids = list(dict.fromkeys(int(line) for line in path.read_text().splitlines() if line.strip()))
    if any(gid <= 0 for gid in ids):
        raise ValueError(f"Gallery IDs must be positive integers: {path}")
    return ids



def split_failed_log(failed_log=FAILED_LOG, folder=FAILED_FOLDER, chunk_size=100):
    """Split unique IDs, or reuse existing chunks after checking source coverage."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    ids = _read_ids(Path(failed_log))
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        paths = sorted(folder.glob("*.txt"))
        chunk_ids = {gid for path in paths for gid in _read_ids(path)}
        missing = set(ids) - chunk_ids
        if missing:
            raise ValueError(f"{len(missing)} failed IDs are missing from {folder}; use a new empty folder")
        return paths

    folder.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{folder.name}.", dir=folder.parent) as directory:
        staging = Path(directory)
        paths = []
        for start in range(0, len(ids), chunk_size):
            path = staging / f"failed_{start // chunk_size + 1:06d}.txt"
            with path.open("x") as output:
                output.writelines(f"{gid}\n" for gid in ids[start:start + chunk_size])
            paths.append(path.name)
        staging.replace(folder)
    return [folder / name for name in paths]


def run_scraper_on_failed_logs(folder=FAILED_FOLDER):
    """Run scrap.download_failed on each .txt log in the folder."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    paths = sorted(folder.glob("*.txt"))
    for path in paths:
        _read_ids(path)
    if not paths:
        return 0

    # scrap.py loads IDs and writes outputs relative to the project root.
    with chdir(ROOT):
        import scrap

        total = 0
        for path in paths:
            total += scrap.download_failed(batch_id=path.stem, failed_file=path)
        return total


def _self_test():
    """Check splitting and retries without network access or scraper dependencies."""
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock, call, patch

    with TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "failed.txt"
        source.write_text("1\n2\n\n2\n3\n4\n")
        chunks = root / "chunks"
        paths = split_failed_log(source, chunks, 3)
        assert [path.read_text() for path in paths] == ["1\n2\n3\n", "4\n"]
        assert source.read_text() == "1\n2\n\n2\n3\n4\n"
        assert split_failed_log(source, chunks) == paths
        assert [path.read_text() for path in paths] == ["1\n2\n3\n", "4\n"]
        source.write_text("1\nbad-id\n")
        try:
            split_failed_log(source, root / "invalid")
        except ValueError:
            assert not (root / "invalid").exists()
        else:
            raise AssertionError("Invalid IDs must fail before writing chunks")

        fake = SimpleNamespace(download_failed=Mock(side_effect=[2, 1]))
        with patch.dict(sys.modules, {"scrap": fake}):
            previous_directory = Path.cwd()
            assert run_scraper_on_failed_logs(chunks) == 3
            assert Path.cwd() == previous_directory
            assert fake.download_failed.call_args_list == [
                call(batch_id=path.stem, failed_file=path) for path in paths
            ]
            assert [path.read_text() for path in paths] == ["1\n2\n3\n", "4\n"]

        source.write_text("".join(f"{gid}\n" for gid in range(1, 202)))
        interrupted = root / "interrupted"
        interrupted.mkdir()
        original_open = Path.open

        def interrupt_second_chunk(path, *args, **kwargs):
            if path.name == "failed_000002.txt" and args and args[0] == "x":
                raise OSError("simulated interrupted split")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", interrupt_second_chunk):
            try:
                split_failed_log(source, interrupted)
            except OSError:
                assert not any(interrupted.iterdir())
            else:
                raise AssertionError("Interrupted splits must not publish partial chunks")
        complete = split_failed_log(source, interrupted)
        assert sum(len(_read_ids(path)) for path in complete) == 201

        partial = root / "partial"
        partial.mkdir()
        first = partial / "failed_000001.txt"
        original = "".join(f"{gid}\n" for gid in range(1, 101))
        first.write_text(original)
        try:
            split_failed_log(source, partial)
        except ValueError:
            assert first.read_text() == original
            assert list(partial.iterdir()) == [first]
        else:
            raise AssertionError("Existing incomplete chunks must fail before retrying")
    print("Self-test passed")


if __name__ == "__main__":
    chunk_size = 100
    print(f"Prepared {len(split_failed_log(chunk_size=chunk_size))} logs")
    print(f"Downloaded {run_scraper_on_failed_logs()} galleries")
