"""Create a relocated Hermes seed without modifying any existing file."""
from pathlib import Path
import shutil
import subprocess
import sys


def copy_hermes(source: Path, target: Path, version: str) -> None:
    source, target = source.absolute(), target.absolute()

    def copy_file(src, dst):
        src = Path(src)
        data = src.read_bytes()
        if src.parent == source / "bin" and data.startswith(b"#!"):
            # pip also emits shell launchers with an embedded Python path.
            data = data.replace(bytes(source), bytes(target))
        with open(dst, "xb") as output:
            output.write(data)
        shutil.copystat(src, dst)
        return dst

    def ignore(directory, names):
        return {"TEMPLATE-DO-NOT-USE-DIRECTLY"} if Path(directory) == source else set()

    shutil.copytree(source, target, symlinks=True, copy_function=copy_file, ignore=ignore)
    target.chmod(0o700)
    check = 'import sys, importlib.metadata as m; print(sys.prefix); print(m.version("hermes-agent"))'
    prefix, actual = subprocess.check_output(
        [str(target / "bin/python"), "-I", "-c", check], text=True
    ).splitlines()
    if Path(prefix).resolve() != target.resolve() or actual != version or actual == "0.0.0":
        raise ValueError("Private Hermes copy failed prefix/version verification")
    print(f"== private Hermes copy: {target}; hermes-agent {actual}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: copy_suite_hermes.py SOURCE TARGET VERSION")
    copy_hermes(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
