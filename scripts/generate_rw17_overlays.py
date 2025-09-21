"""Deprecated shim: forwards to scripts/01_prompts/generate_rw17_overlays.py"""
from pathlib import Path
import runpy
import sys


def main() -> None:
	target = Path(__file__).parent / "01_prompts" / "generate_rw17_overlays.py"
	sys.stderr.write(
		"[DEPRECATION] scripts/generate_rw17_overlays.py moved to scripts/01_prompts/generate_rw17_overlays.py.\n"
	)
	runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
	main()
