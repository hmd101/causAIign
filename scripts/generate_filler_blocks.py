"""Deprecated shim: forwards to scripts/01_prompts/generate_filler_blocks.py"""
from pathlib import Path
import runpy
import sys


def main() -> None:
	target = Path(__file__).parent / "01_prompts" / "generate_filler_blocks.py"
	sys.stderr.write(
		"[DEPRECATION] scripts/generate_filler_blocks.py moved to scripts/01_prompts/generate_filler_blocks.py.\n"
	)
	runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
	main()
