#!/usr/bin/env python3
"""Sound Fixer — replace video audio with high-quality dedicated recordings."""

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replace video clip audio with matched segments from a dedicated audio recording."
    )
    parser.add_argument("--input", required=True, help="Directory containing raw audio and video files")
    parser.add_argument("--output", required=True, help="Directory for output video files")
    parser.add_argument("--clip", type=int, help="Process only video clip number N (1-indexed)")
    parser.add_argument("--from-clip", type=int, help="Process video clips from N onwards (1-indexed)")
    parser.add_argument(
        "--it-is-what-it-is",
        action="store_true",
        help="Include low-confidence alignment matches in output instead of skipping them",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
