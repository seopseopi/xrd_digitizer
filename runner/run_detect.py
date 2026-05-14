"""
CLI: XRD 이미지 자동 축 감지

usage:
  python -m runner.run_detect --image_path <path> [--stdout]
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description='XRD auto-detect axes')
    parser.add_argument('--image_path', required=True, help='입력 이미지 경로')
    parser.add_argument('--stdout', action='store_true', help='결과를 stdout JSON으로 출력')
    args = parser.parse_args()

    from preprocess.auto_detect import auto_detect
    result = auto_detect(args.image_path)

    out = json.dumps(result, ensure_ascii=False)
    if args.stdout:
        print(out)
    else:
        print(out)


if __name__ == '__main__':
    main()
