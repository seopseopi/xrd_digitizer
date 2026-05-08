#!/usr/bin/env python3
"""
엔진 모델 코드(core, preprocess, trace, calibrate, runner 일부)를
dist/xrd_digitizer_model_v1_2/ 로 복사한 뒤 dist/xrd_digitizer_model_v1_2_<UTC날짜>.zip 생성.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUNDLE_NAME = "xrd_digitizer_model_v1_2"
BUNDLE = DIST / BUNDLE_NAME

PACKAGE_DIRS = ("core", "preprocess", "trace", "calibrate")
RUNNER_FILES = ("run_local.py", "pipeline_experimental.py", "batch_run.py", "__init__.py")


def _write_meta(bundle: Path) -> None:
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (bundle / "VERSION.txt").write_text(
        "product: xrd_digitizer_model_v1_2\nsemantic_version: 1.2.0\n"
        f"packed_at_utc: {utc}\nengine: same as run_local.run_pipeline (v1_1 code path)\n"
        "debug_tag: calibrate_v1_2 when run with --pipeline v1_2\n",
        encoding="utf-8",
    )
    lines = []
    for p in sorted(bundle.rglob("*.py")):
        lines.append(str(p.relative_to(bundle)))
    (bundle / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (bundle / "BUNDLE.txt").write_text(
        "xrd_digitizer v1.2 모델 코드 스냅샷\n"
        "포함: core, preprocess, trace, calibrate, runner(단일·배치·실험 진입)\n\n"
        "실행(번들 폴더를 리포지토리 루트처럼 두고):\n"
        "  python runner/run_local.py --image_path ... --manual_inputs_path ... \\\n"
        "    --output_json_path ... --debug_dir ... --pipeline v1_2\n\n"
        "의존성: numpy, scipy, pillow  (batch_run.py 사용 시 pandas)\n",
        encoding="utf-8",
    )


def main() -> Path:
    DIST.mkdir(parents=True, exist_ok=True)
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)

    for name in PACKAGE_DIRS:
        src = ROOT / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        shutil.copytree(
            src,
            BUNDLE / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".mypy_cache"),
        )

    runner_dst = BUNDLE / "runner"
    runner_dst.mkdir()
    for fn in RUNNER_FILES:
        shutil.copy2(ROOT / "runner" / fn, runner_dst / fn)

    _write_meta(BUNDLE)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    zip_path = DIST / f"{BUNDLE_NAME}_{stamp}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in BUNDLE.rglob("*"):
            if path.is_file():
                arc = path.relative_to(DIST)
                zf.write(path, arc.as_posix())
    print(str(zip_path))
    return zip_path


if __name__ == "__main__":
    main()
