"""Command-line interface.

Examples::

    autobanner presets
    autobanner analyze design.psd
    autobanner render design.psd --pack meta-ads --pack google-display -o out/
    autobanner render banners/*.png --size 1200x628 --format webp --max-kb 150
    autobanner serve --port 7860
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .exceptions import AutoBannerError
from .export import ExportOptions
from .presets import PACK_DESCRIPTIONS, PACKS, list_presets, resolve_targets
from .service import ANCHOR_PRESETS, MODES, RenderRequest, RenderService
from .validators import safe_filename

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_USAGE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autobanner",
        description="Resize one master banner into every ad and social size.",
    )
    parser.add_argument("--version", action="version", version=f"autobanner {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("presets", help="List output sizes and packs")
    p.add_argument("--platform", help="Only show one platform (iab, meta, google, ...)")
    p.add_argument("--json", action="store_true", help="Machine-readable output")

    a = sub.add_parser("analyze", help="Show detected layers and roles")
    a.add_argument("input", help="PSD, PNG, JPG or WEBP file")

    r = sub.add_parser("render", help="Render one or more designs")
    r.add_argument("inputs", nargs="+", help="Design files (PSD, PNG, JPG, WEBP)")
    r.add_argument("-o", "--out", default="autobanner-output", help="Output directory")
    r.add_argument("-p", "--preset", action="append", default=[], help="Preset id (repeatable)")
    r.add_argument("-k", "--pack", action="append", default=[], help="Pack id (repeatable)")
    r.add_argument("-s", "--size", action="append", default=[], help="Custom WxH (repeatable)")
    r.add_argument("--mode", choices=MODES, default="phase21",
                   help="phase21 = fast relayout, phase3 = target-first redesign")
    r.add_argument("--format", choices=("png", "jpeg", "webp"), default="png")
    r.add_argument("--quality", type=int, default=90, help="JPEG/WebP quality 1-100")
    r.add_argument("--max-kb", type=int, help="File-size budget per image (overrides platform)")
    r.add_argument("--ignore-platform-limits", action="store_true",
                   help="Do not apply each network's file-size cap")
    r.add_argument("--no-safe-zones", action="store_true",
                   help="Do not move key elements out of Story/Reels UI zones")
    r.add_argument("--anchor-preset", choices=ANCHOR_PRESETS, default="none")
    r.add_argument("--anchors", help="JSON file with manual anchor boxes (flat images)")
    r.add_argument("--zip", action="store_true", help="Write one ZIP per input instead of files")
    r.add_argument("--use-ai", action="store_true", help="Enable CLIP classification if installed")
    r.add_argument("-q", "--quiet", action="store_true")

    s = sub.add_parser("serve", help="Run the web studio + REST API")
    s.add_argument("--host", default=None, help="Bind address (default 127.0.0.1)")
    s.add_argument("--port", type=int, default=None, help="Port (default 7860)")
    return parser


def _cmd_presets(args: argparse.Namespace) -> int:
    presets = list_presets(args.platform)
    if args.json:
        payload = {
            "presets": [p.to_dict() for p in presets],
            "packs": {k: list(v) for k, v in PACKS.items()},
        }
        print(json.dumps(payload, indent=2))
        return EXIT_OK
    print(f"{'ID':<28} {'SIZE':>11}  {'MAX':>8}  NAME")
    for p in presets:
        cap = f"{p.max_kb}KB" if p.max_kb else "-"
        print(f"{p.id:<28} {p.width:>5}x{p.height:<5}  {cap:>8}  {p.name}")
    if not args.platform:
        print("\nPacks:")
        for pack, ids in PACKS.items():
            print(f"  {pack:<20} {len(ids):>2} sizes  {PACK_DESCRIPTIONS.get(pack, '')}")
    return EXIT_OK


def _cmd_analyze(args: argparse.Namespace) -> int:
    info = RenderService().analyze(args.input)
    print(json.dumps(info, indent=2))
    return EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    targets = resolve_targets(presets=args.preset, packs=args.pack, sizes=args.size)
    if not targets:
        print("error: choose at least one --preset, --pack or --size", file=sys.stderr)
        return EXIT_USAGE

    anchors = None
    if args.anchors:
        anchors = json.loads(Path(args.anchors).read_text())

    request = RenderRequest(
        targets=targets,
        mode=args.mode,
        export=ExportOptions(format=args.format, quality=args.quality, max_kb=args.max_kb),
        respect_platform_limits=not args.ignore_platform_limits,
        manual_anchors=anchors,
        anchor_preset=args.anchor_preset,
        enforce_safe_zones=not args.no_safe_zones,
    )
    request.validate()

    service = RenderService(use_ai=args.use_ai)
    out_root = Path(args.out)
    multi = len(args.inputs) > 1
    worst = EXIT_OK

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    def progress(done: int, total: int, current: str) -> None:
        if done < total:
            log(f"  [{done + 1}/{total}] {current}")

    used_names: set[str] = set()
    for input_path in args.inputs:
        base = safe_filename(Path(input_path).stem, "design")
        name, n = base, 2
        while name in used_names:  # banners/a/input.png + banners/b/input.png
            name, n = f"{base}_{n}", n + 1
        used_names.add(name)
        log(f"▸ {input_path} → {len(targets)} size(s)")
        try:
            report = service.render(input_path, request, progress=progress)
        except AutoBannerError as e:
            print(f"  ✗ {input_path}: {e}", file=sys.stderr)
            worst = EXIT_PARTIAL
            continue

        if args.zip:
            out_root.mkdir(parents=True, exist_ok=True)
            dest = out_root / f"{name}_autobanner.zip"
            dest.write_bytes(report.to_zip())
            log(f"  ✓ {dest}")
        else:
            dest_dir = out_root / name if multi else out_root
            report.write_to(dest_dir)
            log(f"  ✓ {dest_dir}/")

        for asset in report.assets:
            if not asset.ok:
                worst = EXIT_PARTIAL
                log(f"  ✗ {asset.preset.id}: {asset.error}")
                continue
            assert asset.encoded is not None
            line = f"    {asset.filename:<44} {asset.encoded.size_kb:>8.1f} KB"
            if asset.warnings:
                line += "  ⚠ " + "; ".join(asset.warnings)
            log(line)
        s = report.manifest()["summary"]
        log(f"  {s['succeeded']}/{s['total']} ok in {report.duration_ms / 1000:.1f}s")
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "presets":
            return _cmd_presets(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "render":
            return _cmd_render(args)
        if args.command == "serve":
            from .main import main as serve

            serve(host=args.host, port=args.port)
            return EXIT_OK
    except AutoBannerError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    parser.print_help()
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
