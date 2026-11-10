"""Every command in this project, cross-platform (there is no make on Windows).

    python tasks.py test [--fast]                     unit + end-to-end tests (--fast skips the
                                                      full-length synthetic match)
    python tasks.py submission [--data-dir D] [--out F] [--spec S]
                                                      data bundle -> submission CSV
    python tasks.py validate [F]                      official validator + LF check + SHA-256
    python tasks.py report [F] [--spec S]             aggregate sanity numbers for a submission
    python tasks.py synthetic [--out D] [--preset small|full] [--seed N]
                                                      write a synthetic bundle in ATD format
    python tasks.py external [--games 1 2 3] [--prepare]
                                                      download + convert external games (if
                                                      needed) and score baseline vs v1
    python tasks.py tune                              (M5) Optuna on external games

--data-dir defaults to the MTAH_DATA_DIR environment variable. Output defaults to
outputs/real/submission.csv, which is gitignored: per-frame outputs on the real match never enter
git.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUT = ROOT / "outputs" / "real" / "submission.csv"


def _spec(path):
    from mtah.spec import GameSpec
    return GameSpec.load(path) if path else GameSpec.from_upstream("BAR-ATH")


def cmd_test(args):
    command = [sys.executable, "-m", "pytest", "-q"]
    if args.fast:
        command += ["-m", "not slow"]
    return subprocess.call(command + list(args.pytest_args), cwd=str(ROOT))


def cmd_submission(args):
    from mtah import pipeline

    data_dir = args.data_dir or os.environ.get("MTAH_DATA_DIR")
    if not data_dir:
        print("no data directory: pass --data-dir or set MTAH_DATA_DIR")
        return 2
    spec = _spec(args.spec)
    started = time.perf_counter()
    try:
        result = pipeline.run(data_dir, spec, method=args.method)
    except pipeline.KickoffInconsistent as error:
        print("FAILED  {}".format(error))
        print("Run tools/kickoff_diag.py for the full aggregate breakdown.")
        return 1
    digest = pipeline.write_submission(args.out, result.labels)
    elapsed = time.perf_counter() - started

    # Aggregates only: this runs on the real match, and its output may be shared.
    print("game        {}  ({} frames), method {}".format(spec.name, spec.n_frames, args.method))
    for kickoff in result.kickoffs:
        print("kickoff     {}".format(kickoff.describe(spec.fps)))
        for window, values in kickoff.diagnostics.items():
            print("  diag {:<9} {}".format(window, "  ".join(
                "{}={}".format(k, v) for k, v in values.items())))
    counts = pipeline.label_counts(result.labels)
    print("labels      " + "  ".join("{} {}".format(k, v) for k, v in counts.items()))
    print("timing      parse {:.1f} s, label {:.1f} s, total {:.1f} s".format(
        result.timings["parse_s"], result.timings["label_s"], elapsed))
    print("wrote       {}".format(Path(args.out).resolve()))
    print("sha256      {}".format(digest))
    return 0


def cmd_validate(args):
    from mtah.official import VALIDATE_SCRIPT
    from mtah.pipeline import sha256_of

    path = Path(args.file)
    code = subprocess.call([sys.executable, str(VALIDATE_SCRIPT), str(path)])
    if not path.is_file():
        return code or 1
    data = path.read_bytes()
    has_cr = b"\r" in data
    print("\nline endings  {}".format("FAIL: contains \\r bytes" if has_cr else "LF only (no \\r)"))
    print("sha256        {}".format(sha256_of(path)))
    return code or (1 if has_cr else 0)


def cmd_report(args):
    command = [sys.executable, str(ROOT / "tools" / "report.py"), str(args.file)]
    if args.spec:
        command += ["--spec", args.spec]
    return subprocess.call(command)


def cmd_synthetic(args):
    from mtah import synthetic

    out = Path(args.out or ROOT / "outputs" / "synthetic" / args.preset)
    started = time.perf_counter()
    truth = synthetic.generate(out, args.preset, args.seed)
    print("wrote synthetic bundle ({} preset, seed {}) to {} in {:.1f} s".format(
        args.preset, args.seed, out, time.perf_counter() - started))
    print("kickoffs (video frames): {}".format(truth["kickoffs"]))
    spec_arg = "" if args.preset == "full" else " --spec {}".format(out / "game_spec.json")
    print("next: python tasks.py submission --data-dir {}{} --out {}".format(
        out, spec_arg, out / "submission.csv"))
    return 0


def cmd_external(args):
    sys.path.insert(0, str(ROOT))
    from ext import download, evaluate, metrica

    missing = [g for g in args.games
               if not (metrica.BUNDLES / "metrica_g{}".format(g) / "gt.csv").is_file()]
    if args.prepare or missing:
        download.download_metrica()
        for g in (args.games if args.prepare else missing):
            print("converting Metrica game {} ...".format(g))
            metrica.convert(g)
    return evaluate.main(["--games", *map(str, args.games), "--bootstrap", str(args.bootstrap)])


def cmd_not_yet(milestone):
    def run(args):
        print("not implemented until {} (see CLAUDE.md section 5)".format(milestone))
        return 2
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("test", help="run the test suite")
    p.add_argument("--fast", action="store_true", help="skip tests marked slow")
    p.add_argument("pytest_args", nargs="*", help="extra arguments for pytest")
    p.set_defaults(func=cmd_test)

    p = sub.add_parser("submission", help="data bundle -> submission CSV")
    p.add_argument("--data-dir", help="data bundle (default: $MTAH_DATA_DIR)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--spec", help="game_spec.json for a non-BAR-ATH bundle")
    p.add_argument("--method", default="v1", choices=("v1", "baseline"),
                   help="v1 (rules + decoder, default) or the ported upstream baseline")
    p.set_defaults(func=cmd_submission)

    p = sub.add_parser("validate", help="official validator, LF check and SHA-256")
    p.add_argument("file", nargs="?", default=str(DEFAULT_OUT))
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("report", help="aggregate sanity numbers for a submission")
    p.add_argument("file", nargs="?", default=str(DEFAULT_OUT))
    p.add_argument("--spec", help="game_spec.json for a non-BAR-ATH submission")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("synthetic", help="write a synthetic bundle in ATD format")
    p.add_argument("--out")
    p.add_argument("--preset", choices=("small", "full"), default="small")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("external", help="download/convert external games, score methods")
    p.add_argument("--games", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--prepare", action="store_true",
                   help="download (pinned, SHA-256 checked) and convert the games first")
    p.set_defaults(func=cmd_external)
    sub.add_parser("tune", help="(M5)").set_defaults(func=cmd_not_yet("M5"))

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
