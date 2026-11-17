"""All project commands (no make on Windows).

    python tasks.py test [--fast]
    python tasks.py submission [--data-dir D] [--out F] [--method v5|v4|...]
    python tasks.py validate [F]
    python tasks.py report [F]
    python tasks.py package --team NAME
    python tasks.py synthetic [--preset small|full]
    python tasks.py external [--prepare]
    python tasks.py tune

--data-dir defaults to MTAH_DATA_DIR. Output goes to outputs/real/ (gitignored).
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


RECORDED_SHA = ROOT / "config" / "submission.sha256"


def _expected_sha(value):
    """None, an explicit hex digest, or 'recorded' (the frozen SHA in config/submission.sha256)."""
    if not value:
        return None
    if value == "recorded":
        return RECORDED_SHA.read_text(encoding="utf-8").split()[0].lower()
    if value in ("recorded-v2", "recorded-v3", "recorded-v4"):     # the earlier frozen files
        return (ROOT / "config" / "submission_{}.sha256".format(value[-2:])).read_text(
            encoding="utf-8").split()[0].lower()
    return value.lower()


def _check_sha(digest, expected):
    if expected is None:
        return 0
    if digest == expected:
        print("reproduced  identical bytes to the expected SHA-256")
        return 0
    print("MISMATCH    expected {}\n            got      {}".format(expected, digest))
    return 1


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
    args.method = args.method or pipeline.default_method()
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
    if args.method in ("v2", "v3", "v4", "v5"):
        from mtah import emission_v2, emission_v3, emission_v4, emission_v5
        s = {"v2": emission_v2, "v3": emission_v3, "v4": emission_v4,
             "v5": emission_v5}[args.method].settings()
        print("settings    from {}: c_switch {:.2f} c_die {:.2f} c_restart {:.2f} temperature "
              "{:.2f} dead_bias {:+.2f} refine {}{}".format(
                  s["source"], s["c_switch"], s["c_die"], s["c_restart"], s["temperature"],
                  s["dead_bias"], s["refine"],
                  " change weight {:.2f}".format(s["w_change"]) if s.get("w_change") else ""))
        if s.get("w_tagging"):
            print("tagging     team prior from Smart Tagging, weight {:.2f} per frame".format(
                s["w_tagging"]))
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
    return _check_sha(digest, _expected_sha(args.expect_sha256))


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
    digest = sha256_of(path)
    print("sha256        {}".format(digest))
    return code or (1 if has_cr else 0) or _check_sha(digest, _expected_sha(args.expect_sha256))


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
    from ext import download, evaluate, metrica, skillcorner

    if args.prepare:
        download.download_metrica()
        for g in (1, 2, 3):
            if not (metrica.BUNDLES / "metrica_g{}".format(g) / "gt.csv").is_file():
                print("converting Metrica game {} ...".format(g), flush=True)
                metrica.convert(g)
        download.download_skillcorner()
        for match_id in skillcorner.downloaded_matches():
            if not (skillcorner.BUNDLES / "sc_{}".format(match_id) / "gt.csv").is_file():
                print("converting SkillCorner match {} ...".format(match_id), flush=True)
                skillcorner.convert(match_id)
    argv = ["--bootstrap", str(args.bootstrap)]
    if args.games:
        argv += ["--games", *args.games]
    return evaluate.main(argv)


def cmd_package(args):
    """Copy the frozen submission and the technical document to the team-named upload files,
    then run the official validator on the exact CSV that will be uploaded."""
    import re
    import shutil
    from mtah.official import VALIDATE_SCRIPT
    from mtah.pipeline import sha256_of

    team = args.team.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]*", team):
        print("team name must be letters, digits, space, '_', '-' or '.'")
        return 2
    src = Path(args.file)
    out_dir = ROOT / "outputs" / "real" / "upload"          # gitignored, like all of outputs/
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "{}.csv".format(team)
    doc = out_dir / "{}.md".format(team)
    shutil.copyfile(src, csv)
    shutil.copyfile(ROOT / "docs" / "TECHNICAL.md", doc)
    print("upload files in {}".format(out_dir))
    code = subprocess.call([sys.executable, str(VALIDATE_SCRIPT), str(csv)])
    data = csv.read_bytes()
    has_cr = b"\r" in data
    digest = sha256_of(csv)
    print("\nline endings  {}".format("FAIL: contains \\r bytes" if has_cr else "LF only (no \\r)"))
    print("sha256        {}".format(digest))
    return code or (1 if has_cr else 0) or _check_sha(digest, _expected_sha("recorded"))


def cmd_tune(args):
    from mtah import tune
    argv = ["--variant", args.variant, "--trials", str(args.trials)]
    if args.nested_only:
        argv.append("--nested-only")
    if args.change:
        argv += ["--change", args.change]
    if args.freeze:
        argv.append("--freeze")
    return tune.main(argv)


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
    p.add_argument("--expect-sha256", metavar="HEX|recorded",
                   help="fail unless the file has this SHA-256 ('recorded' = the frozen one)")
    p.add_argument("--method", choices=("v5", "v4", "v3", "v2", "v1", "v1d", "baseline"),
                   help="default: the submitted model (v5: v4 plus a team prior from the Smart "
                        "Tagging). v4, v3 and v2 reproduce the earlier frozen files; v1 is the "
                        "rules; baseline is the ported upstream baseline")
    p.set_defaults(func=cmd_submission)

    p = sub.add_parser("validate", help="official validator, LF check and SHA-256")
    p.add_argument("file", nargs="?", default=str(DEFAULT_OUT))
    p.add_argument("--expect-sha256", metavar="HEX|recorded",
                   help="fail unless the file has this SHA-256 ('recorded' = the frozen one)")
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
    p.add_argument("--games", nargs="+",
                   help="bundle names, e.g. metrica_g3 sc_1874553 (default: every converted game)")
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--prepare", action="store_true",
                   help="download (pinned, SHA-256 checked) and convert Metrica + SkillCorner")
    p.set_defaults(func=cmd_external)
    p = sub.add_parser("package", help="team-named upload files + official validation")
    p.add_argument("--team", required=True, help="team name exactly as in the registration form")
    p.add_argument("file", nargs="?", default=str(DEFAULT_OUT))
    p.set_defaults(func=cmd_package)

    p = sub.add_parser("tune", help="Optuna on external out-of-fold predictions -> config/final.yaml")
    p.add_argument("--variant", default="all")
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--nested-only", action="store_true",
                   help="only the nested 2-fold estimate")
    p.add_argument("--change", metavar="VARIANT",
                   help="drive transition costs with the change-point model of this variant")
    p.add_argument("--freeze", action="store_true",
                   help="write config/final.yaml (without it nothing frozen is touched)")
    p.set_defaults(func=cmd_tune)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
