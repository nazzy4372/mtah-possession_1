"""Download external open datasets into ext_cache/ (gitignored), verified against a manifest.

    python -m ext.download metrica

Metrica Sports sample data, https://github.com/metrica-sports/sample-data (anonymised games 1-3).
Terms (README of that repository): no formal licence; "be responsible with the use of this data"
and "if you use it for anything public, please acknowledge the source". Acknowledged in
docs/label_derivation.md and docs/TECHNICAL.md.

The files are pinned to a commit and checked by SHA-256, so a rerun reproduces the same inputs.
They are data only (CSV / JSON / XML / text), parsed, never executed.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "ext_cache"
MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"

METRICA_REPO = "metrica-sports/sample-data"
METRICA_FILES = {
    "Sample_Game_1": ["Sample_Game_1_RawEventsData.csv",
                      "Sample_Game_1_RawTrackingData_Away_Team.csv",
                      "Sample_Game_1_RawTrackingData_Home_Team.csv"],
    "Sample_Game_2": ["Sample_Game_2_RawEventsData.csv",
                      "Sample_Game_2_RawTrackingData_Away_Team.csv",
                      "Sample_Game_2_RawTrackingData_Home_Team.csv"],
    "Sample_Game_3": ["Sample_Game_3_events.json",
                      "Sample_Game_3_metadata.xml",
                      "Sample_Game_3_tracking.txt"],
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrica_manifest_path() -> Path:
    return MANIFEST_DIR / "metrica.json"


def download_metrica(ref: str = None) -> Path:
    """Fetch the three Metrica games at a pinned commit; verify (or record) their SHA-256."""
    manifest_path = metrica_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.is_file() else {}
    ref = ref or manifest.get("ref")
    if ref is None:
        with urllib.request.urlopen("https://api.github.com/repos/{}/commits/master".format(
                METRICA_REPO)) as response:
            ref = json.load(response)["sha"]
    out_root = CACHE / "metrica"
    files = manifest.get("files", {})
    for game, names in METRICA_FILES.items():
        folder = out_root / game
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = folder / name
            key = "{}/{}".format(game, name)
            if not target.is_file():
                url = "https://raw.githubusercontent.com/{}/{}/data/{}/{}".format(
                    METRICA_REPO, ref, game, name)
                print("downloading {}".format(key))
                tmp = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(url, str(tmp))
                tmp.replace(target)
            digest = sha256_of(target)
            if key in files and files[key]["sha256"] != digest:
                raise RuntimeError("{}: SHA-256 mismatch with ext/manifests/metrica.json".format(key))
            files[key] = {"sha256": digest, "bytes": target.stat().st_size}
    manifest = {"source": "https://github.com/{}".format(METRICA_REPO), "ref": ref,
                "terms": "no formal licence; README asks for responsible use and acknowledgement "
                         "of the source", "files": files}
    MANIFEST_DIR.mkdir(exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_root


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "metrica"
    if which != "metrica":
        sys.exit("unknown source {!r}".format(which))
    print("cached in {}".format(download_metrica()))
