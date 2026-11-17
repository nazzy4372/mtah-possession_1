"""Download Metrica + SkillCorner open data at pinned commits, checked against SHA-256 manifests."""

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


SKILLCORNER_REPO = "SkillCorner/opendata"
SKILLCORNER_N_MATCHES = 10   # the first 10 match ids in sorted order; the repo holds 20


def _get_json(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _fetch(url, target: Path):
    tmp = target.with_suffix(target.suffix + ".part")
    urllib.request.urlretrieve(url, str(tmp))
    tmp.replace(target)


def download_skillcorner(n_matches: int = SKILLCORNER_N_MATCHES) -> Path:
    """SkillCorner open data (MIT licence; the README asks users to credit SkillCorner).

    The tracking files are stored in Git LFS: each file's SHA-256 is checked against the oid in
    its LFS pointer, and every file is then recorded in ext/manifests/skillcorner.json.
    """
    manifest_path = MANIFEST_DIR / "skillcorner.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) \
        if manifest_path.is_file() else {}
    ref = manifest.get("ref") or _get_json(
        "https://api.github.com/repos/{}/commits/master".format(SKILLCORNER_REPO))["sha"]
    raw = "https://raw.githubusercontent.com/{}/{}/data/matches".format(SKILLCORNER_REPO, ref)
    media = "https://media.githubusercontent.com/media/{}/{}/data/matches".format(
        SKILLCORNER_REPO, ref)
    ids = manifest.get("matches") or sorted(
        e["name"] for e in _get_json("https://api.github.com/repos/{}/contents/data/matches?ref={}"
                                     .format(SKILLCORNER_REPO, ref)) if e["type"] == "dir"
    )[:n_matches]
    files = manifest.get("files", {})
    out_root = CACHE / "skillcorner"
    for match_id in ids:
        folder = out_root / match_id
        folder.mkdir(parents=True, exist_ok=True)
        for suffix in ("match.json", "phases_of_play.csv", "dynamic_events.csv"):
            name = "{}_{}".format(match_id, suffix)
            if not (folder / name).is_file():
                print("downloading {}/{}".format(match_id, name), flush=True)
                _fetch("{}/{}/{}".format(raw, match_id, name), folder / name)
        name = "{}_tracking_extrapolated.jsonl".format(match_id)
        target = folder / name
        if not target.is_file():
            with urllib.request.urlopen("{}/{}/{}".format(raw, match_id, name)) as response:
                pointer = response.read().decode("ascii")
            oid = next(line.split("sha256:")[1].strip() for line in pointer.splitlines()
                       if line.startswith("oid sha256:"))
            print("downloading {}/{} (LFS)".format(match_id, name), flush=True)
            _fetch("{}/{}/{}".format(media, match_id, name), target)
            if sha256_of(target) != oid:
                target.unlink()
                raise RuntimeError("{}: SHA-256 does not match its LFS pointer".format(name))
        for path in sorted(folder.iterdir()):
            key = "{}/{}".format(match_id, path.name)
            digest = sha256_of(path)
            if key in files and files[key]["sha256"] != digest:
                raise RuntimeError("{}: SHA-256 mismatch with ext/manifests/skillcorner.json"
                                   .format(key))
            files[key] = {"sha256": digest, "bytes": path.stat().st_size}
    manifest = {"source": "https://github.com/{}".format(SKILLCORNER_REPO), "ref": ref,
                "licence": "MIT", "terms": "README asks users to credit SkillCorner",
                "matches": ids, "files": files}
    MANIFEST_DIR.mkdir(exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out_root


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "metrica"
    if which == "metrica":
        print("cached in {}".format(download_metrica()))
    elif which == "skillcorner":
        print("cached in {}".format(download_skillcorner()))
    else:
        sys.exit("unknown source {!r}".format(which))
