#!/usr/bin/env bash
# Regenerate the anonymous artifact from a source branch.
#
#   ./scripts/update_anon.sh [SOURCE_BRANCH]
#
# Rebuilds the `anon` branch as a single orphan commit (no author history),
# scrubs identifying strings, and force-pushes it to the anonymous repo.
# Anonymous GitHub tracks the branch tip with Auto-update on, so the public
# share link picks the change up within the hour. Nothing else to do.
#
# SRC_REPO and ANON_REPO are overridable so the artifact can be built from a
# local checkout that has not been pushed yet:
#
#   SRC_REPO=. ANON_REPO=git@github.com:cai4cai/SinkSLOT.git ./scripts/update_anon.sh main
#
# Without that, the clone below silently sources from whatever the REMOTE branch
# holds, which is not necessarily what you just built and reviewed locally.
set -euo pipefail

SRC_BRANCH="${1:-main}"
SRC_REPO="${SRC_REPO:-git@github.com:cai4cai/SinkSLOT.git}"
ANON_REPO="${ANON_REPO:-git@github.com:aymuos15/sinkslot.git}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning $SRC_BRANCH"
git clone -q --branch "$SRC_BRANCH" --depth 1 "$SRC_REPO" "$WORK/src"
cd "$WORK/src"
SRC_SHA="$(git rev-parse --short HEAD)"

echo "==> scrubbing"
python3 - <<'PY'
import pathlib, re

# The provenance line naming the source lab. Matched loosely (any line mentioning
# the org, plus the sentence that follows it) so a reflow of the docstring cannot
# silently turn this into a no-op -- the earlier exact-text regex could.
# The solver moved from flash_sinkhorn/bench/sinkslot.py into its own package;
# both spellings are tried so this keeps working either side of that change,
# and it is an error for neither to exist rather than a silent no-op -- an
# unguarded read_text() here would abort the build with a bare traceback, and
# a bare exists() check would skip the scrub and leave the gate to catch it.
for cand in ("torch-ext/sinkslot/solver.py",
             "torch-ext/flash_sinkhorn/bench/sinkslot.py"):
    p = pathlib.Path(cand)
    if p.exists():
        s = p.read_text()
        s = re.sub(r"Ported from [^\n]*cai4cai[^\n]*\n(?:[^\n]*accompanying the SLOT paper\. )?", "", s)
        p.write_text(s)
        break
else:
    raise SystemExit("!! solver source not found at any known path -- scrub cannot run")

# Internal engineering notes: not part of the artifact, and they name the cluster
# and carry candid claims that belong in the paper's own words if anywhere.
# Matched by basename anywhere in the tree, not just at the root: memory.md moved
# to scripts/ and the root-only unlink silently stopped finding it, so the notes
# were only kept out of the artifact by the grep gate below tripping.
for name in ("memory.md", "megakernel-findings.md", "cleanup.md", "handoff.md", "analysis.md",
             "kernelreport.md"):
    for p in pathlib.Path(".").rglob(name):
        if ".git" not in p.parts:
            p.unlink()

# This script itself. It is build tooling rather than artifact content, and it
# carries the very list of names and institutions it exists to scrub -- shipping
# it would hand a reviewer the deanonymisation key. It also means the grep gate
# below would otherwise always match itself and abort.
pathlib.Path("scripts/update_anon.sh").unlink(missing_ok=True)

# Upstream HF-kernel packaging. Removed from the source repo itself (build.toml
# and flake.nix along with them), so these are belt-and-braces for an older
# checkout: the workflow published to the upstream author's Hub namespace and
# could not run anonymously, and CARD.md was upstream's model card describing
# FlashSinkhorn rather than this artifact.
import shutil
shutil.rmtree(".github", ignore_errors=True)
for f in ("CARD.md", "build.toml", "flake.nix"):
    pathlib.Path(f).unlink(missing_ok=True)

# NOTE: ot-triton-lab is NOT us. FlashSinkhorn is third-party prior work (Ye et
# al., ICML 2026, cited as ye2026flashsinkhorn), forked here on 2026-07-20; every
# commit touching sinkslot/, configs/ or gradient_flow/ postdates that. So its
# name and URLs are not identifying and must NOT be scrubbed:
#   - LICENSE keeps "Copyright (c) 2025 OT Triton Contributors". MIT requires
#     retaining it on code we redistribute.
#   - pyproject.toml keeps {name = "OT Triton Contributors"} and its ot-triton-lab
#     Homepage/Repository. An earlier version rewrote these to "Anonymous Authors"
#     and our anon URL, which relabelled upstream's package as ours -- passing off,
#     not anonymisation. Anonymise our identity, never someone else's.
#   - README's FlashSinkhorn attribution stays; the paper cites it as the primary
#     baseline, so a reviewer already knows we build on it.
# pyproject.toml is also what installs the sinkslot package (packages.find over
# torch-ext/), so it cannot simply be dropped the way build.toml was.

# Cluster QoS names are site-specific and identify the facility. Swept over the
# whole configs/ tree rather than one named file: the earlier version pointed at
# a config that has since been renamed, so it silently did nothing. The grep gate
# below is what actually catches a miss, but this keeps it from firing.
for p in pathlib.Path("configs").glob("*.py"):
    s = p.read_text()
    t = s.replace("qos_gpu_h100-dev 2-hour cap.", "2-hour dev-queue cap.")
    if t != s:
        p.write_text(t)
PY

# Fail loudly if anything identifying survives, rather than pushing a leak.
# Added after a near-miss: "Jean Zay" reached the public artifact because the
# pattern below listed only people and orgs, not facilities.
PATTERN='cai4cai|kcl\.ac\.uk|King.?s College|KCL|Soumya|Snigdha|Kundu|aymuos|albany\.edu'
PATTERN="$PATTERN"'|yurikifer|ihsieh|reuben|dorent|inria|localssk23|soumyawork15'
PATTERN="$PATTERN"'|sie236|jean.?zay|idris|qos_gpu|overleaf|olp_[A-Za-z0-9]{20,}'
PATTERN="$PATTERN"'|/home/[a-z0-9_]+|/Users/[a-z0-9_]+|@gmail|@[a-z0-9.-]+\.(ac\.uk|edu|fr)'
# Internal sibling repositories, named by path in docstrings. Added after
# mva-internship-2026/SROT reached the published artifact: the branch feeding
# the build had forked before the commit that scrubbed it, and every pattern
# above lists people, orgs or facilities -- none would ever match a repo name.
# Note: match the internship repo's own name only. A bare /SROT would also hit
# khainb/SROT, the public paper repo this work cites legitimately, and the
# SinkSLOT/-CUDA/SROT method labels throughout configs/ -- all fine to ship.
PATTERN="$PATTERN"'|mva-internship'
if grep -rniE "$PATTERN" . --exclude-dir=.git --binary-files=without-match; then
  echo "!! identifying strings found above -- aborting, nothing pushed" >&2
  exit 1
fi
# Text greps skip binaries, so check document metadata separately rather than
# silently passing every PDF in the tree.
if command -v pdfinfo >/dev/null 2>&1; then
  while IFS= read -r pdf; do
    if pdfinfo "$pdf" 2>/dev/null | grep -iE '^(Author|Creator|Producer|Title)' | grep -qiE "$PATTERN"; then
      echo "!! identifying metadata in $pdf -- aborting, nothing pushed" >&2
      exit 1
    fi
  done < <(find . -name '*.pdf' -not -path './.git/*')
fi
echo "    clean"

echo "==> building orphan commit"
git config user.name  "Anonymous Authors"
git config user.email "anonymous@example.com"
git checkout -q --orphan anon
git add -A
git commit -q -m "SinkSLOT: anonymous artifact for double-blind review"

echo "==> pushing to anonymous repo"
git push -q --force "$ANON_REPO" anon
echo "    $ANON_REPO"

# Mirrors get the identical orphan commit in the same run. Keeping a copy on the
# source repo is convenient -- you can see what was published without cloning the
# anonymous one -- but a mirror updated by hand is a mirror that goes stale, and
# a stale copy of this branch is exactly what leaked mva-internship-2026/SROT.
# So it is pushed here or not at all. Set ANON_MIRRORS="" to skip.
for mirror in ${ANON_MIRRORS-git@github.com:cai4cai/SinkSLOT.git}; do
  [ "$mirror" = "$ANON_REPO" ] && continue
  git push -q --force "$mirror" anon
  echo "    $mirror (mirror)"
done

echo "OK: anon updated from $SRC_BRANCH@$SRC_SHA -> $(git rev-parse --short HEAD)"
echo "    history depth: $(git rev-list --count HEAD) commit"
