#!/bin/bash
# molfile_to_params.py is NOT included in the PyRosetta pip wheel (verified by
# searching the installed package). It lives in the public RosettaCommons/rosetta
# repo. This vendors it plus the rosetta_py modules it imports.
#
# These are downloaded third-party files: they live in their own directory, and
# everything that runs them does so with `python -I` so the vendored tree cannot
# shadow stdlib imports.
set -euo pipefail

DEST=/work/hdd/bisa/sshah21/external/rosetta_tools
BASE=https://raw.githubusercontent.com/RosettaCommons/rosetta/main/source/scripts/python/public

mkdir -p "$DEST/rosetta_py/io" "$DEST/rosetta_py/utility"
cd "$DEST"

curl -sSf -o molfile_to_params.py "$BASE/molfile_to_params.py"
for f in rosetta_py/__init__.py \
         rosetta_py/io/__init__.py \
         rosetta_py/io/mdl_molfile.py \
         rosetta_py/utility/__init__.py \
         rosetta_py/utility/rankorder.py \
         rosetta_py/utility/r3.py; do
  curl -sSf -o "$f" "$BASE/$f"
done

date -u +"fetched %Y-%m-%dT%H:%M:%SZ from RosettaCommons/rosetta@main" > DOWNLOADED_ON
echo "ROSETTA_TOOLS_OK -> $DEST"
