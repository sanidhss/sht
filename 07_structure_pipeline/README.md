# 07_structure_pipeline — structure prediction → ligand prep → interface scoring

Installs and verifies the four tools in the pipeline diagram:

```
Boltz-2 / AF3  →  mmCIF + confidence JSON
       ↓  (QC gate here, before any prep)
   LIGAND PREP  →  protonated SDF + receptor PDB      ← shared
       ↓                              ↓
  molfile_to_params            ProLIF fingerprint
       ↓
  PyRosetta pose → relax → InterfaceAnalyzer
```

**Scope of this folder: installation + smoke tests only.** The QC gate and the pipeline
glue are not written yet. What exists here proves each tool runs on this machine and
that the ligand-prep chain connects end to end.

## What is installed where

| component | location | notes |
|---|---|---|
| `structprep` env | `/work/hdd/bisa/sshah21/envs/structprep` (4.7 G) | py3.11, RDKit 2026.09.2, OpenBabel, MDAnalysis 2.11.0, ProLIF 2.2.3, PyRosetta, OpenMM/PDBFixer, gemmi |
| `boltz2` env | `/work/hdd/bisa/sshah21/envs/boltz2` (6.3 G) | py3.11, Boltz-2 2.2.1, torch 2.15.0+cu132 |
| Boltz-2 weights | `/work/hdd/bisa/sshah21/boltz_cache` (7.6 G) | `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, CCD `mols/` |
| AF3 container + DBs | `module load alphafold/3.0.2` | NCSA-provided, on `/taiga` — does **not** count against your quota |
| AF3 weights | `/work/hdd/bisa/sshah21/af3_models/af3_parameters.bin.zst` (974 M) | yours, from DeepMind |
| `molfile_to_params.py` | `/work/hdd/bisa/sshah21/external/rosetta_tools` | vendored — **not** in the PyRosetta wheel |

Two envs on purpose: Boltz-2's torch stack and the prep/analysis stack have no reason to
share a solver, and PyRosetta is CPU-only.

## Why PyRosetta needed no license key

RosettaCommons made non-commercial use license-free, so `pyrosetta-installer` pulls the
wheel with no credentials. No `graylab` username/password step.

## `molfile_to_params.py` is not in the PyRosetta wheel

Verified by searching the installed package. It lives in the public
`RosettaCommons/rosetta` repo; `setup/get_rosetta_tools.sh` vendors it plus the five
`rosetta_py` modules it imports.

Because it is downloaded third-party code it lives in its own directory, and callers run
it with `python -s -E`.

> **Do not "fix" this to `python -I`.** Isolated mode implies `-P`/`-E`, which strips
> `PYTHONPATH` *and the script's own directory* — which is exactly where `rosetta_py`
> lives — so `-I` makes the tool unimportable (`No module named 'rosetta_py'`).
> `-s -E` still drops user site-packages and the environment, and since this is a script
> invocation `sys.path[0]` is the script's directory rather than the cwd, so the ligand
> work directory cannot inject a module.

### In-process alternative

PyRosetta also exposes Rosetta's C++ SDF reader directly:
`pyrosetta.rosetta.core.chemical.sdf.MolFileIOReader` + `convert_to_ResidueTypes`.
That skips the subprocess and the on-disk `.params` entirely. For batch work over many
ligands it is likely the better path; `molfile_to_params` is kept as primary because the
diagram specifies it and it is the better-trodden route.

## ⚠ Aromatic bonds in ligand prep — silent correctness bug

RDKit's `Chem.MolToMolFile` defaults to writing a **Kekulé** structure (MDL bond types
1/2 only). Rosetta needs aromatic bonds flagged as **type 4**; without them it types
aromatic rings as alternating single/double — wrong atom types for essentially any
drug-like ligand.

It does not fail. It prints `WARNING: structure contains double bonds but no aromatic
bonds` into stdout and writes a plausible-looking `.params` anyway.

**Always write the SDF with `kekulize=False`:**

```python
Chem.MolToMolFile(mol, "lig.sdf", kekulize=False)   # bond types 1,2,4
# NOT: Chem.MolToMolFile(mol, "lig.sdf")            # bond types 1,2 only
```

`smoke/smoke_molfile_to_params.py` fails if that warning reappears, so this cannot
silently regress.

## Running the smoke tests

```bash
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate /work/hdd/bisa/sshah21/envs/structprep
python smoke/smoke_structprep.py          # RDKit, obabel, ProLIF, PyRosetta, IA, m2p
python smoke/smoke_molfile_to_params.py   # SDF -> .params -> reload into PyRosetta

sbatch smoke/smoke_boltz.sb               # GPU
sbatch smoke/smoke_af3.sb                 # GPU
```

### `structprep` — all pass

```
[PASS] rdkit            aspirin embedded, 21 atoms, MW=180.2
[PASS] openbabel        obabel gen3d -> SDF ok
[PASS] prolif           prolif 2.2.3, 21 interactions on 1 frame
[PASS] pyrosetta relax  relaxed 10-mer, score 30.0 -> 1.8
[PASS] InterfaceAnalyzer
[PASS] molfile_to_params
[PASS] sdf in-process   MolFileIOReader + convert_to_ResidueTypes
```

### Ligand-prep chain — all pass

RDKit protonated 3D SDF → `molfile_to_params.py` → `LIG.params` (21 ATOM records) +
`LIG_0001.pdb` → loaded back into PyRosetta via `-extra_res_fa` and scored
(1 residue, 21 atoms, score 1.96). Generating the file is not proof; loading it is.

### Boltz-2 / AF3 — see trial folders

GPU jobs 24502364 (Boltz-2) and 24502365 (AF3), submitted 24-11-2026.

## AF3 notes

The weights file is **not** named `af3.bin`, and does not need to be. AF3's
`select_model_files()` globs the model dir for `*.bin` / `*.bin.zst` and derives the
model name from whatever it finds — confirmed by reading
`/alphafold3/src/alphafold3/model/params.py` inside the container. It *does* raise
`Multiple models matched` if more than one params file is present, so
`setup/run_af3.sb` rejects that case up front with a readable message.

```bash
sbatch setup/run_af3.sb <fold_input.json> <output_dir>
```

AF3 3.0.2 writes outputs **zstd-compressed** (`.zst`) — the QC gate that reads the
confidence JSON will need to decompress (`zstandard` is installed in `structprep`).

## MSA note

`smoke_boltz.sb` runs single-sequence (`msa: empty`) to stay fast and self-contained.
That is *not* because the nodes lack internet — compute nodes do have outbound HTTPS
(verified 13-11-2026, job 23850441, per `04_setup/build_vllm_env.sb`). Real runs should
drop `msa: empty` and pass `--use_msa_server`.

## Files

```
setup/build_structprep_env.sh   # env A
setup/build_boltz2_env.sh       # env B
setup/get_rosetta_tools.sh      # vendor molfile_to_params.py + rosetta_py
setup/run_af3.sb                # AF3 runner (container + DBs + your weights)
smoke/smoke_structprep.py       # env A, 7 checks
smoke/smoke_molfile_to_params.py# full SDF -> .params -> PyRosetta chain
smoke/smoke_boltz.sb            # GPU
smoke/smoke_af3.sb              # GPU
```

The `.sb` files here deliberately do **not** use the `_lib/new_trial.sh` trial pattern,
following the precedent of `04_setup/build_vllm_env.sb`: these are setup/verification
jobs, not experiments whose exact script version needs preserving. Pipeline runs built
on top of this should use the trial pattern.
