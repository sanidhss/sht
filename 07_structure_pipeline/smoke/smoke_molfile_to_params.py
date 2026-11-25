"""Smoke test: RDKit 3D/protonated SDF -> Rosetta .params via molfile_to_params.py.

molfile_to_params.py is NOT shipped in the PyRosetta pip wheel; it lives in the
public RosettaCommons/rosetta repo and is vendored to
/work/hdd/bisa/sshah21/external/rosetta_tools by setup/get_rosetta_tools.sh.
"""
import os, subprocess, sys, tempfile

TOOLS = "/work/hdd/bisa/sshah21/external/rosetta_tools"
M2P = os.path.join(TOOLS, "molfile_to_params.py")


def main():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    scratch = os.environ.get("M2P_SCRATCH", "/work/hdd/bisa/sshah21/results/structure")
    os.makedirs(scratch, exist_ok=True)
    work = tempfile.mkdtemp(prefix="m2p_smoke_", dir=scratch)
    sdf = os.path.join(work, "asp.sdf")

    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    AllChem.EmbedMolecule(m, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(m)
    # kekulize=False writes aromatic bonds as MDL bond type 4. RDKit's DEFAULT
    # writes a Kekule structure (types 1/2 only), and molfile_to_params then warns
    # "structure contains double bonds but no aromatic bonds" and types the ring
    # as alternating single/double -- wrong atom types for any aromatic ligand.
    # This is a silent correctness bug, not a cosmetic warning: keep kekulize=False.
    Chem.MolToMolFile(m, sdf, kekulize=False)
    print(f"[ok] RDKit wrote {sdf} ({m.GetNumAtoms()} atoms, Hs added, aromatic bonds preserved)")

    # NOT -I here: isolated mode implies -P/-E, which strips both PYTHONPATH and the
    # script's own directory from sys.path -- and rosetta_py lives next to the script,
    # so -I makes molfile_to_params.py unimportable ("No module named 'rosetta_py'").
    # -s -E still drops user site-packages and the environment. Because this is a
    # script invocation, sys.path[0] is the SCRIPT's directory (the vendored tree we
    # intend to run), never cwd -- so the ligand work dir cannot inject a module.
    proc = subprocess.run(
        [sys.executable, "-s", "-E", M2P, "-n", "LIG", "-p", "LIG", sdf],
        cwd=work, capture_output=True, text=True, timeout=300,
    )
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print("[FAIL] molfile_to_params exited", proc.returncode)
        print(proc.stderr[-2000:])
        return 1

    # Guard the aromatic-bond regression described above.
    if "no aromatic bonds" in proc.stdout:
        print("[FAIL] molfile_to_params saw a Kekule structure -- aromatic rings "
              "will be mistyped. The SDF must be written with kekulize=False.")
        return 1
    print("[PASS] aromatic bonds carried through to Rosetta (no Kekule warning)")

    params = os.path.join(work, "LIG.params")
    pdb = os.path.join(work, "LIG_0001.pdb")
    for f in (params, pdb):
        if not os.path.exists(f):
            print(f"[FAIL] expected output missing: {f}")
            print("dir:", os.listdir(work))
            return 1

    natoms = sum(1 for ln in open(params) if ln.startswith("ATOM"))
    print(f"[PASS] wrote {params} ({natoms} ATOM records) and {pdb}")

    # The .params must actually load into Rosetta's residue type set.
    import pyrosetta
    pyrosetta.init(f"-mute all -extra_res_fa {params}", silent=True)
    pose = pyrosetta.pose_from_file(pdb)
    from pyrosetta.rosetta.core.scoring import get_score_function
    score = get_score_function()(pose)
    print(f"[PASS] PyRosetta loaded ligand pose: {pose.total_residue()} residue(s), "
          f"{pose.residue(1).natoms()} atoms, score={score:.2f}")
    print("\nRESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
