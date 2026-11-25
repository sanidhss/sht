"""Smoke test for env A: RDKit + OpenBabel + ProLIF + PyRosetta + molfile_to_params."""
import os, sys, glob, tempfile, subprocess

ok = True
def check(name, fn):
    global ok
    try:
        r = fn()
        print(f"[PASS] {name}: {r}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")

def rdkit_test():
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    AllChem.EmbedMolecule(m, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(m)
    return f"aspirin embedded, {m.GetNumAtoms()} atoms, MW={Descriptors.MolWt(m):.1f}"

def obabel_test():
    out = subprocess.run(["obabel", "-:CCO", "-osdf", "--gen3d"],
                         capture_output=True, text=True, timeout=120)
    assert "$$$$" in out.stdout, out.stderr[-300:]
    return "obabel gen3d -> SDF ok"

def prolif_test():
    import prolif, MDAnalysis as mda
    from prolif.datafiles import TOP, TRAJ
    u = mda.Universe(TOP, TRAJ)
    lig = u.select_atoms("resname LIG")
    prot = u.select_atoms("protein and byres around 6.0 group ligand",
                          ligand=lig)
    # fp.run() takes MDAnalysis AtomGroups and converts them internally;
    # passing prolif.Molecule here raises "'Molecule' object has no attribute 'atoms'".
    fp = prolif.Fingerprint()
    fp.run(u.trajectory[:1], lig, prot, progress=False)
    df = fp.to_dataframe()
    return f"prolif {prolif.__version__}, {df.shape[1]} interactions on 1 frame"

def pyrosetta_test():
    import pyrosetta
    pyrosetta.init("-mute all", silent=True)
    from pyrosetta import pose_from_sequence
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.scoring import get_score_function
    pose = pose_from_sequence("AAAAAAAAAA")
    sf = get_score_function()
    e0 = sf(pose)
    fr = FastRelax(sf, 1)
    fr.apply(pose)
    return f"relaxed 10-mer, score {e0:.1f} -> {sf(pose):.1f}"

def interface_analyzer_test():
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    return f"InterfaceAnalyzerMover importable: {InterfaceAnalyzerMover.__name__}"

def molfile_to_params_test():
    # NOT in the pip wheel -- vendored from RosettaCommons/rosetta by
    # setup/get_rosetta_tools.sh. Full conversion test: smoke_molfile_to_params.py
    p = "/work/hdd/bisa/sshah21/external/rosetta_tools/molfile_to_params.py"
    assert os.path.exists(p), f"missing {p}; run setup/get_rosetta_tools.sh"
    return p


def sdf_in_process_test():
    # In-process alternative to molfile_to_params: Rosetta's C++ SDF reader.
    import pyrosetta
    pyrosetta.init("-mute all", silent=True)
    from pyrosetta.rosetta.core.chemical import sdf
    assert hasattr(sdf, "convert_to_ResidueTypes")
    return "core.chemical.sdf.MolFileIOReader + convert_to_ResidueTypes available"

for n, f in [("rdkit", rdkit_test), ("openbabel", obabel_test), ("prolif", prolif_test),
             ("pyrosetta relax", pyrosetta_test),
             ("InterfaceAnalyzer", interface_analyzer_test),
             ("molfile_to_params", molfile_to_params_test),
             ("sdf in-process", sdf_in_process_test)]:
    check(n, f)

print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
