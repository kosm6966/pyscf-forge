import os
import re
import tempfile
import time

import numpy
import scipy
from pyscf.mgpaw import isdfx
from pyscf import lib
from pyscf.lib import chkfile
from pyscf.pbc.dft import multigrid
from pyscf.pbc.gto import Cell
from pyscf.scf.uhf import mulliken_spin_pop
from ase.io import read

TMPDIR = os.environ.get("PYSCF_TMPDIR", tempfile.gettempdir())
PYSCF_MAX_MEMORY = int(os.getenv("PYSCF_MAX_MEMORY", 800))
CONV_TOL = 1.0e-6


def dump_check(mf, dm=None, vj=None, vk=None):
    # Extract only serializable attributes from mf.cell
    cell_attrs = {
        "a": mf.cell.a,
        "unit": mf.cell.unit,
        "basis": mf.cell.basis,
        "atom": mf.cell.atom,
        "pseudo": mf.cell.pseudo,
        "mesh": mf.cell.mesh,
        "ke_cutoff": mf.cell.ke_cutoff,
        "_basis": mf.cell._basis,
        "_pseudo": mf.cell._pseudo,
    }

    chkfile.dump(mf.chkfile, "cell", cell_attrs)
    chkfile.dump(mf.chkfile, "hcore", mf.get_hcore())
    chkfile.dump(mf.chkfile, "overlap", mf.get_ovlp())
    if dm is None:
        dm = mf.make_rdm1()
    chkfile.dump(mf.chkfile, "dm", dm)
    if getattr(dm, "mo_occ", None) is not None:
        chkfile.dump(mf.chkfile, "mo_occ", dm.mo_occ)
    if getattr(dm, "mo_coeff", None) is not None:
        chkfile.dump(mf.chkfile, "mo_coeff", dm.mo_coeff)
    if vj is not None:
        chkfile.dump(mf.chkfile, "vj", vj)
    if vk is not None:
        chkfile.dump(mf.chkfile, "vk", vk)
    chkfile.dump(mf.chkfile, "en", mf.e_tot)
    return dm


def get_dm0_from_chk(cfn):
    dm = chkfile.load(cfn, "dm")
    mo_occ = chkfile.load(cfn, "mo_occ")
    mo_coeff = chkfile.load(cfn, "mo_coeff")
    dm = lib.tag_array(dm, mo_occ=mo_occ, mo_coeff=mo_coeff)
    en_pyscf = chkfile.load(cfn, "en")
    return en_pyscf, dm


def get_en(mf, dm, with_j=True, with_k=True):
    if with_j:
        t0 = time.time()
        vj = mf.with_df.get_jk(dm=dm, with_j=with_j, with_k=False, exxdiv='ewald')[0]
        t_jk = time.time() - t0
        print()
        print("get_j walltime:", round(t_jk, 2), flush=True)
        print()

    if with_k:
        t0 = time.time()
        vk = mf.with_df.get_jk(dm=dm, with_j=False, with_k=with_k, exxdiv='ewald')[1]
        t_jk = time.time() - t0
        print()
        print("get_k walltime:", round(t_jk, 2), flush=True)
        print()

    vhf = numpy.zeros_like(dm)
    if with_j:
        vhf += vj.real.astype(numpy.float64).reshape(vhf.shape)
    if with_k:
        vhf -= 0.5 * vk.real.astype(numpy.float64).reshape(vhf.shape)

    if dm.ndim == 2:
        vhf = vhf.reshape(-1, *dm.shape)
        dm = dm.reshape(-1, *dm.shape)

    e_coul = sum(numpy.einsum("ij,ji->", vi, di).real * 0.5 for vi, di in zip(vhf, dm))
    # e_tot = mf.energy_tot(dm=dm, vhf=vhf)
    return e_coul

def flip_spin(mf, dm, afm_guess, bias):
    cell = mf.cell

    # Function to strictly match the full AO label
    def find_exact_ao_indices(cell, target_label):
        pattern = rf"^{re.escape(target_label)}$"  # Ensure exact match
        return [i for i, label in enumerate(cell.ao_labels()) if re.match(pattern, label)]

    # Find AO indices for Cu 3dx2-y2 orbitals
    alpha_indices = []
    beta_indices = []

    for key, ao_labels in afm_guess.items():
        print(key)
        for label in ao_labels:
            ao_idx = find_exact_ao_indices(cell, label)  # Find indices of specific AOs
            print("flipping:", ao_idx, label, dm[0][ao_idx, ao_idx],dm[1][ao_idx, ao_idx])
            if key == "alpha":
                alpha_indices.extend(ao_idx)
            else:
                beta_indices.extend(ao_idx)

    # Apply AFM ordering by modifying the density matrix
    for ao_idx in alpha_indices:
        dm[0][ao_idx, ao_idx] += bias # α (spin-up)
        dm[1][ao_idx, ao_idx] *= 0.0  # Remove β (spin-down)

    for ao_idx in beta_indices:
        dm[0][ao_idx, ao_idx] *= 0.0  # Remove α (spin-up)
        dm[1][ao_idx, ao_idx] += bias

    s1e = mf.get_ovlp(cell)
    ne = numpy.einsum("xij,ji->x", dm, s1e).real
    nelec = cell.nelec
    if numpy.any(abs(ne - nelec) > 0.01):
        print(
            "Spin flip causes error in the electron number "
            "of initial guess density matrix (Ne/cell = %s)!\n"
            "  This can cause huge error in Fock matrix and "
            "lead to instability in SCF for low-dimensional "
            "systems.\n  DM is normalized wrt the number "
            "of electrons %s",
            ne,
            nelec,
        )
        dm *= (nelec / ne).reshape(2, 1, 1)

def get_dm0(cell, method, ncopy, init_guess="1e", afm_guess=None, bias=0.):
    # Use UHF to allow AFM ordering
    mf = method(cell)
    mf.chkfile = TMPDIR + "/dm0.chk"
    # mf.with_df = multigrid.multigrid_pair.MultiGridFFTDF2(cell)
    # mf.with_df.ngrids = 4
    dm = mf.get_init_guess(key=init_guess)
    if afm_guess is not None:
        afm_guess = afm_guess_for_supercell(afm_guess, numpy.prod(ncopy), cell.natm)
        flip_spin(mf, dm, afm_guess, bias)
    if "uhf" in str(method).lower():
        mulliken_spin_pop(cell, dm)
    dump_check(mf, dm)
    return dm

def make_rdm1(mo_occ, mo_coeff):
    tol = 1.e-6
    return (mo_coeff[:,mo_occ>tol] * mo_occ[mo_occ>tol]) @ mo_coeff[:,mo_occ>tol].T

def test_gdf(cell, dm, method, converged, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------GDF----------")
    print()
    mf = method(cell).density_fit()
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/gdf.chk"
    mf.init_guess = dm
    if not converged:
        t0 = time.time()
        mf.with_df.build(j_only=not with_k)
        print()
        print("build walltime         :", round(time.time() - t0, 2), flush=True)
        print()
        en = get_en(mf, dm, with_j=with_j, with_k=with_k)
    else:
        en = mf.kernel(dm0=dm)
        dm = dump_check(mf)
        if "uhf" in str(method).lower():
            mulliken_spin_pop(cell, dm)
    print(f"GDF FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()

    return en


def test_pymg(cell, dm, method, converged, mg_version, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------PYSCF-MG----------")
    print()
    if "ks" not in str(method).lower():
        raise ValueError("Must use DFT")

    mf = method(cell)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/fftdf.chk"
    mf.init_guess = dm

    if mg_version == 2:
        mf.with_df = multigrid.multigrid_pair.MultiGridFFTDF2(cell)
        mf.with_df.ngrids = 4
    elif mg_version == 1:
        mf.with_df = multigrid.multigrid.MultiGridFFTDF(cell)

    if not converged:
        en = get_en(mf, dm, with_j=with_j, with_k=with_k)
    else:
        en = mf.kernel()
        dm = dump_check(mf)
        if "uhf" in str(method).lower():
            mulliken_spin_pop(cell, dm)
    print(f"FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return en, dm


def test_fftdf(cell, dm, method, converged, with_k=True, with_j=True, en_pyscf=None, max_cycle=50):
    print()
    print("----------FFTDF----------")
    print()
    mf = method(cell)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/fftdf.chk"
    mf.init_guess = dm
    mf.max_cycle = max_cycle
    if not converged:
        en = get_en(mf, dm, with_j=with_j, with_k=with_k)
    else:
        en = mf.kernel(dm0=dm)
        dm = dump_check(mf)
        if "uhf" in str(method).lower():
            mulliken_spin_pop(cell, dm)
    print(f"FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return en, dm


def test_occRI_fftdf(cell, dm, method, converged, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------occRI-FFTDF----------")
    print()
    mf = method(cell)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/occRI.chk"
    mf.with_df = isdfx.ISDFX(
        cell,
        multigrid_on=False,
        fit_dense_grid=False,
        fit_sparse_grid=False,
        direct_k_sparse=True
    )
    mf.init_guess = dm
    mf.cell = mf.with_df.cell
    if not converged:
        t0 = time.time()
        mf.with_df.build(incore=False, with_j=with_j, with_k=with_k)
        print()
        print("build walltime         :", round(time.time() - t0, 2), flush=True)
        print()
        en = get_en(mf, dm, with_j=with_j, with_k=with_k)
    else:
        en = mf.kernel(dm0=dm)
        dm = dump_check(mf)

        # if "uhf" in str(method).lower():
        #     mulliken_spin_pop(mf.cell, dm)
    mf.with_df.print_times()
    print(f"occRI FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / mf.cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return en, dm


def test_mg_isdf_occRI(
    cell,
    ncopy,
    dm,
    method,
    converged,
    ke_eps,
    isdf_thresh,
    alpha_cut=2.8,
    with_k=True,
    with_j=True,
    en_pyscf=None,
    max_cycle=50
):
    print()
    print("----------MG-ISDF-occRI----------")
    print()
    mf = method(cell)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/mg-isdf.chk"
    mf.init_guess = dm
    mf.max_cycle = max_cycle
    mf.with_df = isdfx.ISDFX(
        cell,
        ncopy=ncopy,
        ke_epsilon=ke_eps,
        isdf_thresh=isdf_thresh,
        alpha_cutoff=alpha_cut,
        multigrid_on=True,
        fit_dense_grid=True,
        fit_sparse_grid=False,
        # get_j_from_pyscf=True
    )
    mf.cell = mf.with_df.cell 
    if not converged:
        t0 = time.time()
        mf.with_df.build(with_j=with_j, with_k=with_k)
        print()
        print("build walltime         :", round(time.time() - t0, 2), flush=True)
        print()
        en = get_en(mf, dm, with_j=with_j, with_k=with_k)
    else:
        en = mf.kernel(dm0=dm)
        dm = dump_check(mf)
        if "uhf" in str(method).lower():
            mulliken_spin_pop(mf.with_df.cell, dm)
    mf.with_df.print_times()
    print(f"ISDFX Energy (Ha)     : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / mf.with_df.cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return dm


def afm_guess_for_supercell(afm_guess, ncopies, natm):
    # New dictionary to store expanded values
    expanded_afm_guess = {"alpha": [], "beta": []}

    for spin in afm_guess:
        for entry in afm_guess[spin]:
            atom_index, element, orbital = entry.split(maxsplit=2)
            atom_index = int(atom_index)

            # Duplicate for the supercell
            for i in range(ncopies):
                new_index = atom_index + i * natm // ncopies
                expanded_afm_guess[spin].append(f"{new_index} {element} {orbital}")

    print(expanded_afm_guess)
    return expanded_afm_guess



# Copy of pyscf.pbc.tools.lattice

A2B = 1.889725989

def get_ase_cell(formula, kecut, basis="gth-dzv", pseudo="gth-pbe",  exp_to_discard=0., spin=0, verbose=4):
    ase_atom = get_ase_atom(formula)
    atom = [[atom.symbol, atom.position] for atom in ase_atom]
    cell = Cell(
        a=ase_atom.cell[:],
        unit="B",
        atom=atom,
        exp_to_discard=exp_to_discard,
        ke_cutoff = kecut,
        basis = basis,
        pseudo = pseudo,
        spin = spin,
        verbose = verbose,
    )
    cell.build()
    return cell

def get_ase_atom(formula):
    from ase.build import bulk
    formula = formula.lower()
    assert formula in ['lih','c', 'nio'] or formula[-5:] == '.vasp'
    if formula == 'lih':
        ase_atom = bulk('LiH', 'rocksalt', a=4.0834*A2B, cubic=True) # E. Zintl and A. Harder, Z. Phys. Chem. B 32 (1936) 113
    elif formula == 'c':
        from ase.lattice.cubic import Diamond
        ase_atom = Diamond(symbol='C', latticeconstant=3.5668*A2B) # CRC Handbook of Chemistry and Physics
    elif formula == 'nio':
        # AFM
        ase_atom = bulk('NiO', 'rocksalt', a=4.17*A2B, cubic=True) #Terakura et al., Phys. Rev. B 30, 4734 (1984)
    elif formula[-5:] =='.vasp':
        ase_atom = read(formula)
        ase_atom.positions *= A2B
        ase_atom.cell.array *= A2B
    return ase_atom

