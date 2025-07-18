import os
import re
import tempfile
import time

import numpy
from pyscf.mgpaw import isdfx
from pyscf import lib
from pyscf.lib import chkfile
from pyscf.pbc.dft import multigrid

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
        vj = mf.with_df.get_jk(dm=dm, kpts=mf.kpts, with_j=with_j, with_k=False, exxdiv='ewald')[0]
        t_jk = time.time() - t0
        print()
        print("get_j walltime:", round(t_jk, 2), flush=True)
        print()

    if with_k:
        t0 = time.time()
        vk = mf.with_df.get_jk(dm=dm, kpts=mf.kpts, with_j=False, with_k=with_k, exxdiv='ewald')[1]
        t_jk = time.time() - t0
        print()
        print("get_k walltime:", round(t_jk, 2), flush=True)
        print()

    vhf = numpy.zeros_like(dm, numpy.complex128)
    if with_j:
        vhf += vj.reshape(vhf.shape)
    if with_k:
        vhf -= 0.5 * vk.reshape(vhf.shape)

    if dm.ndim == 2:
        vhf = vhf.reshape(-1, *dm.shape)
        dm = dm.reshape(-1, *dm.shape)

    e_coul = sum(numpy.einsum("ij,ji->", vi, di).real * 0.5 for vi, di in zip(vhf, dm)) / dm.shape[0]
    return e_coul


def flip_spin(mf, dm, afm_guess):
    cell = mf.cell

    # Function to strictly match the full AO label
    def find_exact_ao_indices(cell, target_label):
        pattern = rf"^{re.escape(target_label)}$"  # Ensure exact match
        return [i for i, label in enumerate(cell.ao_labels()) if re.match(pattern, label)]

    # Find AO indices for Cu 3dx2-y2 orbitals
    alpha_indices = []
    beta_indices = []

    for key, ao_labels in afm_guess.items():
        for label in ao_labels:
            ao_idx = find_exact_ao_indices(cell, label)  # Find indices of specific AOs
            print("flipping:", ao_idx, label, dm.mo_occ[0][ao_idx], dm.mo_occ[1][ao_idx])
            if key == "alpha":
                alpha_indices.extend(ao_idx)
            else:
                beta_indices.extend(ao_idx)

    # Apply AFM ordering by modifying the density matrix
    for ao_idx in alpha_indices:
        dm[0][ao_idx, ao_idx] *= 1.0  # α (spin-up)
        dm[1][ao_idx, ao_idx] *= 0.0  # Remove β (spin-down)

        dm.mo_occ[0][ao_idx] *= 1.0  # α (spin-up)
        dm.mo_occ[1][ao_idx] *= 0.0  # Remove β (spin-down)

    for ao_idx in beta_indices:
        dm[0][ao_idx, ao_idx] *= 0.0  # Remove α (spin-up)
        dm[1][ao_idx, ao_idx] *= 1.0  # β (spin-down)

        dm.mo_occ[0][ao_idx] *= 0.0  # Remove α (spin-up)
        dm.mo_occ[1][ao_idx] *= 1.0  # β (spin-down)

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
        coeff, occ = dm.mo_coeff, dm.mo_occ  ###
        dm *= (nelec / ne).reshape(2, 1, 1)
        occ *= (nelec / ne).reshape(2, 1)  ###
        dm = lib.tag_array(dm, mo_coeff=coeff, mo_occ=occ)
    return dm


def get_dm0(cell, method, kmesh, init_guess="minao", afm_guess=None, smearing=None):
    # Use UHF to allow AFM ordering
    kpts = cell.make_kpts(kmesh)
    mf = method(cell, kpts=kpts)
    mf.chkfile = TMPDIR + "/dm0.chk"
    # mf.with_df = multigrid.multigrid_pair.MultiGridFFTDF2(cell)
    # mf.with_df.ngrids = 4
    dm = mf.get_init_guess(key=init_guess)
    if smearing is not None:
        dm[0] += smearing  # Add small positive density to α
        dm[1] -= smearing  # Subtract from β    

    dump_check(mf, dm)
    return dm


def test_gdf(cell, dm, method, converged, kmesh, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------GDF----------")
    print()
    kpts = cell.make_kpts(kmesh)
    mf = method(cell, kpts=kpts).density_fit()
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/gdf.chk"
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
    print(f"GDF FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()

    return en, dm


def test_pymg(cell, dm, method, converged, mg_version, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------PYSCF-MG----------")
    print()
    if "ks" not in str(method).lower():
        raise ValueError("Must use DFT")

    mf = method(cell)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/fftdf.chk"

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
    print(f"FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return en, dm


def test_fftdf(cell, dm, method, converged, kmesh, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------FFTDF----------")
    print()
    kpts = cell.make_kpts(kmesh)
    mf = method(cell, kpts=kpts)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/fftdf.chk"
    if not converged:
        en = get_en(mf, dm, with_j=with_j, with_k=with_k)
    else:
        en = mf.kernel(dm0=dm)
        dm = dump_check(mf)
    print(f"FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return en, dm


def test_occRI_fftdf(cell, dm, method, converged, kmesh, with_k=True, with_j=True, en_pyscf=None):
    print()
    print("----------occRI-FFTDF----------")
    print()
    kpts = cell.make_kpts(kmesh)
    mf = method(cell, kpts=kpts)
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/occRI.chk"
    mf.with_df = isdfx.ISDFX(
        cell,
        kmesh=kmesh,
        multigrid_on=False,
        fit_dense_grid=False,
        fit_sparse_grid=False,
        use_kpt_symm=True,
        direct_k_sparse=True,
    )
    mf.init_guess = dm
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
    mf.with_df.print_times()
    print(f"occRI FFTDF Energy (Ha) : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / mf.cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
    return en, dm


def test_mg_isdf_occRI(
    cell,
    kmesh,
    dm,
    method,
    converged,
    ke_eps,
    isdf_thresh,
    alpha_cut=2.8,
    with_k=True,
    with_j=True,
    en_pyscf=None,
):
    print()
    print("----------MG-ISDF-occRI----------")
    print()
    kpts = cell.make_kpts(kmesh)
    mf = method(cell, kpts=kpts)
    mf.with_df = isdfx.ISDFX(
        cell,
        kmesh=kmesh,
        ke_epsilon=ke_eps,
        isdf_thresh=isdf_thresh,
        alpha_cutoff=alpha_cut,
        multigrid_on=True,
        fit_dense_grid=True,
        fit_sparse_grid=False,
        use_kpt_symm=True,
        direct_k_sparse=True,
        isdf_pts_from_gamma_point=False
    )
    mf.conv_tol = CONV_TOL
    mf.chkfile = TMPDIR + "/mg-isdf.chk"
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
    mf.with_df.print_times()
    print(f"ISDFX Energy (Ha)     : {en:8.10f}", flush=True)
    if en_pyscf is not None:
        err = abs(en - en_pyscf) / mf.with_df.cell.natm
        print(f"Error per atom (Ha)   : {err:8.2e}", flush=True)
        print()
