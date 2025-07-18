import pyscf
import numpy
import re
from pyscf.pbc.gto import Cell
from ase.build import bulk
from pyscf.scf.uhf import mulliken_spin_pop

A2B = 1.889725989

def get_ase_cell():
    ase_atom = bulk('NiO', 'rocksalt', a=4.17*A2B, cubic=True) # 10.1103/PhysRevB.74.155108
    atom = [[atom.symbol, atom.position] for atom in ase_atom]
    cell = Cell(
        a=ase_atom.cell[:],
        unit="B",
        atom=atom,
        ke_cutoff = 190,
        basis = "gth-dzvp-molopt-sr",
        pseudo = "gth-pbe",
        spin = 0,
        verbose = 4,
    )
    cell.build()
    return cell

def flip_spin(mf, dm, afm_guess):
    cell = mf.cell

    # Function to strictly match the full AO label
    def find_exact_ao_indices(cell, target_label):
        pattern = rf"^{re.escape(target_label)}$"  # Ensure exact match
        return [i for i, label in enumerate(cell.ao_labels()) if re.match(pattern, label)]

    # Find AO indices for 3dx2-y2 orbitals
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
    bias = 0.9 # I've found that this give ~ +/- 1.9 on Ni in line with AFM.
    #DOI: 10.1103/PhysRevB.74.155108
    for ao_idx in alpha_indices:
        dm[0][ao_idx, ao_idx] += bias
        dm[1][ao_idx, ao_idx] *= 0.0

    for ao_idx in beta_indices:
        dm[0][ao_idx, ao_idx] *= 0.0
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

def get_dm0(cell, init_guess, afm_guess):
    # Use UHF to allow AFM ordering
    mf = pyscf.pbc.scf.UHF(cell)
    dm = mf.get_init_guess(key=init_guess)
    flip_spin(mf, dm, afm_guess)
    return dm

def get_en(cell, dm):
    mf = pyscf.pbc.scf.UHF(cell)
    mf.conv_tol = 1.e-6
    mf.init_guess = dm0
    mf.kernel(dm0=dm0)
    dm = mf.make_rdm1()
    mulliken_spin_pop(cell, dm)


if __name__ == "__main__":

    cell = get_ase_cell()

    afm_guess = {
        "alpha": ["0 Ni 3dx2-y2", "2 Ni 3dx2-y2"],  # Spin-up on Ni 0 and Ni 2
        "beta":  ["4 Ni 3dx2-y2", "6 Ni 3dx2-y2"],  # Spin-down on Ni 1 and Ni 3
    }

    dm0 = get_dm0(cell, "minao", afm_guess)
    get_en(cell, dm0)
    
    dm0 = get_dm0(cell, "atom", afm_guess)
    get_en(cell, dm0)