#!/usr/bin/env python3
"""
PySCF calculation for antiferromagnetic NiO in rock-salt structure.

NiO has a face-centered cubic (fcc) structure with antiferromagnetic ordering
along the (111) direction. The unit cell contains 4 Ni atoms with opposite 
spins and 4 O atoms.

Expected Results with PBE0:
- Magnetic moment: ~1.6-1.8 μB per Ni (exp: ~1.64-1.9 μB)
- AFM state is ground state (more stable than FM or NM)
"""

import re
import numpy as np
from pyscf.pbc import gto, scf
from pyscf.scf.uhf import mulliken_pop
from ase.build import bulk

# Build AFM NiO structure
# Rock-salt structure with AFM ordering along (111) direction
# Set up the magnetic unit cell for AFM ordering
# Use conventional fcc cell with 4 formula units to capture AFM
# Two Ni atoms with spin up, two with spin down
# Terakura et al., Phys. Rev. B 30, 4734 (1984)
ase_atom = bulk('NiO', 'rocksalt', a=4.17*1.889725989, cubic=True)
atom = [[atom.symbol, atom.position] for atom in ase_atom]
cell = gto.Cell(
    a=ase_atom.cell[:],
    unit="B",
    atom=atom,
    ke_cutoff = 190,
    verbose = 4,
    basis = "gth-dzvp-molopt-sr",
    pseudo = "gth-hf",
)
cell.build()

print("="*70)
print("Antiferromagnetic NiO Calculation with PySCF")
print("="*70)

# Set k-point mesh for Brillouin zone sampling
# Using smaller 2x2x2 k-point grid for hybrid functional (computationally expensive)
kpts = cell.make_kpts([1, 1, 1])

# Perform spin-polarized HF calculation with AFM initial guess using hybrid functional
mf = scf.KUHF(cell, kpts=kpts)
mf.conv_tol = 1.e-6

print(f"Number of k-points: {len(kpts)}")
print(f"FFT mesh: {cell.mesh}")

# Create AFM spin configuration
# Manipulate the density matrix to set initial magnetization
# Positive values for spin up Ni, negative for spin down Ni
afm_guess = {
    "alpha": ["0 Ni 3dx2-y2", "2 Ni 3dx2-y2"],  # Spin-up on Ni 0 and Ni 2
    "beta":  ["4 Ni 3dx2-y2", "6 Ni 3dx2-y2"]   # Spin-down on Ni 1 and Ni 3
}
ni_bias = 0.8

# Get the default initial guess and modify it for AFM
dm = mf.get_init_guess(key="minao")

def print_spin_pop(dm):
    print("Atom  Element  Moment")
    print("-" * 25)
    for k_idx in range(len(mf.kpts)):
        dma = dm[0][k_idx]
        dmb = dm[1][k_idx]
        (pop_a,pop_b), _ = mulliken_pop(cell, (dma, dmb), verbose=0)
        nelec_a = np.zeros(cell.natm)
        nelec_b = np.zeros(cell.natm)
        for i, s in enumerate(cell.ao_labels(fmt=None)):
            nelec_a[s[0]] += pop_a[i]
            nelec_b[s[0]] += pop_b[i]
        for ia in range(cell.natm):
            print(f"{ia:3d}   {cell.atom_symbol(ia):>2s}      {(nelec_a[ia]-nelec_b[ia]):+5.2f}")

print("\nInitial spin density:")
print_spin_pop(dm)

# dm is a tuple (dm_alpha, dm_beta) for unrestricted calculations
# We need to modify the density matrices to encode the spin polarization
# For each k-point, adjust the density matrix based on atomic spin configuration

# Create initial spin density by scaling atomic densities
# This is a more robust approach than manually manipulating DM
def bias_afm_state(mf, dm, afm_guess):
    nk = len(mf.kpts)
    cell = mf.cell
    
    # Function to strictly match the full AO label
    def find_exact_ao_indices(cell, target_label):
        pattern = rf"^{re.escape(target_label)}$"  # Ensure exact match
        return [i for i, label in enumerate(cell.ao_labels()) if re.match(pattern, label)]

    # Apply AFM ordering by modifying the density matrix
    dm_alpha, dm_beta = dm
    for k_idx in range(nk):

        # Find AO indices for Cu 3dx2-y2 orbitals
        alpha_indices = []
        beta_indices = []

        for key, ao_labels in afm_guess.items():
            for label in ao_labels:
                ao_idx = find_exact_ao_indices(cell, label)  # Find indices of specific AOs
                if key == "alpha":
                    alpha_indices.extend(ao_idx)
                else:
                    beta_indices.extend(ao_idx)

        for ao_idx in alpha_indices:
            dm_alpha[k_idx][ao_idx, ao_idx] += ni_bias
            dm_beta[k_idx][ao_idx, ao_idx]  -= ni_bias

        for ao_idx in beta_indices:
            dm_alpha[k_idx][ao_idx, ao_idx] -= ni_bias
            dm_beta[k_idx][ao_idx, ao_idx]  += ni_bias

    return dm_alpha, dm_beta

# Apply the AFM initial guess
dm = bias_afm_state(mf, dm, afm_guess)
print("\nAFM spin density:")
print_spin_pop(dm)
print("\n✓ AFM initial density matrix generated and will be used in SCF")

# Run the self-consistent field calculation
print("\n" + "="*70)
print("Starting SCF calculation...")
print("="*70)

try:
    e_tot = mf.kernel(dm0=dm)

    # Check convergence
    if mf.converged:
        print("\n✓ SCF calculation converged successfully!")
    else:
        print("\n✗ Warning: SCF did not converge. Try:")
        print("  - Adjusting initial guess")
        print("  - Increasing max_cycle")
        print("  - Using level shift or damping")    
    
    print("\n" + "="*70)
    print("SCF Calculation Results")
    print("="*70)
    print(f"Total energy: {e_tot:.6f} Ha ({e_tot * 27.2114:.3f} eV)")

    # Analyze magnetic moments via Mulliken population
    print("\n" + "="*70)
    print("Magnetic Moments (Mulliken Analysis)")
    print("="*70)
    print_spin_pop(mf.make_rdm1())
    
    # Expected comparison
    print("\n" + "="*70)
    print("Comparison with Literature")
    print("="*70)
    print("Expected magnetic moment per Ni: ~1.6-1.8 μB")
    
except Exception as e:
    print(f"\nError during calculation: {e}")
    print("\nTroubleshooting tips:")
    print("1. Check if basis set and pseudopotentials are available")
    print("2. Reduce k-point mesh for faster testing")
    print("3. Try different initial guess methods")

print("\n" + "="*70)
print("Additional Notes")
print("="*70)
print("""
AFM Configuration:
- This setup uses type-II AFM (ferromagnetic (111) planes)
- Spins alternate along (111) direction
- This is the experimentally observed ground state

Citations:
- "Predicting the properties of NiO with density functional theory"
  APL Materials 11, 060702 (2023)
- Franchini et al., Phys. Rev. B 72, 045132 (2005) - PBE0 for NiO
""")

# === Using ISDFX ===
from pyscf.occri.isdfx import ISDFX
mf = scf.KUHF(cell, kpts=kpts)
mf.conv_tol = 1.e-6
mf.with_df = ISDFX.from_mf(mf)

try:
    e_tot_isdfx = mf.kernel(dm0=np.asarray(dm))

    # Check convergence
    if mf.converged:
        print("\n✓ SCF calculation converged successfully!")
    else:
        print("\n✗ Warning: SCF did not converge. Try:")
        print("  - Adjusting initial guess")
        print("  - Increasing max_cycle")
        print("  - Using level shift or damping")    
    
    print("\n" + "="*70)
    print("SCF Calculation Results")
    print("="*70)
    print(f"Total energy: {e_tot_isdfx:.6f} Ha")
    print(f"Total FFTDF energy: {e_tot:.6f} Ha")
    print(f"Error per atom: {abs(e_tot_isdfx-e_tot)/cell.natm:.6f} Ha")

    # Analyze magnetic moments via Mulliken population
    print("\n" + "="*70)
    print("Magnetic Moments (Mulliken Analysis)")
    print("="*70)
    print_spin_pop(mf.make_rdm1())

except Exception as e:
    print(f"\nError during calculation: {e}")