"""
Interpolative Separable Density Fitting with eXchange (ISDFX) - Grid-based FFT operations.

This module implements advanced grid-based algorithms for efficient evaluation of
Coulomb and exchange integrals in periodic quantum chemistry calculations. The core
methodology uses interpolative separable density fitting combined with fast Fourier
transforms for computational efficiency.

This file is part of a proprietary software owned by Sandeep Sharma (sanshar@gmail.com).
Unless the parties otherwise agree in writing, users are subject to the following terms.

(0) This notice supersedes all of the header license statements.
(1) Users are not allowed to show the source code to others, discuss its contents, 
    or place it in a location that is accessible by others.
(2) Users can freely use resulting graphics for non-commercial purposes. 
    Credits shall be given to the future software of Sandeep Sharma as appropriate.
(3) Sandeep Sharma reserves the right to revoke the access to the code any time, 
    in which case the users must discard their copies immediately.

Key Concepts:
-------------
- Multigrid approach: Multiple grids with different densities for hierarchical accuracy
- Atom-centered grids: Grids focused around atomic positions for local basis functions
- Universal grids: Single grid covering entire unit cell for global operations
- Voronoi partitioning: Assigning grid points to nearest atoms for load balancing
- ISDF decomposition: Interpolative separable density fitting for rank reduction
- FFT acceleration: Fast Fourier transforms for efficient convolutions

Main Components:
----------------
- FFT execution functions for real/complex transforms
- Grid construction and partitioning algorithms  
- Coulomb (J) and exchange (K) integral evaluation
- k-point sampling support for periodic boundary conditions
- Memory management and parallel execution utilities

Dependencies:
-------------
- numpy: Array operations and linear algebra
- scipy: Scientific computing utilities  
- pyscf: Quantum chemistry framework
- pyfftw: Fast Fourier transform library
- joblib: Parallel processing utilities
"""

import os
import contextlib
import threading
import time
import functools

# Scientific computing libraries
import numpy
import scipy
import pyscf
from pyscf.mgpaw import libevalao
from pyscf import lib
from pyscf.pbc import tools
import pyfftw
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed, parallel_backend


# =============================================================================
# FFT Execution Functions
# =============================================================================

def r2c_fftw_execute(x, mesh, normalize=True, return_x=False):
    """
    Execute real-to-complex FFT using FFTW.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input real array to transform
    mesh : tuple
        Grid dimensions (nx, ny, nz)
    normalize : bool, default=True
        Whether to normalize by 1/sqrt(N)
    return_x : bool, default=False
        If True, return new array; if False, modify input array
        
    Returns:
    --------
    numpy.ndarray
        Complex FFT result
    """
    fft_in, fft_out, fft, _, idx = get_fft_plan(mesh)
    
    # Copy input data to FFT input buffer
    fft_in[:] = x.real.reshape(mesh)
    
    # Execute FFT
    fft()
    
    # Apply normalization if requested
    ng = numpy.prod(mesh)
    if normalize:
        fft_out *= 1.0 / ng**0.5
    
    # Prepare output array
    if return_x:
        out = numpy.zeros(ng, numpy.complex128)
    else:
        out = x.view()
        out *= 0.0
    
    # Copy FFT result to output
    out[idx] = fft_out.ravel()
    return out


def c2c_fftw_execute(x, mesh, normalize=True, return_x=False):
    """
    Execute complex-to-complex FFT using FFTW.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input complex array to transform
    mesh : tuple
        Grid dimensions (nx, ny, nz)
    normalize : bool, default=True
        Whether to normalize by 1/sqrt(N)
    return_x : bool, default=False
        If True, return new array; if False, modify input array
        
    Returns:
    --------
    numpy.ndarray
        Complex FFT result
    """
    fft_in, fft_out, fft = get_fft_plan(mesh)[:3]
    
    # Copy input data to FFT input buffer
    fft_in[:] = x.reshape(mesh)
    
    # Execute FFT
    fft()
    
    # Apply normalization if requested
    ng = numpy.prod(mesh)
    if normalize:
        fft_out *= 1.0 / ng**0.5
    
    # Prepare output array
    if return_x:
        out = numpy.zeros(ng, numpy.complex128)
    else:
        out = x.view()
        out *= 0.0
    
    # Copy FFT result to output
    out[:] = fft_out.ravel()
    return out


def c2r_ifftw_execute(x, mesh, normalize=True, return_x=False):
    """
    Execute complex-to-real inverse FFT using FFTW.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input complex array to inverse transform
    mesh : tuple
        Grid dimensions (nx, ny, nz)
    normalize : bool, default=True
        Whether to normalize by sqrt(N)
    return_x : bool, default=False
        If True, return new array; if False, modify input array
        
    Returns:
    --------
    numpy.ndarray
        Real inverse FFT result
    """
    fft_in, fft_out, _, ifft = get_fft_plan(mesh)[:4]
    
    # Prepare input data for inverse FFT
    y = x.view()
    y.shape = mesh
    fft_out[..., :(mesh[2] // 2 + 1)] = y[..., :(mesh[2] // 2 + 1)]
    
    # Execute inverse FFT
    ifft()
    
    # Apply normalization if requested
    ng = numpy.prod(mesh)
    if normalize:
        fft_in *= ng**0.5
    
    # Prepare output array
    if return_x:
        out = numpy.zeros(ng, numpy.float64)
    else:
        out = x.view()
        out *= 0.0
    
    # Copy inverse FFT result to output
    out[:] = fft_in.ravel()
    return out


def c2c_ifftw_execute(x, mesh, normalize=True, return_x=False):
    """
    Execute complex-to-complex inverse FFT using FFTW.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Input complex array to inverse transform
    mesh : tuple
        Grid dimensions (nx, ny, nz)
    normalize : bool, default=True
        Whether to normalize by sqrt(N)
    return_x : bool, default=False
        If True, return new array; if False, modify input array
        
    Returns:
    --------
    numpy.ndarray
        Complex inverse FFT result
    """
    fft_in, fft_out, _, ifft = get_fft_plan(mesh)[:4]
    
    # Prepare input data for inverse FFT
    y = x.view()
    y.shape = mesh
    fft_out[:] = y[:]
    
    # Execute inverse FFT
    ifft()
    
    # Apply normalization if requested
    ng = numpy.prod(mesh)
    if normalize:
        fft_in *= ng**0.5
    
    # Prepare output array
    if return_x:
        out = numpy.zeros(ng, numpy.complex128)
    else:
        out = x.view()
        out *= 0.0
    
    # Copy inverse FFT result to output
    out[:] = fft_in.ravel()
    return out


# =============================================================================
# FFT Plan Management
# =============================================================================

# Global storage for FFT plan factories and thread-local storage
_fft_plan_factories = {}
_fft_plan_tls = threading.local()


def register_fft_factory(mesh, threads):
    """
    Register an FFT plan factory for real-to-complex transforms.
    
    Parameters:
    -----------
    mesh : tuple
        Grid dimensions (nx, ny, nz)
    threads : int
        Number of threads for FFT execution
    """
    mesh = tuple(map(int, mesh))

    def cut_idx(mesh):
        """
        Generate flat indices for the reduced complex output grid.
        For real-to-complex FFT, only half of the z-dimension is stored.
        """
        nx, ny, nz = mesh
        # Compute flat indices efficiently using broadcasting
        i_idx = numpy.arange(nx)[:, None, None]
        j_idx = numpy.arange(ny)[None, :, None]
        k_idx = numpy.arange(nz // 2 + 1)[None, None, :]
        
        return ((i_idx * ny + j_idx) * nz + k_idx).ravel()

    if mesh not in _fft_plan_factories:
        # Output shape for real-to-complex FFT (reduced in z-dimension)
        shape_out = (mesh[0], mesh[1], mesh[2] // 2 + 1)

        def factory():
            """Create FFTW plans and aligned arrays for this mesh size."""
            # Create aligned arrays for optimal performance
            a = pyfftw.zeros_aligned(mesh, dtype="float64")
            b = pyfftw.zeros_aligned(shape_out, dtype="complex128")
            
            # Create forward FFT plan
            fft = pyfftw.FFTW(
                a, b,
                axes=(0, 1, 2),
                direction="FFTW_FORWARD",
                threads=threads,
                flags=("FFTW_MEASURE",),
            )
            
            # Create inverse FFT plan
            ifft = pyfftw.FFTW(
                b, a,
                axes=(0, 1, 2),
                direction="FFTW_BACKWARD",
                threads=threads,
                flags=("FFTW_MEASURE",),
            )
            
            # Generate index mapping for reduced output
            idx = cut_idx(mesh)
            return a, b, fft, ifft, idx

        _fft_plan_factories[mesh] = factory


def register_fft_factory_kpts(mesh, threads):
    """
    Register an FFT plan factory for complex-to-complex transforms (k-point sampling).
    
    Parameters:
    -----------
    mesh : tuple
        Grid dimensions (nx, ny, nz)
    threads : int
        Number of threads for FFT execution
    """
    mesh = tuple(map(int, mesh))

    if mesh not in _fft_plan_factories:
        def factory():
            """Create FFTW plans and aligned arrays for complex transforms."""
            # Create aligned arrays for complex data
            a = pyfftw.zeros_aligned(mesh, dtype="complex128")
            b = pyfftw.zeros_aligned(mesh, dtype="complex128")
            
            # Create forward FFT plan
            fft = pyfftw.FFTW(
                a, b,
                axes=(0, 1, 2),
                direction="FFTW_FORWARD",
                threads=threads,
                flags=("FFTW_MEASURE",),
            )
            
            # Create inverse FFT plan
            ifft = pyfftw.FFTW(
                b, a,
                axes=(0, 1, 2),
                direction="FFTW_BACKWARD",
                threads=threads,
                flags=("FFTW_MEASURE",),
            )
            
            # No index mapping needed for complex-to-complex
            return a, b, fft, ifft, None

        _fft_plan_factories[mesh] = factory


def get_fft_plan(mesh):
    """
    Get or create FFT plan for the specified mesh.
    Uses thread-local storage to ensure thread safety.
    
    Parameters:
    -----------
    mesh : tuple
        Grid dimensions (nx, ny, nz)
        
    Returns:
    --------
    tuple
        (input_array, output_array, fft_plan, ifft_plan, index_mapping)
    """
    mesh = tuple(map(int, mesh))
    
    # Initialize thread-local storage if needed
    if not hasattr(_fft_plan_tls, "plans"):
        _fft_plan_tls.plans = {}
    
    # Create plan for this mesh if it doesn't exist
    if mesh not in _fft_plan_tls.plans:
        _fft_plan_tls.plans[mesh] = _fft_plan_factories[mesh]()
    
    return _fft_plan_tls.plans[mesh]


# =============================================================================
# Grid Classes
# =============================================================================

class Grid:
    """
    Represents a uniform grid for electronic structure calculations.
    """
    
    def __init__(self, myisdf, ke_cutoff, mesh=None):
        """
        Initialize grid with specified kinetic energy cutoff.
        
        Parameters:
        -----------
        myisdf : object
            ISDF object containing cell information
        ke_cutoff : float
            Kinetic energy cutoff in Hartree
        mesh : tuple, optional
            Grid dimensions; if None, calculated from cutoff
        """
        cell = myisdf.cell
        
        # Calculate mesh from cutoff if not provided
        if mesh is None:
            mesh = tools.cutoff_to_mesh(cell.lattice_vectors(), ke_cutoff)

        self.mesh = numpy.array(mesh * 1, numpy.int32)
        self.Ng = numpy.prod(mesh)  # Total number of grid points
        self.universal = False
        
        # Generate uniform grid coordinates
        self.coords = cell.get_uniform_grids(mesh=self.mesh, wrap_around=True)

        # Initialize counters
        self.Ng_nonzero = 0  # Number of non-zero grid points
        self.nthc = 0        # Number of ... (purpose unclear from context)
        self.nao = 0         # Number of atomic orbitals

    def copy(self):
        """
        Returns a shallow copy of the grid.
        
        Returns:
        --------
        Grid
            Shallow copy of this grid object
        """
        return self.view(self.__class__)


class AtomGrid:
    """
    Represents a grid associated with a specific atom and its basis functions.
    """
    
    def __init__(self, shells_on_grid, alpha, l, rcut, shell_idx, ao_index):
        """
        Initialize atom-specific grid information.
        
        Parameters:
        -----------
        shells_on_grid : array_like
            Shell indices that are included on this grid
        alpha : array_like
            Gaussian exponents for each shell
        l : array_like
            Angular momentum quantum numbers
        rcut : array_like
            Cutoff radii for each shell
        shell_idx : array_like
            Global shell indices
        ao_index : array_like
            Atomic orbital indices
        """
        # Find which shells are actually present on this grid
        sharp_idx = numpy.isin(shell_idx, shells_on_grid)

        # Handle case where no shells are on this grid
        if sum(sharp_idx) == 0:
            self.rcut = numpy.asarray([-1.0], numpy.float64)
            self.max_exp = 0.0
            self.min_exp = 0.0
            self.Ng = 0
            self.nao = 0
            self.nthc = 0
            self.n_sharp = 0
            self.n_diffuse = 0
            return

        # Store information for shells present on this grid
        self.shell_index_sharp = shell_idx[sharp_idx]
        self.max_exp = max(alpha[sharp_idx])  # Maximum exponent
        self.min_exp = min(alpha[sharp_idx])  # Minimum exponent
        self.rcut = rcut[sharp_idx]           # Cutoff radii
        self.l = l[sharp_idx]                 # Angular momenta
        
        # Account for angular momentum multiplicity (2l+1 orbitals per shell)
        multiplicity = 2 * l + 1
        sharp_idx = numpy.repeat(sharp_idx, multiplicity)
        
        # Map to actual atomic orbital indices
        self.ao_index_sharp = ao_index[sharp_idx]
        self.n_sharp = len(self.ao_index_sharp)  # Number of sharp orbitals
        self.nao = len(self.ao_index_sharp)      # Total number of AOs
        self.nthc = 0                            # Initialize counter


class Atoms:
    """
    Represents atomic information for grid-based calculations.
    """
    
    def __init__(self, cell, rcut_epsilon, atom_index, shell_index):
        """
        Initialize atomic information for a specific atom.
        
        Parameters:
        -----------
        cell : pyscf.pbc.gto.Cell
            Unit cell object
        rcut_epsilon : float
            Cutoff threshold for basis function truncation
        atom_index : int
            Index of the atom
        shell_index : array_like
            Indices of shells belonging to this atom
        """
        self.grid_empty = False
        
        # Handle case with no shells
        if len(shell_index) == 0:
            self.grid_empty = True
            return
        
        self.shell_index = shell_index
        
        # Extract angular momentum quantum numbers for each shell
        self.l = numpy.ascontiguousarray(
            [cell.bas_angular(j) for j in shell_index], 
            dtype=numpy.int32
        )
        
        # Extract and concatenate Gaussian exponents
        self.exponents = numpy.ascontiguousarray(
            numpy.concatenate([cell.bas_exp(j) for j in shell_index]),
            dtype=numpy.float64
        )
        
        # Get atomic orbital indices for this atom
        ao_slice = cell.aoslice_by_atom()[atom_index]
        ao_index = numpy.arange(ao_slice[2], ao_slice[3], dtype=numpy.int32)
        
        # Get angular momenta for all shells on this atom
        atom_shells = cell.atom_shell_ids(atom_index)
        l = numpy.ascontiguousarray(
            [cell.bas_angular(j) for j in atom_shells], 
            dtype=numpy.int32
        )
        
        # Account for angular momentum multiplicity
        multiplicity = 2 * l + 1
        shell_idx = numpy.isin(atom_shells, shell_index)
        shell_idx = numpy.repeat(shell_idx, multiplicity)
        
        # Filter AO indices to include only relevant shells
        self.ao_index = ao_index[shell_idx]
        
        # Calculate cutoff radii for primitive Gaussians
        self.rcut = numpy.ascontiguousarray(
            numpy.concatenate(primitive_gto_cutoff(cell, rcut_epsilon, shell_index)),
            dtype=numpy.float64,
        )
        
        # Store atomic center coordinates
        self.atom_center = cell.atom_coords()[atom_index]


def build_grids(myisdf, with_j=True, with_k=True):
    """
    Construct hierarchical multi-grid system for ISDF calculations.
    
    This is a core function that builds the grid infrastructure for both
    Coulomb (J) and exchange (K) matrix calculations. The function creates
    either a multi-grid hierarchy or a single universal grid depending on
    configuration settings.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing computational parameters and cell information
    with_j : bool, default=True
        Whether to build grids for Coulomb (J) matrix calculations
    with_k : bool, default=True
        Whether to build grids for exchange (K) matrix calculations
        
    Grid Construction:
    ------------------
    Two modes are supported:
    
    1. Multi-grid mode (multigrid_on=True):
       - Creates hierarchical grids with different densities
       - Exchange grids: Optimized for local exchange interactions
       - Coulomb grids: May reuse exchange grids or create separate ones
       - Supports different grid sets for J and K if same_jk_grids=False
       
    2. Single grid mode (multigrid_on=False):
       - Creates one universal grid covering the entire unit cell
       - Uses cell.ke_cutoff to determine grid density
       - All basis functions are placed on the same grid
       
    Attributes Set:
    ---------------
    - myisdf.full_grids_j : Grid objects for Coulomb calculations
    - myisdf.atom_grids_j : Atom-centered grids for Coulomb calculations
    - myisdf.full_grids_k : Grid objects for exchange calculations  
    - myisdf.atom_grids_k : Atom-centered grids for exchange calculations
    
    Output:
    -------
    Prints detailed grid information including:
    - Number of grid points and mesh dimensions
    - Basis function statistics per grid level
    - Kinetic energy cutoffs and exponent ranges
    """
    cell = myisdf.cell_unc
    print("cell mesh:", cell.mesh, flush=True)
    print("nbas: ", cell.nbas, flush=True)
    print("nao: ", cell.nao, flush=True)
    if myisdf.multigrid_on:
        mg_k = None
        if with_k:
            # Exchange grids
            mg_k, atomgrids_k = make_exchange_lists(myisdf)
            myisdf.full_grids_k = mg_k
            myisdf.atom_grids_k = atomgrids_k
        if with_j:
            # Coulomb grids
            if myisdf.same_jk_grids:
                if mg_k is not None:
                    mg_j, atomgrids_j = mg_k, atomgrids_k
                else:
                    mg_j, atomgrids_j = make_exchange_lists(myisdf)
            else:
                mg_j, atomgrids_j = make_coulomb_lists(myisdf)
            myisdf.full_grids_j = mg_j
            myisdf.atom_grids_j = atomgrids_j
    else:
        ##### Alternate mesh specification ######
        # (1) cell.mesh
        #     When calling tools.pbc.super_cell, the mesh is not from cell.ke_cutoff
        #     i.e., cutoff_to_mesh(supcell.ke_cutoff) != supcell.mesh
        mesh = None
        if cell.mesh is not None:
            mesh = cell.mesh

        mg = [Grid(myisdf, cell.ke_cutoff, mesh)]
        mg[0].universal = True
        exp_on_grid = numpy.arange(cell.nbas, dtype=numpy.int32)
        atoms = myisdf.atoms
        natm = cell.natm
        exponents = numpy.asarray(
            numpy.concatenate([atoms[i].exponents for i in range(natm)]), numpy.float64
        )
        print("Grids:", flush=True)
        print(
            "{0:8s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s} {6:8s} {7:8s} {8:8s} {9:8s} {10:12s}".format(
                "Grid",
                "ke_max",
                "ke_min",
                "ke_grid",
                "expMax",
                "expMin",
                "nexp",
                "nao",
                "lmax",
                "PWs",
                "mesh",
            ),
            flush=True,
        )
        atomgrids = make_universal_grid(myisdf, exp_on_grid, mg[0])

        print(
            "{0:<8d} {1:<8.1f} {2:<8.1f} {3:<8.1f} {4:<8.1f} {5:<8.1f} {6:<8d} \
                {7:<8d} {8:<8d} {9:<8d} {10:<12s}".format(
                0,
                cell.ke_cutoff,
                cell.ke_cutoff,
                cell.ke_cutoff,
                exponents.max(),
                exponents.min(),
                len(numpy.unique(exponents)),
                atomgrids[0].nao,
                max([numpy.max(atoms[i].l) for i in range(natm)]),
                mg[-1].Ng,
                numpy.array2string(mg[-1].mesh),
            ),
            flush=True,
        )
        if with_k:
            myisdf.full_grids_k = mg
            myisdf.atom_grids_k = atomgrids
        if with_j:
            myisdf.full_grids_j = mg
            myisdf.atom_grids_j = atomgrids


def primitive_gto_cutoff(cell, rcut_epsilon, shell_idx):
    """Cutoff raidus, above which each shell decays to a value less than the
    required precsion"""
    rcut = []
    for ib in shell_idx:
        es = cell.bas_exp(ib)
        r = (-numpy.log(rcut_epsilon) / es) ** 0.5
        rcut.append(r)
    return rcut


def primitive_gto_exponent(rmin, rcut_epsilon):
    """
    Calculate primitive Gaussian-type orbital exponent based on minimum radius and cutoff precision.
    
    Parameters:
    -----------
    rmin : float
        Minimum radius for the Gaussian function
    rcut_epsilon : float
        Cutoff precision threshold
        
    Returns:
    --------
    float
        Calculated exponent for the primitive GTO
    """
    return -numpy.log(rcut_epsilon) / max(rmin**2, 1e-12)


def estimate_ke_cutoff(cell, precision):
    """Energy cutoff estimation for the entire system"""
    ke_cut = []
    for i in range(cell.nbas):
        es = cell.bas_exp(i)
        ke_guess = _estimate_ke_cutoff(es, precision)
        ke_cut.append(ke_guess.max())
    return numpy.array(ke_cut, numpy.float64)


def _estimate_ke_cutoff(es, precision, l=1):
    """
    Internal function to estimate kinetic energy cutoff for given exponents.
    
    Parameters:
    -----------
    es : numpy.ndarray
        Array of Gaussian exponents
    precision : float
        Required precision threshold
        
    Returns:
    --------
    numpy.ndarray
        Estimated kinetic energy cutoffs
    """
    return -numpy.log(precision) * 2.0 * es

def make_atoms(cell, rcut_epsilon):
    """
    Create Atoms objects for all atoms in the unit cell.
    
    Parameters:
    -----------
    cell : pyscf.pbc.gto.Cell
        PySCF Cell object containing atomic and basis set information
    rcut_epsilon : float
        Cutoff precision for determining atomic radii
        
    Returns:
    --------
    list
        List of Atoms objects, one for each atom in the cell
    """
    natm = cell.natm
    atoms = []
    for i in range(natm):
        shell_index = numpy.ascontiguousarray(cell.atom_shell_ids(i), dtype=numpy.int32)
        atoms.append(Atoms(cell, rcut_epsilon, i, shell_index))
    return atoms


def voronoi_partition(ug, Ls, Rgrid):
    """
    Partition universal grid points using Voronoi tessellation.
    
    Each grid point is assigned to the nearest atom (including periodic images).
    This creates atom-centered regions for efficient local operations.
    
    Parameters:
    -----------
    ug : UniversalGrid
        Universal grid object to be partitioned
    Ls : numpy.ndarray
        Array of atomic positions including periodic images (shape: 27*natm x 3)
    Rgrid : numpy.ndarray  
        Grid point coordinates (shape: Ng x 3)
        
    Note:
    -----
    Modifies ug.coord_idx in-place with coordinate indices for each atom
    """
    natm = Ls.shape[0] // 27
    # List of distances between all grid pts and all atoms, including periodic images
    argmindist = numpy.argmin(scipy.spatial.distance.cdist(Rgrid, Ls, "euclidean"), axis=1)
    coord_idx = []
    for i in range(natm):
        coord_idx.append(
            numpy.isin(argmindist, numpy.arange(i, 27 * natm, natm)).nonzero()[0].astype(numpy.int32)
        )  # Store coordinate indices for atom i
    ug.coord_idx = coord_idx


def make_universal_grid(myisdf, exp_on_grid, mg):
    """
    Construct universal sparse grid covering the entire unit cell.
    
    This function creates a single grid that encompasses all atoms and basis
    functions in the unit cell. It aggregates all atomic basis functions
    onto one grid and sets up the data structures for universal grid operations.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing computational parameters
    exp_on_grid : numpy.ndarray
        Array of exponent indices to include on this grid
    mg : Grid
        Grid object containing spatial coordinates and mesh information
        
    Returns:
    --------
    list
        List containing single AtomGrid object representing the universal grid
        
    Notes:
    -----
    - Combines basis functions from all atoms into one data structure
    - Performs Voronoi partitioning to assign grid points to nearest atoms
    - Sets up indexing for efficient access to atomic contributions
    - Universal grid is used for long-range interactions and sparse operations
    """
    cell = myisdf.cell_unc
    atoms = myisdf.atoms
    natm = len(atoms)
    exponents = numpy.asarray(
        numpy.concatenate([atoms[i].exponents for i in range(natm)]), numpy.float64
    )
    la = numpy.asarray(numpy.concatenate([atoms[i].l for i in range(natm)]), numpy.int32)
    nao = int(sum(2 * la + 1))
    nbas = len(exponents)
    ao_index = numpy.asarray(numpy.arange(nao), numpy.int32)
    shell_index = numpy.asarray(numpy.arange(nbas), numpy.int32)
    rcut = numpy.asarray(numpy.concatenate([atoms[i].rcut for i in range(natm)]), numpy.float64)
    na = numpy.asarray(
        [atom for atom in range(natm) for i in range(len(atoms[atom].exponents))], numpy.int32
    )
    ug = AtomGrid(exp_on_grid, exponents, la, rcut, shell_index, ao_index)

    # Shells
    ug.l = numpy.array(la[ug.shell_index_sharp], numpy.int32)
    ug.exponents = numpy.array(exponents[ug.shell_index_sharp], numpy.float64)
    ug.nexp = ug.exponents.shape[0]
    ug.atoms = numpy.array(na[ug.shell_index_sharp], numpy.int32)
    ug.shells = list_to_slices(numpy.array(ug.shell_index_sharp))

    # AOs
    ug.ao_index = numpy.sort(ug.ao_index_sharp)
    ug.nao = len(ug.ao_index)
    ug.ao_index_by_atom = []
    ug.shell_index_by_atom = []
    for i in range(natm):
        idx = numpy.isin(ug.ao_index, atoms[i].ao_index).nonzero()[0]
        ug.ao_index_by_atom.append(numpy.array(idx, numpy.int32))
        ug.shell_index_by_atom.append(
            numpy.sort(
                numpy.array(
                    ug.shell_index_sharp[numpy.isin(ug.shell_index_sharp, atoms[i].shell_index)],
                    numpy.int32,
                )
            )
        )

    # Grid
    Rgrid = mg.coords.round(10)
    nG = Rgrid.shape[0]
    ug.Ng = nG
    ug.coord_idx = numpy.arange(0, nG)
    centers = (
        tools.pbc.cell_plus_imgs(cell, [1, 1, 1]).atom_coords().astype(numpy.float64)
    )  # Make 3x3x3 periodic images
    voronoi_partition(ug, centers, Rgrid)  # NOTE: Need to change this for k-pts?

    #### FRACTIONAL??? ####
    a =  cell.lattice_vectors()
    orth = abs(a - numpy.diag(a.diagonal())).max() < cell.precision
    if orth and not myisdf.get_aos_from_pyscf:
        # x0, y0, z0 are the grid points for which the functions are non-zero.
        ug.x0 = numpy.unique(Rgrid[:, 0])
        ug.y0 = numpy.unique(Rgrid[:, 1])
        ug.z0 = numpy.unique(Rgrid[:, 2])
        # nx, ny, nz are the number of non-zero grid points.
        ug.nx, ug.ny, ug.nz = ug.x0.shape[0], ug.y0.shape[0], ug.z0.shape[0]

        x, y, z = numpy.meshgrid(ug.x0, ug.y0, ug.z0, indexing="ij")
        c_cube = numpy.column_stack([x.ravel(), y.ravel(), z.ravel()])

        cube_keys = structured_coords(c_cube)
        rgrid_keys = structured_coords(Rgrid)

        # # Sort cube keys once
        sort_idx = numpy.argsort(cube_keys)
        cube_keys_sorted = cube_keys[sort_idx]            

        # Use searchsorted to find insertion points
        insert_idx = numpy.searchsorted(cube_keys_sorted, rgrid_keys)

        # Now map back to unsorted c_cube indices
        ug.cube_to_sphere_index = sort_idx[insert_idx]        
        # ug.cube_to_sphere_index = pyscf_coord_sort(c_cube)
        # assert numpy.allclose(Rgrid, c_cube[ug.cube_to_sphere_index]), "Mapping is incorrect!"

        # Init AOs
        ug.norm = numpy.array(
            [
                (cell.vol / mg.Ng) ** 0.5 / RawGaussNorm(ug.exponents[i], ug.l[i])
                for i in range(len(ug.exponents))
            ],
            numpy.float64,
        )
        I = ug.exponents.argmin()
        nimg = numpy.zeros((3,), numpy.int32)
        La = numpy.ascontiguousarray(a.diagonal(), numpy.float64)  # Assumes orthogonal
        libevalao.getNimg(ug.exponents[I], ug.l[I], La, nimg)
        ug.nLs = int(numpy.prod(2 * nimg + 1))
        ug.Ls = numpy.zeros((ug.nLs, 3), numpy.float64)
        ug.nimages = numpy.zeros_like(ug.l)
        ug.images = numpy.zeros((ug.nLs * ug.nexp,), numpy.int32)
        libevalao.formImages(ug.exponents, ug.l, ug.nexp, La, ug.Ls, nimg, ug.nimages, ug.images)
    return [ug]


def pyscf_coord_sort(arr):
    """
    Sort coordinate array using PySCF-compatible ordering scheme.
    
    This function implements a coordinate sorting algorithm that maintains
    compatibility with PySCF's internal coordinate ordering conventions.
    The sorting is performed hierarchically: first by x, then by y, then by z.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        Array of 3D coordinates to sort (shape: N x 3)
        
    Returns:
    --------
    numpy.ndarray
        Array of indices that would sort the input coordinates
        
    Algorithm:
    ----------
    1. Round coordinates to avoid floating point precision issues
    2. Sort by x-coordinate (positive values first, then negative)
    3. Within each x-slice, sort by y-coordinate
    4. Within each (x,y)-slice, sort by z-coordinate
    
    Notes:
    ------
    - Uses 10 decimal places for rounding to handle numerical precision
    - Positive coordinates are ordered before negative coordinates
    - Essential for maintaining consistent grid point ordering
    """
    arr = numpy.around(arr, decimals=10)

    def split_index(sortarr):
        """Split array indices into positive and negative values, then sort each."""
        upper_index = sortarr >= 0
        lower_index = numpy.invert(upper_index).nonzero()[0]
        upper_index = upper_index.nonzero()[0]
        upper_index = upper_index[numpy.argsort(sortarr[upper_index])]
        lower_index = lower_index[numpy.argsort(sortarr[lower_index])]
        return numpy.concatenate([upper_index, lower_index])

    idx = split_index(arr[:, 0])
    x0 = numpy.unique(arr[:, 0])
    for x in x0:
        xidx = (abs(arr[idx][:, 0] - x) < 1.0e-10).nonzero()[0]
        idx[xidx] = idx[xidx[split_index(arr[idx][xidx][:, 1])]]
        y0 = numpy.unique(arr[idx[xidx]][:, 1])
        for y in y0:
            yidx = (abs(arr[idx[xidx]][:, 1] - y) < 1.0e-10).nonzero()[0]
            idx[xidx[yidx]] = idx[xidx[yidx[split_index(arr[idx[xidx[yidx]]][:, 2])]]]

    return idx

def make_coulomb_lists(myisdf):
    """
    Create hierarchical multi-grid system for Coulomb (J) matrix calculations.
    
    This function constructs a series of grids with increasing kinetic energy
    cutoffs for efficient evaluation of Coulomb integrals. The grids range from
    coarse (low cutoff) to fine (high cutoff) to capture different length scales
    of the Coulomb interaction.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing cell and computational parameters
        
    Returns:
    --------
    tuple
        (mg, atomgrids) where:
        - mg: list of Grid objects with increasing resolution
        - atomgrids: list of AtomGrid objects for each grid level and atom
        
    Algorithm:
    ----------
    1. Determine kinetic energy cutoffs based on basis set exponents
    2. Create grids with cutoffs increasing by KE_STEP factor
    3. Assign basis functions to appropriate grid levels 
    4. Optionally add universal sparse grid for remaining functions
    5. Create atom-centered grids for each grid level
    
    Grid Hierarchy:
    ---------------
    - Grid 0: Lowest cutoff, fastest basis functions
    - Grid 1: Medium cutoff, intermediate basis functions  
    - Grid 2+: Higher cutoffs for more diffuse basis functions
    - Universal: Sparse grid for very diffuse/remaining functions
    """
    KE_STEP = 2.0
    GRID_VOL_CUT = 0.05  # Don't add more Coulomb grids if
    # ( max volume of atom centered grid ) > ( GRID_VOLUME_CUT * cell volume )
    MAX_GRIDS = 4
    if myisdf.j_grid_mesh is not None:
        MAX_GRIDS = len(myisdf.j_grid_mesh)
    cell = myisdf.cell_unc
    atoms = myisdf.atoms
    natm = len(atoms)

    ##### Alternate mesh specifications ######
    # (1) j_grid_mesh can store a list of user-given meshes.
    # (2) cell.mesh
    #     When calling tools.pbc.super_cell, the mesh is not from cell.ke_cutoff
    #     i.e., cutoff_to_mesh(supcell.ke_cutoff) != supcell.mesh
    #     For do_real_isdfx, set supcell.mesh = primitive_cell.mesh * ncopy.

    mesh = None
    if myisdf.j_grid_mesh is not None:
        mesh = myisdf.j_grid_mesh[0]
    elif cell.mesh is not None:
        mesh = cell.mesh
    ke_cutoffs = estimate_ke_cutoff(cell, cell.precision)
    # ke_cutoffs = estimate_ke_cutoff(cell, myisdf.ke_epsilon * 0.01)
    ke_step = 1.0 / KE_STEP
    ke_grid = (
        tools.mesh_to_cutoff(cell.lattice_vectors(), mesh)[0] if mesh is not None else cell.ke_cutoff
    )
    ke_grid_min = min(ke_cutoffs)
    ke_max = max(ke_cutoffs)
    ke_min = max(min(ke_cutoffs) + 1.0e-6, ke_grid * ke_step)
    while abs(ke_min) > ke_max:
        ke_min *= ke_step
    atoms = make_atoms(cell, cell.precision)
    # atoms = myisdf.atoms
    def make_atom_centered_grids(mg, exponents_on_grid):
        cell = myisdf.cell_unc
        natm = cell.natm
        atomgrids = []
        for i in range(natm):
            atomgrids.append(
                AtomGrid(
                    exponents_on_grid,
                    atoms[i].exponents,
                    atoms[i].l,
                    atoms[i].rcut,
                    atoms[i].shell_index,
                    atoms[i].ao_index,
                )
            )
        get_atomgrid_coords(myisdf, atomgrids, mg)
        place_functions_on_atomgrids(myisdf, atomgrids, mg)
        return atomgrids

    t0 = time.time()
    next_min_greater_than_gridmin = True
    vol_rcut_less_than_cell_vol = True
    atomgrids, mg = [], []
    all_l = numpy.concatenate([atoms[i].l for i in range(natm)])
    exponents = numpy.concatenate([atoms[i].exponents for i in range(natm)])
    grid_num = 0
    print("Coulomb Grids:", flush=True)
    print(
        "{0:8s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s} {6:8s} {7:8s} {8:8s} {9:8s} {10:8s} {11:12s}".format(
            "Grid",
            "ke_max",
            "ke_min",
            "ke_grid",
            "expMax",
            "expMin",
            "rcut",
            "nexp",
            "nao",
            "lmax",
            "PWs",
            "mesh",
        ),
        flush=True,
    )    
    while next_min_greater_than_gridmin and vol_rcut_less_than_cell_vol and grid_num < MAX_GRIDS - 1:
        if myisdf.j_grid_mesh is not None:
            mesh = myisdf.j_grid_mesh[grid_num]
        mg.append(Grid(myisdf, ke_grid, mesh))
        mg[-1].ke_min = ke_min
        mg[-1].ke_max = ke_max
        mg[-1].ke_grid = ke_grid
        exp_on_grid = ((ke_cutoffs >= ke_min) * (ke_cutoffs <= ke_max)).nonzero()[0]
        for i in range(natm):
            atomgrids.append(
                AtomGrid(
                    exp_on_grid,
                    atoms[i].exponents,
                    atoms[i].l,
                    atoms[i].rcut,
                    atoms[i].shell_index,
                    atoms[i].ao_index,
                )
            )
        # t1 = time.time()
        get_atomgrid_coords(myisdf, atomgrids[grid_num*natm:(grid_num+1)*natm], mg[-1])
        # print("Make Grids:", time.time()-t1, flush=True)
        # t1 = time.time()
        place_functions_on_atomgrids(myisdf, atomgrids[grid_num*natm:(grid_num+1)*natm], mg[-1])
        # print("Functions on Grids:", time.time()-t1, flush=True)
        N = sum(ag.nao for ag in atomgrids[grid_num*natm:(grid_num+1)*natm])
        grid_max_rcut = max(ag.rcut.max() for ag in atomgrids[grid_num*natm:(grid_num+1)*natm])
        print(
            "{0:<8d} {1:<8.1f} {2:<8.1f} {3:<8.1f} {4:<8.1f} {5:<8.1f} {6:<8.1f} {7:<8d} {8:<8d} {9:<8d} {10:<8d} {11:<10s}".format(
                grid_num,
                ke_max,
                ke_min,
                ke_grid,
                exponents[exp_on_grid].max(),
                exponents[exp_on_grid].min(),
                grid_max_rcut,
                len(numpy.unique(exponents[exp_on_grid])),
                N,
                all_l[exp_on_grid].max(),
                mg[-1].Ng,
                numpy.array2string(mg[-1].mesh),
            ),
            flush=True,
        )
        ke_max = max(ke_cutoffs[ke_cutoffs < ke_min])
        if myisdf.j_grid_mesh is not None:
            ke_grid = tools.mesh_to_cutoff(cell.lattice_vectors(), myisdf.j_grid_mesh[grid_num])[0]
            ke_max = min(ke_max, ke_grid)
        else:
            ke_grid = min(ke_min, ke_max)
            # Ensure that the KE cutoff will give a new mesh.
            while (tools.cutoff_to_mesh(cell.lattice_vectors(), ke_grid) == mg[-1].mesh).all():
                ke_grid -= 0.1
        ke_min = min(ke_max * ke_step, ke_grid)
        next_min_greater_than_gridmin = max(
            ke_cutoffs[ke_cutoffs < ke_min], default=0.0
        ) * ke_step > min(ke_grid_min, ke_grid)
        rcut_vol_max = 4.189 * max([atomgrids[-natm:][i].rcut.max() for i in range(natm)]) ** 3.0
        vol_rcut_less_than_cell_vol = GRID_VOL_CUT * cell.vol > rcut_vol_max
        grid_num += 1
        mesh = None

    # print("Dense Coulomb Grids:", time.time()-t0, flush=True)
    # t0 = time.time()
    if myisdf.j_grid_mesh is not None:
        mg.append(Grid(myisdf, ke_grid, myisdf.j_grid_mesh[-1]))
        mg[-1].ke_max = ke_max
    else:
        mg.append(Grid(myisdf, ke_grid))
        mg[-1].ke_max = max(ke_max, ke_grid)
    mg[-1].ke_min = min(ke_cutoffs)
    mg[-1].ke_grid = ke_grid
    mg[-1].universal = True
    exp_on_grid = (ke_cutoffs <= mg[-1].ke_max).nonzero()[0]
    if mg[-1].universal:
        atomgrids.extend(make_universal_grid(myisdf, exp_on_grid, mg[-1]))
    else:
        atomgrids.extend(make_atom_centered_grids(mg[-1], exp_on_grid))
    grid_max_rcut = atomgrids[-1].rcut.max()
    print(
        "{0:<8d} {1:<8.1f} {2:<8.1f} {3:<8.1f} {4:<8.1f} {5:<8.1f} {6:<8.1f} {7:<8d} {8:<8d} {9:<8d} {10:<8d} {11:<10s}".format(
            grid_num,
            mg[-1].ke_max,
            mg[-1].ke_min,
            mg[-1].ke_grid,
            exponents[exp_on_grid].max(),
            exponents[exp_on_grid].min(),
            grid_max_rcut,
            len(numpy.unique(exponents[exp_on_grid])),
            atomgrids[-1].nao,
            all_l[exp_on_grid].max(),
            mg[-1].Ng,
            numpy.array2string(mg[-1].mesh),
        ),
        flush=True,
    )
    # print("Sparse Coulomb Grids:", time.time()-t0, flush=True)
    return mg, atomgrids


def make_exchange_lists(myisdf):
    """
    Create multi-grid system optimized for exchange (K) matrix calculations.
    
    This function constructs grids specifically optimized for exchange integral
    evaluation. Unlike Coulomb grids, exchange grids use different criteria for
    basis function assignment based on the local nature of exchange interactions.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing cell and computational parameters
        
    Returns:
    --------
    tuple
        (mg, atomgrids) where:
        - mg: list of Grid objects optimized for exchange
        - atomgrids: list of AtomGrid objects for each grid level and atom
        
    Algorithm:
    ----------
    1. Determine kinetic energy cutoffs from ke_epsilon parameter
    2. Create atom-centered grids based on exponent thresholds
    3. Assign basis functions using alpha_cutoff criteria
    4. Add universal sparse grid for diffuse functions
    5. Optimize grid spacing for exchange integral locality
    
    Key Differences from Coulomb Grids:
    -----------------------------------
    - Uses alpha_cutoff to separate sharp/diffuse functions
    - Optimized for the locality of exchange interactions
    - Typically requires fewer grid levels than Coulomb
    - Grid construction based on ke_epsilon precision parameter
    """
    print("Exchange Grids:", flush=True)
    print(
        "{0:8s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s} {6:8s} {7:8s} {8:8s} {9:8s} {10:8s} {11:12s}".format(
            "Grid",
            "ke_max",
            "ke_min",
            "ke_grid",
            "expMax",
            "expMin",
            "rcut",
            "nexp",
            "nao",
            "lmax",
            "PWs",
            "mesh",
        ),
        flush=True,
    )
    cell = myisdf.cell_unc
    atoms = myisdf.atoms
    natm = len(atoms)
    log_ke_epsilon = -numpy.log(myisdf.ke_epsilon) * 2.0
    atomgrids = []
    mg = []

    ##### Alternate mesh specifications ######
    # (1) k_grid_mesh can store a list of user-given meshes.
    # (2) cell.mesh
    #     When calling tools.pbc.super_cell, the mesh is not from cell.ke_cutoff
    #     i.e., cutoff_to_mesh(supcell.ke_cutoff) != supcell.mesh
    #     For do_real_isdfx, set supcell.mesh = primitive_cell.mesh * ncopy.

    t0 = time.time()
    mesh = None
    if myisdf.k_grid_mesh is not None:
        mesh = myisdf.k_grid_mesh[0]
    if cell.mesh is not None:
        mesh = cell.mesh

    # Put exponents from all atoms on the dense grid.
    ke_max = 0.0
    ke_min = 1.0e8
    # If a user specified mesh is used, take the ke_cutoff from it.
    ke_grid = (
        tools.mesh_to_cutoff(cell.lattice_vectors(), mesh)[0] if mesh is not None else cell.ke_cutoff
    )

    natm = len(atoms)
    grid_max_exp = 0.0
    grid_min_exp = 1.0e8
    l_list = []
    exp_list = []
    atoms2 = [None] * natm
    lmax = 0
    max_exp = max([atoms[i].exponents.max() for i in range(natm)])
    if myisdf.alpha_cutoff > max_exp:
        myisdf.alpha_cutoff = max_exp

    for i, atomi in enumerate(atoms):
        idx = numpy.greater(atomi.exponents, myisdf.alpha_cutoff - cell.precision)
        diffuse_exp = (1 - idx).astype(numpy.bool_)
        sharp_exp = atomi.exponents[idx]
        exp_list.extend(sharp_exp)
        l_list.extend(atomi.l[idx])
        atoms2[i] = Atoms(cell, myisdf.rcut_epsilon, i, atomi.shell_index[diffuse_exp])
        lmax = max(max(atomi.l[idx], default=0), lmax)
        atomgrids.append(
            AtomGrid(
                atomi.shell_index[idx],
                atomi.exponents,
                atomi.l,
                atomi.rcut,
                atomi.shell_index,
                atomi.ao_index,
            ),
        )

    grid_max_exp = max(exp_list)
    max_l = (numpy.asarray(l_list)[(exp_list==max(exp_list)).nonzero()[0]]).max()
    exp_list = numpy.unique(exp_list)
    nexp = exp_list.shape[0]
    grid_min_exp = min(exp_list)
    ke_max = log_ke_epsilon * grid_max_exp
    ke_min = log_ke_epsilon * grid_min_exp
    
    # Based on the range of exponents on it, we build the dense grid.
    mg.append(Grid(myisdf, ke_grid, mesh))
    mg[-1].ke_min = ke_min
    mg[-1].ke_max = ke_max
    mg[-1].ke_grid = ke_grid

    # The atom-centered grid points are assigned to each atom-grid.
    get_atomgrid_coords(myisdf, atomgrids, mg[-1])
    # Non-zero functions are assigned to each atom-grid.
    place_functions_on_atomgrids(myisdf, atomgrids, mg[-1])

    N = sum(ag.nao for ag in atomgrids)
    grid_max_rcut = max(ag.rcut.max() for ag in atomgrids)
    print(
        "{0:<8d} {1:<8.1f} {2:<8.1f} {3:<8.1f} {4:<8.1f} {5:<8.1f} {6:<8.1f} {7:<8d} {8:<8d} {9:<8d} {10:<8d} {11:<10s}".format(
            0,
            mg[-1].ke_max,
            mg[-1].ke_min,
            mg[-1].ke_grid,
            grid_max_exp,
            grid_min_exp,
            grid_max_rcut,
            nexp,
            N,
            lmax,
            mg[-1].Ng,
            numpy.array2string(mg[-1].mesh),
        ),
        flush=True,
    )
    # print("Dense Exchange Grids:", time.time()-t0, flush=True)

    exp_list = numpy.unique(numpy.concatenate([a2.exponents for a2 in atoms2]))
    l_list = numpy.unique(numpy.concatenate([a2.l for a2 in atoms2]))
    sparse_shell_idx = numpy.concatenate([a2.shell_index for a2 in atoms2])
    nexp = exp_list.shape[0]
    grid_max_exp = max(exp_list)
    grid_min_exp = min(exp_list)
    lmax = max(l_list)
    ke_min = log_ke_epsilon * grid_min_exp
    # If a user specified mesh is used, take the ke_cutoff from it.
    if myisdf.k_grid_mesh is not None:
        ke_max = max(tools.mesh_to_cutoff(cell.lattice_vectors(), myisdf.k_grid_mesh[-1]))
        ke_grid = ke_max
        mg.append(Grid(myisdf, ke_grid, myisdf.k_grid_mesh[-1]))
    else:
        # ke_max = _estimate_ke_cutoff(grid_max_exp, 1.e-3)
        sparse_grid_ke = pyscf.pbc.gto.cell._estimate_ke_cutoff(grid_max_exp, lmax, 1)
        dense_grid_ke = pyscf.pbc.gto.cell._estimate_ke_cutoff(max_exp, max_l, 1)
        ke_max = sparse_grid_ke/dense_grid_ke * max(tools.mesh_to_cutoff(cell.lattice_vectors(), cell.mesh))
        ke_grid = ke_max
        while (tools.cutoff_to_mesh(cell.lattice_vectors(), ke_grid) >= mg[-1].mesh).all():
            ke_grid -= 10.
        mg.append(Grid(myisdf, ke_grid))

    mg[-1].ke_min = ke_min
    mg[-1].ke_max = ke_max
    mg[-1].ke_grid = ke_grid
    mg[-1].universal = True

    if mg[-1].Ng > mg[0].Ng:
        msg = f"sparse grid mesh ({mg[-1].Ng}) > dense grid mesh ({mg[-2].Ng})! \n Lower ke_epsilon or raise ke_cutoff."
        raise ValueError(msg)

    atomgrids.extend(make_universal_grid(myisdf, sparse_shell_idx, mg[-1]))
    grid_max_rcut = atomgrids[-1].rcut.max()
    print(
        "{0:<8d} {1:<8.1f} {2:<8.1f} {3:<8.1f} {4:<8.1f} {5:<8.1f} {6:<8.1f} {7:<8d} {8:<8d} {9:<8d} {10:<8d} {11:<10s}".format(
            1,
            mg[-1].ke_max,
            mg[-1].ke_min,
            mg[-1].ke_grid,
            grid_max_exp,
            grid_min_exp,
            grid_max_rcut,
            nexp,
            atomgrids[-1].nao,
            lmax,
            mg[-1].Ng,
            numpy.array2string(mg[-1].mesh),
        ),
        flush=True,
    )
    # print("Sparse Exchange Grids:", time.time()-t0, flush=True)
    return mg, atomgrids


def structured_coords(arr):
    # Make keys lex-sortable
    return numpy.core.records.fromarrays(arr.T, names='x,y,z', formats='f8,f8,f8')

def get_atomgrid_coords(myisdf, atomgrids, mg):
    cell = myisdf.cell_unc
    natm = len(atomgrids)
    a = cell.lattice_vectors()
    orth = abs(a - numpy.diag(a.diagonal())).max() < cell.precision
    Rgrid = mg.coords.round(10)
    atom_centers = tools.pbc.cell_plus_imgs(cell, [1, 1, 1]).atom_coords().astype(numpy.float64)

    # Parallelize this.
    def atom_loop(i):
        ag = atomgrids[i]
        if ag.nao == 0:
            return
        # list of distances b/w atom i and all grid pts, inc periodic imgs
        centers = atom_centers[i : 27 * natm : natm]
        diff = Rgrid[:, None, :] - centers[None, :, :]
        dist_from_atoms = numpy.linalg.norm(diff, axis=-1).min(axis=1)

        # list of all distances < rcut
        ag.coord_idx = numpy.flatnonzero(dist_from_atoms < ag.rcut.max())  # spherical coordinate indices
        ag.Ng = ag.coord_idx.shape[0]  # store number of grid points
        if orth and not myisdf.get_aos_from_pyscf:
            grid_i = Rgrid[ag.coord_idx]
            # x0, y0, z0 are the unique xyz coords for which the functions are non-zero. They do not need to be the full grid.
            ag.x0 = numpy.unique(grid_i[:, 0])
            ag.y0 = numpy.unique(grid_i[:, 1])
            ag.z0 = numpy.unique(grid_i[:, 2])

            # nx, ny, nz are the number of non-zero grid points.
            ag.nx = ag.x0.shape[0]
            ag.ny = ag.y0.shape[0]
            ag.nz = ag.z0.shape[0]

            x, y, z = numpy.meshgrid(ag.x0, ag.y0, ag.z0, indexing="ij")
            c_cube = numpy.column_stack([x.ravel(), y.ravel(), z.ravel()])

            cube_keys = structured_coords(c_cube)
            rgrid_keys = structured_coords(grid_i)

            # Sort cube keys once
            sort_idx = numpy.argsort(cube_keys)
            cube_keys_sorted = cube_keys[sort_idx]            

            # Use searchsorted to find insertion points
            insert_idx = numpy.searchsorted(cube_keys_sorted, rgrid_keys)

            # Now map back to unsorted c_cube indices
            ag.cube_to_sphere_index = sort_idx[insert_idx]

            # assert numpy.allclose(grid_i, c_cube[ag.cube_to_sphere_index]), "Mapping is incorrect!"

    with threadpool_limits(limits=1):
        with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
            Parallel()(
                delayed(atom_loop)(i) for i in range(natm)
            )

def place_functions_on_atomgrids(myisdf, atomgrids, mg):
    cell = myisdf.cell_unc
    Rgrid = mg.coords
    atom_centers = tools.pbc.cell_plus_imgs(cell, [1, 1, 1]).atom_coords().astype(numpy.float64)
    atoms = myisdf.atoms
    natm = len(atoms)
    nG = Rgrid.shape[0]
    # loop over all other atoms
    min_exp = numpy.asarray([atomgrids[j].min_exp for j in range(natm)], numpy.float64)
    min_exp = min_exp[
        min_exp != 0.0
    ]  # We set min_exp = 0 if no exp on grid. Remove these. Use nao = 0 instead?
    min_exp = min(min_exp)
    max_exp = max([atomgrids[j].max_exp for j in range(natm)])
    a = cell.lattice_vectors()
    orth = abs(a - numpy.diag(a.diagonal())).max() < cell.precision
    
    def atom_loop(i):
        if atomgrids[i].nao == 0:
            return
        ag = atomgrids[i]
        diffuse_indices = []
        ao_on_grid_indices = []
        atoms_on_grid = []
        shells = []
        alpha = []
        la = []
        Rgridatom = Rgrid[ag.coord_idx]
        for j in range(natm):
            if j == i:
                # Added this bc bug when cell.ke_cutoff too low.
                # Distance to closest grid point too large for sharpest exponent.
                # This puts all sharp + diffuse exponents on the atom's own grid.
                max_exponent = ag.max_exp
            else:
                centers = atom_centers[j : 27 * natm : natm]
                # distance between atom-i centered grid n points and atom j (inc pbc)
                # Rmax^2 = -ln(epsilon)/mu
                diff = Rgridatom[:, None, :] - centers[None, :, :]
                dist_from_atoms = numpy.linalg.norm(diff, axis=-1).min(axis=1)

                # closest atom-i grid n point to atom j
                # if Exponent > max_exponent then zero on grid.
                min_diffuseist = dist_from_atoms.min()
                max_exponent = min(primitive_gto_exponent(min_diffuseist, myisdf.rcut_epsilon), max_exp)

            # select all atom j functions that have exponent less than max_exponent
            # any greater than max_exponent will be zero on this grid
            # also remove exponents greater than max_exp bc they are not supported by this grid
            ###### NOTE: THIS IS PROBLEMATIC IF USING l. COMPARE RCUT?############

            idx = numpy.less_equal(atoms[j].exponents, max_exponent)  # original
            shells.extend(atoms[j].shell_index[idx])
            alpha.extend(atoms[j].exponents[idx])
            la.extend(atoms[j].l[idx])
            atoms_on_grid.extend(numpy.repeat(j, sum(idx)))

            multiplicity = 2 * atoms[j].l + 1
            on_grid = numpy.asarray(atoms[j].ao_index[numpy.repeat(idx, multiplicity)], numpy.int32)
            ao_on_grid_indices.extend(on_grid)

            # store idx for exponents that are sharp to less dense grids
            idx = numpy.less(atoms[j].exponents, min_exp) * idx
            diffuse_fxns = numpy.asarray(atoms[j].ao_index[numpy.repeat(idx, multiplicity)], numpy.int32)
            diffuse_indices.extend(diffuse_fxns)

        ag.ao_index = numpy.sort(ao_on_grid_indices)
        ag.nao = len(ag.ao_index)
        ag.ao_index_sharp_on_grid = (
            numpy.isin(ag.ao_index, ag.ao_index_sharp).nonzero()[0].astype(numpy.int32)
        )  # Same atom Sharp

        ag.ao_index_diffuse = numpy.sort(diffuse_indices)
        ag.ao_index_diffuse_on_grid = (
            numpy.isin(ag.ao_index, ag.ao_index_diffuse).nonzero()[0].astype(numpy.int32)
        )
        ag.n_diffuse = len(ag.ao_index_diffuse)

        ag.l = numpy.asarray(la, numpy.int32)
        ag.exponents = numpy.asarray(alpha, numpy.float64)
        ag.shells = list_to_slices(numpy.asarray(shells))
        ag.nexp = ag.exponents.shape[0]
        ag.atoms = numpy.asarray(atoms_on_grid, numpy.int32)

        ###### Store for calculating AOs ######
        if orth and not myisdf.get_aos_from_pyscf:
            nexp = ag.exponents.shape[0]
            ag.norm = numpy.array(
                [(cell.vol / nG) ** 0.5 / RawGaussNorm(ag.exponents[ii], ag.l[ii]) for ii in range(nexp)]
            )
            I = ag.exponents.argmin()
            nimg = numpy.zeros((3,), numpy.int32)
            # PYSCF: nimgs = numpy.ceil(rcut * libevalao.norm(cell.reciprocal_vectors(norm_to=1), axis=1) + boundary_penalty).astype(int)
            La = numpy.ascontiguousarray(cell.lattice_vectors().diagonal(), numpy.float64)
            libevalao.getNimg(ag.exponents[I], ag.l[I], La, nimg)
            ag.nLs = numpy.prod(2 * nimg + 1)
            ag.Ls = numpy.zeros((ag.nLs, 3), numpy.float64)
            ag.nimages = numpy.zeros_like(ag.l)
            ag.images = numpy.zeros((ag.nLs * nexp,), numpy.int32)
            libevalao.formImages(ag.exponents, ag.l, nexp, La, ag.Ls, nimg, ag.nimages, ag.images)

    with threadpool_limits(limits=1):
        with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
            Parallel()(
                delayed(atom_loop)(i) for i in range(natm)
            )


def shell_to_ao_indices(mol, shell_list):
    ao_start = 0
    ao_list = []
    for shell_idx in shell_list:
        # ao_loc lists the starting AO index for each shell
        ao_start = mol.ao_loc[shell_idx]
        ao_end = mol.ao_loc[shell_idx + 1]
        ao_list.extend(list(range(ao_start, ao_end)))
    return ao_list


def list_to_slices(arr):
    arr = numpy.sort(arr)
    it = iter(arr)
    start = next(it)
    slices = []
    for i, x in enumerate(it):
        if x - arr[i] != 1:
            end = arr[i]
            slices.append([start, end + 1])
            start = x
    slices.append([start, arr[-1] + 1])

    return slices


def get_Gmesh(nmesh):
    """
    Generate momentum space (G-vector) mesh for FFT operations.
    
    This function creates a 3D mesh of reciprocal space vectors (G-vectors)
    corresponding to the FFT frequencies for a given real space mesh.
    
    Parameters:
    -----------
    nmesh : array_like
        Number of grid points in each direction [nx, ny, nz]
        
    Returns:
    --------
    numpy.ndarray
        Array of G-vectors (shape: Ng x 3) where Ng = nx*ny*nz
        Each row contains the [Gx, Gy, Gz] components
        
    Notes:
    -----
    - Uses numpy.fft.fftfreq for proper FFT frequency ordering
    - G-vectors are in units of 2π/L where L is the box length
    - Essential for momentum space operations in FFT-based algorithms
    """
    rx = numpy.fft.fftfreq(nmesh[0], 1.0 / nmesh[0])
    ry = numpy.fft.fftfreq(nmesh[1], 1.0 / nmesh[1])
    rz = numpy.fft.fftfreq(nmesh[2], 1.0 / nmesh[2])
    return lib.cartesian_prod((rx, ry, rz)).astype(numpy.float64)


def get_reduced_freq_coords(newMesh, oldMesh):
    """
    Map frequency coordinates from a coarse mesh to a fine mesh.
    
    This function creates an index mapping to interpolate frequency domain
    data from a smaller (coarse) mesh to a larger (fine) mesh. It identifies
    which frequency components from the fine mesh correspond to valid
    frequencies in the coarse mesh.
    
    Parameters:
    -----------
    newMesh : array_like
        Dimensions of the new (typically smaller) mesh [nx, ny, nz]
    oldMesh : array_like  
        Dimensions of the old (typically larger) mesh [nx, ny, nz]
    cell : optional
        Cell object (currently unused)
        
    Returns:
    --------
    numpy.ndarray
        Boolean array indicating which points in the old mesh
        correspond to valid frequencies in the new mesh
        
    Notes:
    -----
    - Essential for multi-grid interpolation in Fourier space
    - Handles aliasing by properly mapping frequency components
    - Used to combine results from different grid resolutions
    """
    Gmesh = get_Gmesh(oldMesh)
    new_indices = numpy.ones(Gmesh.shape[0], numpy.bool_)
    highfreq = numpy.empty(3, numpy.int32)
    lowfreq = numpy.empty(3, numpy.int32)

    # new freq range is ~ -1/2freqcut, 1/2freqcut
    # see: https://numpy.org/doc/stable/reference/routines.fft.html#module-numpy.fft
    for i in range(3):
        if bool(newMesh[i] % 2 != 0):
            lowfreq[i] = numpy.amax(Gmesh[numpy.less_equal(Gmesh[:, i], -(newMesh[i] - 1) // 2), i])
            highfreq[i] = numpy.amin(Gmesh[numpy.greater_equal(Gmesh[:, i], (newMesh[i] - 1) // 2), i])
        else:
            lowfreq[i] = numpy.amax(Gmesh[numpy.less_equal(Gmesh[:, i], -(newMesh[i]) // 2), i])
            highfreq[i] = numpy.amin(Gmesh[numpy.greater_equal(Gmesh[:, i], (newMesh[i]) // 2 - 1), i])

    for i in range(3):
        new_indices[numpy.less(Gmesh[:, i], lowfreq[i] - 1e-12)] = False
        new_indices[numpy.greater(Gmesh[:, i], highfreq[i] + 1e-12)] = False

    a = numpy.nonzero(new_indices)[0]
    a = numpy.reshape(a, numpy.sum(new_indices))

    return a


def DoubleFactR(l):
    r = 1
    while l > 1:
        r *= l
        l -= 2
    return abs(r)


def RawGaussNorm(fExp, l):
    return (numpy.pi / (2.0 * fExp)) ** 0.75 * numpy.sqrt(DoubleFactR(2 * l - 1) / ((4.0 * fExp) ** l))


def eval_all_ao(myisdf, atomgrids, mg, return_aos=False, get_aos_from_pyscf=False):
    print("Storing AOs", flush=True)
    cell = myisdf.cell_unc
    all_aos = None
    grid_id = 0
    # calculate it only on a single multigrid
    if return_aos:
        all_aos = []  # should return index from eval_ao and sort this.
    for k, ag in enumerate(atomgrids):
        if ag.nao == 0:
            continue
        grid_id = k // cell.natm
        if return_aos:
            all_aos.append(eval_ao(myisdf, ag, mg[grid_id], get_aos_from_pyscf))
        else:
            ag.aovals = eval_ao(myisdf, ag, mg[grid_id], get_aos_from_pyscf)
    return all_aos


def eval_ao(myisdf, ag, mg, from_pyscf=False):
    """
    Evaluate atomic orbitals on grid points.
    
    This function computes the values of atomic basis functions on the
    specified grid points. It supports both custom C-code evaluation and
    PySCF's built-in evaluation routines, with k-point sampling capabilities.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing cell and computational parameters
    ag : AtomGrid
        Atom-centered grid object specifying which orbitals to evaluate
    mg : Grid
        Grid object containing spatial coordinates
    from_pyscf : bool, default=False
        Force use of PySCF evaluation instead of optimized C-code
        
    Returns:
    --------
    numpy.ndarray or list
        Orbital values on grid points. Shape depends on k-point usage:
        - Gamma-point: (nao, ng) 
        - k-points: list of (nao, ng) arrays for each k-point
        
    Notes:
    -----
    - Automatically chooses between C-code and PySCF based on lattice vectors
    - Handles both orthogonal and non-orthogonal unit cells
    - Applies volume normalization for proper grid scaling
    - Supports cutoff-based screening for numerical efficiency
    """
    PRECISION = 1.0e-12
    t0 = time.time()
    cell = myisdf.cell_unc
    nK = myisdf.Nk
    a = cell.lattice_vectors()
    coord_idx = ag.coord_idx
    from_pyscf = myisdf.get_aos_from_pyscf or from_pyscf
    if mg.universal:
        coord_idx = numpy.arange(mg.Ng)
    if abs(a - numpy.diag(a.diagonal())).max() > 1e-12 or from_pyscf:
        coords = mg.coords
        if myisdf.use_kpt_symm:
            ao_vals = numpy.empty((nK, ag.nao, ag.Ng), numpy.complex128, order="C")
            ag_coords = coords[coord_idx]
            ao_index = 0
            for shell in ag.shells:
                naos = len(shell_to_ao_indices(cell, range(shell[0], shell[1])))
                out = myisdf._numint.eval_ao(
                    cell, ag_coords, shls_slice=shell, kpts=myisdf.kpts, cutoff=PRECISION
                )
                for nk in range(nK):
                    ao_vals[nk, ao_index : ao_index + naos] = out[nk].T
                ao_index += naos
            ao_vals *= (cell.vol / mg.Ng) ** 0.5
        else:
            ao_vals = numpy.empty((ag.nao, ag.Ng), numpy.float64, order="C")
            ag_coords = coords[coord_idx]
            ao_index = 0
            for shell in ag.shells:
                naos = len(shell_to_ao_indices(cell, range(shell[0], shell[1])))
                ao_vals[ao_index : ao_index + naos] = myisdf._numint.eval_ao(
                    cell, ag_coords, shls_slice=shell, cutoff=PRECISION
                )[0].T
                ao_index += naos
            ao_vals *= (cell.vol / mg.Ng) ** 0.5
    else:
        # nG != ag.Ng
        # ag.Ng refers to the non-zero points in the spherical region
        # nG refers to the rhobizoid we build the aos on in the c-code.
        nG = ag.nx * ag.ny * ag.nz
        centers = cell.atom_coords()

        if myisdf.use_kpt_symm:
            expLk = numpy.asarray(
                numpy.exp(1.0j * numpy.dot(ag.Ls, myisdf.kpts.T)), order="C"
            )  # nLs x Nk
            ao_vals = numpy.zeros((nK, ag.nao, nG), numpy.complex128, order="C")
            libevalao.pbceval_all_aos(
                ag.nexp,
                ao_vals,
                ag.x0,
                ag.y0,
                ag.z0,
                ag.nx,
                ag.ny,
                ag.nz,
                ag.exponents,
                ag.l,
                centers,
                ag.atoms,
                ag.norm,
                ag.nimages,
                ag.images,
                ag.nLs,
                ag.Ls,
                nK,
                expLk,
            )
            ao_vals = [ao_vals_k[:, ag.cube_to_sphere_index] for ao_vals_k in ao_vals]  # nLs x Nk
        else:
            ao_vals = numpy.zeros((ag.nao, nG), numpy.float64, order="C")
            libevalao.eval_all_aos(
                ag.nexp,
                ao_vals,
                ag.x0,
                ag.y0,
                ag.z0,
                ag.nx,
                ag.ny,
                ag.nz,
                ag.exponents,
                ag.l,
                centers,
                ag.atoms,
                ag.norm,
                ag.nimages,
                ag.images,
                ag.nLs,
                ag.Ls,
            )
            ao_vals = numpy.array(ao_vals[:, ag.cube_to_sphere_index], numpy.float64, order="C")

    if myisdf.use_kpt_symm:
        ao_vals = [
            numpy.ascontiguousarray(aok) for aok in ao_vals
        ]  # change to list so that we can have some real, some imag.
        gamma_pt = (numpy.sum(abs(myisdf.kpts), axis=1) < cell.precision).nonzero()[0][0]
        ao_vals[gamma_pt] = ao_vals[gamma_pt].real.astype(numpy.float64)

    myisdf.Times_["AOs"] += time.time() - t0
    return ao_vals


def eval_ao_1d(N, R, alpha, A, l, vals):
    for i in range(N):
        x_A = R[i] - A

        vals[i * (l + 1)] = numpy.exp(-alpha * x_A * x_A)
        for mx in range(1, l + 1):
            vals[i * (l + 1) + mx] = vals[i * (l + 1) + mx - 1] * x_A


def lstsq(aovals, Rg, out, sharp_fxns=None):
    if isinstance(aovals, numpy.ndarray):
        aovals = [aovals]

    def in_place_product(A, B):
        """Computes the element-wise product of A and B, storing the result in A."""
        A *= B
        return A

    phiLocalR = aovals if sharp_fxns is None else [ao[sharp_fxns] for ao in aovals]
    local_Rg_Rg = numpy.sum(numpy.matmul(aok[:, Rg].T, aok[:, Rg].conj()) for aok in phiLocalR)
    global_Rg_Rg = numpy.sum(numpy.matmul(aok[:, Rg].T.conj(), aok[:, Rg]) for aok in aovals)

    local_Rg_R = numpy.sum(numpy.matmul(aok[:, Rg].T, aok.conj()) for aok in phiLocalR)
    global_Rg_R = numpy.sum(numpy.matmul(aok[:, Rg].T.conj(), aok) for aok in aovals)

    X_Rg_Rg = in_place_product(local_Rg_Rg, global_Rg_Rg)
    X_Rg_R = in_place_product(local_Rg_R, global_Rg_R)
    # Expect: assume_a='her'
    out[:] = scipy.linalg.solve(
        X_Rg_Rg, X_Rg_R, overwrite_a=True, overwrite_b=True, check_finite=False
    ).real


def cholesky(aovals, isdf_thresh, sharp_fxns=None):
    """
    Select ISDF interpolation points using pivoted Cholesky decomposition.
    
    This function identifies the most important grid points for interpolative
    separable density fitting by performing a pivoted Cholesky decomposition
    of the orbital product matrix Z = LRR * GRR.
    
    Parameters:
    -----------
    aovals : list of numpy.ndarray or numpy.ndarray
        Atomic orbital values on grid points for each k-point
    isdf_thresh : float
        Threshold for Cholesky decomposition accuracy
    nK : int
        Number of k-points 
    sharp_fxns : numpy.ndarray, optional
        Indices of sharp (local) basis functions
        
    Returns:
    --------
    numpy.ndarray
        Indices of selected ISDF interpolation points (dtype: int32)
        
    Algorithm:
    ----------
    1. Compute orbital overlap matrices:
       - LRR: Local (sharp) orbital overlaps
       - GRR: Global (all) orbital overlaps  
    2. Form product matrix Z = LRR * GRR
    3. Perform pivoted Cholesky: Z = P^T L L^T P
    4. Select points based on threshold criterion
    """

    GRR = numpy.matmul(aovals.T.conj(), aovals)
    if sharp_fxns is None:
        LRR = GRR
    else:
        phiLocalR = aovals[sharp_fxns]
        LRR = numpy.matmul(phiLocalR.T, phiLocalR)
    Z = LRR * GRR
    # Real symmetric positive semidefinite matrix decomposition
    pp, nfit = scipy.linalg.lapack.dpstrf(Z, tol=isdf_thresh**2, overwrite_a=True)[1:3]
    P = pp[:nfit] - 1  # Convert to 0-based indexing
    return numpy.asarray(P, numpy.int32)


def cholesky_kpts(aovals, isdf_thresh, nK, sharp_fxns=None):
    """
    Select ISDF interpolation points using pivoted Cholesky decomposition.
    
    This function identifies the most important grid points for interpolative
    separable density fitting by performing a pivoted Cholesky decomposition
    of the orbital product matrix Z = LRR * GRR.
    
    Parameters:
    -----------
    aovals : list of numpy.ndarray or numpy.ndarray
        Atomic orbital values on grid points for each k-point
    isdf_thresh : float
        Threshold for Cholesky decomposition accuracy
    nK : int
        Number of k-points 
    sharp_fxns : numpy.ndarray, optional
        Indices of sharp (local) basis functions
        
    Returns:
    --------
    numpy.ndarray
        Indices of selected ISDF interpolation points (dtype: int32)
        
    Algorithm:
    ----------
    1. Compute orbital overlap matrices:
       - LRR: Local (sharp) orbital overlaps
       - GRR: Global (all) orbital overlaps  
    2. Form product matrix Z = LRR * GRR
    3. Perform pivoted Cholesky: Z = P^T L L^T P
    4. Select points based on threshold criterion
    """
    
    P = cholesky(aovals[0], isdf_thresh, sharp_fxns)

    if nK > 1:
        # nao = aovals[0].shape[0]
        # nfit = nK * nao
        # aovals = numpy.vstack(aovals)
        # sharp_fxns = numpy.concatenate([sharp_fxns + kn * nao for kn in range(nK)])
        # Z = scipy.stats.ortho_group.rvs(nao)[:nfit]
        # GRR = Z @ aovals
        # GRR = (GRR.T @ GRR.conj())
        # LRR = aovals[sharp_fxns]
        # LRR = (LRR.T @ LRR.conj())

        # nao = aovals[0].shape[0]
        # nfit = P.shape[0]
        # aovals = numpy.vstack(aovals[1:])
        # sharp_fxns = numpy.concatenate([sharp_fxns + kn * nao for kn in range(nK-1)])
        # Z = scipy.stats.ortho_group.rvs(aovals.shape[0])[:nfit]
        # GRR = Z @ aovals
        # GRR = (GRR.T @ GRR.conj())
        # LRR = aovals[sharp_fxns]
        # LRR = (LRR.T @ LRR.conj())
        
        GRR = numpy.sum(numpy.matmul(aok.T.conj(), aok) for aok in aovals)
        if sharp_fxns is None:
            LRR = GRR
        else:
            phiLocalR = [ao[sharp_fxns] for ao in aovals]
            LRR = numpy.sum(numpy.matmul(aok.T, aok.conj()) for aok in phiLocalR)

        # Real symmetric positive semidefinite matrix decomposition
        Z = ((LRR * GRR)/nK**2).real.astype(numpy.float64)
        pp, nfit = scipy.linalg.lapack.dpstrf(Z, tol=isdf_thresh**2, overwrite_a=True)[1:3]
        P = pp[:nfit] - 1  # Convert to 0-based indexing
        # P = numpy.unique(numpy.concatenate([P, pp[:nfit] - 1]))  # Convert to 0-based indexing

    return numpy.asarray(P, numpy.int32)


def isdf_points(myisdf, atomgrid):
    """
    Select ISDF interpolation points for an atom-centered grid.
    
    This function determines the optimal set of grid points for interpolative
    separable density fitting on a specific atom-centered grid using Cholesky
    decomposition of the orbital overlap matrix.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing threshold and k-point settings
    atomgrid : AtomGrid
        Atom-centered grid object containing orbital values and indices
        
    Returns:
    --------
    numpy.ndarray
        Indices of selected ISDF interpolation points for this atom grid
        
    Notes:
    ------
    - Uses only sharp (local) basis functions for point selection
    - Supports k-point sampling with optional gamma-point-only mode
    - Selection quality controlled by myisdf.isdf_thresh parameter
    """
    aovals = atomgrid.aovals
    sharpFxns = atomgrid.ao_index_sharp_on_grid
    nK = myisdf.Nk

    if myisdf.use_kpt_symm:
        P = cholesky_kpts(aovals, myisdf.isdf_thresh, nK, sharpFxns)
    else: 
        P = cholesky(aovals, myisdf.isdf_thresh, sharpFxns)
    return P


def isdf_pointsSG(myisdf, aovals):
    """
    Select ISDF interpolation points for the universal sparse grid.
    
    This function determines ISDF points on the universal sparse grid by
    performing separate point selection for each atom's contribution to
    the grid, then combining all selected points.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing grids and computational parameters
    aovals : list of numpy.ndarray or numpy.ndarray
        Atomic orbital values on universal grid for each k-point
        
    Returns:
    --------
    numpy.ndarray
        Combined set of ISDF interpolation points from all atoms
        
    Algorithm:
    ----------
    1. Loop over all atoms in the system
    2. Extract orbital values for each atom's grid region  
    3. Perform Cholesky decomposition for each atom separately
    4. Combine all selected points into final ISDF point set
    
    Notes:
    ------
    - Operates on the last (universal) grid in the hierarchy
    - Handles both k-point arrays and gamma-point-only calculations
    """
    ug = myisdf.atom_grids_k[-1]
    nao = ug.nao
    if myisdf.use_kpt_symm:
        if myisdf.isdf_pts_from_gamma_point:
            aovals = [aovals[0]]
        else:
            aovals = numpy.vstack(aovals)
    nK = myisdf.Nk

    nG = myisdf.full_grids_k[-1].Ng
    t0 = time.time()
    natm = len(ug.ao_index_by_atom)
    allPivots = []
    for i in range(natm):
        ao_id = ug.ao_index_by_atom[i]
        nao_i = ao_id.shape[0]
        if nao_i == 0:
            continue
        Ri = ug.coord_idx[i]

        if not isinstance(aovals, numpy.ndarray):
            aovals_i = [aok[:, Ri] for aok in aovals]
        else:
            ao_id = numpy.concatenate([ao_id + kn * nao for kn in range(nK)])
            aovals_i = aovals[:, Ri]

        Rg = Ri[cholesky(aovals_i, myisdf.isdf_thresh, myisdf.Nk, ao_id)]
        allPivots.extend(Rg)

    # Full Grid
    Rg = numpy.array(allPivots, numpy.int32)
    print(
        "atomic calculations done with Voronoi + cholesky! ",
        nG,
        Rg.shape[0],
        round(time.time() - t0, 2),
        "sec",
        flush=True,
    )

    # Do cholesky on universal grid with fitting points.
    t0 = time.time()
    if not isinstance(aovals, numpy.ndarray):
        aovals = [aok[:, Rg] for aok in aovals]
    else:
        aovals = aovals[:, Rg]

    Rg = Rg[cholesky(aovals, myisdf.isdf_thresh, myisdf.Nk)]
    print(
        "atomic calculations done with cholesky! ",
        nG,
        Rg.shape[0],
        round(time.time() - t0, 2),
        "sec",
        flush=True,
    )

    return numpy.sort(Rg)


def do_isdf_loop(myisdf, ag, mg):
    if ag.nao == 0:
        return []
    if getattr(ag, "aovals", None) is None:
        ag.aovals = eval_ao(myisdf, ag, mg)
    P = isdf_points(myisdf, ag)
    ag.IP_idx_on_grid = P
    ag.IP_idx = ag.coord_idx[P]
    ag.nthc = P.shape[0]
    myisdf.nao_max = max(ag.nao, myisdf.nao_max)
    myisdf.nthc_max = max(ag.nthc, myisdf.nthc_max)
    mg.Ng_nonzero += ag.Ng
    mg.nthc += ag.nthc
    mg.nao += ag.nao
    c_array = numpy.empty((ag.nthc, ag.Ng), numpy.float64)
    lstsq(ag.aovals, ag.IP_idx_on_grid, c_array, ag.ao_index_sharp_on_grid)
    if myisdf.use_kpt_symm:
        ag.aovals = [numpy.ascontiguousarray(aovals[:, ag.IP_idx_on_grid]) for aovals in ag.aovals]
    else:
        ag.aovals = numpy.ascontiguousarray(ag.aovals[:, ag.IP_idx_on_grid])
    return c_array


def do_isdf_and_build_w(myisdf):
    mg_k = myisdf.full_grids_k  # all universal grids
    ag_k = myisdf.atom_grids_k  # atoms on atom-centered & sparse universal grid

    print(flush=True)
    print("ISDF", flush=True)
    print(flush=True)
    print(
        "{0:8s} {1:8s} {2:8s} {3:8s} {4:8s} {5:8s} {6:8s} {7:8s} {8:8s} {9:8s} {10:8s} {11:8s}".format(
            "atm",
            "grid",
            "ng",
            "Ni",
            "Nxi",
            "sharp",
            "on_grid",
            "diffuse",
            "maxexp",
            "minexp",
            "lmax",
            "lmin",
        ),
        flush=True,
    )

    if myisdf.multigrid_on:
        mg0 = mg_k[0]
        if myisdf.fit_dense_grid:
            t0 = time.time()
            # with threadpool_limits(limits=1):
            #     with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
            #         c_array = Parallel()(
            #             delayed(do_isdf_loop)(myisdf, ag, mg0) for i, ag in enumerate(ag_k[:-1])
            #         )
            c_array = []
            for i, ag in enumerate(ag_k[:-1]):
                    c_array.append(do_isdf_loop(myisdf, ag, mg0))
            myisdf.Times_["ISDF-pts"] += time.time() - t0
        else:
            for i, ag in enumerate(ag_k[:-1]):
                ag.IP_idx = ag.IP_idx_on_grid = range(ag.Ng)
                ag.nthc = ag.Ng
                mg0.Ng_nonzero += ag.Ng
                mg0.nthc += ag.nthc
                mg0.nao += ag.nao
                myisdf.nao_max = max(myisdf.nao_max, ag.nao)
                myisdf.nthc_max = max(myisdf.nthc_max, ag.nthc)
                if getattr(ag, "aovals", None) is None:
                    ag.aovals = eval_ao(myisdf, ag, mg0)
            c_array = [numpy.eye(ag.nthc) for ag in ag_k[:-1]]

        for i, ag in enumerate(ag_k[:-1]):
            if ag.nao == 0:
                print(
                    "{0:<8d} {1:<8d} {2:<8d} {3:<8d} {4:<8d} {5:<8d} {6:<8d} {7:<8d} {8:<8d} {9:<8d} {10:<8d} {11:<8d}".format(
                        i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ),
                    flush=True,
                )
            else:
                print(
                    "{0:<8d} {1:<8d} {2:<8d} {3:<8d} {4:<8d} {5:<8d} {6:<8d} {7:<8d} {8:<8.2f} {9:<8.2f} {10:<8d} {11:<8d}".format(
                        i,
                        0,
                        mg0.Ng,
                        ag.Ng,
                        ag.nthc,
                        ag.n_sharp,
                        ag.nao,
                        ag.n_diffuse,
                        ag.max_exp,
                        ag.min_exp,
                        ag.l.max(),
                        ag.l.min(),
                    ),
                    flush=True,
                )

    else:
        c_array = []
    dense_time = myisdf.Times_["ISDF-pts"] + myisdf.Times_["ISDF-vec"]

    ag = ag_k[-1]
    if getattr(ag, "aovals", None) is None:
        ag.aovals = eval_ao(myisdf, ag, mg_k[-1])
    if myisdf.fit_sparse_grid:
        t0 = time.time()
        P = isdf_pointsSG(myisdf, ag.aovals)
        myisdf.Times_["ISDF-pts"] += time.time() - t0
        ag.IP_idx = ag.IP_idx_on_grid = P
        ag.nthc = P.shape[0]
        myisdf.nao_max = max(ag.nao, myisdf.nao_max)
        myisdf.nthc_max = max(ag.nthc, myisdf.nthc_max)
        c_array.append(numpy.empty((ag.nthc, ag.Ng), numpy.float64))
        t0 = time.time()
        lstsq(ag.aovals, ag.IP_idx_on_grid, c_array[-1])
        myisdf.Times_["ISDF-vec"] += time.time() - t0
        if myisdf.use_kpt_symm:
            ag.aovals = [numpy.asarray(aos[:, ag.IP_idx_on_grid], order="C") for aos in ag.aovals]
        else:
            ag.aovals = numpy.asarray(ag.aovals[:, ag.IP_idx_on_grid], order="C")
    else:
        ag.IP_idx = ag.IP_idx_on_grid = range(ag.Ng)
        ag.nthc = ag.Ng
        myisdf.nao_max = max(myisdf.nao_max, ag.nao)
        myisdf.nthc_max = max(myisdf.nthc_max, ag.nthc)
        c_array.append([None])

    mg_k[-1].Ng_nonzero += ag.Ng
    mg_k[-1].nthc += ag.nthc
    mg_k[-1].nao += ag.nao
    print(
        "{0:<8s} {1:<8d} {2:<8d} {3:<8d} {4:<8d} {5:<8d} {6:<8d} {7:<8d} {8:<8.2f} {9:<8.2f} {10:<8d} {11:<8d}".format(
            "-",
            1,
            mg_k[-1].Ng,
            ag.Ng,
            ag.nthc,
            ag.nao,
            ag.nao,
            ag.nao,
            ag.max_exp,
            ag.min_exp,
            ag.l.max(),
            ag.l.min(),
        ),
        flush=True,
    )

    print("ISDF-pts: {0:10.2f}".format(myisdf.Times_["ISDF-pts"]), flush=True)
    print("ISDF-vec: {0:10.2f}".format(myisdf.Times_["ISDF-vec"]), flush=True)
    print("Fit dense grids ISDF: {0:10.2f}".format(dense_time), flush=True)
    print(
        "Fit sparse grids ISDF: {0:10.2f}".format(
            myisdf.Times_["ISDF-pts"] + myisdf.Times_["ISDF-vec"] - dense_time
        ),
        flush=True,
    )
    print()
    print(
        "{0:8s} {1:8s} {2:8s} {3:8s} {4:8s}".format("Grid", "PWs", "ng_i", "N_xi", "N_mu"),
        flush=True,
    )
    print(
        "{0:<8d} {1:<8d} {2:<8d} {3:<8d} {4:<8d}".format(
            0, mg_k[0].Ng, mg_k[0].Ng_nonzero, mg_k[0].nthc, mg_k[0].nao
        ),
        flush=True,
    )
    print(
        "{0:<8d} {1:<8d} {2:<8d} {3:<8d} {4:<8d}".format(
            1, mg_k[-1].Ng, mg_k[-1].Ng_nonzero, mg_k[-1].nthc, mg_k[-1].nao
        ),
        flush=True,
    )
    print()
    print("nthc:", mg_k[0].nthc + mg_k[-1].nthc, flush=True)
    myisdf.nthc = mg_k[0].nthc + mg_k[-1].nthc

    print()
    print("Build W", flush=True)
    t0 = time.time()
    isdf_fft(myisdf, c_array)
    myisdf.Times_["ISDF-fft"] += time.time() - t0
    print("ISDF-fft: {0:10.2f}".format(myisdf.Times_["ISDF-fft"]), flush=True)
    print()


def isdf_fft(myisdf, c_list):
    atomgrids = myisdf.atom_grids_k
    mg = myisdf.full_grids_k
    natm = myisdf.cell.natm
    w_mem = 0.0
    myisdf.W = [[] for _ in range(len(atomgrids))]
    if myisdf.multigrid_on:
        for i in range(natm):
            ai = atomgrids[i]
            w_mem += ai.nthc * ai.nthc
            for j in range(i + 1, natm):
                aj = atomgrids[j]
                w_mem += ai.nthc * aj.nthc
            w_mem += ai.nthc * atomgrids[-1].nthc
    if not myisdf.direct_k_sparse:
        myisdf.W[-1].append([])
        w_mem += atomgrids[-1].nthc * (atomgrids[-1].nthc + 1) // 2
    w_mem *= myisdf.Nk * 8e-9
    ao_mem = sum(ag.nao * ag.nthc for ag in atomgrids)
    ao_mem *= myisdf.Nk * 8e-9
    c_mem = sum(ag.Ng * ag.nthc for ag in atomgrids[:-1])
    if not myisdf.direct_k_sparse:
        c_mem += atomgrids[-1].Ng * atomgrids[-1].nthc
    c_mem *= 8e-9
    current_mem = lib.current_memory()[0] / 1000 + w_mem
    avail_memory = myisdf.max_memory / 1000 - current_mem
    print("Curent RAM Use (GB): {0:10.2f} ".format(current_mem), flush=True)
    print("Avail  RAM     (GB): {0:10.2f} ".format(avail_memory), flush=True)
    print("W Mem          (GB): {0:10.2f}".format(w_mem), flush=True)

    if myisdf.multigrid_on:
        nthc_max = max(ag.nthc for ag in atomgrids[:-1])
        req_mem = (2.0 * (mg[0].Ng + mg[1].Ng) * nthc_max) * myisdf.Nk * 8e-9
        njobs = myisdf.joblib_njobs
        req_mem_parallel = req_mem * njobs
        if njobs > 1:
            # print(njobs, req_mem_parallel)
            while req_mem_parallel > avail_memory and njobs > 1:
                njobs = max(njobs // 2, 1)
                req_mem_parallel = req_mem * njobs
                # print(njobs, req_mem_parallel)
        mem_limited = True if njobs == 1 else False
        blas_jobs = myisdf.joblib_njobs if njobs == 1 else myisdf.fftw_njobs
        print("Build W RAM Use  (GB): {0:10.2f}".format(req_mem_parallel), flush=True)
        print(f"Build W using {njobs} joblib threads and {blas_jobs} BLAS threads.", flush=True)
        myisdf.fft_time = 0
        myisdf.blas_time = 0
        t0 = time.time()
        if myisdf.use_kpt_symm:
            with threadpool_limits(limits=blas_jobs):
                with parallel_backend("threading", n_jobs=njobs):
                    Parallel()(
                        delayed(get_thc_potential_kpts)(myisdf, i, c_list, mem_limited)
                        for i in range(len(atomgrids) - 1)
                    )
            # for i in range(len(atomgrids) - 1):
            #     get_thc_potential_kpts(myisdf, i, c_list, mem_limited)

        else:
            with threadpool_limits(limits=blas_jobs):
                with parallel_backend("threading", n_jobs=njobs):
                    Parallel()(
                        delayed(get_thc_potential)(myisdf, i, c_list, mem_limited)
                        for i in range(len(atomgrids) - 1)
                    )
            # for i in range(len(atomgrids) - 1):
            #     get_thc_potential(myisdf, i, c_list, mem_limited)                    
        print("ISDF-fft dense grids (sec): {0:4.2f}".format(time.time() - t0), flush=True)
        print("BLAS time (sec): {0:4.2f}".format(myisdf.blas_time), flush=True)
        print("FFT  time (sec): {0:4.2f}".format(myisdf.fft_time), flush=True)
        c_list = c_list[-1]

    t0 = time.time()
    if not myisdf.direct_k_sparse:
        t0 = time.time()
        if myisdf.use_kpt_symm:
            get_thc_potential_sparse_grid_kpts(myisdf, c_list[-1])
        else:
            get_thc_potential_sparse_grid(myisdf, c_list)
    print("ISDF-fft sparse grids (sec): {0:4.2f}".format(time.time() - t0), flush=True)


def get_thc_potential(myisdf, atomID, C, mem_limited):
    """
    Compute tensor hypercontraction (THC) potential for a specific atom.
    
    This function evaluates the THC potential matrix W for exchange integral
    calculations. The THC decomposition represents four-center integrals as
    products of two-center quantities, enabling efficient exchange evaluation.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing grids and fitting parameters
    atomID : int
        Index of the atom for which to compute THC potential
    C : numpy.ndarray
        Orbital coefficients contracted with basis functions
    mem_limited : bool
        Whether to use memory-limited algorithm for large systems
        
    Returns:
    --------
    None
        Modifies myisdf.W[atomID] in-place with computed THC potential
        
    Algorithm:
    ----------
    1. Set up FFT grid mapping from atom grid to universal grid
    2. Compute basis function values at THC interpolation points
    3. Apply Coulomb operator via FFT convolution
    4. Contract with orbital coefficients to form W matrix
    
    Notes:
    -----
    - THC enables O(N^3) scaling for exchange matrix construction
    - Memory usage can be controlled via mem_limited parameter
    - Results stored in myisdf.W for later use in exchange evaluation
    """
    atomgrids = myisdf.atom_grids_k
    ai = atomgrids[atomID]
    if ai.nao == 0:
        myisdf.W[atomID] = [[]]
        return
    coord_idx = ai.coord_idx
    mg = myisdf.full_grids_k
    nG_dense = mg[0].Ng
    nG_sparse = mg[-1].Ng
    cell = myisdf.cell_unc
    natm = cell.natm
    nthc = ai.nthc
    mesh0 = mg[0].mesh
    njobs = myisdf.joblib_njobs if mem_limited else 1
    w_array = myisdf.W[atomID]

    ### FFT
    t0 = time.time()
    if nthc == ai.Ng:
        Gv = cell.get_Gv(mesh0)
        V = numpy.exp(-1.0j * (Gv[coord_idx] @ mg[0].coords.T))
        V *= 1.0 / (nG_dense**0.5)
    else:
        V = numpy.zeros(
            (nthc, nG_dense),  # <-- the issue for parallel for large systems
            numpy.complex128,
        )
        V[:, coord_idx] = C[atomID]
        with parallel_backend("threading", n_jobs=njobs):
            Parallel()(delayed(r2c_fftw_execute)(v, mesh0) for v in V)
    V *= tools.get_coulG(cell, mesh=mesh0)
    myisdf.fft_time += time.time() - t0

    ### iFFT
    reduced_coords = get_reduced_freq_coords(mg[-1].mesh, mesh0)
    v_on_universal_grid = numpy.take(V, reduced_coords, axis=1)
    mesh1 = mg[-1].mesh
    t0 = time.time()
    with parallel_backend("threading", n_jobs=njobs):
        Parallel()(delayed(c2r_ifftw_execute)(v0, mesh0) for v0 in V)
        Parallel()(delayed(c2r_ifftw_execute)(v1, mesh1) for v1 in v_on_universal_grid)
    myisdf.fft_time += time.time() - t0

    # Do Dense Grid
    V = V.real.astype(numpy.float64)
    f = nG_dense * 1.0 / cell.vol
    if nthc == ai.Ng:
        w_array[:] = [(V[:, coord_idx] * f)[numpy.triu_indices(nthc)]]
    else:
        w_array[:] = [(V[:, coord_idx] * f @ C[atomID].T)[numpy.triu_indices(nthc)]]
    for j in range(atomID + 1, natm):
        aj = atomgrids[j]
        if aj.nao != 0:
            if aj.nthc == aj.Ng:
                w_array.append(V[:, aj.coord_idx] * f)
            else:
                w_array.append(V[:, aj.coord_idx] * f @ C[j].T)
        else:
            w_array.append([])
    myisdf.blas_time += time.time() - t0

    # Do Sparse Grid
    v_on_universal_grid = v_on_universal_grid.real.astype(numpy.float64)
    f = nG_dense**0.5 * nG_sparse**0.5 * 1.0 / cell.vol
    if atomgrids[-1].nthc == nG_sparse:
        w_array.append(v_on_universal_grid * f)
    else:
        w_array.append(v_on_universal_grid * f @ C[-1].T)

def get_thc_potential_kpts(myisdf, atomID, C, mem_limited):
    atomgrids = myisdf.atom_grids_k
    mg = myisdf.full_grids_k
    cell = myisdf.cell_unc
    ai = atomgrids[atomID]
    if ai.nao == 0:
        myisdf.W[atomID] = [[]]
        return

    nG0 = mg[0].Ng
    nG_universal = mg[-1].Ng
    mesh0 = mg[0].mesh
    mesh_universal = mg[-1].mesh
    coords0 = mg[0].coords
    coords_universal = mg[-1].coords
    coord_idx = ai.coord_idx
    natm = cell.natm

    nthc = ai.nthc
    kpts = -myisdf.kpts
    nK = myisdf.Nk

    fft_factor0 = 1.0 / (nG0** 0.5)
    ifft_factor0 = nG0 ** 0.5
    ifft_factor_universal = nG_universal ** 0.5
    f0 = 2.0 * nG0 / cell.vol / nK
    fu = 2.0 * (nG_universal * nG0) ** 0.5 / cell.vol / nK
    njobs = myisdf.joblib_njobs if mem_limited else 1
    if nthc == ai.Ng:
        Gv = cell.get_Gv(mesh0)

    out = [None] * nK
    for k, kpt in enumerate(kpts):
        expmikr = numpy.exp(-1.0j * (coords0 @ kpt))
        if nthc == ai.Ng:
            V = numpy.exp(-1j * (coords0[coord_idx] @ (Gv + kpt).T))
            V *= fft_factor0
        else:
            V = numpy.zeros((nthc, nG0), numpy.complex128)
            V[:, coord_idx] = C[atomID] * expmikr[coord_idx]
            with parallel_backend("threading", n_jobs=njobs):
                Parallel()(delayed(c2c_fftw_execute)(v, mesh0) for v in V)

        V *= tools.get_coulG(cell, k=kpt, mesh=mesh0)
        
        reduced_coords = get_reduced_freq_coords(mesh_universal, mesh0)
        v_on_universal_grid = numpy.take(V, reduced_coords, axis=1)
        with parallel_backend("threading", n_jobs=njobs):
            Parallel()(delayed(c2c_ifftw_execute)(v, mesh0) for v in V)
            Parallel()(delayed(c2c_ifftw_execute)(v, mesh_universal) for v in v_on_universal_grid)
        V *= expmikr.conj()

        if nthc == ai.Ng:
            W = V[:, coord_idx].T
        else:
            W = C[atomID] @ V[:, coord_idx].T
        W *= f0
        out[k] = [W.T]

        for j in range(atomID - 1, -1, -1):
            aj = atomgrids[j]
            if aj.nao == 0:
                out[k].append([])
                continue
            if aj.nthc == aj.Ng:
                W = V[:, aj.coord_idx].T
            else:
                W = C[j] @ V[:, aj.coord_idx].T
            W *= f0
            out[k].append(W.T)
        V = None

        # Do Sparse Grid
        v_on_universal_grid *= numpy.exp(1.0j * (coords_universal @ kpt))
        if atomgrids[-1].nthc == nG_universal:
            W = v_on_universal_grid.T
        else:
            W = C[-1] @ v_on_universal_grid.T
        W *= fu
        out[k].append(W.T)

    # out[k][n] --> out[n][k]
    out = [numpy.asarray([out[k][n] for k in range(nK)]) for n in range(len(out[0]))]
    out_fft = []
    for n, out_n in enumerate(out):
        out_fft.append(
            scipy.fft.hfftn(
                out_n.reshape(*myisdf.kmesh, ai.nthc, atomgrids[atomID - n].nthc),
                s=(myisdf.kmesh),
                axes=[0, 1, 2],
                overwrite_x=True,
            )
        )
    myisdf.W[atomID] = out_fft


def get_thc_potential_sparse_grid(myisdf, C):
    cell = myisdf.cell_unc
    ug = myisdf.atom_grids_k[-1]
    mg = myisdf.full_grids_k[-1]
    nthc = ug.nthc
    mesh = mg.mesh
    W = myisdf.W[-1][-1]

    f = ug.Ng / cell.vol
    if nthc != mg.Ng:
        V = (C * 1.0).astype(numpy.complex128)
        for v in V:
            r2c_fftw_execute(v, mesh, False)
        V *= tools.get_coulG(cell, mesh=mesh)
        for v in V:
            c2r_ifftw_execute(v, mesh, False)
        W[:] = numpy.matmul(C, V.T.real.astype(numpy.float64))[numpy.triu_indices(nthc)]
    else:
        coords = mg.coords
        Gv = cell.get_Gv(mesh)
        Gv = numpy.asarray(Gv.T, order="C")
        V = numpy.exp(-1.0j * numpy.matmul(coords, Gv)).T
        V *= tools.get_coulG(cell, mesh=mesh)
        for v in V:
            c2r_ifftw_execute(v, mesh, False)
        W[:] = V.real.astype(numpy.float64)[numpy.triu_indices(nthc)]

    W *= f


def get_thc_potential_sparse_grid_kpts(myisdf, C=None):
    cell = myisdf.cell_unc
    ug = myisdf.atom_grids_k[-1]
    mg = myisdf.full_grids_k[-1]
    nG = mg.Ng
    nthc = ug.nthc
    nK = myisdf.Nk
    f = 2.0 * nG / cell.vol * 1.0 / nK
    kpts = myisdf.kpts
    mesh = mg.mesh
    coords = mg.coords
    W = numpy.empty((nK, nthc, nthc), numpy.complex128)
    njobs = myisdf.joblib_njobs 
    if nG == nthc:
        Gv = cell.get_Gv(mesh)
        for kpt, Wk in zip(kpts, W):
            Wk[:] = numpy.exp(-1j * (coords @ (Gv + kpt).T))  # FFT of delta function.
            Wk *= tools.get_coulG(cell, kpt, mesh=mesh)
            with parallel_backend("threading", n_jobs=njobs):
                Parallel()(delayed(c2c_ifftw_execute)(wk, mesh, False) for wk in Wk)
            Wk *= numpy.exp(1.0j * (coords @ kpt))
    else:
        for kpt, Wk in zip(kpts, W):
            expmikr = numpy.exp(-1.0j * (coords @ kpt))
            V = C * expmikr
            with parallel_backend("threading", n_jobs=njobs):
                Parallel()(delayed(c2c_fftw_execute)(v, mesh, False) for v in V)
            V *= tools.get_coulG(cell, kpt, mesh=mesh).reshape(1, -1)
            with parallel_backend("threading", n_jobs=njobs):
                Parallel()(delayed(c2c_ifftw_execute)(v, mesh, False) for v in V)
            V *= expmikr.conj()
            numpy.matmul(V, C.T.conj(), Wk)
    W = W.conj()
    W *= f
    W = scipy.fft.hfftn(
        W.reshape(*myisdf.kmesh, nthc, nthc), s=(myisdf.kmesh), axes=[0, 1, 2], overwrite_x=True
    )
    myisdf.W[-1] = [W]


def make_rho(ag, dm, aovals):
    """
    Construct electron density on atom-centered grid points.
    
    This function computes the electron density using a mixed sharp/diffuse 
    orbital approach for efficient evaluation on atom-centered grids.
    
    Parameters:
    -----------
    ag : AtomGrid
        Atom-centered grid object containing orbital indices
    dm : numpy.ndarray
        Density matrix (nao x nao)
    aovals : numpy.ndarray
        Atomic orbital values on grid points
        
    Returns:
    --------
    numpy.ndarray
        Electron density values on grid points
        
    Algorithm:
    ----------
    - Uses sharp orbital basis functions for efficient grid evaluation
    - Applies 2x weighting for diffuse orbital contributions
    - Computes density as ρ(r) = Σ_μν D_μν φ_μ(r) φ_ν(r)
    """
    sharp_idx = ag.ao_index_sharp
    sharp_idx_on_grid = ag.ao_index_sharp_on_grid
    all_idx = ag.ao_index
    diff1 = ag.ao_index_diffuse_on_grid
    local_dm = dm[numpy.ix_(sharp_idx, all_idx)]
    local_dm[:, diff1] *= 2.0  # Weight diffuse orbital contributions
    density = local_dm @ aovals
    density *= aovals[sharp_idx_on_grid]
    return numpy.sum(density, axis=0)


def make_rho_sg(ag, dm, aovals):
    """
    Construct electron density on sparse grid points.
    
    Parameters:
    -----------
    ag : AtomGrid
        Atom-centered grid object
    dm : numpy.ndarray
        Density matrix
    aovals : numpy.ndarray
        Atomic orbital values on grid points
        
    Returns:
    --------
    numpy.ndarray
        Electron density values on sparse grid points
    """
    aoidx = ag.ao_index
    density = dm[numpy.ix_(aoidx, aoidx)] @ aovals
    density *= aovals
    return numpy.sum(density, axis=0)


def get_pot(ag, aovals, pot_on_grid, f, j):
    """
    Compute potential matrix elements from grid potential.
    
    Parameters:
    -----------
    ag : AtomGrid
        Atom-centered grid object containing orbital indices
    aovals : numpy.ndarray
        Atomic orbital values on grid points
    pot_on_grid : numpy.ndarray
        Potential values on grid points
    f : float
        Scaling factor for potential
    j : numpy.ndarray
        Output potential matrix (modified in-place)
        
    Note:
    -----
    Updates j matrix elements for sharp-diffuse and diffuse-sharp orbital pairs
    """
    sharp_idx = ag.ao_index_sharp
    sharp_idx_on_grid = ag.ao_index_sharp_on_grid
    all_idx = ag.ao_index
    pot = numpy.multiply(pot_on_grid[ag.coord_idx] * f, aovals[sharp_idx_on_grid])
    pot = pot @ aovals.T
    j[sharp_idx[:, None], all_idx[None, :]] = pot
    j[all_idx[:, None], sharp_idx[None, :]] = pot.T


def get_pot_sg(ag, aovals, pot_on_grid, f, j):
    """
    Compute potential matrix elements for sparse grid.
    
    Parameters:
    -----------
    ag : AtomGrid
        Atom-centered grid object
    aovals : numpy.ndarray
        Atomic orbital values on grid points
    pot_on_grid : numpy.ndarray
        Potential values on grid points
    f : float
        Scaling factor
    j : numpy.ndarray
        Output matrix (modified in-place)
    """
    aoidx = ag.ao_index
    pot = numpy.multiply(pot_on_grid[numpy.newaxis, :] * f, aovals)
    j[aoidx[:, None], aoidx[None, :]] = pot @ aovals.T


def inner_density(myisdf, ag, mg, dm):
    """
    Calculate electron density on atom-centered grid points.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object with calculation parameters
    ag : AtomGrid
        Atom-centered grid object
    mg : Grid
        Full grid object
    dm : numpy.ndarray
        Density matrix or array of density matrices
        
    Returns:
    --------
    numpy.ndarray
        Electron density values on grid points (shape: nset x Ng)
    """
    nset = dm.shape[0]
    density_on_grid = numpy.zeros((nset, mg.Ng), numpy.float64)
    if ag.nao == 0:
        return density_on_grid
    aovals = ag.aovals if getattr(ag, "aovals", None) is not None else eval_ao(myisdf, ag, mg)
    for j in range(nset):
        density_on_grid[j, ag.coord_idx] += make_rho(ag, dm[j], aovals)
    return density_on_grid


def inner_pot(myisdf, ag, mg, pot_on_grid, vj, f):
    """
    Compute potential matrix contributions from atom-centered grid.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object with calculation parameters  
    ag : AtomGrid
        Atom-centered grid object
    mg : Grid
        Full grid object
    pot_on_grid : numpy.ndarray
        Potential values on grid points (shape: nset x Ng)
    vj : numpy.ndarray
        Output Coulomb matrix (modified in-place)
    f : float
        Normalization factor
    """
    if ag.nao == 0:
        return
    aovals = ag.aovals if getattr(ag, "aovals", None) is not None else eval_ao(myisdf, ag, mg)
    for j in range(pot_on_grid.shape[0]):
        get_pot(ag, aovals, pot_on_grid[j], f, vj[j])


def get_j(myisdf, dm):
    """
    Compute Coulomb (J) matrix using multi-grid ISDF method.
    
    This function calculates the Coulomb interaction matrix elements using a 
    hierarchical multi-grid approach combined with FFT-based convolutions.
    The algorithm consists of three main steps:
    1. Calculate electron density on multiple grids
    2. Convolve with Coulomb operator in Fourier space  
    3. Integrate potential with basis functions to get J matrix
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing grids, atoms, and computational parameters
    dm : numpy.ndarray
        Density matrix or array of density matrices (shape: [nset,] nao, nao)
        
    Returns:
    --------
    numpy.ndarray
        Coulomb matrix with same shape as input density matrix
        
    Algorithm:
    ----------
    - Multi-grid density construction using atom-centered grids
    - Universal sparse grid for long-range contributions  
    - FFT-based Coulomb operator application: G^{-1} in k-space
    - Parallel evaluation over atoms and grid levels
    - Inverse FFT to get potential in real space
    - Integration with basis functions to form J matrix elements
    """
    # Extract computational parameters and grid information
    cell = myisdf.cell_unc
    mg = myisdf.full_grids_j
    atomgrids = myisdf.atom_grids_j
    natm = cell.natm
    nmg = len(mg)
    if mg[-1].universal:
        nmg -= 1  # Exclude universal grid from main loop
    mesh0 = mg[0].mesh  # Reference grid for interpolation

    # STEP 1: CALCULATE ELECTRON DENSITY ON GRIDS
    t0 = time.time()
    nset = dm.shape[0]  # Number of density matrices to process
    full_density = numpy.zeros((nset, mg[0].Ng), numpy.complex128)
    # Loop over multi-grid levels (atom-centered grids)
    for n in range(nmg):
        # Parallel density calculation over atoms at current grid level
        with threadpool_limits(limits=myisdf.fftw_njobs):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                results = Parallel()(
                    delayed(inner_density)(myisdf, atomgrids[i + natm * n], mg[n], dm)
                    for i in range(natm)
                )

        # Sum atomic contributions and apply volume normalization
        density_on_grid = numpy.sum(results, axis=0)
        density_on_grid *= (mg[n].Ng / cell.vol) ** 0.5

        # Transform to Fourier space and interpolate to reference grid
        mesh = mg[n].mesh
        intp = get_reduced_freq_coords(mesh, mesh0)
        for j in range(nset):
            full_density[j, intp] += r2c_fftw_execute(density_on_grid[j], mesh, return_x=True)

    # Handle universal sparse grid if present
    if mg[-1].universal:
        # Process universal grid contributions to density
        ag = atomgrids[-1]
        aovals_sg = ag.aovals if getattr(ag, "aovals", None) is not None else eval_ao(myisdf, ag, mg[-1])
        mesh = mg[-1].mesh
        f = (mg[-1].Ng / cell.vol) ** 0.5
        intp = get_reduced_freq_coords(mesh, mesh0)
        for j in range(nset):
            density_on_grid = make_rho_sg(ag, dm[j], aovals_sg)
            density_on_grid *= f
            full_density[j, intp] += r2c_fftw_execute(density_on_grid, mesh, return_x=True)

    # STEP 2: APPLY COULOMB OPERATOR IN FOURIER SPACE
    # Multiply density by 1/|G|^2 Coulomb kernel
    full_density *= tools.get_coulG(cell, mesh=mesh0)

    # Initialize output Coulomb matrix
    vj2 = 0.0 * dm
    
    # STEP 3: TRANSFORM BACK TO REAL SPACE AND COMPUTE J MATRIX
    for n in range(nmg):
        mesh = mg[n].mesh
        # Transform potential from Fourier space back to real space
        intp = get_reduced_freq_coords(mesh, mesh0)
        pot_on_grid2 = numpy.empty((nset, mg[n].Ng), numpy.float64)
        for j in range(nset):
            pot_on_grid2[j] = c2r_ifftw_execute(full_density[j, intp], mesh, return_x=True)
        
        # Apply normalization factor and integrate with basis functions
        f = (mg[n].Ng / cell.vol) ** 0.5
        with threadpool_limits(limits=myisdf.fftw_njobs):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                Parallel()(
                    delayed(inner_pot)(myisdf, atomgrids[i + natm * n], mg[n], pot_on_grid2, vj2, f)
                    for i in range(natm)
                )

    # Handle universal sparse grid contributions to J matrix
    if mg[-1].universal:
        # Transform potential to universal grid and compute sparse grid contributions
        mesh = mg[-1].mesh
        intp = get_reduced_freq_coords(mesh, mesh0)
        f = (mg[-1].Ng / cell.vol) ** 0.5
        pot_on_grid2 = numpy.empty((nset, mg[-1].Ng), numpy.float64)
        for j in range(nset):
            pot_on_grid2[j] = c2r_ifftw_execute(full_density[j, intp], mesh, return_x=True)
            get_pot_sg(atomgrids[-1], aovals_sg, pot_on_grid2[j], f, vj2[j])

    return vj2

def inner_density_kpts(myisdf, ag, mg, dm):
    rhoR = numpy.zeros(mg.Ng, numpy.complex128)
    if ag.nao == 0:
        return rhoR

    sharp_idx, sharp_idx_on_grid = ag.ao_index_sharp, ag.ao_index_sharp_on_grid
    all_idx, diffuse_idx = ag.ao_index, ag.ao_index_diffuse_on_grid
    aovals = ag.aovals if getattr(ag, "aovals", None) is not None else eval_ao(myisdf, ag, mg)

    for ao_k, dm_k in zip(aovals, dm):
        local_dm = dm_k[numpy.ix_(sharp_idx, all_idx)]
        local_dm[:, diffuse_idx] *= 2.0
        ao_dm = local_dm @ ao_k.conj()
        ao_dm *= ao_k[sharp_idx_on_grid]
        rhoR[ag.coord_idx] += numpy.sum(ao_dm, axis=0).real
    return rhoR

def get_j_kpts(myisdf, dm):
    """
    Compute Coulomb (J) matrix for k-point sampled calculations.
    
    This function extends the multi-grid ISDF Coulomb matrix evaluation
    to k-point sampling in periodic boundary conditions. The algorithm
    handles complex-valued density matrices and properly accounts for
    k-point phase factors.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing k-point grids and computational parameters
    dm : numpy.ndarray
        Array of density matrices for each k-point (nK x nao x nao)
        
    Returns:
    --------
    numpy.ndarray
        Coulomb matrix summed over all k-points (nao x nao)
        
    Algorithm:
    ----------
    1. Calculate density on grids for all k-points simultaneously
    2. Sum density contributions with proper k-point weights
    3. Apply Coulomb operator in reciprocal space
    4. Transform back and integrate with basis functions
    
    Key Differences from Gamma-point:
    ---------------------------------
    - Uses complex arithmetic for k-point phase factors
    - Sums density over all k-points before Coulomb operation
    - Accounts for k-point symmetry when use_kpt_symm=True
    - Final result is k-point averaged J matrix
    """
    cell = myisdf.cell_unc
    mg = myisdf.full_grids_j
    atomgrids = myisdf.atom_grids_j
    nK = myisdf.Nk
    natm = cell.natm
    nG_dense = mg[0].Ng
    mesh0 = mg[0].mesh
    nmg = len(mg)
    if mg[-1].universal:
        nmg -= 1

    ####CALCULATE THE DENSITY
    full_density = numpy.zeros(nG_dense, numpy.complex128)

    for n in range(nmg):
        with threadpool_limits(limits=myisdf.fftw_njobs):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                results = Parallel()(
                    delayed(inner_density_kpts)(myisdf, atomgrids[i + natm * n], mg[n], dm)
                    for i in range(natm)
                )
        
        rhoR = numpy.sum(results, axis=0)
        nG = mg[n].Ng
        mesh = mg[n].mesh
        rhoR *= ( nG / cell.vol / nK) ** 0.5
        rhoG = tools.fft(rhoR, mesh)
        rhoG *= 1.0 / nG ** 0.5
        intp = get_reduced_freq_coords(mesh, mesh0)
        full_density[intp] += rhoG

    if mg[-1].universal:
        # Do universal grid
        # NOTE: PySCF: vR may be complex if the underlying density is complex
        # What is nec for our systems?
        rhoR = numpy.zeros(mg[-1].Ng, numpy.complex128)
        f = (mg[-1].Ng / cell.vol / nK) ** 0.5

        aoidx = atomgrids[-1].ao_index
        if getattr(atomgrids[-1], "aovals", None) is not None and (
            atomgrids[-1].nthc == atomgrids[-1].Ng
        ):
            aovals_sg = atomgrids[-1].aovals
        else:
            aovals_sg = eval_ao(myisdf, atomgrids[-1], mg[-1])

        for ao_k, dm_k in zip(aovals_sg, dm):
            local_dm = dm_k[numpy.ix_(aoidx, aoidx)]
            ao_dm = local_dm @ ao_k.conj()
            ao_dm *= ao_k
            rhoR += numpy.sum(ao_dm, axis=0).real
        rhoR *= f
        mesh = mg[-1].mesh
        rhoG = tools.fft(rhoR, mesh)
        rhoG *= 1.0 / mg[-1].Ng ** 0.5
        intp = get_reduced_freq_coords(mesh, mesh0)
        full_density[intp] += rhoG
        # --DENSITY ON G DONE--##

    vG = full_density * tools.get_coulG(cell, mesh=mesh0)
    ###--POT ON G DONE--##

    vj = numpy.zeros((nK, cell.nao, cell.nao), numpy.complex128)
    for n in range(nmg):
        ##pot on current grid in G
        mesh = mg[n].mesh
        intp = get_reduced_freq_coords(mesh, mesh0)
        pot_on_grid = vG[intp]

        ##pot on current grid in R
        pot_on_grid = tools.ifft(pot_on_grid, mesh)
        pot_on_grid *= mg[n].Ng ** 0.5
        vR = pot_on_grid * (mg[n].Ng / cell.vol / nK) ** 0.5

        for i in range(natm):
            ag = atomgrids[i + natm * n]
            if ag.nao == 0:
                continue
            sharp_idx, sharp_idx_on_grid = ag.ao_index_sharp, ag.ao_index_sharp_on_grid
            all_idx = ag.ao_index
            aovals = ag.aovals if getattr(ag, "aovals", None) is not None else eval_ao(myisdf, ag, mg[n])

            for ao_k, j_k in zip(aovals, vj):
                ao_vR = ao_k[sharp_idx_on_grid] * vR[ag.coord_idx]
                j = ao_k.conj() @ ao_vR.T
                j_k[all_idx[:, None], sharp_idx[None, :]] = j
                j_k[sharp_idx[:, None], all_idx[None, :]] = j.conj().T

    if mg[-1].universal:
        # Do sparse grid
        aoidx = atomgrids[-1].ao_index
        mesh = mg[-1].mesh
        intp = get_reduced_freq_coords(mesh, mesh0)
        pot_on_grid = vG[intp]

        ##pot on last grid in R
        pot_on_grid = tools.ifft(pot_on_grid, mesh)
        pot_on_grid *= mg[-1].Ng ** 0.5
        vR = pot_on_grid * (mg[-1].Ng / cell.vol / nK) ** 0.5

        for ao_k, j_k in zip(aovals_sg, vj):
            ao_vR = ao_k * vR
            j_k[numpy.ix_(aoidx, aoidx)] = ao_k.conj() @ ao_vR.T

        return vj
    

def integrals_uu_with_thc(myisdf, Kao, Koo, UX, aovals, mo_occ):
    ug = myisdf.atom_grids_k[-1]
    W = numpy.zeros((ug.nthc,ug.nthc), numpy.float64)
    W[numpy.triu_indices(ug.nthc)] += myisdf.W[-1][-1]
    W += numpy.triu(W,1).T
    UU = numpy.matmul(UX.T * mo_occ, UX)  # iRg, iRg' -> RgRg'
    UU *= W # RgRg', RgRg'-> RgRg'
    muX = UX @ UU  # jRg, RgRg'-> oRg'
    Kao[ug.ao_index] += numpy.matmul(aovals, muX.T)  # oRg', Rg'k-> ok


def integrals_uu(i, moIR, IR, coulG, mo_occ, mesh):
    fft_in, fft_out, fft, ifft = get_fft_plan(mesh)[:4]
    moIR_i = moIR[i]
    fft_in = fft_in.ravel()

    for j, moIR_j in enumerate(moIR):
        numpy.multiply(moIR_i, moIR_j, out=fft_in)
        fft()
        fft_out *= coulG
        ifft()
        IR[i] += fft_in.real * moIR_j * mo_occ[j]


def get_universal_grid_integrals(myisdf, Koo, Kao, i_Rg, mu_Rg, mo_occ):
    cell = myisdf.cell_unc
    ug = myisdf.atom_grids_k[-1]
    mg = myisdf.full_grids_k[-1]
    mesh = mg.mesh
    nmo = i_Rg.shape[0]
    f = ug.Ng / cell.vol

    t0 = time.time()
    # print("start SG", flush=True)

    # Parallelize over the outer loop (i) using joblib.
    # The inner loop is handled inside the function. Nec for memory cost.
    coulG = tools.get_coulG(cell, mesh=mesh).reshape(*mesh)[..., : mesh[2] // 2 + 1]
    vR_dm = numpy.zeros((nmo, ug.Ng), numpy.float64)

    with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
        Parallel()(delayed(integrals_uu)(i, i_Rg, vR_dm, coulG, mo_occ, mesh) for i in range(nmo))

    vR_dm *= f
    Kao[ug.ao_index] = mu_Rg @ vR_dm.T
    # print("end SG --> ", time.time() - t0, flush=True)


def integrals_ii(SX, DX, AX, Kao, Koo, ai, mo_coeff, mo_occ, Wi):
    nao = ai.nao
    if nao == 0:
        return
    sidx = ai.ao_index_sharp_on_grid
    didx = ai.ao_index_diffuse_on_grid
    aX = ai.aovals
    sX = aX[sidx]
    dX = aX[didx]
    nocc = mo_coeff.shape[0]
    nthc = ai.nthc
    numpy.matmul(mo_coeff[:, ai.ao_index_sharp], sX, out=SX)
    numpy.matmul(mo_coeff[:, ai.ao_index_diffuse], dX, out=DX)
    numpy.matmul(mo_coeff[:, ai.ao_index], aX, out=AX)

    ############## Atom[i] * Atom[i] ##############
    muX = numpy.empty((nocc, nthc), numpy.float64, order='C')
    W = numpy.zeros((nthc, nthc), numpy.float64, order='C')
    W[numpy.triu_indices(nthc)] += Wi
    W += numpy.triu(W, 1).T

    # DSSD
    SX_T = SX.T * mo_occ  # iRg
    SS = SX_T @ SX  # iRg, iRg' -> RgRg'
    SS *= W  # RgRg', RgRg'-> RgRg'
    numpy.matmul(DX, SS, out=muX)  # jRg, RgRg'-> oRg'

    # SASD (and DSSD)
    AX_T = AX.T * mo_occ  # iRg
    AS = AX_T @ SX  # iRg, iRg' -> RgRg'
    AS *= W  # RgRg', RgRg'-> RgRg'
    muX += SX @ AS  # jRg, RgRg'-> oRg'
    Kao[didx] += dX @ muX.T  # oRg', Rg'k-> ok

    # DSAS
    SA = AS.T
    numpy.matmul(DX, SA, out=muX)  # jRg, RgRg'-> oRg'

    # SAAS
    AA = AX_T @ AX  # iRg, iRg' -> RgRg'
    AA *= W  # RgRg', RgRg'-> RgRg'
    muX += SX @ AA  # jRg, RgRg'-> oRg'
    Kao[sidx] += sX @ muX.T  # oRg', Rg'k-> ok

def integrals_iu(SX, DX, AX, dRg, sRg, muX2, ai, mo_occ, UX, Wi):
    nao = ai.nao
    if nao == 0:
        return
    aX = ai.aovals
    sX = aX[ai.ao_index_sharp_on_grid]
    dX = aX[ai.ao_index_diffuse_on_grid]
    SX_T = SX.T * mo_occ # iRg
    AX_T = AX.T * mo_occ  # iRg

    ############## Atom[i] * Universal ##############
    ## SU
    RgU = SX_T @ UX  # iRg, iRg' -> RgRg'
    RgU *= Wi  # RgRg', RgRg'-> RgRg'
    # UUSd: ng_U x nao
    numpy.matmul(DX, RgU, out=muX2)  # jRg, RgRg'-> oRg'
    numpy.matmul( dX,  RgU, out=dRg)  # jRg, RgRg'-> oRg'

    ## AU
    numpy.matmul(AX_T, UX, out=RgU)  # iRg, iRg' -> RgRg'
    RgU *= Wi  # RgRg', RgRg'-> RgRg'
    # SAUu: nocc x ng_U
    muX2 += SX @ RgU  # jRg, RgRg'-> oRg'
    # UUAs, UUSd: ng_U x nao
    numpy.matmul( sX, RgU, out=sRg)  # jRg, RgRg'-> oRg'    

def get_universal_and_same_atom_integrals(myisdf, Koo, Kao, mo_coeff):
    atomgrids = myisdf.atom_grids_k
    mo_occ = mo_coeff.mo_occ
    ug = atomgrids[-1]
    uidx = ug.ao_index
    tt = time.time()
    if getattr(ug, "aovals", None) is not None:
        uX = ug.aovals
    else:
        uX = eval_ao(myisdf, ug, myisdf.full_grids_k[-1])
        if myisdf.fit_sparse_grid:
            uX = ug.aovals[:, ug.IP_idx]
    UX = numpy.matmul( mo_coeff[:, ug.ao_index], uX, order='C')
    if myisdf.direct_k_sparse:
        get_universal_grid_integrals(myisdf, Koo, Kao, UX, uX, mo_occ)
    else:
        integrals_uu_with_thc(myisdf, Kao, Koo, UX, uX, mo_occ)
    myisdf.Times_["Diffuse-Diffuse"] += time.time() - tt

    # Contract MOs and AOs
    mo_sharp, mo_diffuse, mo_all = [], [], []
    if myisdf.multigrid_on:
        nocc = UX.shape[0]
        nao = Kao.shape[0]
        t0 = time.time()
        mo_sharp = [numpy.empty((nocc, ag.nthc), numpy.float64, order='C') for ag in atomgrids[:-1]]
        mo_diffuse = [numpy.empty((nocc, ag.nthc), numpy.float64, order='C') for ag in atomgrids[:-1]]
        mo_all = [numpy.empty((nocc, ag.nthc), numpy.float64, order='C') for ag in atomgrids[:-1]]
        Kao_list = [numpy.zeros((ag.nao, nocc), numpy.float64, order='C') for ag in atomgrids]
        Koo_list = [numpy.zeros((nocc, nocc), numpy.float64, order='C') for ag in atomgrids]
        with threadpool_limits(limits=1):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                Parallel()(
                    delayed(integrals_ii)(
                        mo_sharp[i],
                        mo_diffuse[i],
                        mo_all[i],
                        Kao_list[i],
                        Koo_list[i],
                        ai,
                        mo_coeff,
                        mo_occ,
                        myisdf.W[i][0],
                    )
                    for i, ai in enumerate(atomgrids[:-1])
                )              
        for i, ai in enumerate(atomgrids[:-1]):
            if ai.nao != 0:
                Kao[ai.ao_index] += Kao_list[i]
        myisdf.Times_["Sharp-Sharp"] += time.time() - t0

        t0 = time.time()
        ung = ug.Ng
        current_mem = lib.current_memory()[0] / 1000
        avail_memory = myisdf.max_memory / 1000 - current_mem
        natm = myisdf.cell.natm
        max_diffuse = max(ag.n_diffuse for ag in atomgrids[:-1])
        max_sharp = max(ag.n_sharp for ag in atomgrids[:-1])
        max_nthc = max(ag.nthc for ag in atomgrids[:-1])

        def req_mem_iu(chunk_size):
            return (
                chunk_size
                * (
                    max_diffuse * ung  # dRg
                    + max_sharp * ung  # sRg
                    + nocc * ung  # muX2_list
                    + 5 * max_nthc * nocc  # mo_sharp, mo_diffuse, mo_all, iRg
                    + max_nthc * ung # (Rg|U)
                )  
                + nao * ung  # muX1
            ) * 8e-9
    
        nchunk = natm
        req_mem_parallel = req_mem_iu(nchunk)
        # print(int(numpy.ceil(natm / nchunk)), nchunk, req_mem_parallel)
        while req_mem_parallel > avail_memory and nchunk > 1:
            nchunk = max(nchunk // 2, 1)
            req_mem_parallel = req_mem_iu(nchunk)
            # print(int(numpy.ceil(natm / nchunk)), nchunk, req_mem_parallel)
        ntasks = int(numpy.ceil(natm / nchunk))
        njobs = min(nchunk, myisdf.joblib_njobs)
        blas_jobs = myisdf.joblib_njobs if njobs == 1 else myisdf.fftw_njobs
        print("Dense-Sparse RAM Use (GB): {0:10.2f} ".format(req_mem_parallel), flush=True)
        print(f"Dense-Sparse split into {ntasks} tasks using {njobs} joblib threads.", flush=True)
        muX1 = numpy.zeros((nao, ung), numpy.float64, order='C')
        muX2 = numpy.zeros((nocc, ung), numpy.float64, order='C')
        for tt in range(ntasks):
            dRg = [
                numpy.empty((ag.n_diffuse, ung), numpy.float64, order='C')
                for ag in atomgrids[tt * nchunk : min(nchunk * (tt + 1), natm)]
            ]
            sRg = [
                numpy.empty((ag.n_sharp, ung), numpy.float64, order='C')
                for ag in atomgrids[tt * nchunk : min(nchunk * (tt + 1), natm)]
            ]
            muX2_list = [
                numpy.zeros((nocc, ung), numpy.float64, order='C')
                for _ in atomgrids[tt * nchunk : min(nchunk * (tt + 1), natm)]
            ]

            with threadpool_limits(limits=blas_jobs):
                with parallel_backend("threading", n_jobs=njobs):
                    Parallel()(
                        delayed(integrals_iu)(
                            mo_sharp[i + tt * nchunk],
                            mo_diffuse[i + tt * nchunk],
                            mo_all[i + tt * nchunk],
                            dRg[i],
                            sRg[i],
                            muX2_list[i],
                            ai,
                            mo_occ,
                            UX,
                            myisdf.W[i + tt * nchunk][-1],
                        )
                        for i, ai in enumerate(atomgrids[tt * nchunk : min(nchunk * (tt + 1), natm)])
                    )
            muX2 += numpy.sum(muX2_list, axis=0)        
            for i, ai in enumerate(atomgrids[tt * nchunk : min(nchunk * (tt + 1), natm)]):
                if ai.nao != 0:
                    muX1[ai.ao_index_diffuse] += dRg[i]
                    muX1[ai.ao_index_sharp] += sRg[i]

        Kao += muX1 @ UX.T  # oRg', Rg'k-> ok
        Kao[uidx] += uX @ muX2.T  # oRg', Rg'k-> ok
        myisdf.Times_["Sharp-Diffuse"] += time.time() - t0
    return mo_sharp, mo_diffuse, mo_all


def integrals_ij( ai, i, atomgrids, mo_all, mo_sharp, mo_diffuse, w_array, nao, mo_occ):
    if ai.nao == 0:
        return None
    # Allocate local result array
    nocc = mo_all[0].shape[0]
    Kao = numpy.zeros((nao, nocc), numpy.float64, order='C')

    if getattr(ai, "aovals", None) is not None:
        aX = ai.aovals
    else:
        aX = ai.aovals[:, ai.IP_idx_on_grid]

    AX_T = mo_all[i].T * mo_occ
    SX_T = mo_sharp[i].T * mo_occ
    nthc = ai.nthc
    SXYY = numpy.zeros((nthc, nocc), numpy.float64, order='C')
    yYXS = numpy.zeros((nao, nthc), numpy.float64, order='C')
    DXYY = numpy.zeros((nthc, nocc), numpy.float64, order='C')
    yYXD = numpy.zeros((nao, nthc), numpy.float64, order='C')
    natmgrids = len(atomgrids) - 1

    for j in range(i + 1, natmgrids):
        aj = atomgrids[j]

        if aj.nao == 0:
            continue

        W = w_array[i][j - i]
        AY = mo_all[j]
        SY = mo_sharp[j]
        DY_T = mo_diffuse[j].T
        dY = aj.aovals[aj.ao_index_diffuse_on_grid]
        sY = aj.aovals[aj.ao_index_sharp_on_grid]
        sidx_l = aj.ao_index_sharp
        didx_l = aj.ao_index_diffuse

        # S x AS'D'
        AS = AX_T @ SY  # iRg, iRg' -> RgRg'
        AS *= W  # RgRg', RgRg'-> RgRg'
        SXYY += AS @ DY_T  # jRg, RgRg'-> oRg'

        # S x AS'd'
        yYXS[didx_l] += dY @ AS.T  # jRg, RgRg'-> oRg'

        # S x AA'S'
        AA = AX_T @ AY  # iRg, iRg' -> RgRg'
        AA *= W  # RgRg', RgRg'-> RgRg'
        SXYY += AA @ SY.T  # jRg, RgRg'-> oRg'

        # S x AA's'
        yYXS[sidx_l] += sY @ AA.T  # jRg, RgRg'-> oRg'

        # D x SS'D'
        SS = SX_T @ SY  # iRg, iRg' -> RgRg'
        SS *= W  # RgRg', RgRg'-> RgRg'
        DXYY += SS @ DY_T  # jRg, RgRg'-> oRg'

        # D x SS'd'
        yYXD[didx_l] += dY @ SS.T  # jRg, RgRg'-> oRg'

        # D x SA'S'
        SA = SX_T @ AY  # iRg, iRg' -> RgRg'
        SA *= W  # RgRg', RgRg'-> RgRg'
        DXYY += SA @ SY.T  # jRg, RgRg'-> oRg'

        # D x SA's'
        yYXD[sidx_l] += sY @ SA.T  # jRg, RgRg'-> oRg'

    sidx = ai.ao_index_sharp
    didx = ai.ao_index_diffuse
    sX = aX[ai.ao_index_sharp_on_grid]
    dX = aX[ai.ao_index_diffuse_on_grid]
    SX = mo_sharp[i]
    DX = mo_diffuse[i]

    Kao[sidx] += sX @ SXYY # oRg', Rg'k-> ok
    # S'A'AS, D'S'AS, SAA'S', SAS'D'
    # AS'd', AA's' -> SAS'd', SAA's'
    Kao +=  yYXS @ SX.T

    # SS'D', SA'S' -> D'S'Sd, S'A'Sd
    Kao[didx] += dX @ DXYY # oRg', Rg'k-> ok
    # SS'd', SA's' -> DSS'd', DSA's'
    Kao += yYXD @ DX.T
    return Kao


def get_k_occRI(myisdf, mo_coeff):
    """
    Compute exchange (K) matrix using occupied orbital resolution of identity (occRI).
    
    This function implements the tensor hypercontraction (THC) approach for exchange
    matrix evaluation using interpolative separable density fitting (ISDF). The 
    algorithm uses occupied molecular orbitals to reduce computational complexity
    from O(N^4) to approximately O(N^3) scaling.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing grids, fitting points, and computational parameters
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients (nao x nocc)
        Must have mo_occ attribute for occupation numbers
        
    Returns:
    --------
    numpy.ndarray
        Exchange matrix in AO basis (nao x nao)
        
    Algorithm:
    ----------
    1. Contract molecular orbitals with ISDF fitting functions
    2. Compute occupied-occupied and occupied-all interactions 
    3. Use multi-grid approach for different interaction ranges
    4. Build full exchange matrix using resolution of identity
    
    Notes:
    ------
    - Uses parallel execution when myisdf.joblib_njobs > 1
    - Supports both single-grid and multi-grid modes
    - Memory efficient through sparse grid representations
    - Exchange matrix is Hermitian: K_μν = K_νμ
    """
    cell = myisdf.cell_unc
    nocc = mo_coeff.shape[0]
    nao = cell.nao
    
    # Pre-allocate matrices (avoid redundant nocc assignment)
    Koo = numpy.zeros((nocc, nocc), numpy.float64)
    Kao = numpy.zeros((nao, nocc), numpy.float64)
    
    # Choose sequential or parallel execution with optimized functions
    # Optimizations include: C-contiguous arrays, reduced allocations, BLAS optimization
    mo_sharp, mo_diffuse, mo_all = get_universal_and_same_atom_integrals(
        myisdf, Koo, Kao, mo_coeff)

    
    if myisdf.multigrid_on:
        t0 = time.time()
        # Multi-grid contributions with optimized memory usage
        atomgrids = myisdf.atom_grids_k
        natmgrids = len(atomgrids) - 1
        mo_occ = mo_coeff.mo_occ
        
        # Use single accumulation matrix instead of list for better memory efficiency
        Kao_accum = numpy.zeros((nao, nocc), numpy.float64)
        w_array = myisdf.W
        
        with threadpool_limits(limits=myisdf.fftw_njobs):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                results = Parallel()(
                    delayed(integrals_ij)(
                        ai, i, atomgrids, mo_all, mo_sharp, mo_diffuse,
                        w_array, nao, mo_occ)
                    for i, ai in enumerate(atomgrids[:-2])
                )
        
        # Accumulate results more efficiently
        for result in results:
            if result is not None:
                Kao_accum += result
        
        Kao += Kao_accum
        myisdf.Times_["Sharp-Sharp"] += time.time() - t0
        
    # Pre-compute overlap matrix multiplication once
    Kuv = build_full_exchange(myisdf.S_unc, Kao, mo_coeff)
    return Kuv


def build_full_exchange(S, Kao, mo_coeff):
    """
    Construct full exchange matrix from occupied orbital components.
    
    This function builds the complete exchange matrix in the atomic orbital (AO)
    basis from the occupied-occupied (Koo) and occupied-all (Koa) components
    computed using the resolution of identity approximation.
    
    Parameters:
    -----------
    Sa : numpy.ndarray
        Overlap matrix times MO coefficients (nao x nocc)
    Kao : numpy.ndarray
        Occupied-all exchange matrix components (nao x nocc)
    Koo : numpy.ndarray
        Occupied-occupied exchange matrix components (nocc x nocc)
        
    Returns:
    --------
    numpy.ndarray
        Full exchange matrix in AO basis (nao x nao)
        
    Algorithm:
    ----------
    K_μν = Sa_μi * Koa_iν + Sa_νi * Koa_iμ - Sa_μi * Koo_ij * Sa_νj
    
    This corresponds to the resolution of identity expression:
    K_μν ≈ Σ_P C_μP W_PP' C_νP' where C are fitting coefficients
    """

    # First term: Sa @ Koa  
    Sa = S @ mo_coeff.T
    tmp = numpy.matmul(Sa, Kao.T, order='C')  # Ensure C-contiguous for better cache performance
    
    # Initialize with first term
    Kuv = tmp.copy()
    
    # Second term: add transpose in-place (leverages symmetry)
    Kuv += tmp.T
    
    # Third term: use temporary for the inner multiplication to avoid double allocation
    Koo = mo_coeff @ Kao
    Sa_Koo = numpy.matmul(Sa, Koo, order='C')
    Kuv -= numpy.matmul(Sa_Koo, Sa.T)
    return Kuv


def get_k_direct(myisdf, mo_coeff):
    """
    Compute exchange matrix using direct (non-ISDF) method on sparse grid.
    
    This function evaluates the exchange matrix using a direct approach
    without tensor hypercontraction or ISDF fitting. It uses FFT-based
    convolutions with the Coulomb kernel for each orbital pair.
    
    Parameters:
    -----------
    myisdf : ISDF
        ISDF object containing grids and computational parameters
    mo_coeff : numpy.ndarray
        Molecular orbital coefficients with mo_occ attribute
        
    Returns:
    --------
    numpy.ndarray
        Exchange matrix in AO basis (nao x nao)
        
    Algorithm:
    ----------
    For each orbital pair (i,j):
    1. Compute orbital product ψ_i(r) * φ_j(r) on grid
    2. FFT to momentum space 
    3. Multiply by Coulomb kernel 1/|k|^2
    4. Inverse FFT back to real space
    5. Contract with orbital i and occupation
    
    Notes:
    ------
    - Uses universal sparse grid only
    - Direct O(N^4) scaling algorithm  
    - More expensive than ISDF/THC methods but exact on grid
    """
    cell = myisdf.cell_unc
    ug = myisdf.atom_grids_k[-1]
    nG = ug.Ng
    nao = cell.nao
    f = ug.Ng / cell.vol
    mg = myisdf.full_grids_k[-1]
    mesh = mg.mesh
    nmo = mo_coeff.shape[0]
    if getattr(ug, "aovals", None) is not None:
        aovals = ug.aovals
    else:
        aovals = eval_ao(myisdf, ug, mg)

    # Transform MO coefficients to grid representation
    moorbs = numpy.matmul(mo_coeff, aovals).astype(numpy.float64)
    occ = mo_coeff.mo_occ
    coulG = tools.get_coulG(cell, mesh=mesh)  # Coulomb kernel in reciprocal space
    vR_dm = numpy.zeros((nao, nG), numpy.float64)
    
    # Double loop over AO (j) and occupied MO (i) indices
    for j in range(nao):
        for i in range(nmo):
            # Form orbital product ψ_i(r) * φ_j(r) on grid
            V = numpy.multiply(moorbs[i], aovals[j]).astype(numpy.complex128)
            
            # Transform to reciprocal space
            r2c_fftw_execute(V, mesh, False)
            
            # Apply Coulomb interaction: V(k) *= 1/|k|^2
            V *= coulG
            
            # Transform back to real space
            c2r_ifftw_execute(V, mesh, False)
            
            if vR_dm.dtype == numpy.float64:  # Type conversion for consistency
                V = V.real
                
            # Accumulate exchange contribution: K_jl = Σ_i occ_i * (jl|ii)
            vR_dm[j] += numpy.multiply(moorbs[i] * occ[i], V)

    vR_dm *= f  # Apply volume normalization
    Kuv = numpy.matmul(aovals, vR_dm.T)  # Transform back to AO basis

    return Kuv


def integrals_uu_kpts(Kao, Koo, ao_index, cell, mesh, kpts, coords, i_Rg, mu_Rg, k):
    nK = len(mu_Rg)
    nocc = i_Rg[k].shape[0]
    nG = mu_Rg[k].shape[1]
    f = 2.0 * nG / cell.vol * 1.0 / nK
    vR_dm = numpy.zeros((nocc, nG), numpy.complex128)
    for j in range(nocc):
        for k_prim in range(nK):
            nocc = i_Rg[k_prim].shape[0]
            coulG = tools.get_coulG(cell, kpts[k] - kpts[k_prim], False, mesh=mesh)
            expmikr = numpy.exp(-1j * (coords @ (kpts[k] - kpts[k_prim])))

            for i in range(nocc):
                V = numpy.multiply(i_Rg[k_prim][i].conj() * expmikr, i_Rg[k][j])  # ( __ , __ | k'i kj)
                c2c_fftw_execute(V, mesh, False)
                V *= coulG
                c2c_ifftw_execute(V, mesh, False)
                vR_dm[j] += numpy.multiply(
                    i_Rg[k_prim][i] * expmikr.conj(), V
                )  # ( __ , k'i | k'i kj) sum over k_prim & i

    vR_dm *= f
    Kao[k][ao_index] = mu_Rg[k].conj() @ vR_dm.T
    Koo[k][:] = i_Rg[k].conj() @ vR_dm.T


def get_universal_grid_integrals_kpts(myisdf, Koo, Kao, i_Rg, mu_Rg):
    cell = myisdf.cell_unc
    ug = myisdf.atom_grids_k[-1]
    ao_index = ug.ao_index
    kpts = myisdf.kpts
    nK = myisdf.Nk

    mg = myisdf.full_grids_k[-1]
    mesh = mg.mesh
    coords = mg.coords

    with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
        Parallel()(delayed(integrals_uu_kpts)(Kao, Koo, ao_index, cell, mesh, kpts, coords, i_Rg, mu_Rg, k) for k in range(nK))

def integrals_uu_with_thc_kpts(myisdf, Kao, Koo, UX, uX):
    nthc = myisdf.atom_grids_k[-1].nthc
    nK = myisdf.Nk

    U = numpy.empty((nK, nthc, nthc), numpy.complex128)
    for Uk, ix in zip(U, UX):
        numpy.matmul(ix.T, ix.conj(), Uk)

    convolve_with_W(U, myisdf.W[-1][-1], myisdf.kmesh)

    uidx = myisdf.atom_grids_k[-1].ao_index
    for k in range(nK):
        XUU = U[k] @ UX[k].T
        Kao[k][uidx] += uX[k].conj() @ XUU
        Koo[k] += UX[k].conj() @ XUU

def integrals_ii_kpts(ai, mo_coeff, Wi, kmesh):
    if ai.nao == 0:
        return [None] * 4
    nK = numpy.prod(kmesh)
    ############## Atom[i] * Atom[i] ##############
    nthc = Wi.shape[-1]
    aX = ai.aovals
    aidx = ai.ao_index
    sidx = ai.ao_index_sharp
    sidx_on_grid = ai.ao_index_sharp_on_grid
    didx = ai.ao_index_diffuse
    didx_on_grid = ai.ao_index_diffuse_on_grid

    AX = [mo_coeff[k][:, aidx] @ aX[k] for k in range(nK)]
    SX = [mo_coeff[k][:, sidx] @ aX[k][sidx_on_grid] for k in range(nK)]
    DX = [mo_coeff[k][:, didx] @ aX[k][didx_on_grid] for k in range(nK)]    

    SS = numpy.empty((nK, nthc, nthc), numpy.complex128)
    AA = numpy.empty((nK, nthc, nthc), numpy.complex128)
    SA = numpy.empty((nK, nthc, nthc), numpy.complex128)
    for k in range(nK):
        numpy.matmul(SX[k].T, SX[k].conj(), SS[k])
        numpy.matmul(AX[k].T, AX[k].conj(), AA[k])
        numpy.matmul(SX[k].T, AX[k].conj(), SA[k])

    convolve_with_W(SS, Wi, kmesh)
    convolve_with_W(AA, Wi, kmesh)
    convolve_with_W(SA, Wi, kmesh)

    Kao = [numpy.zeros((ai.nao,DX[k].shape[0]), numpy.complex128) for k in range(nK)]
    Koo = [numpy.zeros((DX[k].shape[0],DX[k].shape[0]), numpy.complex128) for k in range(nK)]
    for k in range(nK):
        # SAAS
        XXX = AA[k] @ SX[k].T

        # SASD
        XXX += SA[k].conj().T @ DX[k].T

        # SASD + SAAS
        Kao[k][sidx_on_grid] += aX[k][sidx_on_grid].conj() @ XXX
        Koo[k] += SX[k].conj() @ XXX

        # DSSD
        XXX[:] = SS[k] @ DX[k].T

        # DSAS
        XXX += SA[k] @ SX[k].T

        # DSAS + DSSD
        Kao[k][didx_on_grid] += aX[k][didx_on_grid].conj() @ XXX
        Koo[k] += DX[k].conj() @ XXX

    return AX, SX, DX, Kao, Koo

def integrals_iu_kpts(ai, Wi, kmesh, AX, SX, DX, UX):
    if ai.nao == 0:
        return 
    nK = numpy.prod(kmesh)

    ############## Atom[i] * Universal ##############
    nthc = AX[0].shape[-1]
    nthcu = UX[0].shape[-1]
    SU = numpy.empty((nK, nthc, nthcu), numpy.complex128)
    AU = numpy.empty((nK, nthc, nthcu), numpy.complex128)
    for k in range(nK):
        numpy.matmul(SX[k].T, UX[k].conj(), SU[k])
        numpy.matmul(AX[k].T, UX[k].conj(), AU[k])    

    convolve_with_W(SU, Wi, kmesh)
    convolve_with_W(AU, Wi, kmesh)  

    xXU = numpy.zeros((nK, ai.nao, nthcu), numpy.complex128)
    XXU = [numpy.zeros((UX[k].shape[0], nthcu), numpy.complex128) for k in range(nK)]
    sidx = ai.ao_index_sharp_on_grid
    didx = ai.ao_index_diffuse_on_grid
    aX = ai.aovals
    for k in range(nK):
        # Accumulate DSU for integrals uUSD and DSUU
        XXU[k] += DX[k].conj() @ SU[k]

        # Accumulate SAU for integrals uUAS and SAUU
        XXU[k] += SX[k].conj() @ AU[k]

        # Accumulate dSU for integrals dSUU
        xXU[k][didx] += aX[k][didx].conj() @ SU[k]

        # Accumulate sAU for integrals sAUU
        xXU[k][sidx] += aX[k][sidx].conj() @ AU[k]

    return XXU, xXU


def get_universal_and_same_atom_integrals_kpts(myisdf, Koo, Kao, mo_coeff):
    nK = myisdf.Nk
    kmesh = myisdf.kmesh
    natm = myisdf.cell.natm

    # A: all
    # S: sharp
    # D: diffuse
    # Lower case indicates AOs
    # Upper case indicates contracted MO/AOs

    #### Universal-Universal ####
    ug = myisdf.atom_grids_k[-1]
    uidx = ug.ao_index

    if getattr(ug, "aovals", None) is not None:
        uX = ug.aovals
    else:
        uX = eval_ao(myisdf, ug, myisdf.full_grids_k[-1])
        if myisdf.fit_sparse_grid:
            uX = [aovals_k[:, ug.IP_idx] for aovals_k in ug.aovals]
    UX = [mo_coeff[k][:, uidx] @ uX[k] for k in range(nK)]
    if myisdf.direct_k_sparse:
        get_universal_grid_integrals_kpts(myisdf, Koo, Kao, UX, uX)
    else:
        integrals_uu_with_thc_kpts(myisdf, Kao, Koo, UX, uX)
    ############################

    mo_sharp, mo_diffuse, mo_all = [], [], []
    if myisdf.multigrid_on:
        atomgrids = myisdf.atom_grids_k

        #### For the integral ( A B | C D), XXU and xXU accumulate ( B | C D) and ( A B | C).
        #### We sum over A (or D) first to ensure that the final contraction is over the
        #### universal grid. Doing this contraction within the loop over atoms is prohibitive.
        #### Here, X indicates in general A, S, and D.
        W = myisdf.W
        with threadpool_limits(limits=1):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                results = Parallel()(
                    delayed(integrals_ii_kpts)(
                        ai,
                        mo_coeff, 
                        W[i][0],
                        kmesh,
                    )
                    for i, ai in enumerate(atomgrids[:-1])
                )
        mo_all, mo_sharp, mo_diffuse = [], [], []
        for i, ai in enumerate(atomgrids[:-1]):
            mo_all.append(results[i][0])
            mo_sharp.append(results[i][1])
            mo_diffuse.append(results[i][2])
            if ai.nao != 0:
                for k in range(nK):
                    Kao[k][ai.ao_index] += results[i][3][k]
                    Koo[k] += results[i][4][k]

        with threadpool_limits(limits=1):
            with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
                results = Parallel()(
                    delayed(integrals_iu_kpts)(
                        ai,
                        W[i][-1],
                        kmesh,
                        mo_all[i],
                        mo_sharp[i],
                        mo_diffuse[i],
                        UX
                    )
                    for i, ai in enumerate(atomgrids[:-1])
                )

        for i in range(natm):
            XXU, xXU = results[i][0], results[i][1]
            aidx = atomgrids[i].ao_index
            for k in range(nK):
                Kao[k][uidx] += (XXU[k] @ uX[k].T).T.conj()  # xXUU
                Kao[k][aidx] += xXU[k] @ UX[k].T  # uUXX

                tmp = XXU[k] @ UX[k].T
                Koo[k] += tmp.T.conj()  # UUXX
                Koo[k] += tmp  # XXUU

    return mo_sharp, mo_diffuse, mo_all


def integrals_ij_kpts(ai, i, atomgrids, kmesh, AX, SX, DX, mo_coeff, w_array, nao):
    if ai.nao == 0:
        return None

    nthc_i = ai.nthc
    nK = len(mo_coeff)   
    Ai = AX[i]
    Si = SX[i]
    Di = DX[i]
    sXYi = [numpy.zeros((nthc_i, mo_k.shape[0]), numpy.complex128) for mo_k in mo_coeff] # (\mu_sharp|Rg)(Rg|Rg')(Rg'|i)
    dXYi = [numpy.zeros((nthc_i, mo_k.shape[0]), numpy.complex128) for mo_k in mo_coeff] # (\mu_diffuse|Rg)(Rg|Rg')(Rg'|i)
    SXYmu = numpy.zeros((nK, nthc_i, nao), numpy.complex128) # (i|Rg)(Rg|Rg')(Rg'|\mu_sharp)
    DXYmu = numpy.zeros((nK, nthc_i, nao), numpy.complex128) # (i|Rg)(Rg|Rg')(Rg'|\mu_diffuse)
    natmgrids = len(atomgrids) - 1
    
    # for j in range(natmgrids):
    for j in range(i + 1, natmgrids):
        aj = atomgrids[j]
        # if aj.nao == 0 or i<=j:
        if aj.nao == 0:
            continue
        
        # W = w_array[i][j]
        W = w_array[i][j-i]
        nthc_j = aj.nthc
        Aj = AX[j]
        Sj = SX[j]
        Dj = DX[j]        
        AS = numpy.empty((nK, nthc_i, nthc_j), numpy.complex128)
        SS = numpy.empty((nK, nthc_i, nthc_j), numpy.complex128)
        SA = numpy.empty((nK, nthc_i, nthc_j), numpy.complex128)
        AA = numpy.empty((nK, nthc_i, nthc_j), numpy.complex128)

        for k in range(nK):
            numpy.matmul(Si[k].T, Sj[k].conj(), SS[k])
            numpy.matmul(Ai[k].T, Sj[k].conj(), AS[k])
            numpy.matmul(Si[k].T, Aj[k].conj(), SA[k])
            numpy.matmul(Ai[k].T, Aj[k].conj(), AA[k])

        convolve_with_W(AS, W, kmesh)
        convolve_with_W(SS, W, kmesh)
        convolve_with_W(SA, W, kmesh)
        convolve_with_W(AA, W, kmesh)     

        sidx = aj.ao_index_sharp
        didx = aj.ao_index_diffuse
        sidx_on_grid = aj.ao_index_sharp_on_grid
        didx_on_grid = aj.ao_index_diffuse_on_grid
        aY = aj.aovals
        for k in range(nK):
            sY_T = aY[k][sidx_on_grid].T
            dY_T = aY[k][didx_on_grid].T
            SY_T = Sj[k].T
            DY_T = Dj[k].T

            # SAAS
            sXYi[k] += AA[k] @ SY_T
            SXYmu[k][:,sidx] += AA[k] @ sY_T

            # DSSD
            dXYi[k] += SS[k] @ DY_T
            DXYmu[k][:,didx] += SS[k] @ dY_T

            # SASD
            sXYi[k] += AS[k] @ DY_T
            SXYmu[k][:,didx] += AS[k] @ dY_T

            # DSAS
            dXYi[k] += SA[k] @ SY_T
            DXYmu[k][:,sidx] += SA[k] @ sY_T

    Kao = numpy.zeros((nK, nao, AX[0][0].shape[0]), numpy.complex128)
    Koo = numpy.zeros((nK, AX[0][0].shape[0], AX[0][0].shape[0]), numpy.complex128)
    sidx = ai.ao_index_sharp
    didx = ai.ao_index_diffuse
    sidx_on_grid = ai.ao_index_sharp_on_grid
    didx_on_grid = ai.ao_index_diffuse_on_grid
    aX = ai.aovals
    for k in range(nK):
        Kao[k] += (Si[k].conj() @ SXYmu[k]).conj().T  # SAAs, SASd
        Kao[k] += (Di[k].conj() @ DXYmu[k]).conj().T   # DSsd, DSAs
        
        Kao[k][didx] += aX[k][didx_on_grid].conj() @ dXYi[k] # dSSD, dSAS
        Kao[k][sidx] += aX[k][sidx_on_grid].conj() @ sXYi[k] # sAAS, sASD

        # tmp = Di[k].conj() @ dXYi[k] # DSSD, DSAS
        # Koo[k] += tmp
        # Koo[k] += tmp.conj().T
        # tmp = Si[k].conj() @ sXYi[k] # SAAS, SASD        
        # Koo[k] += tmp
        # Koo[k] += tmp.conj().T
    return Kao, Koo


def integrals_ij_kpts(ai, i, atomgrids, kmesh, AX, SX, DX, mo_coeff, w_array, nao):
    if ai.nao == 0:
        return None

    nthc_i = ai.nthc
    nK = len(mo_coeff)   
    Ai = AX[i]
    Si = SX[i]
    Di = DX[i]
    # sXYi = [numpy.zeros((nthc_i, mo_k.shape[0]), numpy.complex128) for mo_k in mo_coeff] # (\mu_sharp|Rg)(Rg|Rg')(Rg'|i)
    # dXYi = [numpy.zeros((nthc_i, mo_k.shape[0]), numpy.complex128) for mo_k in mo_coeff] # (\mu_diffuse|Rg)(Rg|Rg')(Rg'|i)
    # SXYmu = numpy.zeros((nK, nthc_i, nao), numpy.complex128) # (i|Rg)(Rg|Rg')(Rg'|\mu_sharp)
    # DXYmu = numpy.zeros((nK, nthc_i, nao), numpy.complex128) # (i|Rg)(Rg|Rg')(Rg'|\mu_diffuse)

    XXXD = [numpy.zeros((mo_k.shape[0], nthc_i), numpy.complex128) for mo_k in mo_coeff] 
    xXXD = numpy.zeros((nK, nao, nthc_i), numpy.complex128)
    XXXS = [numpy.zeros((mo_k.shape[0], nthc_i), numpy.complex128) for mo_k in mo_coeff] 
    xXXS = numpy.zeros((nK, nao, nthc_i), numpy.complex128)
    natmgrids = len(atomgrids) - 1
    
    for j in range(i + 1, natmgrids):
        aj = atomgrids[j]
        if aj.nao == 0:
            continue
        
        # W = w_array[i][j]
        W = w_array[j][j-i]
        nthc_j = aj.nthc
        Aj = AX[j]
        Sj = SX[j]
        Dj = DX[j]        
        AS = numpy.empty((nK, nthc_j, nthc_i), numpy.complex128)
        SS = numpy.empty((nK, nthc_j, nthc_i), numpy.complex128)
        SA = numpy.empty((nK, nthc_j, nthc_i), numpy.complex128)
        AA = numpy.empty((nK, nthc_j, nthc_i), numpy.complex128)

        for k in range(nK):
            numpy.matmul(Sj[k].T, Si[k].conj(), SS[k])
            numpy.matmul(Aj[k].T, Si[k].conj(), AS[k])
            numpy.matmul(Sj[k].T, Ai[k].conj(), SA[k])
            numpy.matmul(Aj[k].T, Ai[k].conj(), AA[k])

        convolve_with_W(AS, W, kmesh)
        convolve_with_W(SS, W, kmesh)
        convolve_with_W(SA, W, kmesh)
        convolve_with_W(AA, W, kmesh)     

        sidx = aj.ao_index_sharp
        didx = aj.ao_index_diffuse
        sidx_on_grid = aj.ao_index_sharp_on_grid
        didx_on_grid = aj.ao_index_diffuse_on_grid
        aY = aj.aovals
        for k in range(nK):
            sY_conj = aY[k][sidx_on_grid].conj()
            dY_conj = aY[k][didx_on_grid].conj()
            SY_conj = Sj[k].conj()
            DY_conj = Dj[k].conj()
            XXXD[k] += numpy.matmul(SY_conj, AS[k])  # S'A' S (D) x
            XXXD[k] += numpy.matmul(DY_conj, SS[k])  # D'S' S (D) x

            xXXD[k][sidx] += numpy.matmul(sY_conj, AS[k])  # s'A' S (D) x
            xXXD[k][didx] += numpy.matmul(dY_conj, SS[k])  # d'S' S (D) x

            XXXS[k] += numpy.matmul(DY_conj, SA[k])  # D'S' A (S) x
            XXXS[k] += numpy.matmul(SY_conj, AA[k])  # S'A' A (S) x

            xXXS[k][didx] += numpy.matmul(dY_conj, SA[k])  # d'S' A (S) x
            xXXS[k][sidx] += numpy.matmul(sY_conj, AA[k])  # s'A' A (S) x

    Kao = numpy.zeros((nK, nao, AX[0][0].shape[0]), numpy.complex128)
    Koo = numpy.zeros((nK, AX[0][0].shape[0], AX[0][0].shape[0]), numpy.complex128)
    sidx = ai.ao_index_sharp
    didx = ai.ao_index_diffuse
    sidx_on_grid = ai.ao_index_sharp_on_grid
    didx_on_grid = ai.ao_index_diffuse_on_grid
    aX = ai.aovals
    for k in range(nK):
        Kao[k][didx] += numpy.matmul(aX[k][didx_on_grid].conj(), XXXD[k].T.conj())  # dS A'S', dS S'D'
        Kao[k] += numpy.matmul(xXXD[k], Di[k].T)  # s'A'SD, d'S'SD
        Kao[k][sidx] += numpy.matmul(aX[k][sidx_on_grid].conj(), XXXS[k].T.conj())  # sAS'D', sAA'S'
        Kao[k] += numpy.matmul(xXXS[k], Si[k].T)  # d'S'AS, s'A'AS

        # tmp = numpy.matmul(XXXD[k], Di[k].T)  # S'A'SD, D'S'SD, DSS'D', DSA'S'
        # Koo[k] += tmp
        # Koo[k] += tmp.T.conj()
        # tmp = numpy.matmul(XXXS[k], Si[k].T)  # D'S'AS, S'A'AS, SAS'D', SAA'S'
        # Koo[k] += tmp
        # Koo[k] += tmp.T.conj()
    return Kao, Koo


def get_k_occRI_kpts(myisdf, mo_coeff):
    cell = myisdf.cell_unc
    nao = cell.nao
    nK = myisdf.Nk
    atomgrids = myisdf.atom_grids_k

    ############## Same-grid and Universal grid integrals ##############
    Koo = [numpy.zeros((mo_k.shape[0], mo_k.shape[0]), numpy.complex128) for mo_k in mo_coeff]
    Kao = [numpy.zeros((nao, mo_k.shape[0]), numpy.complex128) for mo_k in mo_coeff]
    mo_sharp, mo_diffuse, mo_all = get_universal_and_same_atom_integrals_kpts(
        myisdf, Koo, Kao, mo_coeff
    )

    ############## Atom[i] * Atom[j] ##############
    if myisdf.multigrid_on:
        w_array = myisdf.W
        kmesh = myisdf.kmesh
        # with parallel_backend("threading", n_jobs=myisdf.joblib_njobs):
        #     results = Parallel()(
        #         delayed(integrals_ij_kpts)(ai, i, atomgrids, kmesh, mo_all, mo_sharp, mo_diffuse, mo_coeff, w_array, nao)
        #             for i, ai in enumerate(atomgrids[:-1])
        #     )

        results = []
        for i, ai in enumerate(atomgrids[:-1]):
                results.append(integrals_ij_kpts(ai, i, atomgrids, kmesh, mo_all, mo_sharp, mo_diffuse, mo_coeff, w_array, nao))

        natm = cell.natm
        for i in range(natm):
            Kao_ij, Koo_ij = results[i][0], results[i][1]
            Kao += Kao_ij
            # Koo += Koo_ij     

    # Manzer et al. J. Chem. Phys. 143, 024113 (2015)
    Kuv = numpy.empty((nK, nao, nao), numpy.complex128)
    S = myisdf.S_unc
    Koo = [mo_coeff[k].conj() @ Kao[k] for k in range(nK)]
    for k in range(nK):
        CS_k = mo_coeff[k].conj() @ S[k]
        KCS_k = Kao[k] @ CS_k
        Kuv[k] = KCS_k
        Kuv[k] += KCS_k.T.conj()
        Kuv[k] -= CS_k.T.conj() @ (Koo[k] @ CS_k)

    return Kuv


def convolve_with_W(U, W, kmesh):
    n0, n1 = W.shape[-2], W.shape[-1]
    Nk = numpy.prod(kmesh)
    U_fft = scipy.fft.hfftn(
        U.reshape(*kmesh, n0, n1), s=(kmesh), axes=[0, 1, 2], overwrite_x=True
    ).astype(numpy.complex128)
    U_fft *= W
    U[:] = scipy.fft.ifftn(U_fft, axes=[0, 1, 2], overwrite_x=True).reshape(Nk, n0, n1)
    U_fft = None

    # # M_k = sum_k' W_{k'-k} * U_k' ###
    # Mk = numpy.zeros_like(U, numpy.complex128)
    # q_idx = pyscf.pbc.tools.k2gamma.double_translation_indices(kmesh) # k' - k
    # for k in range(Nk):
    #     for k_prim in range(Nk):
    #         q = q_idx[k,k_prim] # D[M,N] = D[N-M]
    #         Mk[k] += W[q] * U[k_prim]
    # U[:] = Mk


def to_uncontracted_basis(myisdf):
    cell = myisdf.cell
    unc_bs = {}
    for symb, bs in cell._basis.items():
        unc_bs[symb] = pyscf.gto.uncontract(bs)

    cell_unc = cell.copy(deep=True)
    cell_unc.basis = unc_bs
    cell_unc.build(dump_input=False)
    myisdf.cell_unc = cell_unc
    c = pyscf.gto.mole.decontract_basis(cell, aggregate=True)[1]

    # s_unc = cell_unc.intor_symmetric('int1e_ovlp')
    # s_unc2c = pyscf.gto.mole.intor_cross('int1e_ovlp', cell_unc, cell)
    # c = lib.cho_solve(s_unc, s_unc2c, strict_sym_pos=False)

    # evals, evecs = numpy.linalg.eigh(s_unc)
    # keep = evals > 1e-8
    # s_half_inv = evecs[:, keep]
    # c = s_half_inv @ s_half_inv.T @ c

    if myisdf.use_kpt_symm:
        myisdf.c = c.astype(numpy.complex128)
    else:
        myisdf.c = c.astype(numpy.float64)

def make_natural_orbitals(dms, S):
    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((dms.shape[0], dms.shape[1]) )
    for i, dm in enumerate(dms):
        #also see: make_natorbs
        # Diagonalize the DM in AO (using Eqn. (1) referenced above)
        A = lib.reduce(numpy.dot, (S, dm, S))
        w, v = scipy.linalg.eigh(A, b=S)

        # Flip NOONs (and NOs) since they're in increasing order
        mo_occ[i] = numpy.flip(w)
        mo_coeff[i] = numpy.flip(v, axis=1)
    return mo_coeff, mo_occ


class ISDFX(pyscf.pbc.df.fft.FFTDF):
    """
    Interpolative Separable Density Fitting with eXchange (ISDFX) class.
    
    This class implements a highly efficient method for computing Coulomb (J) and 
    exchange (K) matrices in periodic boundary condition systems using:
    - Multi-grid hierarchical approach for different length scales
    - Interpolative Separable Density Fitting (ISDF) for rank reduction
    - Fast Fourier Transforms (FFT) for efficient convolutions
    - Tensor hypercontraction (THC) for exchange matrix compression
    
    The method significantly reduces computational scaling compared to traditional
    approaches while maintaining chemical accuracy for periodic systems.
    
    Key Features:
    -------------
    - Atom-centered multi-grid construction for local accuracy
    - Universal sparse grid for long-range interactions
    - Parallel execution over atoms and k-points  
    - Memory-efficient algorithms for large systems
    - Support for k-point sampling in periodic calculations
    - Tunable precision parameters for accuracy vs. speed tradeoff
    
    Parameters:.
    -----------
    cell : pyscf.pbc.gto.Cell
        PySCF Cell object containing atomic and basis set information
    df2copy : ISDFX, optional
        Existing ISDFX instance to copy settings from
    **kwargs : dict, optional
        Configuration parameters (see default_kwargs for full list)
        
    Key Configuration Options:
    --------------------------
    alpha_cutoff : float, default=2.8
        Basis function exponent threshold for sparse grid assignment
    rcut_epsilon : float, default=1e-5
        Precision threshold for atom-centered grid radii
    ke_epsilon : float, default=1e-8
        Precision threshold for kinetic energy cutoff (grid density)
    isdf_thresh : float, default=1e-6
        ISDF decomposition accuracy threshold
    multigrid_on : bool, default=True
        Enable hierarchical multi-grid approach
    fit_dense_grid : bool, default=True
        Perform ISDF fitting on dense grids
    fit_sparse_grid : bool, default=False
        Perform ISDF fitting on sparse grid
    direct_k : bool, default=False
        Use direct exchange evaluation instead of ISDF/THC
    kmesh : list, default=[1,1,1]
        k-point mesh for Brillouin zone sampling
        
    Examples:
    ---------
    >>> from pyscf.pbc import gto
    >>> cell = gto.Cell()
    >>> # ... set up cell ...
    >>> mydf = ISDFX(cell, rcut_epsilon=1e-5, isdf_thresh=1e-6)
    >>> mydf.build()
    >>> j_matrix = mydf.get_j(density_matrix)
    >>> k_matrix = mydf.get_k(mo_coefficients)
    
    References:
    -----------
    - Interpolative Separable Density Fitting: doi.org/XXX
    - Multi-grid methods for periodic systems: doi.org/XXX
    """
    def __init__(
        self,
        cell=None,  # PySCF cell object. Pass in all cases.
        df2copy=None,  # Send an instance from a reference unit cell to make integer
        # ncopy translations of original grids for benchmarking or real isdf.
        # Kpts are turned off. Must pass ncopy.
        # Note: Even grids may cause issues with Exchange.
        **kwargs,
    ):
        # Default values for keyword arguments
        default_kwargs = {
            "alpha_cutoff": 2.8,  # Exponents < alpha_cut go on sparse grid.
            "rcut_epsilon": 1.0e-6,  # Determines atom-centered grid radii.
            "ke_epsilon": 1.0e-7,  # Determines sparse grid mesh.
            "isdf_thresh": 1.0e-6,  # Determines number of THC points selected. Larger value gives less.
            "multigrid_on": True,  # Use two grids when building the Exchange and up to 4 grids for Coulomb.
            "get_aos_from_pyscf": False,  # Use C-code or PySCF. Defaults to PySCF if C-code is not compiled.
            "incore": True,  # Use C-code or PySCF. Defaults to PySCF if C-code is not compiled.
            ########## ISDF options ##########
            "fit_dense_grid": True,  # If true do ISDF on dense grids.
            "fit_sparse_grid": False,  # If true, do ISDF on sparse grid.
            ##################################
            ### Useful for debugging/testing ###
            "direct_k_sparse": True,  # Whether to do THC on sparse grid or Direct exchange, regardless of fitting.
            "same_jk_grids": None,
            "coulomb_only": False,
            "get_j_from_pyscf": False,
            "joblib_par": True,
            ###################################
            ### K-point needs to be tested ###
            "use_kpt_symm": False,
            "kmesh": [1, 1, 1],  # K-point mesh
            "isdf_pts_from_gamma_point": True,  # Do ISDF only on the gamma point densities
            ##################################
            ##### Useful for benchmarking #####
            # Send unit cell and ncopy > [1,1,1] to make copies of all grids proportional to ncopy.
            # Adjusted for even ncopy.
            # You have to manually set mf.cell = mf.with_df.cell after initializing. Idk how to do this otherwise.
            "ncopy": [1, 1, 1],  # Supercell (Gamma-point) mesh
            "j_grid_mesh": None,
            "k_grid_mesh": None,
            ###################################
            ##### Useful for large systems #####
            # This is currently not enabled (commented out below)
            # But can be useful when the original PySCF code
            # OOM when calculating nuc_pp for HCore.
            "nuc_pp_version": 0,  # PySCF nuc_pp(): (0) original, (1) multigrid v1, (2) multigrid v2. Used (2) for MG Paper.
            "vppnl_ver": 0,  # v0 is original and v1 is modified
            ####################################
        }

        if df2copy and isinstance(df2copy, ISDFX):
            cell = df2copy.cell
        if cell is None:
            raise KeyError(f"Must pass cell!")

        # Validate provided kwargs
        invalid_keys = set(kwargs) - set(default_kwargs)
        if invalid_keys:
            raise KeyError(f"Unexpected keyword arguments: {', '.join(invalid_keys)}")

        # Start with the default values
        attributes = default_kwargs.copy()

        # If an existing instance is provided, update with its attributes
        if df2copy and isinstance(df2copy, ISDFX):
            for attr in default_kwargs:
                attributes[attr] = getattr(df2copy, attr)

        # Override with explicitly provided kwargs
        attributes.update(kwargs)

        # Set all attributes
        for key, value in attributes.items():
            setattr(self, key, value)

        # Print all attributes
        if not df2copy and numpy.prod(self.ncopy) == 1:
            print()
            print("******** <class 'ISDFX'> ********", flush=True)
            for key, value in vars(self).items():
                print(f"{key}: {value}", flush=True)

        # If ncopy, make copy of grids.
        if numpy.prod(self.ncopy) > 1:
            kwargs["ncopy"] = [1, 1, 1]
            self.cell_reference = cell.copy()
            refdf = ISDFX(cell, **kwargs)
            print("Reference cell:", flush=True)
            build_grids(refdf, True, True)
            self.k_grid_mesh = numpy.array(
                [
                    (kgrid.mesh * self.ncopy) + (kgrid.mesh * self.ncopy + 1) % 2
                    for kgrid in refdf.full_grids_k
                ],
            )
            self.j_grid_mesh = numpy.array(
                [
                    (jgrid.mesh * self.ncopy) + (jgrid.mesh * self.ncopy + 1) % 2
                    for jgrid in refdf.full_grids_j
                ],
            )
            cell = tools.pbc.super_cell(cell, self.ncopy, wrap_around=True)
            cell.build()

        print()
        self.StartTime = time.time()
        super().__init__(cell=cell)  # Need this for pyscf's eval_ao function
        self.exxdiv = "ewald"
        self.cell = cell

        self.Ns = numpy.prod(self.ncopy)
        self.Nk = numpy.prod(self.kmesh)
        self.kpts = self.cell.make_kpts(
            self.kmesh, space_group_symmetry=False, time_reversal_symmetry=False, wrap_around=True
        )

        if self.joblib_par:
            self.joblib_njobs = lib.numpy_helper._np_helper.get_omp_threads()
            self.fftw_njobs = 1
        else:
            self.joblib_njobs = 1
            self.fftw_njobs = lib.numpy_helper._np_helper.get_omp_threads()

        if self.Nk > 1:
            self.use_kpt_symm = True

        if not self.multigrid_on and (self.fit_dense_grid or self.fit_sparse_grid):
            self.fit_sparse_grid = True
            self.fit_dense_grid = True

        if self.fit_sparse_grid:
            self.direct_k_sparse = False

        self.stdout = self.cell.stdout
        self.verbose = self.cell.verbose
        self.max_memory = self.cell.max_memory

        self.nao_max = 0
        self.nthc_max = 0
        self.nthc = 0
        self.scf_iter = 0

        self.Times_ = {
            "Diagonalize": 0.0,
            "Exchange": 0.0,
            "Coulomb": 0.0,
            "ISDF-pts": 0.0,
            "ISDF-fft": 0.0,
            "ISDF-vec": 0.0,
            "Grids": 0.0,
            "Init": 0.0,
            "AOs": 0.0,
            "ISDF": 0.0,
            "Fock": 0.0,
            "Sharp-Sharp": 0.0,
            "Sharp-Diffuse": 0.0,
            "Diffuse-Diffuse": 0.0,
        }

        if self.use_kpt_symm:
            self.get_jk = self.get_jk_kpts
            self.get_j = get_j_kpts
            self.get_k = get_k_occRI_kpts
        else:
            self.get_k = get_k_occRI  # getSquareK
            self.get_j = get_j

        if not df2copy:
            print("Exchange function: {0:12s}".format(self.get_k.__name__), flush=True)
        print()

    """
    def get_pp(mydf, kpts=None):
        from pyscf.pbc.dft import multigrid
        cell = mydf.cell.copy()
        a = cell.lattice_vectors()
        if abs(a-numpy.diag(a.diagonal())).max() > 1e-12 and mydf.nuc_pp_version==2:
            mydf.nuc_pp_version = 1
            print("Non-ortho lattice! Changing nuc_pp to MG v1.", flush=True)
        if mydf.use_kpt_symm:
            mf = pyscf.pbc.dft.KRKS(cell, kpts)
        else:
            mf = pyscf.pbc.dft.RKS(cell)
        mf.xc = "LDA"
        if mydf.nuc_pp_version==0:
            print("Using original Nuc PP!", flush=True)
            if cell.pseudo:
                v_pp = mf.with_df.get_pp(kpts)
            else:
                v_pp = mf.with_df.get_nuc(kpts)
        elif mydf.nuc_pp_version==1:
            print("Using MG Nuc PP v1!", flush=True)
            mf.with_df = multigrid.multigrid.MultiGridFFTDF(cell)
            mf.with_df.vppnl_ver = mydf.vppnl_ver
            v_pp = mf.with_df.get_pp(kpts, max_memory=cell.max_memory)
        elif mydf.nuc_pp_version==2:
            print("Using MG Nuc PP v2!", flush=True)
            mf.with_df = multigrid.multigrid_pair.MultiGridFFTDF2(cell)
            mf.with_df.Ngrids = 4 # number of sets of grid points
            v_pp_loc2_nl = mf.with_df.get_pp()
            v_pp_loc1_G = mf.with_df.vpplocG_part1
            v_pp_loc1 = multigrid.multigrid_pair._get_j_pass2(mf.with_df, v_pp_loc1_G, kpts)[0]
            v_pp = v_pp_loc1 + v_pp_loc2_nl
        
        if mydf.Nk == 1: 
            return v_pp.astype(numpy.float64)
        else: 
            return v_pp.astype(numpy.complex128)
            """

    def get_jk(
        self,
        dm=None,
        hermi=1,
        kpt=None,
        kpts_band=None,
        with_j=None,
        with_k=None,
        omega=None,
        exxdiv=None,
        **kwargs,
    ):
        skip_k = False
        dm_shape = dm.shape
        if self.coulomb_only:
            skip_k = True
            with_k = False        

        # Build Grids
        build_k = getattr(self, "full_grids_k", None) is None and with_k
        build_j = getattr(self, "full_grids_j", None) is None and with_j and not self.get_j_from_pyscf
        if build_j or build_k:
            self.build(build_j, build_k, incore=self.incore)
        # if self.get_j_from_pyscf:
        #     log = logger.Logger(self.stdout, self.verbose)
        #     self.tasks = pyscf.pbc.dft.multigrid.multigrid.multi_grids_tasks(self.cell, self.cell.mesh, log)
        #     log.debug('Multigrid ntasks %s', len(self.tasks))

        nao = dm.shape[-1]
        if getattr(dm, "mo_coeff", None) is not None:
            mo_coeff = numpy.asarray(dm.mo_coeff)
            mo_occ = numpy.asarray(dm.mo_occ)
        else:
            mo_coeff, mo_occ = make_natural_orbitals(dm.reshape(-1, nao, nao), self.S)
            # raise ValueError("Must send MO Coefficients with DM!!!")
        dm = dm.reshape(-1, nao, nao)
        nsets = dm.shape[0]
        if mo_coeff.ndim != 3:
            mo_coeff = mo_coeff.reshape(-1, mo_coeff.shape[-2], mo_coeff.shape[-1])
        if mo_occ.ndim != 2:
            mo_occ = mo_occ.reshape(-1, mo_occ.shape[-1])    
        is_contracted_basis = self.cell.nao != self.cell_unc.nao
        c2unc = self.c
        
        if with_j:
            t0 = time.time()
            if self.get_j_from_pyscf:
                vj = pyscf.pbc.df.fft_jk.get_j(self, dm)
            else:
                if is_contracted_basis:
                    unc_dm = numpy.asarray([self.c @ dmii @ self.c.T for dmii in dm], numpy.float64)
                    vj = self.get_j(self, unc_dm)
                    vj = [c2unc.T @ vji @ c2unc for vji in vj]
                else:
                    vj = self.get_j(self, dm)
            t_jk = time.time() - t0
            self.Times_["Coulomb"] += t_jk

        vk = []
        if with_k:
            tol = 1.0e-6
            is_occ = mo_occ > tol
            # print("nocc:", sum(is_occ[0]), flush=True)
            mo_coeff = [coeff[:, is_occ[i]] for i, coeff in enumerate(mo_coeff)]
            if is_contracted_basis:
                mo_coeff = [c2unc @ coeff for coeff in mo_coeff]
            mo_coeff = [numpy.ascontiguousarray(coeff.T) for coeff in mo_coeff]
            mo_coeff = [
                lib.tag_array(coeff, mo_occ=mo_occ[i][is_occ[i]]) for i, coeff in enumerate(mo_coeff)
            ]

            t0 = time.time()
            for ii in range(nsets):
                Kuv = self.get_k(self, mo_coeff[ii])
                if is_contracted_basis:
                    Kuv = c2unc.T @ Kuv @ c2unc
                if exxdiv is not None:
                    Kuv += self.madelung * functools.reduce(numpy.matmul, (self.S, dm[ii], self.S))
                vk.append(Kuv)
            t_jk = time.time() - t0
            self.Times_["Exchange"] += t_jk

        if with_j:
            vj = numpy.asarray(vj, dtype=dm.dtype).reshape(dm_shape)
        else:
            vj = None

        if with_k:
            vk = numpy.asarray(vk, dtype=dm.dtype).reshape(dm_shape)
        else:
            vk = None

        if skip_k:
            vk = vj * 0.0

        self.scf_iter += 1
        return vj, vk

    def get_jk_kpts(
        self,
        dm=None,
        hermi=1,
        kpts=None,
        kpts_band=None,
        with_j=None,
        with_k=None,
        omega=None,
        exxdiv=None,
        **kwargs,
    ):  
        skip_k = False
        if self.coulomb_only:
            skip_k = True
            with_k = False

        nK = self.Nk
        if with_k:
            if getattr(dm, "mo_coeff", None) is not None:
                mo_coeff = numpy.asarray(dm.mo_coeff)
                mo_occ = numpy.asarray(dm.mo_occ)
                if mo_coeff.ndim == 3:
                    mo_coeff = mo_coeff.reshape(-1, nK, mo_coeff.shape[-2], mo_coeff.shape[-1])
                if mo_occ.ndim == 2:
                    mo_occ = mo_occ.reshape(-1, nK, mo_occ.shape[-1])
            else:
                raise ValueError("Must send MO Coefficients with DM!!!")

        nao = dm.shape[-1]
        dm = dm.reshape(-1, nK, nao, nao)
        nsets = dm.shape[0]

        # Build Grids
        build_k = getattr(self, "full_grids_k", None) is None and with_k
        build_j = getattr(self, "full_grids_j", None) is None and with_j and not self.get_j_from_pyscf
        if build_j or build_k:
            self.build(build_j, build_k, incore=self.incore)
        # if self.get_j_from_pyscf:
        #     log = logger.Logger(self.stdout, self.verbose)
        #     self.tasks = pyscf.pbc.dft.multigrid.multigrid.multi_grids_tasks(
        #         self.cell, self.cell.mesh, log
        #     )
        #     log.debug("Multigrid ntasks %s", len(self.tasks))

        is_contracted_basis = self.cell.nao != self.cell_unc.nao

        vj = [None] * nsets
        if with_j:
            t0 = time.time()
            if self.get_j_from_pyscf:
                vj = pyscf.pbc.df.fft_jk.get_j_kpts(self, dm, kpts=self.kpts)
            else:
                if is_contracted_basis:
                    unc_dm = [[self.c @ dm[i][k] @ self.c.T for k in range(nK)] for i in range(nsets)]
                    Juv = self.get_j(self, unc_dm[0])  ## <--- address for UHF
                    vj = numpy.asarray([self.c.T @ jii @ self.c for jii in Juv]).reshape(dm.shape)
                else:
                    vj = self.get_j(self, dm[0]).reshape(dm.shape)  ## <--- address for UHF

            t_jk = time.time() - t0
            self.Times_["Coulomb"] += t_jk

        vk = [None] * nsets
        if with_k:
            tol = 1.0e-6
            is_occ = mo_occ > tol
            # dm2 = numpy.asarray(
            #     [
            #         coeff_k[:, occ_k > 0] * occ_k[occ_k > 0] @ coeff_k[:, occ_k > 0].conj().T
            #         for coeff_k, occ_k in zip(mo_coeff[0], mo_occ[0])
            #     ]
            # )
            # print("nocc:", numpy.sum(is_occ, axis=2), flush=True)
            mo_coeff = [[mo_coeff[i][k][:, is_occ[i][k]] for k in range(nK)] for i in range(nsets)]
            if is_contracted_basis:
                mo_coeff = [[self.c @ mo_coeff[i][k] for k in range(nK)] for i in range(nsets)]
            mo_coeff = [[mo_coeff[i][k].T for k in range(nK)] for i in range(nsets)]
            mo_coeff = [
                [lib.tag_array(mo_coeff[i][k], mo_occ=mo_occ[i][k][is_occ[i][k]]) for k in range(nK)]
                for i in range(nsets)
            ]

            t0 = time.time()
            for ii in range(nsets):
                Kuv = self.get_k(self, mo_coeff[ii])
                if is_contracted_basis:
                    Kuv = [self.c.T @ Kuv[k] @ self.c for k in range(nK)]
                if exxdiv is not None:
                    for k in range(nK):
                        Kuv[k] += self.madelung * functools.reduce(numpy.matmul, (self.S[k], dm[ii][k], self.S[k]))
                vk[ii] = numpy.asarray(Kuv)
            t_jk = time.time() - t0
            self.Times_["Exchange"] += t_jk

        if with_k and nsets == 1:
            vk = vk[0]

        if with_j and nsets == 1 and not self.get_j_from_pyscf:
            vj = vj[0]

        if skip_k:
            vk = vj * 0.0

        self.scf_iter += 1
        return vj, vk

    def build(self, with_j=True, with_k=True, incore=False):
        t0 = time.time()

        to_uncontracted_basis(self)
        self.atoms = make_atoms(self.cell_unc, self.rcut_epsilon)
        self.madelung = tools.pbc.madelung(self.cell, self.kpts)
        if self.use_kpt_symm:
            self.S = self.cell.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts)
            self.S_unc = self.cell_unc.pbc_intor("int1e_ovlp", hermi=1, kpts=self.kpts)
        else:
            self.S = self.cell.pbc_intor("int1e_ovlp", hermi=1).astype(numpy.float64)
            self.S_unc = self.cell_unc.pbc_intor("int1e_ovlp", hermi=1).astype(numpy.float64)

        if self.get_j_from_pyscf:
            with_j = False
        build_grids(self, with_j, with_k)
        if self.use_kpt_symm:
            if with_k:
                for kgrid in self.full_grids_k:
                    register_fft_factory_kpts(kgrid.mesh, self.fftw_njobs)
            if with_j:
                for jgrid in self.full_grids_j:
                    register_fft_factory_kpts(jgrid.mesh, self.fftw_njobs)
        else:
            for kgrid in self.full_grids_k:
                register_fft_factory(kgrid.mesh, self.fftw_njobs)
        if with_j:
            if with_k:
                for kgrid in self.full_grids_k:
                    register_fft_factory(kgrid.mesh, self.fftw_njobs)
            if with_j:
                for jgrid in self.full_grids_j:
                    register_fft_factory(jgrid.mesh, self.fftw_njobs)
        self.Times_["Grids"] += time.time() - t0

        if incore and with_j:
            eval_all_ao(self, self.atom_grids_j, self.full_grids_j, return_aos=False)

        if with_k:
            # Get W and intialize ISDF exchange
            t0 = time.time()
            do_isdf_and_build_w(self)
            self.Times_["ISDF"] += time.time() - t0

    def __del__(self):
        return

    def copy(self):
        """Returns a shallow copy"""
        return self.view(self.__class__)

    def get_keyword_arguments(self):
        # Retrieve all attributes, excluding any non-keyword arguments
        return {key: value for key, value in self.__dict__.items() if key != "cell"}

    def print_times(self):
        print()
        print("Wall time: ", time.time() - self.StartTime, flush=True)
        print("  >Initialize      :{0:18.2f}".format(self.Times_["Init"]), flush=True)
        print("  >Grids           :{0:18.2f}".format(self.Times_["Grids"]), flush=True)
        print("  >AOs             :{0:18.2f}".format(self.Times_["AOs"]), flush=True)
        print("  >ISDF-pts        :{0:18.2f}".format(self.Times_["ISDF-pts"]), flush=True)
        print("  >ISDF-vec        :{0:18.2f}".format(self.Times_["ISDF-vec"]), flush=True)
        print("  >ISDF-fft        :{0:18.2f}".format(self.Times_["ISDF-fft"]), flush=True)
        print("  >Coulomb         :{0:18.2f}".format(self.Times_["Coulomb"]), flush=True)
        print("  >Exchange        :{0:18.2f}".format(self.Times_["Exchange"]), flush=True)
        print("  >Diffuse-Diffuse :{0:18.2f}".format(self.Times_["Diffuse-Diffuse"]), flush=True)
        print("  >Sharp-Diffuse   :{0:18.2f}".format(self.Times_["Sharp-Diffuse"]), flush=True)
        print("  >Sharp-Sharp     :{0:18.2f}".format(self.Times_["Sharp-Sharp"]), flush=True)
        print("  >Diagonalize     :{0:18.2f}".format(self.Times_["Diagonalize"]), flush=True)
        print()


def debug_eval_aos(cell):
    mf = pyscf.pbc.scf.RHF(cell)
    mf.with_df = ISDFX(
        cell,
        multigrid_on=False,
    )
    mf.with_df.build(with_j=True, with_k=False, incore=False)
    ag = mf.with_df.atom_grids_j[0]
    nG = ag.nx * ag.ny * ag.nz
    ao_valsSph = numpy.zeros((ag.nao, nG), numpy.float64, order="C")
    centers = cell.atom_coords()

    import struct

    def write_array(file_name, arr):
        arr = numpy.atleast_2d(arr)  # Convert to 2D if not already 2D
        print(f"Saving array with shape: {arr.shape}")

        # Write the shape (rows, cols)
        shape = arr.shape
        with open(file_name, "wb") as f:
            f.write(struct.pack("ii", shape[0], shape[1]))
            # Write the data itself
            f.write(arr.tobytes())

    write_array(f"ao_valsSph.bin", ao_valsSph)
    write_array(f"Rx.bin", ag.x0)
    write_array(f"Ry.bin", ag.y0)
    write_array(f"Rz.bin", ag.z0)
    write_array(f"alpha.bin", ag.exponents)
    write_array(f"A.bin", centers)
    write_array(f"norm.bin", ag.norm)
    write_array(f"Ls.bin", ag.Ls)
    write_array(f"l.bin", ag.l)
    write_array(f"atoms.bin", ag.atoms)
    write_array(f"nimages.bin", ag.nimages)
    write_array(f"images.bin", ag.images)

    exit()