import pyscf
from pyscf.mgpaw.test import utils
from pyscf.mgpaw.test.utils import get_ase_cell

if __name__ == "__main__":
    refcell = get_ase_cell('pyscf/mgpaw/test/hg1212-conv.vasp', basis= "gth-dzvp-molopt-sr", kecut = 190)
    method = pyscf.pbc.scf.UHF
    with_j = True
    with_k = True
    converged = True

    ncopy = [1,1,1]
    cell = pyscf.pbc.tools.pbc.super_cell(refcell, ncopy)
    cell.build()

    ############ Get dm0 ############
    afm_guess = {"alpha": ["3 Cu 3dx2-y2"], "beta": ["4 Cu 3dx2-y2"]}
    en_exact, dm0 = -736.143335688542, utils.get_dm0(cell, method, ncopy, "minao", afm_guess)

    # ############ FFTDF ############
    # en_exact, _ = utils.test_fftdf(cell, dm0, method, converged, with_k, with_j, en_exact)

    ############ GDF ############
    # en_exact  = utils.test_gdf(cell, dm0, method, converged,
    #                                         with_k, with_j, en_exact)

    # ############ MG ############
    # en_exact, _  = utils.test_pymg(cell, method, converged, 2,
    #                                         with_k, with_j, en_exact)

    # ############ occRI-FFTDF ############
    # en_exact, _ = utils.test_occRI_fftdf(cell, dm0, method, converged, with_k, with_j, en_exact)

    ########### MG-ISDF-occRI ############
    # ke_eps = 1.e-7
    # isdf_thresh = 1.0e-6
    # utils.test_mg_isdf_occRI(
    #     refcell,
    #     ncopy,
    #     dm0,
    #     method,
    #     converged,
    #     ke_eps,
    #     isdf_thresh,
    #     with_k=with_k,
    #     with_j=with_j,
    #     en_pyscf=en_exact,
    # )
