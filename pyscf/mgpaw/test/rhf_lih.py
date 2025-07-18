import pyscf
from pyscf.mgpaw.test import utils
from pyscf.mgpaw.test.utils import get_ase_cell

if __name__ == "__main__":

    refcell = get_ase_cell('lih', kecut = 140)
    method = pyscf.pbc.scf.RHF
    with_j = True
    with_k = True
    converged = True

    ncopy = [1, 1, 1]
    cell = pyscf.pbc.tools.pbc.super_cell(refcell, ncopy)
    cell.build()

    ############ Get dm0 ############
    en_exact, dm0 = -31.9836612613785, utils.get_dm0(cell, method, ncopy)

    # ############ FFTDF ############
    # en_exact, _ = utils.test_fftdf(cell, dm0, method, converged, with_k, with_j, en_exact)

    ############ GDF ############
    # utils.test_gdf(cell, dm0, method, converged, with_k, with_j, en_exact)

    # ############ MG ############
    # en_exact, _  = utils.test_pymg(cell, method, converged, 2,
    #                                         with_k, with_j, en_exact)

    # ############ occRI-FFTDF ############
    # _, _ = utils.test_occRI_fftdf(cell, dm0, method, converged, with_k, with_j, en_exact)

    ########### MG-ISDF-occRI ############
    ke_eps = 1.e-7
    isdf_thresh = 1.0e-6
    utils.test_mg_isdf_occRI(
        refcell,
        ncopy,
        dm0,
        method,
        converged,
        ke_eps,
        isdf_thresh,
        with_k=with_k,
        with_j=with_j,
        en_pyscf=en_exact,
    )