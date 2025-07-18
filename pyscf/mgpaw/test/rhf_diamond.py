import pyscf
from pyscf.mgpaw.test import utils
from pyscf.mgpaw.test.utils import get_ase_cell

if __name__ == "__main__":
    refcell = get_ase_cell('c', kecut = 70)
    refcell.build()

    method = pyscf.pbc.scf.RHF
    with_j = True
    with_k = True
    converged = True

    ncopy = [1,1,1]
    cell = pyscf.pbc.tools.pbc.super_cell(refcell, ncopy)
    cell.incore_anyway = False
    cell.build()

    ############ Get dm0 ############
    en_exact, dm0 = -43.8202311997902, utils.get_dm0(cell, method, ncopy, "1e")
    # en_exact, dm0 = utils.get_dm0_from_chk("diamond-111.chk")

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
    ke_eps = 1.e-4
    isdf_thresh = 1.0e-5
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
