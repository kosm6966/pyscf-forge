import pyscf
from pyscf.mgpaw.test import utils_kpts as utils
from pyscf.mgpaw.test.utils import get_ase_cell

if __name__ == "__main__":

    cell = get_ase_cell('lih', kecut = 140)
    method = pyscf.pbc.scf.KRHF
    with_j = True
    with_k = True
    converged = True
    kmesh = [1,2,2]

    ############ Get dm0 ############
    en_exact, dm0 = -31.9787353043191, utils.get_dm0(cell, method, kmesh, "1e")
    en_exact = -32.0473387593548
    # en_exact, dm0 = utils.get_dm0_from_chk("diamond-111.chk")

    # ############ FFTDF ############
    ### Does not converge with gth-cc-dzvp
    # en_exact, _ = utils.test_fftdf(cell, dm0, method, converged, kmesh, with_k, with_j, en_exact)

    ############ GDF ############
    # en_exact, dm0 = utils.test_gdf(cell, dm0, method, converged, kmesh, with_k, with_j, en_exact)

    # ############ MG ############
    # en_exact, _  = utils.test_pymg(cell, method, converged, 2,
    #                                         with_k, with_j, en_exact)

    # ############ occRI-FFTDF ############
    # _, _ = utils.test_occRI_fftdf(cell, dm0, method, converged, kmesh, with_k, with_j, en_exact)

    ########### MG-ISDF-occRI ############
    ke_eps = 1.0e-7
    isdf_thresh = 1.0e-5
    utils.test_mg_isdf_occRI(
        cell,
        kmesh,
        dm0,
        method,
        converged,
        ke_eps,
        isdf_thresh,
        with_k=with_k,
        with_j=with_j,
        en_pyscf=en_exact,
    )
