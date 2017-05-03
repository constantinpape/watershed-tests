import h5py
import vigra
import numpy as np

from volumina_viewer import volumina_n_layer
from wsdt import grayWsDtSegmentation, signed_distance_transform, wsDtSegmentation


def test_all_vigra_fu(data):
    lambd = 1e-2
    threshold = .2

    dt = signed_distance_transform(data, threshold, 0, False, None)
    opening = vigra.filters.multiGrayscaleOpening(data, lambd)
    closing = vigra.filters.multiGrayscaleClosing(data, lambd)
    dilation = vigra.filters.multiGrayscaleDilation(data, lambd)
    erosion = vigra.filters.multiGrayscaleErosion(data, lambd)
    inv_dilation = 1. - vigra.filters.multiGrayscaleDilation(data, lambd)

    imlyb_dt = vigra.readHDF5('./tmp/imlyb_dt.h5', 'data')

    volumina_n_layer([data, dt, opening, closing, dilation, erosion, inv_dilation, imlyb_dt],
            ['data', 'dt', 'opening', 'closing', 'dilation', 'erosion', 'inv_dilation', 'imlyb'])


def test_vi_graydt(data):
    lambd = 0.015
    threshold = .2

    dt = signed_distance_transform(data, threshold, 0, False, None)
    dilation = vigra.filters.multiGrayscaleDilation(data, lambd)
    erosion = vigra.filters.multiGrayscaleErosion(data, lambd)
    dilation_inv = dilation.max() - dilation

    volumina_n_layer([data, dt, dilation, erosion, dilation_inv],
            ['data', 'dt', 'dilation', 'erosion', 'dilation_inv'])

def simple_test(data):
    lamda = 1e-2
    threshold = .2

    wsdt, _ = wsDtSegmentation(data, threshold, 0, 0, 1.6, 1., False)
    gray_wsdt, _ = grayWsDtSegmentation(data, lamda, 0, 1.6)

    volumina_n_layer([data, wsdt, gray_wsdt], ["data", "wsdt", "gray-wsdt"])


if __name__ == '__main__':
    pmap_p = '/home/consti/Work/data_neuro/CREMI/wsdt_test/cremi_sampleC_probs_cantorV1.h5'
    with h5py.File(pmap_p) as f:
        x = f['data'][1]
        print x.shape

    test_all_vigra_fu(x)
