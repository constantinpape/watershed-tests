import h5py
import vigra
import numpy as np

from volumina_viewer import volumina_n_layer

import sys
sys.path.append('..')
from watersheds import *

# download test data from
#https://drive.google.com/open?id=0B4_sYa95eLJ1ek8yMWozTzhBbGM

def check_consecutive(segmentation):
    prev_max = -1
    for z in xrange(segmentation.shape[0]):
        min_z, max_z = segmentation[z].min(), segmentation[z].max()
        assert prev_max + 1 == min_z, "%i, %i" % (prev_max + 1, min_z)
        prev_max = max_z
    print "Passed"


# TODO install mahotas and run funkey

def test_aniso(view = False):
    pmap = vigra.readHDF5('./test_data/anisotropic/pmap.h5', 'data')

    ws_aniso_dt, n_labels_aniso = ws_anisotropic_distance_transform(pmap, 0.4, 10., 2.)
    check_consecutive(ws_aniso_dt)
    assert n_labels_aniso == ws_aniso_dt.max() + 1
    print "Anisotropic distance transform watershed done"

    res_dt   = []
    res_gray = []
    for n_threads in (1,4):
        ws_dt, n_labels_dt = ws_distance_transform_2d_stacked(
                pmap,
                0.4,
                2.,
                n_threads = n_threads)
        check_consecutive(ws_dt)
        assert n_labels_dt == ws_dt.max() + 1, "%i, %i" % (n_labels_dt, ws_dt.max() + 1)
        res_dt.append(n_labels_dt)
        print "Distance transform watershed done"

        ws_gray, n_labels_gray = ws_grayscale_distance_transform_2d_stacked(
                pmap,
                0.1,
                2.,
                n_threads = n_threads)
        check_consecutive(ws_gray)
        assert n_labels_gray == ws_gray.max() + 1
        res_gray.append(n_labels_gray)
        print "Grayscale distance transform watershed done"

    assert res_dt[0] == res_dt[1]
    assert res_gray[0] == res_gray[1]

    if view:
        raw = vigra.readHDF5('./test_data/anisotropic/raw.h5', 'data')
        volumina_n_layer(
                [raw, pmap, ws_aniso_dt, ws_dt, ws_gray],
                ['raw', 'pmap', 'ws_aniso_dt', 'ws_dt', 'ws_gray']
                )


def test_iso(view = False):
    pmap = vigra.readHDF5('./test_data/isotropic/pmap.h5', 'data')

    ws_dt, n_labels_dt = ws_distance_transform(pmap, 0.4, 2.)
    assert ws_dt.min() == 0
    assert ws_dt.max() + 1 == len(np.unique(ws_dt))
    assert ws_dt.max() + 1 == n_labels_dt
    print "Wsdt done"

    ws_gray, n_labels_gray = ws_grayscale_distance_transform(pmap, 0.1, 2.)
    assert ws_gray.min() == 0
    assert ws_gray.max() + 1 == len(np.unique(ws_gray))
    assert ws_gray.max() + 1 == n_labels_gray
    print "Ws gray done"

    if view:
        raw = vigra.readHDF5('./test_data/isotropic/raw.h5', 'data')
        volumina_n_layer(
                [raw, pmap, ws_dt, ws_gray],
                ['raw', 'pmap', 'ws_dt', 'ws_gray']
                )


if __name__ == '__main__':
    #test_iso()
    test_aniso()
