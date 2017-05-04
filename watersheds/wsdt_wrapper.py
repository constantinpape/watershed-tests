import numpy as np

from concurrent import futures
from wsdt import wsDtSegmentation
from functools import partial


def ws_distance_transform(
        pmap,
        threshold,
        sigma_seeds,
        sigma_weights     = 0.,
        min_membrane_size = 0,
        min_segment_size  = 0,
        group_seeds       = False,
        preserve_membrane = True,
        out_debug_dict    = None
        ):
    """
    """
    fragments, max_label = wsDtSegmentation(
            pmap,
            threshold,
            min_membrane_size,
            min_segment_size,
            sigma_seeds,
            sigma_weights,
            group_seeds,
            preserve_membrane,
            out_debug_dict)
    return fragments - 1, max_label


# NOTE group seeds is not supported, because this does not lift the gil
def ws_distance_transform_2d_stacked(
        pmap,
        threshold,
        sigma_seeds,
        sigma_weights     = 0.,
        min_membrane_size = 0,
        min_segment_size  = 0,
        preserve_membrane = True,
        n_threads         = 1
        ):
    """
    """

    fragments = np.zeros_like(pmap, dtype = 'uint32')
    ws_function = partial(ws_distance_transform,
            threshold     = threshold,
            sigma_seeds   = sigma_seeds,
            sigma_weights = sigma_weights,
            min_membrane_size = min_membrane_size,
            min_segment_size = min_segment_size,
            group_seeds   = False,
            preserve_membrane = preserve_membrane,
            out_debug_dict = None
        )

    # multi-threaded calculation
    if n_threads > 1:

        def segment_slice(z):
            frags_z, n_labels_z = ws_function(pmap[z])
            fragments[z] = frags_z
            return n_labels_z

        # process in parallel
        with futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
            tasks   = [executor.submit(segment_slice, z) for z in xrange(pmap.shape[0])]
            offsets = np.array([t.result() for t in tasks], dtype = 'uint32')

        # total number of labels offset
        offset = np.sum(offsets)

        # add offsets to the slices
        offsets = np.roll(offsets, 1)
        offsets[0] = 0
        offsets = np.cumsum(offsets)
        fragments += offsets[:,None,None]

    # single threaded calculation
    else:

        offset = 0
        for z in xrange(pmap.shape[0]):
            frags_z, n_labels_z = ws_function(pmap[z])
            fragments[z] = frags_z + offset
            offset += n_labels_z

    return fragments, offset
