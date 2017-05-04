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
        grow_on_pmap      = True,
        out_debug_dict    = None
        ):
    """
    Watershed on distance transform on 2d or 3d probabiity map.

    @params:
    pmap: probability map, 2d or 3d numpy.ndarray of type float32.
    threshold: threshold for pixels that are considered in distance transform.
    sigma_seeds: smoothing factor for distance transform used for finding seeds.
    sigma_weights: smoothing factor for heiht map used for the watershed (default 0.).
    min_membrane_size: size filter for connected membrane components after thresholding (default 0 -> no size filtering).
    min_segment_size: size filter for resulting segments (default 0 -> no size filtering).
    group_seeds: use heuristics to group adjacent seeds (default: False).
    preserve_membrane: preserve membrane seeds (default: False).
    grow_on_pmap: grow on the probability map instead of distance transform (default: True).
    out_debug_dict: dictionary to store debug images as chunked arrays (default: None).
    @returns:
    fragments: numpy.ndarray of type uint32
    n_labels:  number of labels
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
            grow_on_pmap,
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
        grow_on_pmap      = True,
        n_threads         = 1
        ):
    """
    Apply 2d distance transform watershed stacked on 3d input parallel over the z slices.

    @params:
    pmap: probability map, 3d numpy.ndarray of type float32.
    threshold: threshold for pixels that are considered in distance transform.
    sigma_seeds: smoothing factor for distance transform used for finding seeds.
    sigma_weights: smoothing factor for heiht map used for the watershed (default 0.).
    min_membrane_size: size filter for connected membrane components after thresholding (default 0 -> no size filtering).
    min_segment_size: size filter for resulting segments (default 0 -> no size filtering).
    preserve_membrane: preserve membrane seeds (default: False).
    grow_on_pmap: grow on the probability map instead of distance transform (default: True).
    n_threads: number of threads (default: 1).
    @returns:
    fragments: numpy.ndarray of type uint32
    n_labels:  number of labels
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
            grow_on_pmap   = grow_on_pmap,
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
