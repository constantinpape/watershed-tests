import vigra
import numpy as np
from functools import partial
from concurrent import futures

from wsdt import binary_seeds_from_distance_transform, group_seeds_by_distance, iterative_inplace_watershed

# TODO include some postprocessing from wsdt
def ws_grayscale_distance_transform(
        pmap,
        sigma_grayscale_dt,
        sigma_seeds,
        sigma_weights = 0.,
        min_segment_size = 0.,
        group_seeds = False,
        grow_on_pmap = True
        ):

    # apply the grayscale distance transform
    # TODO smooth the pmap before ?!
    # TODO properly processing similar to signed_distance_transform
    distance_to_membrane = vigra.filters.multiGrayscaleDilation(pmap, sigma_grayscale_dt)
    distance_to_membrane = 1. - distance_to_membrane

    binary_seeds = binary_seeds_from_distance_transform(distance_to_membrane, sigma_seeds, None)

    if group_seeds:
        labeled_seeds = group_seeds_by_distance( binary_seeds, distance_to_membrane)
    else:
        labeled_seeds = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view('uint8'))

    if grow_on_pmap:
        hmap = pmap
    else:
        hmap = distance_to_membrane
        # Invert the DT: Watershed code requires seeds to be at minimums, not maximums
        hmap[:] *= -1

    if sigma_weights != 0.:
        hmap = vigra.filters.gaussianSmoothing(hmap, sigma_weights)

    max_label = iterative_inplace_watershed(hmap, labeled_seeds, min_segment_size, None)
    return labeled_seeds - 1, max_label


# NOTE group seeds is not supported, because this does not lift the gil
def ws_grayscale_distance_transform_2d_stacked(
        pmap,
        sigma_grayscale_dt,
        sigma_seeds,
        sigma_weights     = 0.,
        min_segment_size  = 0,
        grow_on_pmap      = True,
        n_threads         = 1
        ):
    """
    """
    fragments = np.zeros_like(pmap, dtype = 'uint32')
    ws_function = partial(ws_grayscale_distance_transform,
            sigma_grayscale_dt = sigma_grayscale_dt,
            sigma_seeds   = sigma_seeds,
            sigma_weights = sigma_weights,
            min_segment_size = min_segment_size,
            group_seeds  = False,
            grow_on_pmap = grow_on_pmap
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
