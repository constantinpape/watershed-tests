import vigra
import numpy as np
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

    binary_seeds = binary_seeds_from_distance_transform(distance_to_membrane, sigmaMinima, None)

    if groupSeeds:
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
    return labeled_seeds, max_label


# NOTE group seeds is not supported, because this does not lift the gil
def ws_distance_grayscale_transform_2d_stacked(
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
    pass
