import vigra
import numpy as np
from wsdt import binary_seeds_from_distance_transform, group_seeds_by_distance, iterative_inplace_watershed

# TODO include some postprocessing from wsdt
def ws_grayscale_distance_transform(
        pmap,
        sigma_grayscale_dt,
        sigma_minima,
        sigma_weights = 0.,
        min_segment_size = 0.,
        group_seeds = False,
        out_debug_image_dict = None
        ):

    # apply the grayscale distance transform
    # TODO smooth the pmap before ?!
    # TODO properly processing similar to signed_distance_transform
    distance_to_membrane = vigra.filters.multiGrayscaleDilation(pmap, sigma_grayscale_dt)
    distance_to_membrane = 1. - distance_to_membrane

    binary_seeds = binary_seeds_from_distance_transform(distance_to_membrane, sigmaMinima, out_debug_image_dict)

    if groupSeeds:
        labeled_seeds = group_seeds_by_distance( binary_seeds, distance_to_membrane)
    else:
        labeled_seeds = vigra.analysis.labelMultiArrayWithBackground(binary_seeds.view('uint8'))

    if sigma_weights != 0.:
        vigra.filters.gaussianSmoothing(distance_to_membrane, sigma_weights, out=distance_to_membrane)
        save_debug_image('smoothed DT for watershed', distance_to_membrane, out_debug_image_dict)

    # Invert the DT: Watershed code requires seeds to be at minimums, not maximums
    distance_to_membrane[:] *= -1
    max_label = iterative_inplace_watershed(distance_to_membrane, labeled_seeds, min_segment_size, out_debug_image_dict)

    return labeled_seeds, max_label


def ws_distance_grayscale_transform_2d_stacked(
        pmap,
        sigma_grayscale_dt,
        sigma_seeds,
        sigma_weights     = 0.,
        min_segment_size  = 0,
        group_seeds       = False,
        n_threads         = 1
        ):
    """
    """
    pass
