import numpy as np

from concurrent import futures
from wsdt import wsDtSegmentation


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
    return wsDtSegmentation(
            pmap,
            threshold,
            min_membrane_size,
            min_segment_size,
            sigma_seeds,
            sigma_weights,
            group_seeds,
            preserve_membrane,
            out_debug_dict)


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
    pass
