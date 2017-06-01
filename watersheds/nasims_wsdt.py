import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import watershed
from skimage.segmentation import random_walker
from scipy.ndimage.morphology import distance_transform_edt


__doc__ = "Functions to help generate oversegmentation from probability maps."


def dt(image, normalize_by=0, invert=True, inverted_threshold=0.5, anisotropy=10):
    """Apply distance transform to `image`."""
    # Normalize image if required
    if normalize_by != 0:
        normalized_image = image / 255.
    else:
        normalized_image = image
    # Invert image if required
    if invert:
        inv_image = 1. - normalized_image
    else:
        inv_image = normalized_image
    # Threshold image if required
    if inverted_threshold is not None:
        threshed_inv_image = inv_image > inverted_threshold
    else:
        threshed_inv_image = inv_image
    # Apply dt (in 3D if required)
    if image.ndim == 2:
        dt_image = distance_transform_edt(threshed_inv_image)
    else:
        dt_image = distance_transform_edt(threshed_inv_image, sampling=(anisotropy, 1, 1))
    # Done
    return dt_image


def to_maximap(local_maxi, shape):
    """Convert a list of local maxima pixels to an image."""
    maximap = np.zeros(shape=shape)
    maximap[local_maxi[:, 0], local_maxi[:, 1]] = 1.
    return maximap


def seed_maker(dt_image, dt_gamma=1., dt_smoothing_sigma=0.5, min_seed_distance=5, bucket=None):
    """Make labeled seeds given distance transform."""
    bucket = {} if bucket is None else bucket
    # Smooth
    smoothed_dt_image = gaussian_filter(dt_image ** dt_gamma, dt_smoothing_sigma)
    # Find maxima
    local_maxima = peak_local_max(smoothed_dt_image, min_distance=min_seed_distance,
                                  exclude_border=False)
    # Convert to maximap
    maximap = to_maximap(local_maxima, dt_image.shape)
    # Get labeled seeds and return
    labeled_seeds = label(maximap, background=0)
    # Update bucket
    bucket.update({'smoothed_dt_image': smoothed_dt_image,
                   'labeled_seeds': labeled_seeds})
    return labeled_seeds


def segment(image, seeds, method='watershed', **kwargs):
    """Go from `image` to segmentation with `seeds`."""
    if method == 'watershed':
        segmentation = watershed(image, seeds, **kwargs)
    elif method == 'random-walker':
        segmentation = random_walker(image, seeds, **kwargs)
    else:
        raise NotImplementedError
    return segmentation


def map2overseg(proba_map, normalize=False, threshold=0.5,
                watershed_on='dt', proba_map_smoothing_sigma=5,
                seeding_dt_gamma=1., seeding_dt_smoothing_sigma=0.5, min_seed_distance=5,
                segmentation_method='watershed', bucket=None):
    # Validate
    assert watershed_on in ['dt', 'pmap']
    assert segmentation_method in ['watershed', 'random-walker']
    # Make bucket
    bucket = {} if bucket is None else bucket
    # Do the dt
    dt_map = dt(proba_map, normalize_by=(255. if normalize else 0), inverted_threshold=threshold)
    bucket.update({'dt_map': dt_map})
    # Make seeds
    seeds = seed_maker(dt_map,
                       dt_gamma=seeding_dt_gamma,
                       dt_smoothing_sigma=seeding_dt_smoothing_sigma,
                       min_seed_distance=min_seed_distance, bucket=bucket)
    # Get watershed target
    if watershed_on == 'dt':
        # Invert dt map
        watershed_target = -1. * dt_map
    elif watershed_on == 'pmap':
        # Smooth proba map
        watershed_target = gaussian_filter(proba_map, proba_map_smoothing_sigma)
    else:
        raise NotImplementedError
    bucket.update({'watershed_target': watershed_target})
    # Segment
    oversegmentation = segment(watershed_target, seeds, method=segmentation_method)
    # Done.
    return oversegmentation


def ws_nasim_default(pmap, threshold, smooth_dt, gamma_dt, smooth_pmap):
    return map2overseg(
        pmap,
        threshold=threshold,
        proba_map_smoothing_sigma=smooth_pmap,
        seeding_dt_gamma=gamma_dt,
        seeding_dt_smoothing_sigma=smooth_dt
    )
