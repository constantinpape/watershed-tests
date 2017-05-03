import numpy as np
import mahotas

# Jan Funke Style watershed

def _grid_seeds(
        pmap,
        seed_distance,
        start_id):

    height = pmap.shape[0]
    width  = pmap.shape[1]

    seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
    num_seeds_y = seed_positions[0].size
    num_seeds_x = seed_positions[1].size
    num_seeds = num_seeds_x*num_seeds_y

    seeds = np.zeros_like(pmap).astype('int32')
    seeds[seed_positions] = np.arange(start_id, start_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    return seeds, num_seeds


def _minima_seeds(
        pmap,
        start_id):

    minima = mahotas.regmin(pmap)
    seeds, num_seeds = mahotas.label(minima)
    seeds += start_id
    #seeds[seeds==start_id] = 0 # TODO I don't get this
    return seeds, num_seeds


def _distance_transform_seeds(
        pmap,
        threshold,
        start_id):

    distance = mahotas.distance(pmap<threshold)
    maxima = mahotas.regmax(distance)
    seeds, num_seeds = mahotas.label(maxima)
    seeds += start_id
    #seeds[seeds==start_id] = 0 # TODO I don't get this
    return seeds, num_seeds


def get_seeds(
        pmap,
        method,
        seed_distance,
        threshold,
        start_id
        ):
    assert method in ('grid', 'minima', 'maxima_distance')
    if method == 'grid':
        return _grid_seeds(pmap, seed_distance, start_id)
    elif method == 'minima':
        return _minima_seeds(pmap, start_id)
    elif method == 'maxima_distance':
        return _distance_transform_seeds(pmap, threshold, start_id)


def ws_funkey(
        pmap,
        seed_method = 'grid',
        seed_distance = 10,
        threshold = .5,
        use_affinities   = False
        ):
    """
    """
    if use_affinities:
        assert pmap.ndim == 4
        aff_dim = np.argmin(pmap.shape)[0]
        assert aff_dim in (0,3)
        # TODO is this ok ?
        pmap = pmap.copy()
        if aff_dim == 0:
            pmap = 1. - .5 * (pmap[1] + pmap[2]) # this assumes that the affinities are in the front channel
        else:
            pmap = 1. - .5 * (pmap[...,1] + pmap[...,2]) # this assumes that the affinities are in the front channel
    else:
        assert pmap.ndim == 3
        fragments = np.zeros_like(pmap, dtype = 'uint32')

    start_id = 0
    for z in xrange(depth):
        seeds, n_seeds = get_seeds(pmap[z], seed_method, seed_distance, threshold, start_id)
        fragments[z] = mahotas.cwatershed(pmap, seeds)
        start_id += n_seeds

    return fragments, n_seeds
