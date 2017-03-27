# !!! The order of these imports needs to be preserved !!!
import imglyb
from imglyb import util
from jnius import autoclass, cast
# !!!
import multiprocessing
import numpy as np
import vigra
import h5py

def apply_wsgray(img):

    # TODO we properly want grayscale types instead
    #RealARGBConverter = autoclass( 'net.imglib2.converter.RealARGBConverter')
    #Converters = autoclass( 'net.imglib2.converter.Converters' )
    #ARGBType = autoclass ( 'net.imglib2.type.numeric.ARGBType' )
    #RealType = autoclass ( 'net.imglib2.type.numeric.real.DoubleType' )
    DistanceTransform = autoclass( 'net.imglib2.algorithm.morphology.distance.DistanceTransform' )
    DISTANCE_TYPE = autoclass( 'net.imglib2.algorithm.morphology.distance.DistanceTransform$DISTANCE_TYPE' )
    Views = autoclass( 'net.imglib2.view.Views' )
    Executors = autoclass( 'java.util.concurrent.Executors' )
    #t = ARGBType()

    dt = np.zeros_like( img, dtype=img.dtype )
    cpu_count = multiprocessing.cpu_count()
    DistanceTransform.transform(
            Views.extendBorder( util.to_imglib( -img ) ), # -img or img ?!?
            util.to_imglib( dt ), DISTANCE_TYPE.EUCLIDIAN,
            Executors.newFixedThreadPool( cpu_count ), cpu_count,
            1e-4, 1e-4  )

    return dt



if __name__ == '__main__':
    pmap_p = '/home/consti/Work/data_neuro/CREMI/wsdt_test/cremi_sampleC_probs_cantorV1.h5'
    with h5py.File(pmap_p) as f:
        x = f['data'][1]
        print x.shape

    dt = apply_wsgray(x)
    vigra.writeHDF5(dt, './tmp/imlyb_dt.h5', 'data')
