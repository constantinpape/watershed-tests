# !!! The order of these imports needs to be preserved !!!
import imglyb
from imglyb import util
from jnius import autoclass, cast
# !!!
import multiprocessing
import numpy as np
import vigra
import h5py

def show_bdv(img):

    # converters from numpy real arrays to bdv
    # FIXME: I don't really get all this converter buisness...
    RealARGBConverter = autoclass( 'net.imglib2.converter.DoubleConverter')
    Converters = autoclass( 'net.imglib2.converter.Converters' )
    RealType = autoclass ( 'net.imglib2.type.numeric.real.DoubleType' )
    #ARGBType = autoclass ( 'net.imglib2.type.numeric.ARGBType' ) # FIXME uncommenting this line causes an error in the distance transform
    DistanceTransform = autoclass( 'net.imglib2.algorithm.morphology.distance.DistanceTransform' )
    DISTANCE_TYPE = autoclass( 'net.imglib2.algorithm.morphology.distance.DistanceTransform$DISTANCE_TYPE' )

    Views = autoclass( 'net.imglib2.view.Views' )
    Executors = autoclass( 'java.util.concurrent.Executors' )
    t = RealType()

    print img.shape
    print type(img)

    # dark magic bit shifts for multi-channel RGB
    #argb = (
    #    np.left_shift(img[...,0], np.zeros(img.shape[:-1],dtype=np.uint32) + 16) + \
    #    np.left_shift(img[...,1], np.zeros(img.shape[:-1],dtype=np.uint32) + 8)  + \
    #    np.left_shift(img[...,2], np.zeros(img.shape[:-1],dtype=np.uint32) + 0) ) \
    #    .astype( np.int32 )

    dt = np.zeros_like( img, dtype=img.dtype )
    cpu_count = multiprocessing.cpu_count()
    DistanceTransform.transform(
            Views.extendBorder( util.to_imglib( -img ) ),
            util.to_imglib( dt ), DISTANCE_TYPE.EUCLIDIAN,
            Executors.newFixedThreadPool( cpu_count ), cpu_count,
            1e-2, 1e-2  )
    print "Distance Transform done"

    # convert to float # FIXME this should not be necessary with a different converter, so we don't need to do this memcpy
    img = img.astype('float64')
    dt = dt.astype('float64')

    img_conv = RealARGBConverter( img.min(), img.max() )
    dt_conv  = RealARGBConverter( dt.min(), dt.max() )

    #bdv = util.BdvFunctions.show( util.to_imglib_argb( argb ), "argb", util.options2D().frameTitle( "Distance Transform" ) ) # show rgb image
    # FIXME this fails, because the converter and the type do not match
    bdv = util.BdvFunctions.show(
            Converters.convert(
                cast( 'net.imglib2.RandomAccessibleInterval', util.to_imglib( img ) ), img_conv, t ),
            "probabilities",
            util.options2D().frameTitle( "Distance Transform" ) ) # Get rid of the 'util.options2d()...' for 3d mode

    util.BdvFunctions.show(
            Converters.convert(
                cast( 'net.imglib2.RandomAccessibleInterval', util.to_imglib( dt ) ), dt_conv, t ),
            "Distance Transform (ImgLib2)",
            util.BdvOptions.addTo( bdv ) )

    # Show only one source at a time.
    DisplayMode = autoclass( 'bdv.viewer.DisplayMode' )
    vp = bdv.getBdvHandle().getViewerPanel()
    grouping = vp.getVisibilityAndGrouping()
    grouping.setDisplayMode( DisplayMode.GROUP )
    for idx in range(2):
        grouping.addSourceToGroup( idx, idx )

    # Keep Python running until user closes Bdv window
    check = autoclass( 'net.imglib2.python.BdvWindowClosedCheck' )()
    frame = cast( 'javax.swing.JFrame', autoclass( 'javax.swing.SwingUtilities' ).getWindowAncestor( vp ) )
    frame.addWindowListener( check )
    while check.isOpen():
        time.sleep( 0.1 )




if __name__ == '__main__':
    pmap_p = '/home/consti/Work/data_neuro/CREMI/wsdt_test/cremi_sampleC_probs_cantorV1.h5'
    with h5py.File(pmap_p) as f:
        x = f['data'][1]

    show_bdv(x)
