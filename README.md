# gdistance-python
gdistance python package


The gdistance package makes use of raster, which has an underlying fast implementation of sparse matrices. 
The execution flow is borrowed from a R-gdistance package written by Hijmans and van Etten.

<h2>Requirements</h2>

To use this package, please install Numpy and GDAL.

<h2>Installlation</h2>

Use pip to install the package.

    pip install gdistance

<h2>How to use</h2>
    from gdistance.raster import *
    from gdistance.gdistance import *
    from gdistance.utils import *
