# gdistance-python
gdistance python package


The gdistance package makes use of raster, which has an underlying fast implementation of sparse matrices. 
The execution flow is borrowed from a R-gdistance package written by Hijmans and van Etten.

This is on a study (and course in) facility accessibility in developing countries.  Think roughly of a national marketshed analysis for hospitals and how reachable (travel time) these are for the population.  Hospitals could be other facilities (ATMs, schools, lawyers, drug stores, you name it).  The workflow that we focus on aims to determine trvale time to the nearest (in time) facility fork any location in a country's territory.

Travel can take place by car, train, boat or on foot.  (There could be more modes, but not in our example).  All modes of travel are over a network for which we have data, and each network link max  maximum speed that we shall use.  (Vector data thus ...)  Where there is no travel network, we expect that travel is on foot.  Terrain characteristics such as land cover and hillslope then determine the maximum walking speed.  You will probably immediately understand that such computations will be done in raster mode, but also that you can not really do this on a by-pixel basis: slope for instance depends on in which direction you cross the pixel.  Thus, in one direction you could walk downhill and be fast, while in opposite direction you'd be walking uphill and be slower.  A richer data structure than a raster is required for this.

This package, for the purpose above, takes an input raster and creates a graph out of it.  Each of the NxM raster cells will be a node in the graph, and from any single cell C there will be 4-8-16 links to the neighbour cells.  With a link a cost value can be associated to represent, for instance, slope.  Links can represent one direction (or bi-directional), if the cost associates with travel does not depend on direction.

<h2>Pre-requisite</h2>

To use this package, please install Numpy and GDAL.

<h2>Installlation</h2>

Use pip to install the package.

    pip install gdistance

<h2>How to use</h2>
Import the sub-packages from the main package as follows.

    >> from gdistance.raster import *
    >> from gdistance.gdistance import *
    >> from gdistance.utils import *

The gdistance.raster sub-packages contains raster-based functions for performing raster analysis derived from GDAL. For instance to create and save a raster object, the following lines of codes can be applied.
    
    >> ncols, nrows = 7,6
    >> minX, minY = 0, 0
    >> xres, yres = 1, 1
    >> maxX = minX + (ncols * xres)
    >> maxY = minY + (nrows * xres)
    >> values = [[2, 2, 1, 1, 5, 5, 5], 
              [2, 2, 8, 8, 5, 2, 1], 
              [7, 1, 1, 8, 2, 2, 2], 
              [8, 7, 8, 8, 8, 8, 5], 
              [8, 8, 1, 1, 5, 3, 9], 
              [8, 1, 1, 2, 5, 3, 9]]

    >> raster = Raster(extent=[minX, maxX,minY, maxY], xres=xres, yres=yres, crs=3857, nodatavalue=-9999, pix_values=values)
    >> save_raster(raster, "raster.tif")

Transition function generate an adjacency matrix out of a raster object. It takes a raster object, user-defined function and directions (number of neighbors for a cell).

    >> def mean(x1, x2):
            return np.divide(2, np.add(x1, x2))
    
    >> gd = Gdistance()
    >> trans = gd.transition(raster, function=mean, directions=4)
    
 You can also load a raster object from a file as shown;
 
    >> raster = Raster("processing/friction.tif")
    >> trans =  gd.transition(raster, function=mean, directions=4)

Geocorrection function generates a new transition matrix after applying a distance factor. 

    >> trans = gd.geocorrection(trans)

Accost function calculate the costs of travelling from every cell to target(s).

    >> # targets = coords_from_vector("inputs/bhu_facilities_point.shp")
    >> targets = [(5.5, 1.5)]
    >> accost = gd.acc_cost(trans, targets)
    >> save_raster(min(accost), "accumcost.tif")
