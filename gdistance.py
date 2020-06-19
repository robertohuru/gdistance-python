##############################################
import numpy as np  #
import dask.array as da
from gdistance.core import *
from gdistance.raster import *
from gdistance.utils import *
import sys
import concurrent.futures
from threading import *

##############################################


class TransitionLayer(object):
    def __init__(self, nrows, ncols, extent, crs, xyfromcell, transitionMatrix, transitionCells):
        self.nrows = nrows
        self.ncols = ncols
        self.extent = extent
        self.crs = crs
        self.xyfromcell = xyfromcell
        self.transitionMatrix = transitionMatrix
        self.transitionCells = transitionCells

class GDistance(object):
    """
    Gdistance class
    """
    def __init__(self):
        """
        :param _raster:
        """
        self._raster = None
        self.adj = None
        self.tr = None
        self.no_data_cells = None

    def adjacency(self, directions, include=False):
        """
        :param transitionLayer: Transition Layer
        :param directions: Number of directions for determining adjacent cells; 4, 8 or 16
        :return: Returns a matrix of adjacent cells in a given direction
        """
        width = self._raster.get_width()
        cells = []
        if directions == 4:
            cells = np.array([self.tr.transitionCells,
                              self.tr.transitionCells + 1,
                              self.tr.transitionCells + width])
            mask = np.isin(cells, self.tr.transitionCells, invert=True)
            cells[mask] = -1
            cells = np.transpose([cells[0, :], cells[1, :], cells[0, :], cells[2, :]])
        elif directions == 8:
            cells = np.array([self.tr.transitionCells,
                              self.tr.transitionCells + 1,
                              self.tr.transitionCells + width,
                              self.tr.transitionCells + width + 1,
                              self.tr.transitionCells + width - 1])
            mask = np.isin(cells, self.tr.transitionCells, invert=True)
            cells[mask] = -1
            cells = np.transpose(
                [cells[0, :], cells[1, :], cells[0, :], cells[2, :], cells[0, :], cells[3, :], cells[0, :],
                 cells[4, :]])
        elif directions == 16:
            cells = np.array([
                self.tr.transitionCells,
                self.tr.transitionCells + 1,
                self.tr.transitionCells + width,
                self.tr.transitionCells + width + 1,
                self.tr.transitionCells + width + 2,
                self.tr.transitionCells + (2 * width) + 1,
                self.tr.transitionCells + width - 1,
                self.tr.transitionCells + width - 2,
                self.tr.transitionCells + (2 * width) - 1])
            mask = np.isin(cells, self.tr.transitionCells, invert=True)
            cells[mask] = -1
            cells = np.transpose(
                [cells[0, :], cells[1, :], cells[0, :], cells[2, :], cells[0, :], cells[3, :], cells[0, :], cells[4, :],
                 cells[0, :], cells[5, :], cells[0, :], cells[6, :], cells[0, :], cells[7, :], cells[0, :],
                 cells[8, :]])
        shp = cells.shape
        size = shp[0] * shp[1]
        cells = cells.reshape((int(size / 2), 2))
        mask = (cells[:, 0] != -1) & (cells[:, 1] != -1)
        cells = cells[mask]
        del mask
        return cells

    def transition(self, raster, function, directions, symm=False):
        """
        :param raster: Raster Class
        :param function: transition function
        :param directions: Number of directions for determining adjacent cells; 4, 8 or 16
        :param symm: Boolean value
        :return: Returns a transition matrix from a raster
        """
        self._raster = raster
        array = self._raster.get_pixel_values()
        shape = array.shape

        self.tr = TransitionLayer(
            nrows= shape[0],
            ncols= shape[1],
            extent= self._raster.get_extent(),
            crs = self._raster.get_projection(),
            xyfromcell = None,
            transitionMatrix = None,
            transitionCells = np.flatnonzero(array !=  self._raster.no_data_value).astype(dtype=np.int32)
        )
        adj = self.adjacency(directions)
        if symm:
            adj = adj[:, 1], adj[:, 0]
        else:
            adj = adj[:, 0], adj[:, 1]

        pixvalues = array.flatten()
        fromvalues = pixvalues[adj[0]]
        tovalues = pixvalues[adj[1]]

        del array, pixvalues

        transition_values = self.calculate_transition_values([fromvalues, tovalues], function)
        del fromvalues, tovalues
        self.tr.transitionMatrix = adj[0], adj[1], transition_values
        del adj
        return self.tr

    def calculate_transition_values(self, dataValues, function):
        """
        :param dataValues: List of transition values for adjacent cells
        :param function: Function applied to the values of adjacent cells
        :return: Return a list of the result obtained by applying a function to the values of adjacent cells
        """
        result = np.where((dataValues[0] == self._raster.no_data_value) | (dataValues[1] == self._raster.no_data_value), 0, function(dataValues[0], dataValues[1]))
        return result

    def geocorrection(self,transitionLayer,type ="c", property = "conductance", scaleValue=1, multpl=True, scl=False):
        """
        Correct Transition matrix taking into account local distances
        :param transitionLayer: TransitionLayer object
        :param type: Type of geocorrection to apply; 'r' correction on the N-S direction,  'c', along the E-W direction
        :param property: property of the transition cells; either conductance of resistance
        :param scaleValue: Factor used in scaling transition values
        :param multpl: set true to multipy correctionMatr with transition matrix
        :param scl: set true to scale transition values to a reasonable range
        :return: returns a transition layer with corrected cells
        """
        self.tr = None

        extent = transitionLayer.extent
        isLonLat = False
        if self._raster.isLatLon():
            isLonLat = True
        if scl:
            midpoint = np.array([[np.mean([(extent[0], extent[1])]), np.mean([(extent[2], extent[3])])]])
            point2 = np.add(midpoint, np.array([self._raster.get_resolution()[0], 0]))
            if isLonLat:
                scaleValue = geodetic_distance(midpoint, point2).astype(ndtype=np.float32)
                # scaleValue = np.asarray(distance(midpoint, point2, 0, 0), dtype=np.float32)
            else:
                scaleValue = great_circle_distance(midpoint, point2, method="cp", unit="metres").astype(ndtype=np.float32)
                #scaleValue = np.asarray(distance(midpoint, point2, 1, 0), dtype=np.float32)
            del midpoint, point2

        transitionMatrix = transitionLayer.transitionMatrix
        xCells, yCells = transitionMatrix[0], transitionMatrix[1]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            res1 = executor.submit(self._raster.xy_from_cell, xCells)
            res2 = executor.submit(self._raster.xy_from_cell, yCells)
            xyfirst = res1.result()
            xysecond = res2.result()

        if isLonLat:
            #distances = geodetic_distance(xyfirst, xysecond).astype(dtype=np.float32)
            distances = np.asarray(distance(xyfirst, xysecond, 0, 0), dtype=np.float32)
        else:
            #distances = great_circle_distance(xyfirst, xysecond, method="cp", unit="metres").astype(dtype=np.float32)
            distances = np.asarray(distance(xyfirst, xysecond, 1, 0), dtype=np.float32)

        if property == "conductance":
            correctionValues = np.where(distances == 0, 0.0, np.divide(scaleValue, distances).astype(dtype=np.float32))
        else:
            correctionValues = np.divide(distances/ scaleValue).astype(dtype=np.float32)

        if isLonLat and type == "r":
            rows = self._raster.row_from_cell(xCells) != self._raster.row_from_cell(yCells)
            correction = np.transpose([xyfirst[:, 0], xyfirst[:, 1], xysecond[:, 0], xysecond[:, 1]])
            col3 = np.where(rows == True, correction[:,3], -1)
            col3 = col3[col3 > -1]
            col1 = np.where(rows == True, correction[:, 1], -1)
            col1 = col1[col1 > -1]
            factor = (np.pi/180) * np.mean([col1,col3], axis=0)

            del correction, col3, col1

            if property == "conductance":
                # low near the poles
                 corrFactor = np.cos(factor)
            else:
                # high near the poles
                corrFactor = 1/np.cos(factor)
            # makes conductance lower in N-S direction towards the poles
            correctionValues[rows] = correctionValues[rows] * corrFactor
            del corrFactor, rows,factor

        del xyfirst, xysecond
        if multpl:
            correctedTrans = np.multiply(transitionMatrix[2], correctionValues)
            transitionLayer.transitionMatrix = xCells, yCells, correctedTrans
        else:
            transitionLayer.transitionMatrix = xCells, yCells, correctionValues
        del xCells, yCells, correctedTrans, correctionValues
        return transitionLayer

    def acc_cost(self,transitionLayer, targets):
        """
        :param transitionLayer: TransitionLayer class
        :param targets: List of target cells
        :return: Return a RasterStack of accumulated costs for every target
        """
        targets = self._raster.cell_from_xy(np.array(targets))
        if len(targets) == 0:
            return None

        transitionCells = transitionLayer.transitionCells
        sx, sy = transitionLayer.transitionMatrix[0], transitionLayer.transitionMatrix[1]
        size = len(transitionCells)
        transitionValues = transitionLayer.transitionMatrix[2]
        weights = np.where((transitionValues > 0), np.divide(1, transitionValues), 0)
        del transitionValues, transitionLayer

        max_size = int(np.sum(weights))

        if max_size > sys.maxsize:
            max_size = sys.maxsize
        if self._raster.get_height() * self._raster.get_width() == size:
            acc_costs = np.asarray(acc_cost(sx, sy, weights, targets, size, max_size))
        else:
            # Set the indices to start from zero
            sorter = np.argsort(transitionCells)
            def funcs(arr, tr, sorter):
                return sorter[np.searchsorted(tr, arr, sorter=sorter)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                res1 = executor.submit(funcs, sx, transitionCells, sorter)
                res2 = executor.submit(funcs, sy, transitionCells, sorter)
                res3 = executor.submit(funcs, targets, transitionCells, sorter)
                sx = res1.result()
                sy = res2.result()
                targets = res3.result()
            acc_costs = np.array(acc_cost(sx, sy, weights, targets, size, max_size))

        del weights, sx, sy
        if len(acc_costs) == 0:
            return None
        # Replace max_int values with nodatavalues
        acc_costs = np.where(acc_costs == max_size, self._raster.no_data_value, acc_costs)

        target_ds = gdal.GetDriverByName("Mem").Create('', self._raster.get_width(), self._raster.get_height(),
                            len(targets), self._raster.raster.GetRasterBand(1).DataType)
        target_ds.SetGeoTransform(self._raster.raster.GetGeoTransform())
        target_ds.SetProjection(self._raster.raster.GetProjection())
        pixArr = np.full((self._raster.get_height() * self._raster.get_width()), self._raster.no_data_value)
        for i in range(len(targets)):
            target_band = target_ds.GetRasterBand(i + 1)
            target_band.SetNoDataValue(self._raster.raster.GetRasterBand(1).GetNoDataValue())
            pixArr[transitionCells] = acc_costs[i]
            target_band.WriteArray(pixArr.reshape(self._raster.get_height(), self._raster.get_width()))
            target_band.FlushCache()
        return Raster(target_ds)