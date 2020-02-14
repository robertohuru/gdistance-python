##############################################
import numpy as np  #
from osgeo import osr, gdal, ogr  #
from osgeo import gdal_array as gdarr  #
from gdistance.core import *
import math
##############################################
class Raster:
    def __init__(self, imagePath):
        """
        :param imagePath:
        """
        self.imagePath = imagePath
        self.raster = gdal.Open(self.imagePath)
        self.no_data_value = 9999999999.
        band = self.raster.GetRasterBand(1)
        self._Spatial_ref = osr.SpatialReference(wkt=self.raster.GetProjection())
        if band.GetNoDataValue() is not None:
            self.no_data_value = band.GetNoDataValue()
        self.data_type = gdal.GetDataTypeName(band.DataType)
        band.FlushCache()
        band = None

    def get_pixel_values(self):
        """
        :return:
        """
        x = self.raster.RasterXSize
        y = self.raster.RasterYSize
        pxArray = gdarr.DatasetReadAsArray(self.raster, 0, 0, x, y)
        #pxArray = np.around(pxArray, decimals=2)
        #pxArray = np.where(pxArray == self.no_data_value, 0, pxArray)
        return pxArray

    def get_cell_values(self, indices):
        """
        :param indices:
        :return:
        """
        pixvalues = self.get_pixel_values().flatten()
        result = []
        for i in indices:
           result.append(pixvalues[int(i)])
        return result

    def get_projection(self):
        """
        :return:
        """
        proj = osr.SpatialReference(wkt=self.raster.GetProjection())
        return proj.GetAttrValue('AUTHORITY', 1)

    def get_extent(self):
        """
        :return:
        """
        ulx, xres, xskew, uly, yskew, yres = self.raster.GetGeoTransform()
        lrx = ulx + (self.raster.RasterXSize * xres)
        lry = uly + (self.raster.RasterYSize * yres)
        return [ulx,lry, lrx, uly]

    def get_resolution(self):
        """
        :return:
        """
        ulx, xres, xskew, uly, yskew, yres = self.raster.GetGeoTransform()
        return abs(xres), abs(yres)

    def get_width(self):
        """
        :return:
        """
        return self.raster.RasterXSize

    def get_height(self):
        """
        :return:
        """
        return self.raster.RasterYSize

    def get_cells(self):
        """
        :return:
        """
        cells = {}
        i = 0
        for row in range(self.get_height()):
            for col in range(self.get_width()):
                cells[(row,col)] = i
                i = i + 1
        return cells

    def xyFromCell(self, cells):
        """
        :param cells:
        :return:
        """
        ncols = self.get_width()
        nrows = self.get_height()
        extent = self.get_extent()
        xmin = extent[0]
        xmax = extent[2]
        ymin = extent[1]
        ymax = extent[3]
        temp_cells = []
        for cell in cells:
            temp_cells.append(int(cell))
        return np.array(xyFromCell(ncols, nrows, xmin, xmax, ymin, ymax, temp_cells))

    def rowFromCell(self, cells):
        """
        :param cells:
        :return:
        """
        w = self.get_width()
        h = self.get_height()
        rows = []

        if type(cells) == list or type(cells) == np.ndarray or type(cells) == np.matrix:
            for cell in cells:
                if cell >=0 and cell < w * h:
                    row = math.floor(cell/w)
                    rows.append(row)
            rows = np.array(rows)
        else:
            if cells >= 0 and cells < w * h:
                rows = [math.floor(cells/w)]
        return rows

    def cellFromXY(self, xys):
        """
        :param xys:
        :return:
        """
        ncols = self.get_width()
        nrows = self.get_height()
        extent = self.get_extent()
        xmin = extent[0]
        xmax = extent[2]
        ymin = extent[1]
        ymax = extent[3]
        if type(xys[0]) == list or type(xys[0]) == np.ndarray or type(xys[0]) == np.matrix:
            return np.array(cellFromXY(ncols, nrows, xmin, xmax, ymin, ymax, list(xys[:, 0]), list(xys[:, 1])))
        else:
            return np.array(cellFromXY(ncols, nrows, xmin, xmax, ymin, ymax, [xys[0]], [xys[1]]))