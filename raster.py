##############################################
import numpy as np  #
import dask.array as da
from dask import delayed, compute
from osgeo import osr, gdal, ogr  #
from osgeo import gdal_array as gdarr  #
from gdistance.core import *
import math
import json


##############################################

def _ret_dataset(a, pixArr):
    proj = a.raster.GetProjection()
    geot = a.raster.GetGeoTransform()
    width = a.get_width()
    height = a.get_height()
    nodatavalue = a.no_data_value
    bands = 1
    # Trim the raster
    if a.get_band_count() == 1:
        pixArr = np.where(a.get_pixel_values() == nodatavalue, nodatavalue, pixArr)
    else:
        pixArr = np.where(a.get_pixel_values()[0] == nodatavalue, nodatavalue, pixArr)
    pixArr = np.where(pixArr == np.inf, nodatavalue, pixArr)

    if len(pixArr.shape) > 2:
        bands = pixArr.shape[0]

    datatype = a.raster.GetRasterBand(1).DataType
    if bands == 1:
        dest_ds = gdal.GetDriverByName("Mem").Create("", width, height, bands, datatype)
        dest_ds.SetProjection(proj)
        dest_ds.SetGeoTransform(geot)
        new_band = dest_ds.GetRasterBand(1)
        new_band.SetNoDataValue(a.raster.GetRasterBand(1).GetNoDataValue())
        new_band.WriteArray(pixArr)
    else:
        dest_ds = gdal.GetDriverByName("Mem").Create("", width, height, bands, datatype)
        dest_ds.SetProjection(proj)
        dest_ds.SetGeoTransform(geot)
        for i in range(bands):
            new_band = dest_ds.GetRasterBand(i + 1)
            new_band.SetNoDataValue(a.raster.GetRasterBand(1).GetNoDataValue())
            new_band.WriteArray(pixArr[i])
    return Raster(dest_ds)


class Raster(object):
    def __init__(self, gdal_dataset=None, **kwargs):
        nodatavalue = -9999.0
        data = None
        self.data_type = None
        if gdal_dataset is None:
            if len(kwargs) == 0:
                extent = [-180, 180, -90, 90]
                xres = 1
                yres = -1
                crs = 4326
                self.data_type = gdal.GDT_Float32
            else:
                if 'extent' not in kwargs and 'xres' not in kwargs and 'crs' not in kwargs and 'yres' not in kwargs:
                    raise Exception("Error creating an empty raster. All arguments are required!")
                extent = kwargs['extent']
                xres = kwargs['xres']
                yres = kwargs['yres']
                crs = kwargs['crs']
                if 'pix_values' in kwargs:
                    data = kwargs['pix_values']
                if 'nodatavalue' in kwargs:
                    nodatavalue = kwargs['nodatavalue']
                if 'gdal_type' in kwargs:
                    self.data_type = kwargs['gdal_type']

                if data is not None and self.data_type is None:
                    self.data_type = gdarr.NumericTypeCodeToGDALTypeCode(data.dtype)
                else:
                    self.data_type = gdal.GDT_Float32
            width = int((extent[1] - extent[0]) / xres)
            height = int((extent[3] - extent[2]) / np.abs(yres))
            self.raster = gdal.GetDriverByName("Mem").Create("", width, height, 1, self.data_type)
            self.raster.SetGeoTransform([extent[0], xres, 0., extent[3], 0., -1 * yres])
            target = osr.SpatialReference()
            target.ImportFromEPSG(crs)
            self.raster.SetProjection(target.ExportToWkt())
            if data is not None:
                self.raster.GetRasterBand(1).WriteArray(data)
                self.raster.GetRasterBand(1).SetNoDataValue(nodatavalue)
            self.no_data_value = nodatavalue
        else:
            if type(gdal_dataset) == gdal.Dataset:
                self.raster = gdal_dataset
            else:
                self.raster = gdal.Open(gdal_dataset)
            if self.raster is None:
                raise Exception("Error in openning image!")
            self.no_data_value = 9999999999.
            band = self.raster.GetRasterBand(1)
            self._Spatial_ref = osr.SpatialReference(wkt=self.raster.GetProjection())
            if band.GetNoDataValue() is not None:
                self.no_data_value = band.GetNoDataValue()
                if self.get_pixel_values().dtype != np.array([self.no_data_value]).dtype:
                    self.no_data_value = np.array([self.no_data_value]).astype(self.get_pixel_values().dtype)[0]
                    """if self.no_data_value > 5:
                        self.no_data_value = float(self.no_data_value)
                    else:
                        self.no_data_value = int(self.no_data_value)"""
            self.data_type = gdal.GetDataTypeName(band.DataType)
            band.FlushCache()
            band = None

    def __str__(self):
        summary = "Class : Raster\n"
        extent = self.get_extent()
        values = self.get_pixel_values()
        values = values[np.where(values != self.no_data_value)]
        summary = summary + "\t\tDimensions\t: " + str(self.get_width()) + ", " + str(
            int(self.get_height())) + ", " + str(int(self.get_band_count())) + "\t(width, height, nbands)\n"
        summary = summary + "\t\tExtent\t\t: " + str(round(extent[0], 3)) + ", " + str(
            round(extent[1], 3)) + ", " + str(round(extent[2], 3)) + ", " + str(
            round(extent[3], 3)) + "\t(min-x, max-x, min-y, max-y)\n"
        summary = summary + "\t\tResolution\t: " + str(self.get_resolution()[0]) + ", " + str(
            self.get_resolution()[1]) + "\t(x, y)\n"
        summary = summary + "\t\tSRID\t\t: " + str(self.get_projection()) + "\n"
        summary = summary + "\t\tValues\t\t: " + str(values.min()) + ", " + str(values.max()) + ", " + str(
            values.mean()) + "\t(min, max, mean)\n"
        summary = summary + "\t\tNoDataValue\t: " + str(self.no_data_value)
        return summary

    def __rmul__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            pixArr = np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.multiply(pix1, pix2))
        else:
            pixArr = np.where(pix1 == self.no_data_value, self.no_data_value, np.multiply(pix1, other))
        return _ret_dataset(self, pixArr)

    def __mul__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            pixArr = np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value) , self.no_data_value, np.multiply(pix1, pix2))
        else:
            pixArr = np.where(pix1 == self.no_data_value, self.no_data_value, np.multiply(pix1, other))
        return _ret_dataset(self, pixArr)

    def __abs__(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.abs(pix)))

    def __add__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.add(pix1, pix2)))
        else:
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value, np.add(pix1, other)))

    def __radd__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.add(pix1, pix2)))
        else:
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value, np.add(pix1, other)))

    def __sub__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.subtract(pix1, pix2)))
        else:
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value, np.subtract(pix1, other)))

    def __rsub__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.subtract(pix1, pix2)))
        else:
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value, np.subtract(pix1, other)))

    def __truediv__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            # Replace zero values with no data value to prevent divide by zero warning
            pix2 = np.where(pix2 == 0, self.no_data_value, pix2)
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.true_divide(pix1, pix2)))
        else:
            if other == 0:
                raise RuntimeError("Division by zero not possible")
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value, np.true_divide(pix1, other)))

    def __rtruediv__(self, other):
        pix1 = self.get_pixel_values()
        # Replace zero values with no data value to prevent divide by zero warning
        pix1 = np.where(pix1 == 0, self.no_data_value, pix1)
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == other.no_data_value), self.no_data_value, np.true_divide(pix2, pix1)))
        else:
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value, np.true_divide(other, pix1)))

    def __trunc__(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.trunc(pix)))

    def __gt__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == self.no_data_value),
                                               self.no_data_value,
                                               np.where(pix1 > pix2, True, False)))
        else:
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value,
                                         np.where(pix1 > other, True, False)))

    def __eq__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == self.no_data_value),
                                               self.no_data_value,
                                               np.where(pix1 == pix2, True, False)))
        else:
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value,
                                         np.where(pix1 == other, True, False)))

    def __lt__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == self.no_data_value),
                                               self.no_data_value,
                                               np.where(pix1 < pix2, True, False)))
        else:
            return _ret_dataset(self,
                                np.where(pix1 == self.no_data_value, self.no_data_value,
                                         np.where(pix1 < other, True, False)))

    def __ne__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where((pix1 == self.no_data_value) | (pix2 == self.no_data_value),
                                               self.no_data_value, np.where(pix1 != pix2, True, False)))
        else:
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value,
                                               np.where(pix1 != other, True, False)))

    def __ge__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value, np.where(pix1 >= pix2, True, False)))
        else:
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value,
                                               np.where(pix1 >= other, True, False)))

    def __le__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where(pix1 == self.no_data_value,
                                               self.no_data_value, np.where(pix1 <= pix2, True, False)))
        else:
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value,
                                               np.where(pix1 <= other, True, False)))

    def __setitem__(self, key, value):
        pix = self.get_pixel_values()
        arr = key.get_pixel_values()
        arr = np.where((arr == key.no_data_value) | (arr == False), False, True)
        pix[arr] = value

        nodata = self.raster.GetRasterBand(1).GetNoDataValue()
        bands = self.get_band_count()
        width, height = self.get_width(), self.get_height()

        geot = self.raster.GetGeoTransform()
        proj = self.raster.GetProjection()

        self.raster = gdal.GetDriverByName("Mem").Create("", width, height, bands,
                                                         self.raster.GetRasterBand(1).DataType)
        self.raster.SetGeoTransform(geot)
        self.raster.SetProjection(proj)
        for i in range(bands):
            self.raster.GetRasterBand(i + 1).SetNoDataValue(nodata)
            if len(pix.shape) > 2:
                self.raster.GetRasterBand(i + 1).WriteArray(pix[i])
            else:
                self.raster.GetRasterBand(1).WriteArray(pix)

    def __getitem__(self, item):
        pix = self.get_pixel_values()
        if len(pix.shape) == 2:
            return _ret_dataset(self, pix)
        elif len(pix.shape) > 2 and pix.shape[0] > 1:
            return _ret_dataset(self, pix[item])
        else:
            raise Exception("Cannot slice a Raster object with a single band.")

    def __pow__(self, power, modulo=None):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.power(pix, power, modulo)))

    def __rpow__(self, other):
        pix1 = self.get_pixel_values()
        if type(other) == type(self):
            pix2 = other.get_pixel_values()
            return _ret_dataset(self, np.where(pix1 == self.no_data_value,
                                               self.no_data_value, np.power(pix1, other.get_pixel_values())))
        else:
            return _ret_dataset(self, np.where(pix1 == self.no_data_value, self.no_data_value, np.power(pix1, other)))

    def tan(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.tan(pix)))

    def arctan(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arctan(pix)))

    def arctan2(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arctan2(pix)))

    def arctanh(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arctanh(pix)))

    def sin(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.sin(pix)))

    def arcsin(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arcsin(pix)))

    def arcsinh(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arcsinh(pix)))

    def cos(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.cos(pix)))

    def cosh(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.cosh(pix)))

    def arccosh(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arccosh(pix)))

    def arccos(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.arccos(pix)))

    def sqrt(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self, np.where(pix == self.no_data_value, self.no_data_value, np.sqrt(pix)))

    def exp(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.exp(pix)))

    def exp2(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.exp2(pix)))

    def expm1(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.expm1(pix)))

    def abs(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.abs(pix)))

    def sign(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.sign(pix)))

    def ceil(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.ceil(pix)))

    def floor(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.floor(pix)))

    def trunc(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.trunc(pix)))

    def cumprod(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.cumprod(pix)))

    def cumsum(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.cumsum(pix)))

    def log(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.log(pix)))

    def log10(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.log10(pix)))

    def log2(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.log2(pix)))

    def log1p(self):
        pix = self.get_pixel_values()
        return _ret_dataset(self,
                            np.where(pix == self.no_data_value, self.no_data_value, np.log1p(pix)))

    def min(self, axis=0):
        pix = self.get_pixel_values()
        if len(pix.shape) == 2:
            return _ret_dataset(self, pix)
        elif len(pix.shape) > 2 and pix.shape[0] > 1:
            return _ret_dataset(self, np.min(pix, axis=axis))
        else:
            raise Exception("Expected a raster of 2 or more bands.")

    def argmin(self, axis=0):
        pix = self.get_pixel_values()
        if len(pix.shape) < 3:
            raise Exception("Expected a raster of 2 or more bands")
        return _ret_dataset(self, np.argmin(pix, axis=axis))

    def get_band_count(self):
        return self.raster.RasterCount

    def get_pixel_values(self):
        """
        :return: Returns the pixel values of a raster image
        """
        x = self.raster.RasterXSize
        y = self.raster.RasterYSize
        pxArray = gdarr.DatasetReadAsArray(self.raster, 0, 0, x, y)
        return pxArray

    def index_to_xyposition(self, indices):
        """
         Obtain the x,y position of a cell index e.g index 0 >> (0,0)
        :param indices: A list of cell indices
        """
        row = np.floor(indices / self.get_width())
        col = indices - (row * self.get_width())
        return np.array([row, col]).astype(dtype=np.uint32).transpose()

    def get_cell_values(self, indices, array):
        """
        :param indices: A list of cell indices
        :param array:ndarray of raster values
        :return: Return the pixel/cell values
        """
        pixvalues = array.flatten()
        result = pixvalues[indices]
        return result

    def get_projection(self):
        """
        :return: returns the raster projection
        """
        proj = osr.SpatialReference(wkt=self.raster.GetProjection())
        return int(proj.GetAttrValue('AUTHORITY', 1))

    def isLatLon(self):
        srs = osr.SpatialReference(wkt=self.raster.GetProjection())
        if srs.IsGeographic():
            return True
        else:
            return False

    def get_extent(self):
        """ returns the extent of the raster [xmin, xmax, ymin, ymax]
        :return: returns the extent of the raster
        """
        ulx, xres, xskew, uly, yskew, yres = self.raster.GetGeoTransform()
        lrx = ulx + (self.raster.RasterXSize * xres)
        lry = uly + (self.raster.RasterYSize * yres)
        return [ulx, lrx, lry, uly]

    def get_resolution(self):
        """
        :return: returns the raster resolution
        """
        ulx, xres, xskew, uly, yskew, yres = self.raster.GetGeoTransform()
        # return np.abs(xres), np.abs(yres)
        return np.absolute(xres), np.absolute(yres)

    def get_width(self):
        """
        :return: returns the width of the raster
        """
        return self.raster.RasterXSize

    def get_height(self):
        """
        :return: returns the hieght of the raster
        """
        return self.raster.RasterYSize

    def get_cells(self):
        """
        :return: Retirns a dictionary of cells x,y pixel coordinates and the corresponding cell indices
        """
        cells = {}
        i = 0
        for row in range(self.get_height()):
            for col in range(self.get_width()):
                cells[(row, col)] = i
                i = i + 1
        return cells

    def xy_from_cell(self, cells):
        ncols = self.get_width()
        nrows = self.get_height()
        extent = self.get_extent()
        res =  self.get_resolution()
        xmin = extent[0]
        ymax = extent[3]
        row = np.floor(cells/ncols)
        col = cells - row * ncols
        return np.transpose([(col + 0.5) * res[0] + xmin, ymax - (row + 0.5) * res[1]])

    def row_from_cell(self, cells):
        w = self.get_width()
        return np.floor(cells/w).astype(dtype=np.uint32)

    def cell_from_xy(self, xys):
        ncols = self.get_width()
        nrows = self.get_height()
        xmin, xmax, ymin, ymax = self.get_extent()
        yres_inv = nrows / (ymax - ymin)
        xres_inv = ncols / (xmax - xmin)
        x = xys[:, 0]
        y = xys[:, 1]
        row = np.floor((ymax - y) * yres_inv)
        row = np.where(y == ymin, nrows-1, row)
        col = np.floor((x - xmin) * xres_inv)
        col = np.where(x == xmax, ncols - 1, col)
        cells = np.where(((row < 0) | (row >= nrows)) | ((col < 0) | (col >= ncols)), -1, (row * ncols + col))
        cells = cells[cells != -1]
        return cells.astype(dtype=np.int32)

def tan(_rasterobject):
    """
    Calculate a quadratic tangent of raster pixel values
        :parameter
            :param Raster: Raster Object
        :return: return a raster object after applying the quadratic tangent
    """
    return _rasterobject.tan()


def arctan(_rasterobject):  # _rasterobject
    """:param Raster: Raster Object
    :return: return a raster object after applying the quadratic arctan
    """
    return _rasterobject.arctan()


def arctan2(_rasterobject):
    return _rasterobject.arctan2()


def arctanh(_rasterobject):
    return _rasterobject.arctanh()


def sin(_rasterobject):
    return _rasterobject.sin()


def arcsin(_rasterobject):
    return _rasterobject.arcsin()


def arcsinh(_rasterobject):
    return _rasterobject.arcsinh()


def cos(_rasterobject):
    return _rasterobject.cos()


def cosh(_rasterobject):
    return _rasterobject.cosh()


def arccosh(_rasterobject):
    return _rasterobject.arccosh()


def arccos(_rasterobject):
    return _rasterobject.arccos()


def sqrt(_rasterobject):
    return _rasterobject.sqrt()


def exp(_rasterobject):
    return _rasterobject.exp()


def exp2(_rasterobject):
    return _rasterobject.exp2()


def expm1(_rasterobject):
    return _rasterobject.expm1()


def sign(_rasterobject):
    return _rasterobject.sign()


def ceil(_rasterobject):
    return _rasterobject.ceil()


def floor(_rasterobject):
    return _rasterobject.floor()


def trunc(_rasterobject):
    return _rasterobject.trunc()


def cumprod(_rasterobject):
    return _rasterobject.cumprod()


def cumsum(_rasterobject):
    return _rasterobject.cumsum()


def log(_rasterobject):
    return _rasterobject.log()


def log10(_rasterobject):
    return _rasterobject.log10()


def log2(_rasterobject):
    return _rasterobject.log2()


def abs(_rasterobject):
    return _rasterobject.abs()


def min(_rasterobject, axis=0):
    return _rasterobject.min(axis)


def argmin(_rasterobject, axis=0):
    return _rasterobject.argmin(axis)


def where(condition, x, y):
    if type(x) == Raster:
        x = x.get_pixel_values()
    if type(y) == Raster:
        y = y.get_pixel_values()
    return _ret_dataset(condition, np.where(condition.get_pixel_values() == 1, x, y))


def mask_raster(x, y):
    """
    :param x: Input raster object
    :param y: masking object (vector or raster)
    :param mask_format: Type of mask data object. >> vector or raster
    :return:
    """
    if type(x) == type(y):
        extent = y.get_extent()
        x_min = extent[0]
        y_max = extent[3]

        width = extent[1] - x_min
        height = y_max - extent[2]

        g = y.raster.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(g)
        topPix = gdal.ApplyGeoTransform(inv_gt, x_min, y_max)
        xoff = topPix[0]
        yoff = topPix[1]

        win_xsize = int(np.abs(width / g[1]))
        win_ysize = int(np.abs(height / g[5]))
        raster_ds = gdal.Translate("", x.raster, format="MEM", srcWin=[xoff, yoff, win_xsize, win_ysize],
                                   noData=x.no_data_value)
        return Raster(raster_ds)
    else:
        x = x.raster
        src_ds = ogr.Open(y)
        layer = src_ds.GetLayer()
        raster_ds = gdal.Warp("", x, format="MEM", cutlineLayer=layer, dstNodata=x.no_data_value)
        return Raster(raster_ds)


def save_raster(x, output_raster, format="GTiff"):
    """
    :param x: Raster object
    :param output_raster: path to output raster
    :param format: data format of the output raster
    :return: None
    """
    src_raster = x.raster
    width = src_raster.RasterXSize
    height = src_raster.RasterYSize
    pxArray = x.get_pixel_values()
    dst = gdal.GetDriverByName(format).Create(output_raster, width, height, x.get_band_count(),
                                              src_raster.GetRasterBand(1).DataType)
    dst.GetRasterBand(1).SetNoDataValue(src_raster.GetRasterBand(1).GetNoDataValue())
    dst.GetRasterBand(1).WriteArray(pxArray)
    dst.SetProjection(src_raster.GetProjection())
    dst.SetGeoTransform(src_raster.GetGeoTransform())
    dst.FlushCache()
    dst = None


def rasterize(x, y, **args):
    """
    rasterize(x, **args)

    Return a raster obtained from the `vector_file`.

    Parameters
    ----------

    **args:  dictionary of arguments

    :param vector_file:

    :param args:
            >> x: Raster object,
            >> xres=10,
            >> yres=10,
            >> srcWin=[xmin, xmax, ymin, ymax],
            >> destSRS=SRID,
            >> nodatavalue=-.9999.,
            >> field=None,
            >>gformat="GTiff",
            >> func="max"
    :return: ndarray
        an array of raster pixel values
    """
    if y is not None:
        destSRS = y.get_projection()
        xres, yres = y.get_resolution()
        srcWin = y.get_extent()
        nodatavalue = y.no_data_value

    if 'xres' in args and 'yres' in args:
        xres = args['xres']
        yres = args['yres']
    if 'srcWin' in args:
        srcWin = args['srcWin']
    if 'destSRS' in args:
        destSRS = args['destSRS']
    if 'nodatavalue' in args:
        nodatavalue = args['nodatavalue']
    if 'field' in args:
        field = args['field']
    else:
        field = 1
    if 'format' in args:
        gformat = args['format']
    else:
        gformat = "GTiff"
    if 'func' in args:
        func = args['func']
    else:
        func = None

    datasource = ogr.Open(x)
    layer = datasource.GetLayer(0)

    point_types = (ogr.wkbPoint, ogr.wkbPoint25D, ogr.wkbPointM, ogr.wkbPointZM)
    """line_types = (ogr.wkbLineString, ogr.wkbLineString25D, ogr.wkbLineStringM, ogr.wkbLineStringZM,
                  ogr.wkbMultiLineString, ogr.wkbMultiLineString25D, ogr.wkbMultiLineStringM, ogr.wkbMultiLineStringZM)"""
    poly_types = (ogr.wkbPolygon, ogr.wkbPolygon25D, ogr.wkbPolygonM, ogr.wkbPolygonZM,
                  ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D, ogr.wkbMultiPolygonM, ogr.wkbMultiPolygonZM,
                  ogr.wkbCurvePolygon, ogr.wkbCurvePolygonM, ogr.wkbCurvePolygonZ, ogr.wkbCurvePolygonZM)

    if func is None or func.lower() in ("max", "maximum") or layer.GetGeomType() in poly_types:
        return vector2raster(vector=x, xres=xres, yres=yres, destSRS=destSRS, srcWin=srcWin, nodatavalue=nodatavalue,
                             field=field)

    inSpatialRef = layer.GetSpatialRef()
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(destSRS)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    ncols = int(np.abs((srcWin[1] - srcWin[0]) / xres))  # width
    nrows = int(np.abs((srcWin[3] - srcWin[2]) / yres))  # height
    data = {}
    value = 1
    for feature in layer:
        if field is not None:
            value = feature.GetField(field)
        geom = feature.GetGeometryRef()
        if geom is not None:
            geom.Transform(coordTrans)
            geojsonResult = geom.ExportToJson()
            coord = json.loads(geojsonResult)['coordinates']
            if geom.GetGeometryType() in point_types:
                coord = [coord]
            coord = np.array(coord)
            cells = cellFromXY(ncols, nrows, srcWin[0], srcWin[1], srcWin[2], srcWin[3], coord[:, 0], coord[:, 1])
            for cell in cells:
                if cell in data:
                    values = data[cell]
                    values.append(value)
                else:
                    values = [value]
                data[cell] = values
    pixArr = np.empty((nrows * ncols))
    pixArr.fill(nodatavalue)

    def funct(x):
        if func is None:
            return np.max(x)
        elif func.lower() in ("max", "maximum"):
            return np.max(x)
        elif func.lower() in ("min", "minimum"):
            return np.min(x)
        elif func.lower() == "sum":
            return np.sum(x)
        elif func.lower() in ("avg", "average", "mean"):
            return np.mean(x)
        elif func.lower() in ("median", "med"):
            return np.median(x)
        elif func.lower() in ("mode"):
            return np.mod(x)
        elif func.lower() in ("count"):
            return len(x)
        else:
            return np.max(x)

    for row in data:
        pixArr[row] = funct(data[row])
    pixArr = pixArr.reshape((nrows, ncols))
    datatype = gdarr.NumericTypeCodeToGDALTypeCode(pixArr.dtype)
    target_ds = gdal.GetDriverByName("Mem").Create("", ncols, nrows, 1, gdal.GDT_Float32)
    target_ds.SetGeoTransform([srcWin[0], xres, 0., srcWin[3], 0., -1 * yres])
    target_ds.SetProjection(outSpatialRef.ExportToWkt())
    target_band = target_ds.GetRasterBand(1)
    target_band.WriteArray(pixArr)
    target_band.SetNoDataValue(nodatavalue)
    target_band.FlushCache()
    layer = None
    datasource = None
    return Raster(target_ds)


def vector2raster(vector, xres, yres, destSRS=None, srcWin=None, field=1, nodatavalue=-9999.0):
    """
    :param vector: Path to input vector data to rasterize
    :param xres: x-resolution
    :param: yres: y-resolution
    :param: destSRS: Destination SRS
    :param: srcWin: Extent of the resulting raster
    :param field: Attribute column in the vector table
    :param nodatavalue: The value to assign for nodatavalue cells
    :return: returns the Raster object
    """
    src_ds = ogr.Open(vector)
    src_layer = src_ds.GetLayer()
    minX, maxX, minY, maxY = src_layer.GetExtent()
    if srcWin is not None:
        minX = srcWin[0]
        maxX = srcWin[1]
        minY = srcWin[2]
        maxY = srcWin[3]

    x_size = int(np.abs((maxX - minX) / xres))
    y_size = int(np.abs((maxY - minY) / yres))
    geot = [minX, xres, 0., maxY, 0., -1 * yres]
    target_proj = src_layer.GetSpatialRef()

    if destSRS is not None:
        target_proj = osr.SpatialReference()
        target_proj.ImportFromEPSG(destSRS)

    target_ds = gdal.GetDriverByName("Mem").Create('', x_size, y_size, 1, gdal.GDT_Float32)
    target_ds.SetGeoTransform(geot)
    target_ds.SetProjection(target_proj.ExportToWkt())
    target_band = target_ds.GetRasterBand(1)
    target_band.SetNoDataValue(nodatavalue)
    target_band.Fill(nodatavalue)
    if field == 1:
        gdal.RasterizeLayer(target_ds, [1], src_layer, burn_values=[field])
    else:
        gdal.RasterizeLayer(target_ds, [1], src_layer, options=["ATTRIBUTE={}".format(field), 'MERGE_ALG=REPLACE'])
    target_band.FlushCache()
    target_band = None
    src_layer = None
    return Raster(target_ds)


def project_raster(x, y, **kwargs):
    """
    :param x:
    :param y:
    :param kwargs:
        >> xres,
        >> yres,
        >> extent,
        >> method = "bilinear"
    :return:
    """
    data_type = y.raster.GetRasterBand(1).DataType
    extent = y.get_extent()
    minX, maxX, minY, maxY = extent[0], extent[1], extent[2], extent[3]
    nodata = y.no_data_value
    xres, yres = y.get_resolution()
    crs = y.get_projection()

    if 'xres' in kwargs:
        xres = kwargs['xres']
    if 'yres' in kwargs:
        yres = kwargs['yres']
    if 'extent' in kwargs:
        extent = kwargs['extent']
    if 'crs' in kwargs:
        crs = kwargs['crs']
    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = "bilinear"

    x_size = int(np.abs((maxX - minX) / xres))
    y_size = int(np.abs((maxY - minY) / yres))
    geot = [minX, xres, 0., maxY, 0., -1 * yres]

    dst = gdal.GetDriverByName('MEM').Create('', x_size, y_size, 1, data_type)
    target = osr.SpatialReference()
    target.ImportFromEPSG(crs)
    dst.SetGeoTransform(geot)
    dst.SetProjection(target.ExportToWkt())
    if data_type > 5:
        nodata = float(nodata)
        dst.GetRasterBand(1).WriteArray(np.ones(shape=(y_size, x_size), dtype=np.float32) * nodata)
    else:
        nodata = int(nodata)
        dst.GetRasterBand(1).WriteArray(np.ones(shape=(y_size, x_size), dtype=np.int) * nodata)
    dst.GetRasterBand(1).SetNoDataValue(nodata)

    if "linear" in method.lower():
        method = gdal.GRA_Bilinear
    elif "cubic" in method.lower():
        method = gdal.GRA_Cubic
    elif "nearest" in method.lower():
        method = gdal.GRA_NearestNeighbour
    else:
        method = gdal.GRIORA_Bilinear
    gdal.ReprojectImage(x.raster, dst, x.raster.GetProjection(), y.raster.GetProjection(), method)
    return Raster(dst)

def crop(x, y):
    """
    :param x: raster object to be cropped
    :param y: extent object, or any object from which an Extent object can be extracted
    :return:
    """
    if type(x) == Raster and type(y) == Raster:
        extent = y.get_extent()
        x_min = extent[0]
        y_max = extent[3]
        width = extent[1] - x_min
        height = y_max - extent[2]
        g = y.raster.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(g)
        topPix = gdal.ApplyGeoTransform(inv_gt, x_min, y_max)
        xoff = topPix[0]
        yoff = topPix[1]
        win_xsize = int(np.abs(width / g[1]))
        win_ysize = int(np.abs(height / g[5]))
        raster_ds = gdal.Translate("", x.raster, format="MEM", srcWin=[xoff, yoff, win_xsize, win_ysize],
                                   noData=x.no_data_value)
        pixA = gdarr.DatasetReadAsArray(y.raster, 0, 0, y.raster.RasterXSize, y.raster.RasterYSize)
        pixB = gdarr.DatasetReadAsArray(raster_ds, 0, 0, raster_ds.RasterXSize, raster_ds.RasterYSize)
        pixArr = np.where(pixA == y.no_data_value, x.no_data_value, pixB)
    target_ds = gdal.GetDriverByName("Mem").Create('', win_xsize, win_ysize, x.get_band_count(),
                                                   x.raster.GetRasterBand(1).DataType)
    target_ds.SetGeoTransform(g)
    target_ds.SetProjection(y.raster.GetProjection())
    if x.get_band_count() == 1:
        target_band = target_ds.GetRasterBand(1)
        target_band.SetNoDataValue(x.raster.GetRasterBand(1).GetNoDataValue())
        target_band.WriteArray(pixArr)
        target_band.FlushCache()
    else:
        for i in range(x.get_band_count()):
            target_band = target_ds.GetRasterBand(i + 1)
            target_band.SetNoDataValue(x.raster.GetRasterBand(1).GetNoDataValue())
            target_band.WriteArray(pixArr)
            target_band.FlushCache()
    return Raster(target_ds)


def trim(x):
    """
    :param x: raster object to be cropped
    :return:
    """
    pixArr = x.get_pixel_values()
    pixArr = np.where(pixArr == x.no_data_value, 0, 1)
    cells = np.where(pixArr == 1)
    rows = sorted(cells[0])
    cols = sorted(cells[1])
    win_ysize = int(np.abs(rows[-1] - rows[0]))
    win_xsize = int(np.abs(cols[-1] - cols[0]))
    yoff = rows[0]
    xoff = cols[0]
    raster_ds = gdal.Translate("", x.raster, format="MEM", srcWin=[xoff, yoff, win_xsize, win_ysize],
                               noData=x.no_data_value)
    return Raster(raster_ds)


def reclassify_raster(x, matrix):
    """
    :param x: Input raster object
    :param matrix: matrix of classification
    :return: returns classified raster object
    """
    nodatavalue = x.no_data_value
    px_input = x.get_pixel_values().flatten().tolist()

    for i in range(len(px_input)):
        pix = px_input[i]
        if pix != nodatavalue:
            for row in matrix:
                if matrix.shape[1] == 2:
                    if pix == row[0]:
                        px_input[i] = row[1]
                        break
                elif len(matrix.shape) == 3:
                    if pix >= row[0] and pix < row[1]:
                        px_input[i] = row[2]
                        break
    px_input = np.array(px_input)
    datatype = gdarr.NumericTypeCodeToGDALTypeCode(px_input.dtype)
    new_raster = gdal.GetDriverByName("Mem").Create("", x.get_width(), x.get_height(), 1, gdal.GDT_Float32)
    new_raster.SetProjection(x.raster.GetProjection())
    new_raster.SetGeoTransform(x.raster.GetGeoTransform())
    px_result = px_input.reshape(x.get_height(), x.get_width())
    px_result = np.where(x.get_pixel_values() == nodatavalue, nodatavalue, px_result)
    new_band = new_raster.GetRasterBand(1)
    new_band.WriteArray(px_result)
    new_band.SetNoDataValue(x.raster.GetRasterBand(1).GetNoDataValue())
    new_band.FlushCache()
    new_band = None
    return Raster(new_raster)


def reclassify(x, matrix):
    """
    :param x: Input raster object
    :param matrix: matrix of classification
    :return: returns classified raster object
    """
    # nodatavalue = x.no_data_value
    px_input = x.get_pixel_values()
    px_result = px_input.copy()
    for row in matrix:
        if matrix.shape[1] == 2:
            px_result[np.where(row[0] <= px_input)] = row[1]
        elif len(matrix.shape) == 3:
            px_result[np.where((row[0] <= px_input) & (px_input < row[1]))] = row[2]
    px_result = np.array(px_result)
    datatype = gdarr.NumericTypeCodeToGDALTypeCode(px_result.dtype)
    new_raster = gdal.GetDriverByName("Mem").Create("", x.get_width(), x.get_height(), 1, gdal.GDT_Float32)
    new_raster.SetProjection(x.raster.GetProjection())
    new_raster.SetGeoTransform(x.raster.GetGeoTransform())
    px_result = px_result.reshape(x.get_height(), x.get_width())
    new_band = new_raster.GetRasterBand(1)
    new_band.WriteArray(px_result)
    new_band.SetNoDataValue(x.raster.GetRasterBand(1).GetNoDataValue())
    new_band.FlushCache()
    new_band = None
    return Raster(new_raster)


def cut(x, breaks):
    """
    :param x: raster object
    :param breaks: ndarray
    :return: returns a raster object after applying breaks
    """
    px_input = x.get_pixel_values()
    px_result = px_input.copy()
    breaks = breaks.astype(dtype=px_input.dtype)

    for i in range(len(breaks)):
        if (i + 1) == len(breaks):
            px_result[np.where((breaks[i] <= px_input) & (px_input != x.no_data_value))] = i
        else:
            px_result[np.where((breaks[i] <= px_input) & (px_input < breaks[i + 1]))] = i

    datatype = gdarr.NumericTypeCodeToGDALTypeCode(px_result.dtype)
    new_raster = gdal.GetDriverByName("Mem").Create("", x.get_width(), x.get_height(), 1, gdal.GDT_Float32)
    new_raster.SetProjection(x.raster.GetProjection())
    new_raster.SetGeoTransform(x.raster.GetGeoTransform())
    new_band = new_raster.GetRasterBand(1)
    new_band.WriteArray(px_result)
    new_band.SetNoDataValue(x.raster.GetRasterBand(1).GetNoDataValue())
    new_band.FlushCache()
    new_band = None
    return Raster(new_raster)


def zonal(x, y, func):
    """
    :param x: raster object
    :param y: raster object with codes representing zones
    :param func: aggregate function to be applied to summarize the values by zone. Either as character: 'mean', 'sd', 'min',
            'max', 'sum', 'med', 'mod', 'count'; or, for relatively small Raster* objects, a proper function
    :return: returns an ndarray with a value for each zone (unique value in zones)
    """
    xarray = x.get_pixel_values()
    yarray = y.get_pixel_values()

    funcs = ("max", "min", "sum", "mean", "med", "mod", "count", "sd")
    if func not in funcs or func is None:
        raise Exception("Aggregate function is not specified. Use either: 'mean', 'sd', 'min',"
                        "'max', 'sum', 'med', 'mod', 'count'")

    def aggregate_func(x):
        if func.lower() in ("max"):
            return np.max(x)
        elif func.lower() in ("min"):
            return np.min(x)
        elif func.lower() == "sum":
            return np.sum(x)
        elif func.lower() in ("mean"):
            return np.mean(x)
        elif func.lower() in ("med"):
            return np.median(x)
        elif func.lower() in ("mod"):
            return np.mod(x)
        elif func.lower() in ("count"):
            return len(x)
        elif func.lower() in ("sd"):
            return np.std(x)

    codes = np.unique(yarray)
    codes = codes[codes != y.no_data_value]
    # codes = codes[codes != x.no_data_value]
    zonal_stats = []
    for code in codes:
        index = np.argwhere((yarray == code) & (xarray != x.no_data_value))
        func_val = aggregate_func(xarray[index[:, 0], index[:, 1]])
        zonal_stats.append([code, func_val])
    return np.array(zonal_stats)


def merge(*rasters):
    """
    :param rasters: Array of list for raster pixel values
    :return: Returns the pixel value for the merged raster
    """
    src_raster = rasters[0].raster
    x = src_raster.RasterXSize
    y = src_raster.RasterYSize

    pixArr = []
    for raster in rasters:
        pixArr.append(raster.get_pixel_values())

    pixArr = np.array(pixArr).reshape((len(rasters)), y, x)
    pixArr = np.max(pixArr, axis=0)
    dst = gdal.GetDriverByName("Mem").Create("", x, y, 1, src_raster.GetRasterBand(1).DataType)
    dst.GetRasterBand(1).SetNoDataValue(src_raster.GetRasterBand(1).GetNoDataValue())
    dst.GetRasterBand(1).WriteArray(pixArr)
    dst.SetProjection(src_raster.GetProjection())
    dst.SetGeoTransform(src_raster.GetGeoTransform())
    dst.FlushCache()
    return Raster(dst)


def geodetic_distance(point1, point2, meters=True):
    """
    Vincenty's formula (inverse method) to calculate the distance (in
    kilometers or miles) between two points on the surface of a spheroid
    """

    # WGS 84
    a = 6378137  # meters
    f = 1 / 298.257223563
    b = 6356752.314245  # meters; b = (1 - f)a
    MAX_ITERATIONS = 200
    CONVERGENCE_THRESHOLD = 1e-12  # .000,000,000,001

    U1 = np.arctan((1 - f) * np.tan(np.radians(point1[:, 1])))
    U2 = np.arctan((1 - f) * np.tan(np.radians(point2[:, 1])))
    L = np.radians(np.subtract(point2[:, 0], point1[:, 0]))
    Lambda = L

    sinU1 = np.sin(U1)
    cosU1 = np.cos(U1)
    sinU2 = np.sin(U2)
    cosU2 = np.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda, cosLambda = np.sin(Lambda), np.cos(Lambda)
        sinSigma = np.sqrt((cosU2 * sinLambda) ** 2 +
                           (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)

        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = np.arctan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * np.divide(sinLambda, sinSigma)
        cosSqAlpha = np.subtract(1, sinAlpha ** 2)
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * np.divide(sinU2, cosSqAlpha)
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = (f / 16) * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if np.where(np.abs(np.subtract(Lambda, LambdaPrev)) < CONVERGENCE_THRESHOLD):
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * np.divide((a ** 2 - b ** 2), (b ** 2))
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                                                       (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                                                       (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * np.subtract(sigma, deltaSigma)

    if not meters:
        s = np.divide(s, 1000)  # meters to kilometers
    return s


def great_circle_distance(p1, p2, method="gc", unit="metres"):
    """
    :param p1: ndarray of (x, y) ->> (Longitude, Latitude) origin points
    :param p2: ndarray of (x, y) ->> (Longitude, Latitude) destination points
    :param method: gc -> great_circle, hs -> haversine and cp -> cartesian product. gc and hs works with spherical coordinates while cp works with cartesian coordinates
        >> point_distance(np.array([[41.49008, -71.312796]]), np.array([[41.499498, -81.695391]]), method="gc", unit="miles")
        >>
    :return:
    """
    # using great circle distance formula
    if unit.lower() in ("metres", "meteres", "metre", "meter"):
        r = 6371008.7
    elif unit.lower() in ("kilometres", "kilometeres", "kilometre", "kilometer"):
        r = 6371.0087
    elif unit.lower() in ("miles", "mile"):
        r = 3958.76120954
    elif unit.lower() in ("foot", "feet"):
        r = 20902259.1864
    elif unit.lower() in ("nautical mile", "nm"):
        r = 3440.06943844
    else:
        r = 6371008.7

    if p1.shape != p2.shape:
        raise TypeError(
            "Points must have the same shape"
        )
    lon1 = np.radians(p1[:, 0])
    lat1 = np.radians(p1[:, 1])
    lon2 = np.radians(p2[:, 0])
    lat2 = np.radians(p2[:, 1])

    if method.lower() in ("gc", "great circle", "great_circle", "g"):
        dlon = np.subtract(lon1, lon2)
        d = (np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)))
        return r * d
    elif method.lower() in ("hs", "haversine", "h"):
        dlon = np.subtract(lon2, lon1)
        dlat = np.subtract(lat2, lat1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(a))
    else:
        dlat = np.subtract(p2[:, 0], p1[:, 0])
        dlon = np.subtract(p2[:, 1], p1[:, 1])
        return np.sqrt(
            np.add((dlat ** 2), (dlon ** 2)))


def geodetic_to_cartesian(latitude, longitude, height=0):
    """
     Transforms a geodetic coordinates to cartesian (planar) coordinates
    :param latitude: Latitude
    :param longitude: Langitude
    :param height: height
    :return: returns an x,y planar coordinate
    """
    latitude = np.radians(latitude)
    longitude = np.radians(longitude)
    # flattening
    f = 1 / 298.257223563
    a = 6378137
    # eccentricity squared
    e2 = 2 * f - pow(f, 2)
    # prime vertical radius
    # v = a / np.sqrt(1 - (e2 * pow(sin(latitude), 2)))
    v = np.divide(a, np.sqrt(1 - (e2 * np.power(np.sin(latitude), 2))))
    x = np.add(v, height) * np.cos(latitude) * np.cos(longitude)
    y = np.add(v, height) * np.cos(latitude) * np.sin(longitude)

    # z = (v * (1 - e2) + height) * np.sin(latitude)

    return np.array([x, y]).transpose()


def terrain(x, opt="slope"):
    """
     calculate the slope using the elevation in each point of the map
    :param x: Path to the elevation raster
    :param opt: Option e,g slope
    :return: returns the resulting raster object
    """
    pass
    """gdal.DEMProcessing(destName=output_raster, srcDS=elevation_raster, processing=opt, format="GTiff")
    if os.path.exists(output_raster):
        return Raster(gdal.Open(output_raster))"""
