##############################################
import numpy as np  #
from osgeo import osr, gdal, ogr  #
from osgeo import gdal_array as gdarr  #
import json  #
import os
from math import *
##############################################
class Utils:
    "Utility Class for common functions"
    def coords_from_vector(self, vector_file):
        """
        Obtains coordinates fro a vector file
        :param vector_file: Input vector file e.g shapefile, geojson, kml etc
        :return: returns a list of coordinates
        """
        datasource = ogr.Open(vector_file)
        layer = datasource.GetLayer(0)
        coords = []
        for feature in layer:
            geom = feature.GetGeometryRef()
            geojsonResult = geom.ExportToJson()
            coords.append(json.loads(geojsonResult)['coordinates'])
        layer = None
        datasource = None
        return coords

    def save_raster(self, array, ref_raster, output='Raster.tif', gformat="GTiff", gdal_dtype=gdal.GDT_Float32):
        """
        :param array: List of pixel values
        :param ref_raster:Reference raster image
        :param output: Path to the output raster image
        :param gformat: format of the output raster image
        :param gdal_dtype: Data type of the values
        :return: None
        """
        raster = gdal.Open(ref_raster)
        no_data_value = 9999999999.
        band = raster.GetRasterBand(1)
        if band.GetNoDataValue() is not None:
            no_data_value = band.GetNoDataValue()
        driver = gdal.GetDriverByName('GTiff')
        file = driver.Create(output, array.shape[1], array.shape[0], 1, gdal_dtype)
        file.GetRasterBand(1).SetNoDataValue(no_data_value)
        file.GetRasterBand(1).WriteArray(array)
        file.SetProjection(raster.GetProjection())
        file.SetGeoTransform(raster.GetGeoTransform())
        file.FlushCache()
        band.FlushCache()
        band = None
        file = None

    def rasterize(self, vector,ref_raster, output_raster,field=1, nodatavalue = 0):
        """
        :param vector: Path to input vector data to rasterize
        :param ref_raster: Reference raster image to obtain output raster properties
        :param output_raster: Path to output raster
        :param field: Attribute column in the vector table
        :param nodatavalue: The value to assign for nodatavalue cells
        :return: returns the pixel values for the output raster
        """
        mask_raster = gdal.Open(ref_raster)
        mask_band = mask_raster.GetRasterBand(1)
        if mask_band.GetNoDataValue() is not None:
            nodatavalue = mask_band.GetNoDataValue()

        mask_band.FlushCache()

        driver = mask_raster.GetDriver()
        x = mask_raster.RasterXSize
        y = mask_raster.RasterYSize

        temp_raster = driver.Create("Memory Image", x, y, 1, gdal.GDT_Int16)
        temp_raster.SetProjection(mask_raster.GetProjection())
        temp_raster.SetGeoTransform(mask_raster.GetGeoTransform())

        temp_band = temp_raster.GetRasterBand(1)
        temp_band.SetNoDataValue(nodatavalue)

        source_ds = ogr.Open(vector)
        layer = source_ds.GetLayer()
        if field == 1:
            gdal.RasterizeLayer(temp_raster, [1], layer, burn_values=[field])
        else:
            gdal.RasterizeLayer(temp_raster, [1], layer, options=["ATTRIBUTE={}".format(field)])

        pixArr = gdarr.DatasetReadAsArray(temp_raster, 0, 0, x, y)
        tempxArray = gdarr.DatasetReadAsArray(mask_raster, 0, 0, x, y)

        tempxArray = np.where(tempxArray == nodatavalue, 0, 1)
        mm = pixArr * tempxArray
        pxResult = np.where(mm == 0, nodatavalue, mm)

        new_raster = driver.Create(output_raster, x, y, 1, gdal.GDT_Int16)
        new_raster.SetProjection(mask_raster.GetProjection())
        new_raster.SetGeoTransform(mask_raster.GetGeoTransform())

        new_band = new_raster.GetRasterBand(1)
        new_band.WriteArray(pxResult)
        new_band.SetNoDataValue(nodatavalue)

        temp_band.FlushCache()
        new_band.FlushCache()
        new_band = None
        temp_band = None
        layer = None
        temp_raster = None
        mask_raster = None
        new_raster = None
        return pxResult

    def reclassify(self, inraster, matrix, outraster):
        """
        :param inraster: Path of the input raster
        :param matrix: matrix of classification
        :param outraster: the path of the output raster
        :return: returns the pixel value of the classified raster
        """
        mask_raster = gdal.Open(inraster)
        mask_band = mask_raster.GetRasterBand(1)
        nodatavalue = 9999999.
        if mask_band.GetNoDataValue() is not None:
            nodatavalue = mask_band.GetNoDataValue()

        x = mask_raster.RasterXSize
        y = mask_raster.RasterYSize
        driver = mask_raster.GetDriver()

        pxResult = gdarr.DatasetReadAsArray(mask_raster, 0, 0, x, y)
        matrix = matrix[matrix[:, 2].argsort()]
        for row in matrix:
            pxResult[np.where((pxResult[:] >= row[0]) & (pxResult[:] < row[1]))] = row[2]

        new_raster = driver.Create(outraster, x, y, 1, gdal.GDT_Int16)
        new_raster.SetProjection(mask_raster.GetProjection())
        new_raster.SetGeoTransform(mask_raster.GetGeoTransform())

        new_band = new_raster.GetRasterBand(1)
        new_band.WriteArray(pxResult)
        new_band.SetNoDataValue(nodatavalue)

        mask_band.FlushCache()
        new_band.FlushCache()
        new_band = None
        mask_band = None
        layer = None
        inraster = None
        mask_raster = None
        new_raster = None
        return pxResult

    def get_pixel_values(self, imagepath):
        """
        :param imagepath: Path of the input raster file
        :return: returns the pixel values for the raster
        """
        raster = gdal.Open(imagepath)
        x = raster.RasterXSize
        y = raster.RasterYSize
        pxArray = gdarr.DatasetReadAsArray(raster, 0, 0, x, y)
        raster = None
        return pxArray

    def get_no_data_value(self,imagepath):
        """
        :param imagepath: Path of the input raster file
        :return: Returns the value for the nodatavalue cells
        """
        raster = gdal.Open(imagepath)
        no_data_value = 9999999999.
        band = raster.GetRasterBand(1)
        if band.GetNoDataValue() is not None:
            no_data_value = band.GetNoDataValue()
        band.FlushCache()
        band = None
        raster = None
        return no_data_value

    def merge(self, *rasters):
        """
        :param rasters: Array of list for raster pixel values
        :return: Returns the pixel value for the merged raster
        """
        rasters = np.array(rasters)
        return np.max(rasters, axis=0)

    def merge_raster(self, rasters, outputraster, nodatavalue):
        """
        :param rasters: List of paths of input raster files
        :param outputraster: Path of the output raster file
        :param nodatavalue: Value to assign the nodatavalue cells
        :return: Returns the pixel value for the merged raster
        """
        import subprocess, os
        files_string = " ".join(rasters)
        path = os.path.dirname(__file__)
        gdal_merge = path + os.sep + "gdal_merge.py"
        command = "python %s -a_nodata %s -o %s -of gtiff %s" % (gdal_merge, nodatavalue,outputraster, files_string)
        subprocess.check_output(command)
        if os.path.exists(outputraster):
            return self.get_pixel_values(outputraster)

    def degree_to_radian(self, degree):
        """
            Mathematical function for converting a degree to radian
        :param degree: degree value
        :return: returns a radian value
        """
        return degree * np.pi / 180

    def geodetic_to_cartesian(self, latitude, longitude, height=0):
        """
         Transforms a geodetic coordinates to cartesian (planar) coordinates
        :param latitude: Latitude
        :param longitude: Langitude
        :param height: height
        :return: returns an x,y planar coordinate
        """
        latitude = self.degree_to_radian(latitude)
        longitude = self.degree_to_radian(longitude)
        # flattening
        f=1/298.257223563
        a = 6378137
        # eccentricity squared
        e2 = 2 * f - pow(f, 2)
        # prime vertical radius
        v = a / np.sqrt(1 - (e2 * pow(sin(latitude), 2)))
        x = (v + height) * cos(latitude) * cos(longitude)
        y = (v + height) * cos(latitude) * sin(longitude)
        z = (v * (1 - e2) + height) * sin(latitude)

        return [x,y]

    def terrain(self, elevation_raster, output_raster, opt='slope'):
        """
         calculate the slope using the elevation in each point of the map
        :param elevation_raster: Path to the elevation image
        :param output_raster: Path to the output raster
        :param opt: Option e,g slope
        :return: returns the pixel values of the resulting raster
        """
        gdal.DEMProcessing(output_raster, elevation_raster, opt)
        if os.path.exists(output_raster):
            return self.get_pixel_values(output_raster)