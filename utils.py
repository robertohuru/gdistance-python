##############################################
import numpy as np  #
from osgeo import osr, gdal, ogr  #
from osgeo import gdal_array as gdarr  #
import json  #
##############################################
def data_from_vector(vector_file, fields=None, destSRS=None, includeCoords=True):
    datasource = ogr.Open(vector_file)
    layer = datasource.GetLayer(0)
    outSpatialRef = None
    if includeCoords and destSRS is not None:
        inSpatialRef = layer.GetSpatialRef()
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(int(destSRS))
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    layerDefinition = layer.GetLayerDefn()
    fieldCount=layerDefinition.GetFieldCount()
    if fields is None or len(fields) == 0:
        fields = []
        for i in range(fieldCount):
            fields.append(layerDefinition.GetFieldDefn(i).GetName())
    else:
        fields = fields
    data = []
    for feature in layer:
        featuredict = {}
        for field in fields:
            featuredict[field] = feature.GetField(field)
        if includeCoords:
            geom = feature.GetGeometryRef()
            if destSRS is not None:
                geom.Transform(coordTrans)
            geojsonResult = geom.ExportToJson()
            coord = json.loads(geojsonResult)['coordinates']
            if outSpatialRef is not None and outSpatialRef.IsGeographic():
                coord = [coord[1], coord[0]]
            featuredict["geom"] = coord
        data.append(featuredict)
    layer = None
    datasource = None
    return data

def coords_from_vector(vector_file, destSRS=None):
    """
    Obtains coordinates fro a vector file
    :param vector_file: Input vector file e.g shapefile, geojson, kml etc
    :param destSRS: EPSG code for spatial reference system e.g 4326 for output coordinate
    :return: returns a list of coordinates
    """
    datasource = ogr.Open(vector_file)
    layer = datasource.GetLayer(0)
    coords = []
    outSpatialRef = None
    if destSRS is not None:
        inSpatialRef = layer.GetSpatialRef()
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(destSRS)
        coordTrans  = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    for feature in layer:
        geom = feature.GetGeometryRef()
        if destSRS is not None:
            geom.Transform(coordTrans)
        geojsonResult = geom.ExportToJson()
        coord = json.loads(geojsonResult)['coordinates']
        if outSpatialRef is not None and outSpatialRef.IsGeographic():
            coord = [coord[1], coord[0]]
        coords.append(coord)
    layer = None
    datasource = None
    return coords

def get_pixel_values(imagepath):
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

def get_no_data_value(imagepath):
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

    uSq = cosSqAlpha * np.divide((a ** 2 - b ** 2) , (b ** 2))
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
        r =  3958.76120954
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
        return r *  d
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