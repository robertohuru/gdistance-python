##############################################
from scipy import sparse  #
from scipy.spatial.distance import cdist  #
import numpy as np  #
from gdistance.core import *
from gdistance.raster import *
from gdistance.utils import * #
##############################################

class TransitionLayer:
    def __init__(self, nrows, ncols, extent, crs, transitionMatrix, transitionCells):
        self.nrows = nrows
        self.ncols = ncols
        self.extent = extent
        self.crs = crs
        self.transitionMatrix = transitionMatrix
        self.transitionCells = transitionCells

class GDistance:
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

    def repeat_list(self, a, nr):
        """
        Repeats a initial array/list nr times.
        :param a: Initial array/list to be repeated
        :param nr: number of times it should be repeated
        :return: A repetition of the initial array/list
        """
        result = []
        for i in range(nr):
            result.append(a)
        return np.array(result).flatten()

    def combine_lists(self, lists):
        """
        :param lists:
        :return:
        """
        result = []
        for row in lists:
            a = row.flatten()
            result.extend(a.tolist())
        return np.array(result)

    def adjacency(self, matrix, Cells, directions, include=False):
        """

        :param matrix:
        :param Cells:
        :param directions:
        :param include:
        :return:
        """
        r = self._raster.get_resolution()
        xys = self._raster.xyFromCell(Cells)
        xpoints = xys[:,0]
        ypoints = xys[:,1]
        if directions == 4:
            if include:
                d = self.combine_lists(
                    [xpoints, xpoints - r[0], xpoints + r[0], xpoints, xpoints, ypoints, ypoints, ypoints,
                     ypoints + r[1], ypoints - r[1]])
            else:
                d = self.combine_lists(
                    [xpoints - r[0], xpoints + r[0], xpoints, xpoints, ypoints, ypoints, ypoints + r[1],
                     ypoints - r[1]])
        elif directions == 8:
            if include:
                d = self.combine_lists(
                    [xpoints, self.repeat_list(xpoints - r[0], 3), self.repeat_list(xpoints + r[0], 3), xpoints,
                     xpoints,
                     ypoints, self.repeat_list(self.combine_lists([ypoints + r[1], ypoints, ypoints - r[1]]), 2),
                     ypoints + r[1], ypoints - r[1]])
            else:
                d = self.combine_lists(
                    [self.repeat_list(xpoints - r[0], 3), self.repeat_list(xpoints + r[0], 3), xpoints, xpoints,
                     self.repeat_list(self.combine_lists([ypoints + r[1], ypoints, ypoints - r[1]]), 2), ypoints + r[1],
                     ypoints - r[1]])
        elif directions == 16:
            r2 = np.array(r) * 2
            if include:
                d = self.combine_lists(
                    [xpoints, self.repeat_list(xpoints - r2[0], 2), self.repeat_list(xpoints + r2[0], 2),
                     self.repeat_list(xpoints - r[0], 5), self.repeat_list(xpoints + r[0], 5),
                     xpoints, xpoints, ypoints,
                     self.repeat_list(self.combine_lists([ypoints + r[1], ypoints - r[1]]), 2),
                     self.repeat_list(self.combine_lists(
                         [ypoints + r2[1], ypoints + r[1], ypoints, ypoints - r[1], ypoints - r2[1]]), 2),
                     ypoints + r[1], ypoints - r[1]])
            else:
                d = self.combine_lists([self.repeat_list(xpoints - r2[0], 2), self.repeat_list(xpoints + r2[0], 2),
                                        self.repeat_list(xpoints - r[0], 5), self.repeat_list(xpoints + r[0], 5),
                                        xpoints, xpoints,
                                        self.repeat_list(self.combine_lists([ypoints + r[1], ypoints - r[1]]), 2),
                                        self.repeat_list(self.combine_lists(
                                            [ypoints + r2[1], ypoints + r[1], ypoints, ypoints - r[1],
                                             ypoints - r2[1]]), 2),
                                        ypoints + r[1], ypoints - r[1]])
        elif directions == 'bishop':
            if include:
                d = self.combine_lists(
                    [xpoints, self.repeat_list(xpoints - r[0], 2), self.repeat_list(xpoints + r[0], 2), ypoints,
                     self.repeat_list(self.combine_lists([ypoints + r[1], ypoints - r[1]]), 2)])
            else:
                d = self.combine_lists([self.repeat_list(xpoints - r[0], 2), self.repeat_list(xpoints + r[0], 2),
                                        self.repeat_list(self.combine_lists([ypoints + r[1], ypoints - r[1]]), 2)])
            directions = 4  # to make pairs
        else:
            return None

        d = d.reshape((2, int(len(d) / 2)))
        d = np.transpose(d)
        cell = self.repeat_list(Cells, directions)
        pixCells = []
        cellfromxy = self._raster.cellFromXY(d)
        for index in range(len(cellfromxy)):
            row = cellfromxy[index]
            if row is not None and row != -1:
                pixCells.append([cell[index], row])
        return np.array(pixCells)

    def transition(self, Raster, function, directions, symm=False, include = False):
        """
        :param Raster: Raster Class
        :param function:
        :param directions:
        :param symm:
        :param include:
        :return:
        """
        self._raster = Raster
        array = self._raster.get_pixel_values()
        tmp_arr = array.flatten()
        Cells = np.array(list(range(0, len(tmp_arr))))
        Cells = np.where(tmp_arr == self._raster.no_data_value, -1, Cells)
        Cells = Cells[Cells != -1]
        self.tr = TransitionLayer(
            nrows=array.shape[0],
            ncols=array.shape[1],
            extent= self._raster.get_extent(),
            crs = self._raster.get_projection(),
            transitionMatrix = sparse.csr_matrix((array)),
            transitionCells = Cells)

        transitionMatr = self.tr.transitionMatrix
        self.adj = self.adjacency(transitionMatr, Cells=Cells, directions=directions, include=include)
        if symm:
            self.adj = [self.adj[:, 1], self.adj[:, 0]]
            self.adj = np.array(self.adj).transpose()
        fromvalues = self._raster.get_cell_values(self.adj[:, 0])
        tovalues = self._raster.get_cell_values(self.adj[:,1])
        dataValues = [fromvalues, tovalues]
        dataValues = np.array(dataValues).transpose()

        transition_values = self.calculate_transition_values(dataValues, function)
        nm = self.tr.nrows * self.tr.ncols
        self.tr.transitionMatrix = sparse.csr_matrix((nm, nm))
        self.tr.transitionMatrix._insert_many(self.adj[:, 0], self.adj[:, 1], np.array(transition_values))
        return self.tr

    def calculate_transition_values(self, dataValues, function):
        """
        :param dataValues:
        :param function:
        :return:
        """
        result = []
        for data in dataValues:
            if data[0] == self._raster.no_data_value or data[1] == self._raster.no_data_value:
                result.append(0)
            else:
                result.append(function(data))
        return result

    def geocorrection(self,T,type ="c", property = "conductance", scaleValue=1, multpl=False, scl=False):
        """
        :param T: TransitionLayer class
        :param type:
        :param property:
        :param scaleValue:
        :param multpl:
        :param scl:
        :return:
        """
        utils = Utils()
        extent = T.extent
        isLonLat = False
        if T.crs is not None and str(T.crs) == "4326":
            isLonLat = True
        if scl:
            midpoint = [np.mean([(extent[0], extent[1])]), np.mean([(extent[1], extent[3])])]
            scaleValue =  cdist(midpoint, midpoint + [self._raster.get_resolution[0], 0])

        xCells, yCells = self.adj[:, 0], self.adj[:, 1]

        xyfirst = self._raster.xyFromCell(xCells)
        xysecond = self._raster.xyFromCell(yCells)

        correctionValues = []
        correction = [xyfirst[:][0], xyfirst[:][1], xysecond[:][0], xysecond[:][1]]
        for i in range(len(xyfirst)):
            if isLonLat:
                distances = cdist([utils.geodetic_to_cartesian(xyfirst[i][0], xyfirst[i][1])],
                                  [utils.geodetic_to_cartesian(xysecond[i][0], xysecond[i][1])], 'euclidean')
            else:
                distances = cdist([[xyfirst[i][0], xyfirst[i][1]]],
                                  [[xysecond[i][0], xysecond[i][1]]], 'euclidean')
            if property == "conductance":
                if distances[0][0] == 0:
                    correctionValues.append(0)
                else:
                    correctionValues.append(1*scaleValue/(distances[0][0]))
            else:
                correctionValues.append(distances[0][0] / scaleValue)
        if isLonLat:
            if type == "r":
                adjb = self.adj[self.adj[:, 0].argsort()]
                adjb = adjb[adjb[:, 1].argsort()]
                rows = self._raster.rowFromCell(adjb[:, 0]) != self._raster.rowFromCell(adjb[:, 1])
                cor3 = np.where(rows == True, correction[:,3], -1)
                cor3 = cor3[cor3 > -1]
                cor1 = np.where(rows == True, correction[:, 1], -1)
                cor1 = cor1[cor1 > -1]
                a = (np.pi/180) * np.mean([cor1,cor3], axis=0)
                if property == "conductance":
                    corrFactor = np.cos(a)
                else:
                    corrFactor = 1/np.cos(a)
                b = np.where(rows == True, correctionValues, -1)
                b = b[b > -1]
                j = 0
                for i in range(len(correctionValues)):
                    if rows[i] == True:
                        correctionValues[i] = (b * corrFactor)[j]
                        j = j + 1
        nm = self.tr.nrows * self.tr.ncols
        correctionMatr = sparse.csr_matrix((nm, nm))
        correctionMatr._insert_many(self.adj[:, 0], self.adj[:, 1], np.array(correctionValues))
        if not multpl:
            correctedTrans = T.transitionMatrix.multiply(correctionMatr)
            T.transitionMatrix = correctedTrans
        else:
            T.transitionMatrix = correctionMatr

        return T

    def acc_cost(self,transitionLayer, targets):
        """
        :param transitionLayer:
        :param targets:
        :return:
        """
        transitionValues = transitionLayer.transitionMatrix

        xs = self.adj[:, 0]
        ys = self.adj[:, 1]

        cell_indices = list(self._raster.get_cells().values())
        cell_values = self._raster.get_cell_values(cell_indices)

        data_indexes = np.where(np.array(cell_values) == self._raster.no_data_value, -1, np.array(cell_indices))
        data_indexes = data_indexes[data_indexes != -1]

        no_data_indexes = np.where(np.array(cell_values) != self._raster.no_data_value, -1, np.array(cell_indices))
        no_data_indexes = no_data_indexes[no_data_indexes != -1]
        no_data_dict = dict.fromkeys(list(no_data_indexes), self._raster.no_data_value)

        sx = []
        sy = []
        weights = []
        from_vs = []
        vertices = {}
        for i in range(len(xs)):
            weight = transitionValues[(xs[i],ys[i])]
            keys = vertices.keys()
            if weight != 0 and (xs[i], ys[i]) not in keys and (ys[i], xs[i]) not in keys:
                vertices[(xs[i], ys[i])] = 0
                sx.append(xs[i])
                sy.append(ys[i])
                weights.append(1 / weight)
        temp_keys = {}
        for s in sorted(list(set(sx + sy))):
            if s not in temp_keys:
                temp_keys[s] = len(temp_keys)
        xs = []
        for s in sx:
            xs.append(int(temp_keys[s]))

        ys = []
        for s in sy:
            ys.append(int(temp_keys[s]))
        size = len(data_indexes)

        if len(np.array(targets).shape) > 1:
            for target in targets:
                from_v = self._raster.cellFromXY([target[0], target[1]])
                if from_v is not None and from_v != -1:
                    from_vs.append(int(temp_keys[from_v[0]]))

            acc_costs = acc_cost(xs,ys,weights,from_vs,size)
            acc_costs = np.min(np.array(acc_costs), axis=0)
            if self._raster.data_type == "Float32":
                acc_costs = np.round(acc_costs.flatten(), 1)
            else:
                acc_costs = np.round(acc_costs.flatten(), 0)
            if len(acc_costs) > 0:
                data_dict = dict(zip(list(data_indexes), acc_costs))
                data_dict.update(no_data_dict)
                acc_costs = list(dict(sorted(data_dict.items())).values())
                return np.array(acc_costs).reshape((self._raster.get_height(), self._raster.get_width()))
            else:
                return None
        else:
            from_v = self._raster.cellFromXY([targets[0], targets[1]])
            if from_v[0] == -1:
                return None
            from_vs = []
            for item in from_v:
                from_vs.append(int(temp_keys[item]))

            acc_costs = acc_cost(xs,ys,weights,from_vs,size)[0]
            if self._raster.data_type == "Float32":
                acc_costs = np.round(np.array(acc_costs).flatten(),1)
            else:
                acc_costs = np.round(np.array(acc_costs).flatten(), 0)
            data_dict = dict(zip(list(data_indexes), acc_costs))
            data_dict.update(no_data_dict)
            acc_costs = list(dict(sorted(data_dict.items())).values())
            return np.array(acc_costs).reshape((self._raster.get_height(), self._raster.get_width()))

