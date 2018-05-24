import numpy as np
from scipy.ndimage.interpolation import shift
import struct


class SeismicModel1D:
    def __init__(self, vp=None, vs=None, rho=None, h=None, phi=None):
        # values as 1D model
        self.vp = vp
        self.vs = vs
        self.rho = rho
        self.h = h
        self.phi = phi

    def get_number_of_layers(self):
        return len(self.vp)

    def get_depths(self):
        depths = [0]

        for h in self.h:
            depths.append(h)

            depths[-1] = depths[-1] + depths[-2]

        return depths

    def get_param(self, param_name, index_finish=None, index_start=0):
        param = []

        if param_name.lower() == 'vp':
            param = self.vp

        elif param_name.lower() == 'vs':
            param = self.vs

        elif param_name.lower() == 'rho':
            param = self.rho

        elif param_name.lower() == 'h':
            param = self.h

        elif param_name.lower() == 'phi':
            param = self.phi

        if index_finish is None:
            return param[index_start:]

        else:
            return param[index_start:index_finish]

    def get_max_boundary_depth(self):
        return np.sum(self.h)

    def get_model_from_columns(self, vp_column, vs_column, rho_column, dz):
        # self.vp1D = vp_column
        # self.vs1D = vs_column
        # self.rho1D = rho_column

        # parsing depths
        vp_column_s = shift(vp_column, 1, cval=vp_column[0])
        vs_column_s = shift(vs_column, 1, cval=vs_column[0])
        rho_column_s = shift(rho_column, 1, cval=rho_column[0])

        vp_bounds = np.round(np.abs(vp_column - vp_column_s), 4)
        vp_bounds[vp_bounds > 0] = 1
        vs_bounds = np.round(np.abs(vs_column - vs_column_s), 4)
        vs_bounds[vs_bounds > 0] = 1
        rho_bounds = np.round(np.abs(rho_column - rho_column_s), 4)
        rho_bounds[rho_bounds > 0] = 1

        bounds_indexes = vp_bounds + vs_bounds + rho_bounds
        bounds_indexes[bounds_indexes > 0] = 1
        bounds_values = np.array([dz * i * ind for i, ind in enumerate(bounds_indexes)])
        bounds_values = bounds_values[bounds_values > 0]

        if len(bounds_values) == 0:
            vp1d = [vp_column[0]]
            vs1d = [vs_column[0]]
            rho1d = [rho_column[0]]
            h1d = []

        else:
            empty_list = np.zeros(len(vp_column))

            h1d = bounds_values
            for i in range(1, len(h1d)):
                h1d[i] -= h1d[i-1]

            bounds_indexes[0] = 1

            vp1d = vp_column * bounds_indexes
            vs1d = vs_column * bounds_indexes
            rho1d = rho_column * bounds_indexes

            if np.array_equal(vp1d, empty_list):
                vp1d = np.zeros(len(h1d) + 1)
            else:
                vp1d = vp1d[vp1d > 0]

            if np.array_equal(vs1d, empty_list):
                vs1d = np.zeros(len(h1d) + 1)
            else:
                vs1d = vs1d[vs1d > 0]

            if np.array_equal(rho1d, empty_list):
                rho1d = np.zeros(len(h1d) + 1)
            else:
                rho1d = rho1d[rho1d > 0]

        self.vp = vp1d
        self.vs = vs1d
        self.rho = rho1d
        self.h = h1d

    def find_nearest_value(self, val_list, h_list, h_cur):
        h_list = np.append([0], h_list)
        h_nearest = h_list[h_cur >= h_list][-1]
        nearest_index = h_list.tolist().index(h_nearest)

        return val_list[nearest_index]

    def get_1D_regular_grid(self, param, h_max, dh):
        nz = int(h_max / dh)
        axesz = [i*dh for i in range(nz)]

        hh = list(self.h)

        for i in range(1, len(hh)):
            hh[i] += hh[i-1]

        values_col = [self.find_nearest_value(self.get_param(param), hh, axsz) for axsz in axesz]

        return values_col, axesz


class SeismicModel2D:
    def __init__(self):
        self.vp = []
        self.vs = []
        self.rho = []
        self.dx = 0
        self.dz = 0

    def get_nx(self):
        return self.vp.shape[1]

    def get_nz(self):
        return self.vp.shape[0]

    def read_fwi_model_file(self, file_name_vp, file_name_vs, file_name_rho, nx, nz, dx, dz):

        with open(file_name_vp, 'rb') as f:
            self.vp = np.reshape(struct.unpack('{}f'.format(nx * nz), f.read(4 * nx * nz)), (nx, nz)).T

        with open(file_name_vs, 'rb') as f:
            self.vs = np.reshape(struct.unpack('{}f'.format(nx * nz), f.read(4 * nx * nz)), (nx, nz)).T

        with open(file_name_rho, 'rb') as f:
            self.rho = np.reshape(struct.unpack('{}f'.format(nx * nz), f.read(4 * nx * nz)), (nx, nz)).T

        self.dx = dx
        self.dz = dz

    def write_fwi_model_file(self, file_name_vp, file_name_vs, file_name_rho):
        nx = self.get_nx()
        nz = self.get_nz()

        with open(file_name_vp, 'wb') as f:
            f.write(struct.pack('{}f'.format(nx * nz), *np.reshape(self.vp.T, (1, nx * nz))[0]))

        with open(file_name_vs, 'wb') as f:
            f.write(struct.pack('{}f'.format(nx * nz), *np.reshape(self.vs.T, (1, nx * nz))[0]))

        with open(file_name_rho, 'wb') as f:
            f.write(struct.pack('{}f'.format(nx * nz), *np.reshape(self.rho.T, (1, nx * nz))[0]))

    def get_1d_model(self, column_inex=0):
        model1d = SeismicModel1D()
        model1d.get_model_from_columns(self.vp[:, column_inex],
                                       self.vs[:, column_inex],
                                       self.rho[:, column_inex],
                                       self.dz)

        return model1d

    def find_nearest_value(self, val_list, h_list, h_cur):
        h_list = np.append([0], h_list)
        h_nearest = h_list[h_cur >= h_list][-1]
        nearest_index = h_list.tolist().index(h_nearest)

        return val_list[nearest_index]

    def create_column(self, values, h_input, nz, dz):
        axesz = [i * dz for i in range(nz)]
        h = h_input.copy()

        for i in range(1, len(h)):
            h[i] += h[i - 1]

        values_col = [self.find_nearest_value(values, h, axsz) for axsz in axesz]

        return values_col

    def set_1d_model(self, model1d):
        vp_col = self.create_column(model1d.vp, model1d.h, self.get_nz(), self.dz)
        vs_col = self.create_column(model1d.vs, model1d.h, self.get_nz(), self.dz)
        rho_col = self.create_column(model1d.rho, model1d.h, self.get_nz(), self.dz)

        for i in range(self.get_nx()):
            self.vp[:, i] = vp_col[:]
            self.vs[:, i] = vs_col[:]
            self.rho[:, i] = rho_col[:]

    def set_value_by_index(self, row_index, col_index, vp_val=None, vs_val=None, rho_val=None):
        if vp_val is not None:
            self.vp[row_index, col_index] = vp_val

        if vs_val is not None:
            self.vs[row_index, col_index] = vs_val

        if rho_val is not None:
            self.rho[row_index, col_index] = rho_val

    def get_value_by_index(self, row_index, col_index):

        vp_val = self.vp[row_index, col_index]
        vs_val = self.vs[row_index, col_index]
        rho_val = self.rho[row_index, col_index]

        return vp_val, vs_val, rho_val
