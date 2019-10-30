import json
import os
import random
import warnings

import numpy as np

from inversion.optimizators.optimizations import optimizers_dict
from objects.seismic.seismogram import Seismogram, Trace
from obspy_edited import segy

parameters_invers_1 = [
    'Km',
    'Gm',
    'rho_m',

    'Ks',
    'Gs',
    'rho_s',

    'Kf',
    'rho_f',

    'phi',
    'phi_s',
    'h'
]

parameters_components_1 = [
    'matrix',
    'shale',
    'fluid'
]

parameters_elastic_1 = [
    'K',
    'G',
    'rho',
    'volume'
]

keys_1 = {
    'matrix': {
        'K': 'Km',
        'G': 'Gm',
        'rho': 'rho_m',
    },

    'shale': {
        'K': 'Ks',
        'G': 'Gs',
        'rho': 'rho_s',
        'volume': 'phi_s'
    },

    'fluid': {
        'K': 'Kf',
        'rho': 'rho_f',
        'volume': 'phi'
    }
}


def get_starter_dict():
    params = {key:[] for key in parameters_invers_1}

    return params


def read_input_file(file_name):
    with open(file_name, 'r') as f:
        input = json.load(f)

    nlayers = len(input['model']['layers'])

    params_all_dict = get_starter_dict()
    params_all_dict['nlayers'] = nlayers

    params_to_optimize = []
    bounds_to_optimize = []

    for layer in input['model']['layers']:
        params_all_dict['h'].append(layer['H'])
        for pc in parameters_components_1:
            for pe in parameters_elastic_1:

                if (pc == 'matrix' and pe == 'volume') or (pc == 'fluid' and pe == 'G'):
                    continue

                key = keys_1[pc][pe]

                if layer[pc][pe]['optimize']:
                    params_to_optimize.append({key: layer['index']})
                    bound_min = layer[pc][pe]['min']
                    bound_max = layer[pc][pe]['max']
                    bounds_to_optimize.append(np.array([bound_min, bound_max]))
                    params_all_dict[key].append(random.uniform(bound_min, bound_max))

                else:
                    params_all_dict[key].append(layer[pc][pe]['value'])

    del(params_all_dict['h'][-1])

    return nlayers, params_all_dict, params_to_optimize, np.array(bounds_to_optimize)


def read_input_fp_file(model_folder):
    file_name = os.path.join(model_folder, 'input', 'input_fp.json')
    with open(file_name, 'r') as f:
        input = json.load(f)

    nlayers = len(input['model']['layers'])

    params_all_dict = get_starter_dict()
    params_all_dict['nlayers'] = nlayers

    params_to_optimize = []
    bounds_to_optimize = []

    for layer in input['model']['layers']:
        params_all_dict['h'].append(layer['H'])
        for pc in parameters_components_1:
            for pe in parameters_elastic_1:

                if (pc == 'matrix' and pe == 'volume') or (pc == 'fluid' and pe == 'G'):
                    continue

                key = keys_1[pc][pe]

                params_all_dict[key].append(layer[pc][pe]['value'])

                if layer[pc][pe]['optimize']:
                    params_to_optimize.append({key: layer['index']})
                    bounds_to_optimize.append(np.array([layer[pc][pe]['min'], layer[pc][pe]['max']]))

    del(params_all_dict['h'][-1])

    observation_params = input['observation']

    return nlayers, params_all_dict, params_to_optimize, np.array(bounds_to_optimize), observation_params


def create_start_stop_indexes(indexes_params, x_arr, dt):
    if indexes_params['type'] == 'from_model':
        # TODO при условии, что граница - горзинотальная!
        def formula(v, h, x):
            return np.sqrt(x**2 + 4*h*h) / v
        
        start_indexes = formula(indexes_params['values']['start']['v'], indexes_params['values']['start']['h'], x_arr)
        start_indexes = (start_indexes / dt).astype(int)

        stop_indexes = formula(indexes_params['values']['stop']['v'], indexes_params['values']['stop']['h'], x_arr)
        stop_indexes = (stop_indexes / dt).astype(int)

    return start_indexes, stop_indexes


def read_input_ip_file(model_folder, x_arr, dt):
    input_file_name = os.path.join(model_folder, 'input', 'input_ip.json')
    with open(input_file_name, 'r') as f:
        input = json.load(f)

    input = input['inversion_params']
    seismogram = read_segy(os.path.join(model_folder, 'input', input['segy_observed']))
    err = input['error']
    optimizers = [optimizers_dict[opt['name']](**opt['params']) for opt in input['optimizers'] if opt['use']]

    start_indexes, stop_indexes = create_start_stop_indexes(input['seismic_indexes'], x_arr, dt)

    return seismogram, err, optimizers, start_indexes, stop_indexes


def read_inversion_result_file(result_folder):
    file_name = os.path.join(result_folder, 'model_values.txt')
    with open(file_name, 'r') as f:
        rows = f.readlines()

    params_optimized = []
    vals = []

    for r in rows[:-2]:
        rr = r.split(', ')
        rr = rr[1].split(' = ')
        rr = rr[0].split('[')
        param = r.split(', ')[1].split(' = ')[0].split('[')[0][:-9]
        index = float(r.split(', ')[1].split(' = ')[0].split('[')[1][:-1])
        value = float(r.split(', ')[1].split(' = ')[1])

        params_optimized.append({param: index})
        vals.append(value)

    return params_optimized, vals


def get_results_files_list(folder):
    current_folder_numbers = [int(f.split('_')[-1]) for f in os.listdir(folder) if 'result' in f.lower() and not 'average' in f.lower()]
    current_folder_numbers.sort()

    return current_folder_numbers


def create_res_folder(folder):
    base_folder = os.path.join(folder, 'output')
    current_files_numbers = get_results_files_list(base_folder)

    if len(current_files_numbers) == 0:
        current_res_number = 1

    else:
        current_res_number = current_files_numbers[-1] + 1

    res_folder = os.path.join(base_folder, 'result_{}'.format(current_res_number))

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    return current_res_number


def write_output_file(model_folder, params_all_, inversed_model, params_to_optimize, inverse_duration=None, res_folder_postfix=None):
    base_folder = os.path.join(model_folder, "output")
    print(inversed_model)

    res_folder = os.path.join(base_folder, 'result_{}'.format(res_folder_postfix))

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    file_name = os.path.join(res_folder, 'model_values.txt')

    rows = []
    errs = []
    for m, p in zip(inversed_model, params_to_optimize):
        key = list(p.keys())[0]
        val = list(p.values())[0]
        true_val = params_all_[key][val]

        if true_val != 0:
            errs.append(abs((true_val - m) / true_val))

        rows.append('{}_true[{}] = {}, {}_inversed[{}] = {}\n'.format(key, val, true_val,
                                                                        key, val, m))
    if inverse_duration is not None:
        rows.append('inversion duration: {} min\n'.format(inverse_duration))

    err_average = np.average(errs)

    rows.append('Difference between true values and inverted values: {}\n'.format(err_average))

    with open(file_name, 'w') as f:
        f.writelines(rows)


def write_segy(seismogram, filename):
    segy_obj = segy._read_segy("../Files/example.sgy")
    tmp_name = "tmp_segy.sgy"
    ntraces = seismogram.ntraces

    segy_obj.traces = [segy_obj.traces[0]] * ntraces
    segy_obj.write(tmp_name)
    segy_obj = segy._read_segy(tmp_name)

    segy_obj.binary_file_header.unassigned_1 = b""
    segy_obj.binary_file_header.unassigned_2 = b""
    segy_obj.binary_file_header.number_of_samples_per_data_trace = -1
    segy_obj.binary_file_header.number_of_data_traces_per_ensemble = -1
    segy_obj.binary_file_header.sample_interval_in_microseconds = int(seismogram.traces[0].dt)

    # TODO добавить нормальное заполнение заголовков sou_x и rec_x
    for i in range(ntraces):
        segy_obj.traces[i].data = np.array(seismogram.traces[i].values, dtype=np.float32)

        segy_obj.traces[i].header.source_coordinate_x = 0
        segy_obj.traces[i].header.source_coordinate_y = 0

        segy_obj.traces[i].header.ensemble_number = int(i)
        segy_obj.traces[i].header.original_field_record_number = int(i)
        segy_obj.traces[i].header.energy_source_point_number = int(i)

        segy_obj.traces[i].header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group = \
            int(seismogram.traces[i].offset*1000)

        segy_obj.traces[i].header.number_of_samples_in_this_trace = len(seismogram.traces[i].values)
        segy_obj.traces[i].header.sample_interval_in_ms_for_this_trace = int(seismogram.traces[i].dt * 1000)

        # segy_obj.traces[i]

    os.remove(tmp_name)

    segy_obj.write(filename)

def read_segy(filename):
    segy_obj = segy._read_segy(filename)

    warnings.warn("Offset value is multiplying with 0.001!!!")
    traces = [Trace(trace.data, trace.header.sample_interval_in_ms_for_this_trace * 0.001,
                      trace.header.distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group * 0.001)
              for trace in segy_obj.traces]

    return Seismogram(traces)

