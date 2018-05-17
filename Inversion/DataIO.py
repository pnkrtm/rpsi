import json
import os
import numpy as np

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

                params_all_dict[key].append(layer[pc][pe]['value'])

                if layer[pc][pe]['optimize']:
                    params_to_optimize.append({key: layer['index']})
                    bounds_to_optimize.append(np.array([layer[pc][pe]['min'], layer[pc][pe]['max']]))

    del(params_all_dict['h'][-1])

    return nlayers, params_all_dict, params_to_optimize, np.array(bounds_to_optimize)


def read_inversion_result_file(file_name):
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
    current_files_numbers = [int(f.split('_')[-1]) for f in os.listdir(folder) if 'result' in f.lower() and not 'average' in f.lower()]
    current_files_numbers.sort()

    return current_files_numbers

def write_output_file(folder, params_all_, inversed_model, params_to_optimize, inverse_duration=None, file_name=None):

    print(inversed_model)

    if file_name is None:
        current_files_numbers = get_results_files_list(folder)

        if len(current_files_numbers) == 0:
            file_name = 'result_1'

        else:
            file_name = 'result_{}'.format(current_files_numbers[-1] + 1)

    file_name = os.path.join(folder, file_name)

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
    if not inverse_duration is None:
        rows.append('Inversion duration: {} min\n'.format(inverse_duration))

    err_average = np.average(errs)

    rows.append('Difference between true values and inverted values: {}\n'.format(err_average))

    with open(file_name, 'w') as f:
        f.writelines(rows)


