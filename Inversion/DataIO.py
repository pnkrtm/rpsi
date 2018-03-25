import json
import os

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
                    bounds_to_optimize.append((layer[pc][pe]['min'], layer[pc][pe]['max']))

    del(params_all_dict['h'][-1])

    return nlayers, params_all_dict, params_to_optimize, bounds_to_optimize


def write_output_file(folder, params_all_, inversed_model, params_to_optimize):

    print(inversed_model)

    current_files_numbers = [f.split('_')[-1] for f in os.listdir(folder) if 'result' in f.lower()]

    if len(current_files_numbers) == 0:
        file_name = folder + '/result_1'

    else:
        file_name = folder + '/result_{}'.format(int(current_files_numbers[-1]) + 1)

    rows = []
    for m, p in zip(inversed_model, params_to_optimize):
        key = list(p.keys())[0]
        val = list(p.values())[0]
        rows.append('{}_observed[{}] = {}, {}_inversed[{}] = {}\n'.format(key, val, params_all_[key][val],
                                                                        key, val, m))

    with open(file_name, 'w') as f:
        f.writelines(rows)


