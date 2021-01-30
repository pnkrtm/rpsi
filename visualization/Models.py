import os
from matplotlib import pyplot as  plt
from visualization.Seismic import visualize_model_wellogs


def visualize_wellog(model_obs, model_inv, param_type: str, title: str, xlabel: str, ylabel:str, file_name: str,):
    visualize_model_wellogs(plt, model_obs, param_type, legend_label='истиная модель')
    visualize_model_wellogs(plt, model_inv, param_type, legend_label='результат подбора', linestyle='--')
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=18)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def visualize_model(model_obs, model_inv, pics_folder):
    visualize_wellog(model_obs, model_inv, 'vp', 'vp', 'скорость, м/с', 'глубина, м',
                     os.path.join(pics_folder, 'vp_png'))
    visualize_wellog(model_obs, model_inv, 'vs', 'vs', 'скорость, м/с', 'глубина, м',
                     os.path.join(pics_folder, 'vs_png'))
    visualize_wellog(model_obs, model_inv, 'aip', 'aip', 'акустический импеданс', 'глубина, м',
                     os.path.join(pics_folder, 'aip_png'))
    visualize_wellog(model_obs, model_inv, 'ais', 'ais', 'акустический импеданс', 'глубина, м',
                     os.path.join(pics_folder, 'ais_png'))
    visualize_wellog(model_obs, model_inv, 'rho', 'rho', 'плотность, кг/м3', 'глубина, м',
                     os.path.join(pics_folder, 'rho_png'))
    visualize_wellog(model_obs, model_inv, 'phi', 'phi', 'пористость', 'глубина, м',
                     os.path.join(pics_folder, 'phi_png'))
