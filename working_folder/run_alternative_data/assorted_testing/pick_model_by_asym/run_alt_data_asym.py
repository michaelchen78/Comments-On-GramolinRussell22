import numpy as np

from working_folder.run_alternative_data.assorted_testing.pick_model_by_asym import alt_data_methods_asym
from working_folder.run_alternative_data import modified_fit


def run_set_find_order_param_print(max_order, lambdas, data_filepath, run_name):
    print("\n\nRUN: ", run_name, "\n\n")
    _, n_norm_params = modified_fit.read_cs_data(
        data_filepath)  # this is meant to change N_NORM_PARAMS, BEAM_ENERGIES, SPECTROMETERS TO THE RIGHT VALUE in fit
    cross_val_no_reg, optimal_reg_params, cross_val_best_reg \
        = alt_data_methods_asym.find_fit_order_and_tikhonov_params(data_filepath, max_order, lambdas)
    table2 = alt_data_methods_asym.get_table_ii(data_filepath, max_order, optimal_reg_params, n_norm_params)
    print("Cross-validation without regularization: \n", cross_val_no_reg)
    print("Optimized regularization parameters: \n", max_order, optimal_reg_params)
    print("Cross-validation with optimized regularization: \n", cross_val_best_reg)
    print("Table II: \n", table2)
    with open('{name}_text_file.txt'.format(name=run_name), 'a+') as f:
        f.write("Cross-validation without regularization: \n")
        f.write(str(cross_val_no_reg))
        f.write("Optimized regularization parameters: \n")
        f.write(str(max_order))
        f.write(str(optimal_reg_params))
        f.write("Cross-validation with optimized regularization: \n")
        f.write(str(cross_val_best_reg))
        f.write("Table II: \n")
        f.write(str(table2))

def main():
    '''run 1: OG data'''
    max_N = 9  # change for small Prad. N must always start at 1, and go to max_N
    lambdas = np.linspace(0, 1000, 21)
    data_filepath = "../../data/CrossSections.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 1 (1): OG DATA")

    '''run 2: rebinned+PRad'''
    max_N = 9
    lambdas = np.linspace(0, 1000, 21)
    data_filepath = "../../data/Rebinned+PRadCrossSectionsData.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 2 (1): Rebinned+PRad")

    '''run 3: rebinned'''
    max_N = 9
    lambdas = np.linspace(0, 1000, 21)
    data_filepath = "../../data/RebinnedCrossSectionsData.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 3 (1): Rebinned")

    '''run 4: OG + PRad'''
    max_N = 10
    lambdas = np.linspace(0, 1000, 21)
    data_filepath = "../../data/OG+PRadCrossSectionsData.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 4 (1): OG+PRad")


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()