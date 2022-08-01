# NOTES: (for Michael)
# group validation is always run with norms because we want to measure accuracy of model, and this must be to specific
# data, not data with parametrized normalizations. but the best norms are with the whole data set/model
# I guess I just don't get how cross validations work...
# beware of the BEAM ENERGIES
# CHECK to make sure beam energies and spectrometers are the right values
# figure out what happened why train was different but test the same
# make sure it is all okay, especially the beam energies and spectrometers
import numpy as np

import alt_data_methods
from working_folder.run_alternative_data import fit


def run_set_find_order_param_print(max_N, lambdas, data_filepath, run_name):
    print("\n\nRUN: ", run_name, "\n\n")
    data, n_norm_params = fit.read_cs_data(
        data_filepath)  # this is meant to change N_NORM_PARAMS, BEAM_ENERGIES, SPECTROMETERS TO THE RIGHT VALUE in fit
    cross_val_no_reg, min_N, orders_to_scan, optimal_reg_params, cross_val_best_reg \
        = alt_data_methods.find_fit_order_and_tikhonov_params(data, n_norm_params, max_N, lambdas)
    table2 = alt_data_methods.get_table_ii(orders_to_scan, optimal_reg_params, data, n_norm_params)
    print("Cross-validation without regularization: \n", cross_val_no_reg)
    print("Value of N, n, at which chi^2_test was minimized: ", min_N)
    print("Optimized regularization parameters: \n", orders_to_scan, optimal_reg_params)
    print("Cross-validation with optimized regularization: \n", cross_val_best_reg)
    print("Table II: \n", table2)
    with open('{name}_text_file.txt'.format(name=run_name), 'a+') as f:
        f.write("Cross-validation without regularization: \n")
        f.write(str(cross_val_no_reg))
        f.write("Value of N, n, at which chi^2_test was minimized: \n")
        f.write(str(min_N))
        f.write("Optimized regularization parameters: \n")
        f.write(str(orders_to_scan))
        f.write(str(optimal_reg_params))
        f.write("Cross-validation with optimized regularization: \n")
        f.write(str(cross_val_best_reg))
        f.write("Table II: \n")
        f.write(str(table2))


def main():
    '''run 1: OG data'''
    max_N = 9  # change for small Prad. N must always start at 1, and go to max_N
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/CrossSections.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 1: OG DATA")

    '''run 2: rebinned+PRad'''
    max_N = 9
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/Rebinned+PRadCrossSectionsData.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 2: Rebinned+PRad")

    '''run 3: rebinned'''
    max_N = 9
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/RebinnedCrossSectionsData.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 3: Rebinned")

    '''run 4: OG + PRad'''
    max_N = 10
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/OG+PRadCrossSectionsData.dat"
    run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 4: OG+PRad")


if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()