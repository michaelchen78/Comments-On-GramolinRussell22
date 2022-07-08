# NOTES: (for Michael)
# group validation is always run with norms because we want to measure accuracy of model, and this must be to specific
# data, not data with parametrized normalizations. but the best norms are with the whole data set/model
# I guess I just don't get how cross validations work...
# beware of the BEAM ENERGIES
import alt_data_methods
from run_alternative_data import fit


def run_set_find_order_param_print(max_N, lambdas, data_filepath, run_name):
    print("\n\nRUN: ", run_name, "\n\n")
    data, n_norm_params = fit.read_cs_data(
        data_filepath)
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


'''run 1: OG data'''
max_N = 8  # change for small Prad. N must always start at 1, and go to max_N
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
data_filepath = "data/CrossSections.dat"
run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 1: OG DATA")

'''run 2: rebinned+PRad'''
max_N = 9
lambdas = [0.09, 0.10, 0.11, 0.20, 0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.45, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57]
data_filepath = "data/Rebinned+PRadCrossSectionsData.dat"
run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 2: Rebinned+PRad")

'''run 3: rebinned'''
max_N = 8
lambdas = [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.028, 0.03,0.032,  0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,0.11, 0.115, 0.116, 0.117, 0.118, 0.119,0.12,0.121, 0.122, 0.123, 0.124, 0.13]
data_filepath = "data/RebinnedCrossSectionsData.dat"
run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 3: Rebinned")

'''run 4: OG + PRad'''
max_N = 9
lambdas = [0.16,0.17, 0.18,0.19, 0.20, 0.21, 0.22, 0.23,0.24, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60,0.61, 0.62, 0.63, 0.64, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2]
data_filepath = "data/OG+PRadCrossSectionsData.dat"
run_set_find_order_param_print(max_N, lambdas, data_filepath, "RUN 4: OG+PRad")

