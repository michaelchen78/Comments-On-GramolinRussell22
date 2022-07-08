# NOTES: (for Michael)
# group validation is always run with norms because we want to measure accuracy of model, and this must be to specific
# data, not data with parametrized normalizations. but the best norms are with the whole data set/model
# I guess I just don't get how cross validations work...
# beware of the BEAM ENERGIES
import alt_data_methods
from run_alternative_data import fit_forprad
import run_alt_data

def run_set_find_order_param_print_for_prad(max_N, lambdas, data_filepath, run_name):
    print("\n\nRUN: ", run_name, "\n\n")
    data, n_norm_params = fit_forprad.read_cs_data(
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

'''run 5: PRad'''
max_N = 4
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11,0.12,0.13,0.14,0.15,0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9]
data_filepath = "data/PRadAlone.dat"
run_set_find_order_param_print_for_prad(max_N, lambdas, data_filepath, "RUN 5: PRad Alone")