import numpy as np
import modified_fit
import warnings
import sys


# Returns chi^2_train and chi^2_test from group validation with the best normalizations,
def perform_group_val_with_best_normalizations(order, reg_param, n_norm_params, data):
    # Mirrors behavior in original modified_fit.py lines 173-180
    best_params, _, _, _, _ = modified_fit.fit(data, data, order, reg_param)
    normalizations = best_params[:n_norm_params]
    return modified_fit.group_validation(data, order, normalizations, reg_param)


# Performs cross validation without regularization, then scans for optimal lambda, then performs cross validation with
# optimal lambda. Does the latter 2 steps only for N > n where n is the order at which chi^2_test was minimized.
def replicate_table_i(data, n_norm_params, max_order, lambdas):
    '''Perform 18-fold group cross-validation WITHOUT regularization'''
    # Creates an array of N values with corresponding chi^2 values
    cross_val_no_reg = np.empty((max_order, 3))  # array will have 8 rows of length 3: [N,chi^2_train,chi^2_test]
    for counter in range(max_order):
        # For each row, assign order and both chi^2 values
        order = counter + 1
        cross_val_no_reg[counter][0] = order
        cross_val_no_reg[counter][1], cross_val_no_reg[counter][2] = \
            perform_group_val_with_best_normalizations(order, 0, n_norm_params, data)

    # Finds the order n at which chi^2_test reaches a min, indicating under-fitting for N < n and over-fitting for N > n
    chi_sq_tests = []
    for counter in range(max_order):
        chi_sq_tests.append(int(cross_val_no_reg[counter][2]))
    min_order = chi_sq_tests.index(min(chi_sq_tests)) + 1  # (the N at which chi^2 test reaches a min)
    num_modeled_orders = max_order - min_order + 1

    '''Scan for optimal tikhonov regularization parameter for each order >= min_order (to control over-fitting)'''
    orders_to_scan = np.linspace(min_order, max_order, num=num_modeled_orders).astype(int)
    optimal_reg_params = []  # corresponds to the orders in orders_to_scan
    for order in orders_to_scan:
        # Initial values for min
        min_reg_param = 0
        min_chi_sq_test = perform_group_val_with_best_normalizations(order, 0, n_norm_params, data)[1]
        min_lambda = None  # Purely for the warning

        for best_reg_param in lambdas:
            chi_sq_test = perform_group_val_with_best_normalizations(order, best_reg_param, n_norm_params, data)[1]
            # Warns if two chi^2_test values are the same
            if chi_sq_test == min_chi_sq_test and min_lambda is not None:
                warning_message = str("Two chi^2_test values before rounding were equal during scan. Lambdas: " +
                                      str(best_reg_param) + " and " + str(min_lambda) + "; searching for order: " +
                                      str(order))
                warnings.warn(warning_message)
            if chi_sq_test < min_chi_sq_test:
                min_reg_param = best_reg_param
                min_chi_sq_test = chi_sq_test
                min_lambda = best_reg_param
        optimal_reg_params.append(min_reg_param)

    '''Performs group validation with optimized regularization'''
    cross_val_best_reg = np.empty((num_modeled_orders, 4),
                                  dtype=float)  # array will have x rows of length 4: [N,lambda, chi^2_train,chi^2_test]
    for counter, order in enumerate(orders_to_scan):
        best_reg_param = optimal_reg_params[counter]  # best for the given order, as found above
        cross_val_best_reg[counter][0] = round(order, 1)
        cross_val_best_reg[counter][1] = best_reg_param
        cross_val_best_reg[counter][2], cross_val_best_reg[counter][3] \
            = perform_group_val_with_best_normalizations(order, best_reg_param, n_norm_params, data)

    return cross_val_no_reg, min_order, orders_to_scan, optimal_reg_params, cross_val_best_reg


# Trains N > n with optimized regularization on the full data set
def replicate_table_ii(orders_included, optimal_reg_params, data, n_norm_params):
    num_modeled_orders = len(orders_included)
    table2 = np.empty((num_modeled_orders, 6),
                      dtype=float)  # array will have x rows of length 6: [N,lambda, L, chi^2, b^2_1, r_E]
    for counter, order in enumerate(orders_included):
        best_reg_param = optimal_reg_params[counter]
        # Mirrors behavior in original modified_fit.py lines 173-192
        # Fit the full dataset:
        best_params, chi2, _, L, cov = modified_fit.fit(data, data, order, best_reg_param)
        fit_params = best_params[n_norm_params:]
        fit_cov = cov[n_norm_params:, n_norm_params:]
        # Extract the radii:
        b2, b2_sigma = modified_fit.get_b2(fit_params, fit_cov)
        radius, radius_stat = modified_fit.get_radius(b2, b2_sigma)

        table2[counter] = round(order, 1), best_reg_param, L, chi2, b2, radius
    return table2


# Prints replications of Tables I and II to the console with corollary information
def print_table_replications(max_N, lambdas, data_filepath, data_file_name):
    print("\n\n\nDATA FILE: ", data_file_name)

    data = modified_fit.read_cs_data(data_filepath)  # This will change N_NORM_PARAMS, BEAM_ENERGIES, and SPECTROMETERS
    # to the correct values in modified_fit.py. Avoid running programs in parallel to ensure conservation of constants.
    n_norm_params = modified_fit.N_NORM_PARAMS

    cross_val_no_reg, min_order, orders_to_scan, optimal_reg_params, cross_val_best_reg \
        = replicate_table_i(data, n_norm_params, max_N, lambdas)
    table2 = replicate_table_ii(orders_to_scan, optimal_reg_params, data, n_norm_params)

    print("Cross-validation without regularization: [N, chi^2_train, chi^2_test]\n", cross_val_no_reg, "\n")
    print("Value of N, n, at which chi^2_test was minimized: ", min_order, "\n")
    print("Regularization parameters searched: ", lambdas, "\n")
    print("For orders: ", orders_to_scan, ", the optimized regularization parameters were: ", optimal_reg_params, "\n")
    print("Cross-validation with optimized regularization: [N, lambda, chi^2_train, chi^2_test]\n",
          cross_val_best_reg, "\n")
    print("For orders N > n, fit with optimized regularization on the full data set: [N, lambda, L, chi^2, b^2_1, r_E] "
          "\n", table2)


def main():
    sys.stdout = open("output.txt", "w")

    '''Run 1: A1'''
    max_order = 9
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/a1_cs.dat"
    print_table_replications(max_order, lambdas, data_filepath, "A1")

    '''Run 2: Rebinned A1'''
    max_order = 9
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/rebinned_a1_cs.dat"
    print_table_replications(max_order, lambdas, data_filepath, "Rebinned A1")

    '''Run 3: A1 + PRad'''
    max_order = 9
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/a1_with_prad_cs.dat"
    print_table_replications(max_order, lambdas, data_filepath, "A1 + PRad")

    '''Run 4: Rebinned + PRad'''
    max_order = 9
    lambdas = np.linspace(0, 1.5, 151)
    data_filepath = "data/rebinned_with_prad_cs.dat"
    print_table_replications(max_order, lambdas, data_filepath, "Rebinned + PRad")

    sys.stdout.close()


if __name__ == "__main__":
    main()
