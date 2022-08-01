from working_folder.run_alternative_data import fit
import numpy as np


# params: the file path, the max order tested, the range of lambdas scanned
# returns: cross val w/o regularization, order@ min chi2, scanned orders, optimal_reg_params, cross val w regularization
def find_fit_order_and_tikhonov_params(data, n_norm_params, max_N, lambdas):
    print("PROGRESS:\n")

    print("globals at start (n, b, s): ", fit.N_NORM_PARAMS, fit.BEAM_ENERGIES, fit.SPECTROMETERS)

    print("performing cross validation without regularization...")
    '''Perform 18-fold group cross-validation WITHOUT regularization'''
    # create array of N values with corresponding chi^2 values
    cross_val_no_reg = np.empty((max_N, 3), dtype=int)  # array will have 8 rows of length 3: [N,chi^2_train,chi^2_test]
    for counter in range(max_N):
        order = counter + 1
        cross_val_no_reg[counter][0] = order
        cross_val_no_reg[counter][1], cross_val_no_reg[counter][2] = \
            perform_group_val_with_best_normalizations(order, 0, n_norm_params, data)

    # find the order n at which chi^2_test reaches a min, indicating under-fitting for N < n and over-fitting for N > n
    chi_sq_tests = []
    for counter in range(max_N): chi_sq_tests.append(int(cross_val_no_reg[counter][2]))
    min_N = chi_sq_tests.index(min(chi_sq_tests)) + 1  # (the N at which chi^2 test reaches a min)
    num_modeled_orders = max_N - min_N + 1

    print("performing regularization parameter scan for each order N >= n...")
    '''scan for optimal tikhonov regularization parameter for each order >= min_N (to control over-fitting)'''
    orders_to_scan = np.linspace(min_N, max_N, num=num_modeled_orders).astype(int)
    optimal_reg_params = []  # corresponds to the orders in orders_to_scan
    for order in orders_to_scan:
        print("scanning for order: ", order)
        min_reg_param = 0
        min_chi_sq_test = perform_group_val_with_best_normalizations(order, 0, n_norm_params, data)[1]

        for best_reg_param in lambdas:  # doesn't take into account equal chi^2_test for two diff lambdas...
            chi_sq_test = perform_group_val_with_best_normalizations(order, best_reg_param, n_norm_params, data)[1]
            if chi_sq_test < min_chi_sq_test:
                min_reg_param = best_reg_param
                min_chi_sq_test = chi_sq_test
        optimal_reg_params.append(min_reg_param)

    print("performing group validation with optimized regularization...")
    '''group validation WITH regularization'''
    cross_val_best_reg = np.empty((num_modeled_orders, 4),
                                  dtype=float)  # array will have x rows of length 4: [N,lambda, chi^2_train,chi^2_test]
    for counter, order in enumerate(orders_to_scan):
        best_reg_param = optimal_reg_params[counter]  # best for the given order, as found above
        cross_val_best_reg[counter][0] = round(order, 1)
        cross_val_best_reg[counter][1] = best_reg_param
        cross_val_best_reg[counter][2], cross_val_best_reg[counter][3] \
            = perform_group_val_with_best_normalizations(order, best_reg_param, n_norm_params, data)

    print("globals at end (n, b, s): ", fit.N_NORM_PARAMS, fit.BEAM_ENERGIES, fit.SPECTROMETERS)

    print("DONE!\n\n\n")
    return cross_val_no_reg, min_N, orders_to_scan, optimal_reg_params, cross_val_best_reg


# Train Ns on full data set
def get_table_ii(orders_included, optimal_reg_params, data, n_norm_params):
    num_modeled_orders = len(orders_included)
    table2 = np.empty((num_modeled_orders, 6),
                      dtype=float)  # array will have x rows of length 6: [N,lambda, L, chi^2, b^2_1, r_E]
    for counter, order in enumerate(orders_included):
        best_reg_param = optimal_reg_params[counter]

        # Fit the full dataset:
        best_params, chi2, _, L, cov = fit.fit(data, data, order, best_reg_param)
        fit_params = best_params[n_norm_params:]
        fit_cov = cov[n_norm_params:, n_norm_params:]

        # Extract the radii:
        b2, b2_sigma = fit.get_b2(fit_params, fit_cov)
        radius, radius_stat = fit.get_radius(b2, b2_sigma)

        table2[counter] = round(order, 1), best_reg_param, L, chi2, b2, radius
    return table2


# returns the best normalizations, used before each time cross validation is executed (helper method)
def perform_group_val_with_best_normalizations(order, reg_param, n_norm_params, data):
    best_params, chi2, _, L, cov = fit.fit(data, data, order, reg_param)
    normalizations = best_params[:n_norm_params]  # return best normalizations
    return fit.group_validation(data, order, normalizations, reg_param)
