from working_folder.run_alternative_data import modified_fit
import numpy as np
from working_folder.run_alternative_data import asym_chi2


def find_fit_order_and_tikhonov_params(data_file_name, max_order, lambdas):
    print("PROGRESS:\n")

    print("performing cross validation without regularization...")
    '''Fit WITHOUT regularization'''
    # create array of N values with corresponding chi^2 values
    cross_val_no_reg = np.empty((max_order, 2))  # array will have 8 rows of length 2: [N,chi^2]
    for counter in range(max_order):
        order = counter + 1
        cross_val_no_reg[counter][0] = order
        cross_val_no_reg[counter][1] = asym_chi2.calc_chi2(data_file_name, order, 0)

    print("performing regularization parameter scan for each order N ")
    '''scan for optimal tikhonov regularization parameter for each order N'''
    optimal_reg_params = []  # corresponds to the orders in orders_to_scan
    for counter in range(max_order):
        order = counter + 1
        print("scanning for order: ", order)
        min_reg_param = 0
        min_chi_sq_test = asym_chi2.calc_chi2(data_file_name, order, 0)

        for best_reg_param in lambdas:  # doesn't take into account equal chi^2_test for two diff lambdas...
            chi_sq_test = asym_chi2.calc_chi2(data_file_name, order, best_reg_param)
            print(best_reg_param, " ", chi_sq_test)
            if chi_sq_test < min_chi_sq_test:
                min_reg_param = best_reg_param
                min_chi_sq_test = chi_sq_test
        optimal_reg_params.append(min_reg_param)

    print("printing asym chi^2 with the optimal reg params")
    '''reg param optim asym chi^2'''
    cross_val_best_reg = np.empty((max_order, 3),
                                  dtype=float)  # array will have x rows of length 3: [N,lambda, chi^2]
    for counter in range(max_order):
        order = counter + 1
        best_reg_param = optimal_reg_params[counter]  # best for the given order, as found above
        cross_val_best_reg[counter][0] = round(order, 1)
        cross_val_best_reg[counter][1] = best_reg_param
        cross_val_best_reg[counter][2] = asym_chi2.calc_chi2(data_file_name, order, best_reg_param)

    print("globals at end (n, b, s): ", modified_fit.N_NORM_PARAMS, fit.BEAM_ENERGIES, modified_fit.SPECTROMETERS)

    print("DONE!\n\n\n")
    return cross_val_no_reg, optimal_reg_params, cross_val_best_reg


# Train Ns on full data set
def get_table_ii(data_file_name, max_order, optimal_reg_params, n_norm_params):
    data = fit.read_cs_data(data_file_name)[0]
    table2 = np.empty((max_order, 6),
                      dtype=float)  # array will have x rows of length 6: [N,lambda, L, chi^2, b^2_1, r_E]
    for counter in range(max_order):
        order = counter + 1
        best_reg_param = optimal_reg_params[counter]

        # Fit the full dataset:
        best_params, chi2, _, L, cov = modified_fit.fit(data, data, order, best_reg_param)
        fit_params = best_params[n_norm_params:]
        fit_cov = cov[n_norm_params:, n_norm_params:]

        # Extract the radii:
        b2, b2_sigma = modified_fit.get_b2(fit_params, fit_cov)
        radius, radius_stat = modified_fit.get_radius(b2, b2_sigma)

        table2[counter] = round(order, 1), best_reg_param, L, chi2, b2, radius
    return table2


