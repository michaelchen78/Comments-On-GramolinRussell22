import numpy as np

run_name = "fndgnxfh"
cross_val_no_reg = np.empty((9,9),dtype=float).fill(25)
optimal_reg_params = [2,213,12,31,41,34,134,13,51,345]

with open('output_txts/{name}.txt'.format(name=run_name), 'w') as f:
    f.write("Cross-validation without regularization: \n")
    f.write(str(cross_val_no_reg))
    f.write("Value of N, n, at which chi^2_test was minimized: \n")
    f.write(str(5))
    f.write("Optimized regularization parameters: \n")
    f.write(str(optimal_reg_params))
    f.write(str(optimal_reg_params))
    f.write("Cross-validation with optimized regularization: \n")
    f.write(str(cross_val_no_reg))
    f.write("Table II: \n")
    f.write(str(25))

















'''
N_NORM_PARAMS = -1
max_N = 8  # change for small Prad. N must always start at 1, and go to max_N

data_filepath = "data/CrossSections.dat"
data = fit.read_cs_data(data_filepath)  # this changes N_NORM_PARAMS to the appropriate value
print(N_NORM_PARAMS)
order = -1
tikhonov_regularization_parameter = -1


# create array of N values with corresponding chi^2 values
cross_val_no_reg = np.empty((max_N, 3), dtype=int)  # array will have 8 rows of length 3: [N,chi^2_train,chi^2_test]
for counter in range(max_N):
    order = counter + 1
    best_params, chi2, _, L, cov = fit.fit(data, data, order, 0)
    normalizations = best_params[:N_NORM_PARAMS]  # return best normalizations

    cross_val_no_reg[counter][0] = order
    cross_val_no_reg[counter][1], cross_val_no_reg[counter][2] = fit.group_validation(data, order, normalizations, 0)
#print(cross_val_no_reg)
# find the order n at which chi^2_test reaches a minimum, indicating under-fitting for N < n and over-fitting for N > n
chi_sq_tests = []
for counter in range(max_N): chi_sq_tests.append(int(cross_val_no_reg[counter][2]))
min_N = chi_sq_tests.index(min(chi_sq_tests)) + 1  # (the N at which chi^2 test reaches a min)
#print("min_N: ", min_N)
num_modeled_orders = max_N - min_N + 1

''''''scan for optimal tikhonov regularization parameter for each order >= min_N (to control over-fitting)'''
'''orders_to_scan = np.linspace(min_N, max_N, num=num_modeled_orders)
orders_to_scan = orders_to_scan.astype(int)
#print(orders_to_scan)
optimal_reg_params = []  # corresponds to the orders in orders_to_scan
for order in orders_to_scan:
    # lambdas = np.linspace(0.00, 0.10, num=11)
    lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #print("ORDER ON: ", order)
    min_reg_param = 0
    best_params, chi2, _, L, cov = fit.fit(data, data, order, 0)
    normalizations = best_params[:N_NORM_PARAMS]  # return best normalizations
    min_chi_sq_test = fit.group_validation(data, order, normalizations, 0)[1]

    for best_reg_param in lambdas:  # doesn't take into account equal chi^2_test for two diff lambdas...
        best_params, chi2, _, L, cov = fit.fit(data, data, order, best_reg_param)
        normalizations = best_params[:N_NORM_PARAMS]  # return best normalizations

        chi_sq_test = fit.group_validation(data, order, normalizations, best_reg_param)[1]
        if chi_sq_test < min_chi_sq_test:
            min_reg_param = best_reg_param
            min_chi_sq_test = chi_sq_test
    #print("min lambda: ", min_reg_param)
    optimal_reg_params.append(min_reg_param)

print("RIGHT HALF: ")
'''


'''group validation WITH regularization
# right half of TABLE I
cross_val_best_reg = np.empty((num_modeled_orders, 4),
                              dtype=float)  # array will have x rows of length 4: [N,lambda, chi^2_train,chi^2_test]
for counter, order in enumerate(orders_to_scan):
    best_reg_param = optimal_reg_params[counter]  # best for the given order, as found above

    cross_val_best_reg[counter][0] = round(order, 1)  # since the array is of type float, make it more readable
    cross_val_best_reg[counter][1] = best_reg_param

    best_params, chi2, _, L, cov = fit.fit(data, data, order, best_reg_param)
    normalizations = best_params[:N_NORM_PARAMS]  # return best normalizations
    cross_val_best_reg[counter][2], cross_val_best_reg[counter][3] \
        = fit.group_validation(data, order, normalizations, best_reg_param)
#print(cross_val_best_reg)

print("TABLE2: ")
print(num_modeled_orders)
print(orders_to_scan)
print(optimal_reg_params)
'''

'''print TABLE II -- train Ns on full dataset
table2 = np.empty((num_modeled_orders, 6),
                  dtype=float)  # array will have x rows of length 6: [N,lambda, L, chi^2, b^2_1, r_E]
for counter, order in enumerate(orders_to_scan):
    best_reg_param = optimal_reg_params[counter]
    data = fit.read_cs_data("data/CrossSections.dat")

    print(order)
    print(best_reg_param)    # Fit the full dataset:
    best_params, chi2, _, L, cov = fit.fit(data, data, order, best_reg_param)
    print("best_params", best_params)

    normalizations = best_params[:N_NORM_PARAMS]
    print(normalizations)

    fit_params = best_params[N_NORM_PARAMS:]
    print(fit_params)

    fit_cov = cov[N_NORM_PARAMS:, N_NORM_PARAMS:]
    print(fit_cov)

    fit.print_fit_params(fit_params, fit_cov)

    # Extract the radii:
    b2, b2_sigma = fit.get_b2(fit_params, fit_cov)
    radius, radius_stat = fit.get_radius(b2, b2_sigma)

    table2[counter] = round(order,1), best_reg_param, L, chi2, b2, radius

print(table2)

'''