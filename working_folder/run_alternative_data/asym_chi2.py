import pandas as pd
import numpy as np
import fit
import plot
from plot import plot_ge_gm
import models

cols = {
    0: "Q2",
    1: "ff_ratio",
    2: "stat_error",
    3: "sys_error",
}

max_Q2 = 3.47
data = pd.read_csv("asymdata/combined_asym.dat", sep=" ", skiprows=1, usecols=cols.keys(), names=cols.values())

# remove row if Q2 > max_Q2
drop_index = []
for idx, Q2 in enumerate(data["Q2"]):
    if Q2 > max_Q2: drop_index.append(idx)
data.drop(labels=drop_index, axis=0, inplace=True)


def read_data(data_file_name, order, reg_param):
    cs_data = fit.read_cs_data(data_file_name)[0]
    params, _ = plot.calc_params(cs_data, order, reg_param)
    Q2_range = np.linspace(0, max_Q2, int(max_Q2 * 1000 + 1))  # [MOD HERE]
    ge, gm = models.calc_ge_gm(Q2_range, params, order)
    return Q2_range, ge, gm


def combine_stat_sys_error(stat_error, sys_error):
        return np.sqrt(stat_error**2 + sys_error**2)


def calc_chi2(data_file_name, order, reg_param):
    Q2_range, ge, gm = read_data(data_file_name, order, reg_param)
    model = ge/gm
    for idx, dumb_number in enumerate(Q2_range): Q2_range[idx] = round(dumb_number, 3)
    assert len(model) == len(Q2_range)

    chi2 = 0
    for idx, row in data.iterrows():
        Q2 = row['Q2']
        Q2_idx = np.where(Q2_range == Q2)  # rounds?
        assert len(Q2_idx) == 1
        Q2_idx = Q2_idx[0][0]

        asym_ff_ratio = row['ff_ratio']
        asym_error = combine_stat_sys_error(row['stat_error'], row['sys_error'])
        model_ff_ratio = model[int(Q2_idx)]
        chi2_term = ((model_ff_ratio-asym_ff_ratio)**2) / (asym_error**2)
        chi2 += chi2_term

    return chi2


def main():
    data_file_name, order, reg_param = "data/CrossSections.dat", 5, 0.02
    print("OG: ", calc_chi2(data_file_name, order, reg_param))

    data_file_name, order, reg_param = "data/RebinnedCrossSectionsData.dat", 5, 0.01
    print("Rebinned: ", calc_chi2(data_file_name, order, reg_param))

    data_file_name, order, reg_param = "data/OG+PRadCrossSectionsData.dat", 7, 0.63
    print("OG+PRad: ", calc_chi2(data_file_name, order, reg_param))

    data_file_name, order, reg_param = "data/Rebinned+PRadCrossSectionsData.dat", 6, 0.1
    print("Rebinned+PRad: ", calc_chi2(data_file_name, order, reg_param))


if __name__ == "__main__":
    main()
