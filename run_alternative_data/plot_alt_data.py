import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit
from models import calc_cs, calc_ffs, calc_ge_gm, calc_rho, dipole_ffs, get_b2, hbarc
from plot import fill_between, plot_ge_gm, calc_interval


def calc_ge_over_gm(GE, GM):
    """Calculate GE / GM"""
    return GE / GM


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 26
matplotlib.rcParams["font.family"] = "lmodern"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20

# Number of samples to use when generating statistical uncertainty bands
N_SAMPLES = 1000


def plot_data_set(cs_data, order, reg_param, Q2_max):
    '''Plot results from the transverse charge density analysis'''
    GE, GM, Q2_range, interval, f1_up, f1_low, f2_up, f2_low = plot_ge_gm(cs_data, order, reg_param, Q2_max=Q2_max)

    # Plot the best-fit line for G_E*mu/G_M
    plt.plot(Q2_range, GE / GM, color="black", lw=1, alpha=0.7, label="Gramolin")

    # Find and plot the statistical uncertainties
    # Calculate the statistical uncertainties using standard propagation
    # interval = calc_interval()  this is the alt to find just 68%, other alt is find sigmas differently
    ge_up = interval[1, 0]
    ge_low = interval[0, 0]
    gm_up = interval[1, 1]
    gm_low = interval[0, 1]
    sigma_ge = (ge_up - ge_low) / 2
    sigma_gm = (gm_up - gm_low) / 2
    sigma_f = (GE / GM) * np.sqrt((sigma_ge / GE) ** 2 + (sigma_gm / GM) ** 2)
    f_stat_up = (GE / GM) + sigma_f
    f_stat_low = (GE / GM) - sigma_f
    # Plot the statistical band
    fill_between(Q2_range, f_stat_up, f_stat_low, "#FFAAAA")

    # Find and plot the systematic uncertainties
    # Calculate the systematic uncertainties using standard propagation
    f_sys_up = (GE / GM) * np.sqrt((f1_up / GE) ** 2 + (f2_up / GM) ** 2)
    f_sys_low = (GE / GM) * np.sqrt((f1_low / GE) ** 2 + (f2_low / GM) ** 2)
    # Plot the systematic bands
    fill_between(Q2_range, f_stat_up + f_sys_up, f_stat_up, "red")
    fill_between(Q2_range, f_stat_low, f_stat_low - f_sys_low, "red")
    '''
    fill_between(Q2_range, GE/(f2_up+interval[1,1]), GE/(-f2_low+interval[0,1]), "red")
    fill_between(Q2_range, (f1_up+interval[1,0])/GM, (-f1_low+interval[0,0])/GM , "red")#,label="Gramolin")
    
    fill_between(Q2_range, GE/(interval[1,1]), GE/(interval[0,1]), "#FFAAAA")
    fill_between(Q2_range, (interval[1,0])/GM, (interval[0,0])/GM , "#FFAAAA")
    '''

    '''Plot asymmetric data'''
    # Display choices
    studies = ["asymdata/Punjabi.dat", "asymdata/Paolone.dat", "asymdata/Crawford.dat", "asymdata/Zhan.dat"]
    colors = ["blue", "purple", "darkorange", "green"]
    fmts = ["^", "o", "s", "v"]

    # How statistical and systematic error are combined to make error bars
    def combine_stat_sys_error(stat_error, sys_error):
        return stat_error + sys_error

    # Plot the asymmetry data
    cols = {
        0: "Q2",  # Four-momentum transfer squared (in GeV^2)
        1: "ff_ratio",  # GE/GM
        2: "stat_error",  # statistical uncertainty
        3: "sys_error",  # systematic uncertainty
    }
    for idx, study_name in enumerate(studies):
        # Create a dictionary holding data from a specific study,
        study = pd.read_csv(study_name, sep=" ", skiprows=1, usecols=cols.keys(), names=cols.values())
        # Plot the data
        plt.plot(study["Q2"], study["ff_ratio"], fmts[idx], color=colors[idx], label=studies[idx][9:-4], ms=8)
        plt.errorbar(study["Q2"], study["ff_ratio"], xerr=0,
                     yerr=combine_stat_sys_error(study["stat_error"], study["sys_error"]),
                     fmt=fmts[idx], color=colors[idx], lw=2, ms=8, zorder=666)


fig = plt.figure(figsize=(10, 3.5))  # fig = plt.figure(dpi=150,figsize=(12,6))

'''OG data'''
fig.add_subplot(1, 2, 1)

# parameters
cs_data = fit.read_cs_data("data/CrossSections.dat")[0]
order = 5
reg_param = 0.02
Q2_max = 1.4

# Axes and limits
plot_data_set(cs_data, order, reg_param, Q2_max)
plt.ylim(0., 1.05)
plt.xlim(0, Q2_max)
plt.ylabel(r"$\mu$ $G_{E}$/$G_{M}$")
plt.xlabel(r"Q$^2$ [GeV/c]$^2$")
plt.legend(frameon=False,handletextpad=0.5)


'''Rebinned Data'''
fig.add_subplot(1, 2, 2)

# parameters
cs_data = fit.read_cs_data("data/RebinnedCrossSectionsData.dat")[0]
order = 5
reg_param = 0.005
Q2_max = 1.4

# Axes and limits
plot_data_set(cs_data, order, reg_param, Q2_max)
plt.ylim(0., 1.05)
plt.xlim(0, Q2_max)
plt.ylabel(r"$\mu$ $G_{E}$/$G_{M}$")
plt.xlabel(r"Q$^2$ [GeV/c]$^2$")
plt.legend(frameon=True,handletextpad=0.5)


'''OG+PRad Data'''
fig.add_subplot(2, 2, 3)

# parameters
cs_data = fit.read_cs_data("data/OG+PRadCrossSectionsData.dat")[0]
order = 6
reg_param = 0.22
Q2_max = 1.4

# Axes and limits
plot_data_set(cs_data, order, reg_param, Q2_max)
plt.ylim(0., 1.05)
plt.xlim(0, Q2_max)
plt.ylabel(r"$\mu$ $G_{E}$/$G_{M}$")
plt.xlabel(r"Q$^2$ [GeV/c]$^2$")
plt.legend(frameon=True,handletextpad=0.5)


'''Rebinned+PRad Data'''
fig.add_subplot(2, 2, 4)

# parameters
cs_data = fit.read_cs_data("data/Rebinned+PRadCrossSectionsData.dat")[0]
order = 6
reg_param = 0.1
Q2_max = 1.4

# Axes and limits
plot_data_set(cs_data, order, reg_param, Q2_max)
plt.ylim(0., 1.05)
plt.xlim(0, Q2_max)
plt.ylabel(r"$\mu$ $G_{E}$/$G_{M}$")
plt.xlabel(r"Q$^2$ [GeV/c]$^2$")
plt.legend(frameon=True,handletextpad=0.5)


# plt.tight_layout()
# plt.savefig("OriginalRatio.png")
plt.show()
