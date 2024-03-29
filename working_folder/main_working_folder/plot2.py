"""Plotting for manuscript figures."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit
from models import calc_cs, calc_ffs, calc_ge_gm, calc_rho, dipole_ffs, get_b2, hbarc

from scipy.stats import shapiro

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 26
matplotlib.rcParams["font.family"] = "lmodern"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20

# Number of samples to use when generating statistical uncertainty bands
N_SAMPLES = 1000


def read_Rosenbluth_data():
    """Read data for G_E and G_M from "Rosenbluth.dat"."""
    col_names = ["Q2", "GE", "delta_GE", "GM", "delta_GM"]
    data = pd.read_csv("data/Rosenbluth.dat", sep=" ", skiprows=5, names=col_names)
    return data


def calc_interval(calc_func, x_range, param_list, order):
    """Calculate 68% ("1 sigma") percentile interval from param sample."""
    out = np.array([calc_func(x_range, param, order) for param in param_list])
    # print("INTERVA", np.percentile(out, (15.9, 84.1), 0))
    # print(out.shape)
    # print(np.percentile(out, (15.9, 84.1), 0).shape)

    '''
    over_90 = False
    counter = 0
    while not over_90:
        Q2_range = np.linspace(0, 1.4, 200)
        yes_Gaussian = 0
        no_Gaussian = 0
        not_Gaussians = []
        for q2 in range(len(Q2_range)):
            ge_s_at_q2 = np.empty((1,1000))
            for idx, sample in enumerate(out):
                ge_s = sample[0]
                ge_s_at_q2[0][idx] = ge_s[q2]
            stat, p = shapiro(ge_s_at_q2)
            # print('stat=%.3f, p=%.3f\n' % (stat, p))
            if p > 0.05:
                yes_Gaussian += 1
            else:
                no_Gaussian += 1
                not_Gaussians.append(p)
        # print(yes_Gaussian, no_Gaussian)
        # print(not_Gaussians)
        if yes_Gaussian >= 180:
            over_90 = True
        counter += 1
    print(counter)
    print(yes_Gaussian, no_Gaussian)
    print(not_Gaussians)
    '''


    # the following code demonstrates the shape of each line of G_Es (at a specific Q^2)
    num = 1
    Q2_range = np.linspace(0, 1.4, 141)
    q2_value = Q2_range[num]
    samples = np.linspace(0, 999, 1000)
    y = np.empty((1, 1000))
    for idx, line in enumerate(out):
        y[0][idx] = line[0][num]
    y = np.sort(y)
    plt.plot(samples, y[0])


    return np.percentile(out, (15.9, 84.1), 0)


def calc_params(data, order, reg_param):
    """Run fit and get model parameters and covariance."""
    params, _, _, _, cov = fit.fit(data, data, order, reg_param)
    params = params[fit.N_NORM_PARAMS :]
    cov = cov[fit.N_NORM_PARAMS :, fit.N_NORM_PARAMS :]
    return params, cov


def calc_sys_bands(calc_func, x_range, data, order, reg_param):
    """Calculate systematic error bands for given quantity."""
    params, _ = calc_params(data, order, reg_param)
    f1, f2 = calc_func(x_range, params, order)
    mincut_params = fit.fit_systematic_variant("cs_mincut", data, order, reg_param)[0]
    maxcut_params = fit.fit_systematic_variant("cs_maxcut", data, order, reg_param)[0]
    sysup_params = fit.fit_systematic_variant("cs_sysup", data, order, reg_param)[0]
    syslow_params = fit.fit_systematic_variant("cs_syslow", data, order, reg_param)[0]
    mincut_f1, mincut_f2 = calc_func(x_range, mincut_params, order)
    maxcut_f1, maxcut_f2 = calc_func(x_range, maxcut_params, order)
    sysup_f1, sysup_f2 = calc_func(x_range, sysup_params, order)
    syslow_f1, syslow_f2 = calc_func(x_range, syslow_params, order)
    # Calculate upper and lower limits for each of the systematic variations:
    f1_cut_up = np.clip(np.max(np.stack([mincut_f1 - f1, maxcut_f1 - f1]), 0), 0, None)
    f1_cut_low = np.clip(np.min(np.stack([mincut_f1 - f1, maxcut_f1 - f1]), 0), None, 0)
    f1_sys_up = np.clip(np.max(np.stack([sysup_f1 - f1, syslow_f1 - f1]), 0), 0, None)
    f1_sys_low = np.clip(np.min(np.stack([sysup_f1 - f1, syslow_f1 - f1]), 0), None, 0)
    f2_cut_up = np.clip(np.max(np.stack([mincut_f2 - f2, maxcut_f2 - f2]), 0), 0, None)
    f2_cut_low = np.clip(np.min(np.stack([mincut_f2 - f2, maxcut_f2 - f2]), 0), None, 0)
    f2_sys_up = np.clip(np.max(np.stack([sysup_f2 - f2, syslow_f2 - f2]), 0), 0, None)
    f2_sys_low = np.clip(np.min(np.stack([sysup_f2 - f2, syslow_f2 - f2]), 0), None, 0)
    # Add two systematic "errors" in quadrature:
    f1_up = np.sqrt(f1_cut_up ** 2 + f1_sys_up ** 2)
    f1_low = np.sqrt(f1_cut_low ** 2 + f1_sys_low ** 2)
    f2_up = np.sqrt(f2_cut_up ** 2 + f2_sys_up ** 2)
    f2_low = np.sqrt(f2_cut_low ** 2 + f2_sys_low ** 2)
    return f1_up, f1_low, f2_up, f2_low


def fill_between(x_range, y_up, y_low, color, hbarc_scale=False):
    """Plot confidence interval."""
    if hbarc_scale:
        x_range = hbarc * x_range
        y_up = y_up / (hbarc * hbarc)
        y_low = y_low / (hbarc * hbarc)
    plt.fill_between(x_range, y_up, y_low, color=color, lw=0, alpha=0.7)


def plot_f1_f2(data, order, reg_param):
    """Plot the Dirac and Pauli form factors."""
    params, cov = calc_params(data, order, reg_param)
    Q2_range = np.linspace(0, 1, 100)
    F1, F2 = calc_ffs(Q2_range, params, order)

    # Transverse charge radius and the slope of F1:
    b2, _ = get_b2(params, cov)
    slope_x = np.linspace(0, 0.15, 10)
    slope_y = 1 - slope_x * b2 / 4
    # Plot the form factor slope:
    plt.plot(slope_x, slope_y, ls="--", color="black", lw=1)

    if fit.covariance_bad(cov):
        print("Warning: Covariance ill-conditioned, will not plot confidence intervals")
        draw_confidence = False
    else:
        draw_confidence = True

    if draw_confidence:
        # Calculate statistical uncertainties:
        params = np.random.multivariate_normal(params, cov, size=N_SAMPLES)
        interval = calc_interval(calc_ffs, Q2_range, params, order)
        # Calculate systematic uncertainties:
        f1_up, f1_low, f2_up, f2_low = calc_sys_bands(calc_ffs, Q2_range, data, order, reg_param)

        # Plot the systematic band for F2:
        fill_between(Q2_range, interval[1, 1] + f2_up, interval[1, 1], "blue")
        fill_between(Q2_range, interval[0, 1], interval[0, 1] - f2_low, "blue")
        # Plot the statistical band for F2:
        fill_between(Q2_range, interval[1, 1], interval[0, 1], "#AAAAFF")
    # Plot the best-fit line for F2:
    plt.plot(Q2_range, F2, color="blue", lw=0.6, alpha=0.7)

    # Plot the same things for F1:
    if draw_confidence:
        fill_between(Q2_range, interval[1, 0] + f1_up, interval[1, 0], "red")
        fill_between(Q2_range, interval[0, 0], interval[0, 0] - f1_low, "red")
        fill_between(Q2_range, interval[1, 0], interval[0, 0], "#FFAAAA")
    plt.plot(Q2_range, F1, color="red", lw=0.6, alpha=0.7)

    # Axes and labels:
    plt.xlim(0, 1)
    plt.xlabel(r"$Q^2~\left(\mathrm{GeV}^2\right)$")
    plt.ylabel(r"$F_1, \, F_2$", labelpad=11)
    if order == 5:
        plt.text(0.45, 0.46, r"$F_1$", color="#FF0000")
        plt.text(0.36, 0.31, r"$F_2$", color="#0000FF")


def plot_rhos(data, order, reg_param):
    """Plot the transverse densities rho1 and rho2."""
    rho_range = np.linspace(0, 10.1, 100)
    params, cov = calc_params(data, order, reg_param)
    rho1, rho2 = calc_rho(rho_range, params, order)

    if fit.covariance_bad(cov):
        print("Warning: Covariance ill-conditioned, will not plot confidence intervals")
        draw_confidence = False
    else:
        draw_confidence = True

    if draw_confidence:
        # Calculate statistical uncertainties:
        params = np.random.multivariate_normal(params, cov, size=N_SAMPLES)
        interval = calc_interval(calc_rho, rho_range, params, order)
        # Calculate systematic uncertainties:
        rho1_up, rho1_low, rho2_up, rho2_low = calc_sys_bands(calc_rho, rho_range, data, order, reg_param)

        # Plot the systematic band for rho1:
        fill_between(rho_range, interval[1, 0] + rho1_up, interval[1, 0], "red", hbarc_scale=True)
        fill_between(rho_range, interval[0, 0], interval[0, 0] - rho1_low, "red", hbarc_scale=True)
        # Plot the statistical band for rho1:
        fill_between(rho_range, interval[1, 0], interval[0, 0], "#FFAAAA", hbarc_scale=True)
    # Plot the best-fit line for rho1:
    plt.plot(hbarc * rho_range, rho1 / (hbarc * hbarc), color="red", alpha=0.7, lw=0.6)

    # Plot the same things for rho2:
    if draw_confidence:
        fill_between(rho_range, interval[1, 1] + rho2_up, interval[1, 1], "blue", hbarc_scale=True)
        fill_between(rho_range, interval[0, 1], interval[0, 1] - rho2_low, "blue", hbarc_scale=True)
        fill_between(rho_range, interval[1, 1], interval[0, 1], "#AAAAFF", hbarc_scale=True)
    plt.plot(hbarc * rho_range, rho2 / (hbarc * hbarc), color="blue", alpha=0.7, lw=0.6)

    # Axes and labels:
    plt.xlim(0, 2)
    plt.yscale("log")
    plt.xlabel(r"$b~(\mathrm{fm})$", labelpad=6)
    plt.ylabel(r"$\rho_1, \, \rho_2~\left(\mathrm{fm}^{-2}\right)$")
    if order == 5:
        plt.text(0.94, 0.013, r"$\rho_1$", color="#FF0000")
        plt.text(1.1, 0.079, r"$\rho_2$", color="#0000FF")


def plot_ge_gm(cs_data, R_data, order, reg_param):
    """Plot the Sachs electric and magnetic form factors."""
    params, cov = calc_params(cs_data, order, reg_param)
    Q2_range = np.linspace(0, 1.4, 200)
    GE, GM = calc_ge_gm(Q2_range, params, order)

    if fit.covariance_bad(cov):
        print("Warning: Covariance ill-conditioned, will not plot confidence intervals")
        draw_confidence = False
    else:
        draw_confidence = True

    # Calculate statistical uncertainties:
    if draw_confidence:
        params = np.random.multivariate_normal(params, cov, size=N_SAMPLES)
        print(params.shape)
        interval = calc_interval(calc_ge_gm, Q2_range, params, order)
        print(interval.shape)
        # Calculate systematic uncertainties:
        f1_up, f1_low, f2_up, f2_low = calc_sys_bands(calc_ge_gm, Q2_range, cs_data, order, reg_param)

    fun = np.array([calc_ge_gm(Q2_range, par, order) for par in params])
    print(len(fun[10]))

    fig = plt.figure(figsize=(10, 3.5))
    plt.subplots_adjust(wspace=0.35)

    # Left panel (electric form factor):
    fig.add_subplot(1, 2, 1)
    GE_dip, GM_dip = dipole_ffs(R_data["Q2"])
    GE_R = R_data["GE"] / GE_dip
    delta_GE_R = R_data["delta_GE"] / GE_dip
    # Plot the experimental data points for G_E:
    plt.errorbar(R_data["Q2"], GE_R, yerr=delta_GE_R, fmt="ob", ms=1.5, lw=1, zorder=0)
    if draw_confidence:
        # Plot the systematic band for G_E:
        fill_between(Q2_range, interval[1, 0] + f1_up, interval[1, 0], "red")
        fill_between(Q2_range, interval[0, 0], interval[0, 0] - f1_low, "red")
        # Plot the statistical band for G_E:
        print("STAT ERROR", interval[1, 0] - GE, "\n",  GE - interval[0, 0])
        fill_between(Q2_range, interval[1, 0], interval[0, 0], "#FFAAAA")
    # Plot the best-fit line for G_E:
    plt.plot(Q2_range, GE, color="black", lw=1, alpha=0.7)

    # Axes and labels:
    plt.xlim(0, 1)
    if order == 5:
        plt.ylim(0.6, 1.02)
    plt.xlabel(r"$Q^2~\left(\mathrm{GeV}^2\right)$")
    plt.ylabel(r"$G_{E} / G_{\mathrm{dip}}$")

    # Right panel (magnetic form factor):
    fig.add_subplot(1, 2, 2)
    GM_R = R_data["GM"] / GM_dip
    delta_GM_R = R_data["delta_GM"] / GM_dip
    # Plot the experimental data points for G_M:
    plt.errorbar(R_data["Q2"], GM_R, yerr=delta_GM_R, fmt="ob", ms=1.5, lw=1, zorder=0)
    if draw_confidence:
        # Plot the systematic band for G_M:
        fill_between(Q2_range, interval[1, 1] + f2_up, interval[1, 1], "red")
        fill_between(Q2_range, interval[0, 1], interval[0, 1] - f2_low, "red")
        # Plot the statistical band for G_M:
        fill_between(Q2_range, interval[1, 1], interval[0, 1], "#FFAAAA")
    # Plot the best-fit line for G_M:
    plt.plot(Q2_range, GM, color="black", lw=1, alpha=0.7)

    # Axes and labels:
    plt.xlim(0, 1)
    if order == 5:
        plt.ylim(0.98, 1.09)
    plt.xlabel(r"$Q^2~\left(\mathrm{GeV}^2\right)$")
    plt.ylabel(r"$G_{M} / (\mu \, G_{\mathrm{dip}})$")


    #
    #
    # Form Factor Ratio
    #
    #
    fig = plt.figure(dpi=150,figsize=(12,6))
    plt.xlim(0,1.4)
    plt.ylim(0.6,1.05)
    plt.ylabel(r"$\mu$ $G_{E}$/$G_{M}$")
    plt.xlabel(r"Q$^2$ [GeV/c]$^2$")

    # [MC'S NOTES: so it appears this makes a double layer each of the red and light red
    # it combines them and simply uses the lines and divides...
    # I am pretty sure that if you delete bottom 3, you'll get a thick red layer which is not
    # how it is in Gramolin, and there is two layers of this, the below is less weird but still weird]
    fill_between(Q2_range, GE/(f2_up+interval[1,1]), GE/(-f2_low+interval[0,1]), "red")
    fill_between(Q2_range, (f1_up+interval[1,0])/GM, (-f1_low+interval[0,0])/GM , "red")#,label="Gramolin")

    fill_between(Q2_range, GE/(interval[1,1]), GE/(interval[0,1]), "#FFAAAA")
    fill_between(Q2_range, (interval[1,0])/GM, (interval[0,0])/GM , "#FFAAAA")

    plt.errorbar(cq2,cffratio, xerr=0, yerr=ctffratio, fmt='^', color='blue',lw=2,ms=8,zorder=666)
    plt.plot(cq2,cffratio, '^', color='blue', \
             label='Crawford',ms=8)

    plt.errorbar(zq2,zffratio, xerr=0, yerr=ztffratio, fmt='o',color='purple',lw=2,ms=8,zorder=666 )
    plt.plot(zq2,zffratio, 'o', color='purple', zorder=10, \
             label='Zhan',ms=8)


    plt.plot(vq2,vffratio, 's', color='darkorange', \
             label='Punjabi',ms=8)
    plt.errorbar(vq2,vffratio, xerr=0, yerr=vtffratio, fmt='s', color='darkorange',lw=2,ms=8,zorder=666)


    plt.errorbar(pq2,pffratio, xerr=0, yerr=ptffratio, fmt='v', color='black',lw=2,ms=8,zorder=666)
    plt.plot(pq2,pffratio,'v',color='black', \
             label='Paolone',ms=10)

    #plt.plot(0, 1, color="red",label="Gramolin")
    plt.plot(Q2_range, GE/GM, color="black", lw=1, alpha=0.7, label="Gramolin")


    plt.legend(frameon=False,handletextpad=0.5)
    plt.tight_layout()
    plt.savefig("OriginalRatio.png")





def plot_cs(data, order, reg_param):
    """Plot the measured cross sections with best fits."""
    params, _, _, _, _ = fit.fit(data, data, order, reg_param)

    # Renormalize the cross sections:
    norm_params = np.concatenate([[1], params[: fit.N_NORM_PARAMS]])
    norm = np.prod(norm_params[data["norms"]], axis=1)
    data["cs"] = norm * data["cs"]
    data["delta_cs"] = norm * data["delta_cs"]

    fig_S1 = plt.figure(figsize=(10, 13))
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    for i, energy in enumerate(fit.BEAM_ENERGIES):
        ax = fig_S1.add_subplot(3, 2, i + 1)

        Q2max = np.amax(data["Q2"][data["E"] == energy])
        Q2val = np.linspace(0, Q2max, 100)
        curve = calc_cs(0.001 * energy, Q2val, params[fit.N_NORM_PARAMS :], order)

        # Spectrometer A:
        Q2 = data["Q2"][(data["E"] == energy) & (data["spec"] == "A")]
        cs = data["cs"][(data["E"] == energy) & (data["spec"] == "A")]
        delta_cs = data["delta_cs"][(data["E"] == energy) & (data["spec"] == "A")]
        plt.errorbar(Q2, cs, delta_cs, fmt="sr", ms=3, lw=1)

        # Spectrometer B:
        Q2 = data["Q2"][(data["E"] == energy) & (data["spec"] == "B")]
        cs = data["cs"][(data["E"] == energy) & (data["spec"] == "B")]
        delta_cs = data["delta_cs"][(data["E"] == energy) & (data["spec"] == "B")]
        plt.errorbar(Q2, cs, delta_cs, fmt="ob", ms=3, lw=1)

        # Spectrometer C:
        Q2 = data["Q2"][(data["E"] == energy) & (data["spec"] == "C")]
        cs = data["cs"][(data["E"] == energy) & (data["spec"] == "C")]
        delta_cs = data["delta_cs"][(data["E"] == energy) & (data["spec"] == "C")]
        plt.errorbar(Q2, cs, delta_cs, fmt="^g", ms=3, lw=1)

        plt.plot(Q2val, curve, "k-", linewidth=2, alpha=0.7, zorder=3)
        plt.xlim(left=0)
        plt.xlabel(r"$Q^2~\left(\mathrm{GeV}^2\right)$")
        plt.ylabel(r"$\sigma_{\mathrm{red}} / \sigma_{\mathrm{dip}}$")
        plt.text(0.5, 0.92, str(energy) + " MeV", horizontalalignment="center", transform=ax.transAxes)


def save_fig(path):
    """Save figures to path."""
    print("Saving to '{}'".format(path))
    plt.savefig(path, bbox_inches="tight")


def main(order, reg_param, data_file_name):
    print("Model: N = {}, lambda = {}".format(order, reg_param))

    # Read the cross section and Rosenbluth data:
    cs_data = fit.read_cs_data(data_file_name)[0]
    Rosenbluth_data = read_Rosenbluth_data()

    '''
    # Figure 1:
    print("Plotting F1, F2, and transverse charge densities...")
    fig_1 = plt.figure(figsize=(10, 3.5))
    plt.subplots_adjust(wspace=0.35)

    # Figure 1, left panel (Dirac and Pauli form factors):
    ax1 = fig_1.add_subplot(1, 2, 1)
    plot_f1_f2(cs_data, order, reg_param)
    plt.text(0.9, 0.91, "(a)", transform=ax1.transAxes, fontsize=14)

    # Figure 1, right panel (transverse charge densities):
    ax2 = fig_1.add_subplot(1, 2, 2)
    plot_rhos(cs_data, order, reg_param)
    plt.text(0.9, 0.91, "(b)", transform=ax2.transAxes, fontsize=14)

    save_fig("figures/fig_1.pdf")
    
    
    # Figure S2 (fitted cross sections):
    print("Plotting fitted cross sections...")
    plot_cs(cs_data, order, reg_param)
    save_fig("figures/fig_S2.pdf")
    
    '''

    # Figure S1 (electric and magnetic form factors):
    print("Plotting GE and GM...")
    plot_ge_gm(cs_data, Rosenbluth_data, order, reg_param)
    # save_fig("figures/fig_S1.pdf")
    plt.show()




# Data from X. Zhan et al., PLB 705 (2011) 59.
# https://reader.elsevier.com/reader/sd/pii/S0370269311012342?token=7AD5025DC4F9ECEEC89911837FAD59A46111F8FEFE59CC1952935760E5F7C7161D1D6D72623381E9735FE0272B2E6E95&originRegion=us-east-1&originCreation=20220712173056
zq2       = [ 0.298, 0.346, 0.402, 0.449, 0.494, 0.547, 0.599, 0.695 ]
zffratio  = [ 0.927, 0.943, 0.932, 0.931, 0.929, 0.927, 0.908, 0.912 ]
zeffratio = [ 0.011, 0.009, 0.007, 0.006, 0.005, 0.006, 0.005, 0.005 ]
zsffratio = [ 0.007, 0.009, 0.008, 0.007, 0.008, 0.007, 0.010, 0.011 ]
ztffratio = list(map(np.add,zeffratio,zsffratio))

# Data from M. Paolone et al., PRL 105 (2010) 072001.
# https://arxiv.org/abs/1002.2188
pq2       = [ 0.800, 1.300 ]
pffratio  = [ 0.901, 0.858 ]
peffratio = [ 0.007, 0.008 ]
psffratio = [ 0.010, 0.018 ]
ptffratio = list(map(np.add,peffratio,psffratio))

# Data from C. Crawford et al., PRL 98 (2007) 052301.
# https://arxiv.org/abs/nucl-ex/0609007
cq2       = [ 0.162,0.191,0.232,0.282,0.345,0.419,0.500,0.591]
cffratio  = [ 1.019,1.006,0.999,0.973,0.973,0.980,0.993,0.961]
ceffratio = [ 0.013,0.012,0.012,0.012,0.014,0.016,0.019,0.025]
csffratio = [ 0.015,0.014,0.012,0.011,0.010,0.009,0.008,0.007]
ctffratio = list(map(np.add,ceffratio,csffratio))

# Data from V. Punjabi et al., PRC 71 (2005) 055202.
vq2       = [ 0.49,  0.79, 1.18, 1.48, 1.77, 1.88 ]
vffratio  = [ 0.979, 0.951, 0.883, 0.798, 0.789, 0.777 ]
veffratio = [ 0.016, 0.012, 0.013, 0.029, 0.024, 0.024 ]
vsffratio = [ 0.006, 0.010, 0.018, 0.026, 0.035, 0.033 ]
vtffratio = list(map(np.add,veffratio,vsffratio))


if __name__ == "__main__":

    main(6, 0.1, "data/Rebinned+PRadCrossSectionsData.dat")
    #main(0, 0, "data/RebinnedCrossSectionsData.dat")
    #main(0, 0, "data/OG+PRadCrossSectionsData.dat")
    #main(0, 0, "data/Rebinned+PRadCrossSectionsData.dat")

