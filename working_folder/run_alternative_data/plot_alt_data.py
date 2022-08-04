import copy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

import scipy
import scipy.stats as st
from matplotlib.lines import Line2D
from scipy.stats import norm


import fit
import models
from models import calc_cs, calc_ffs, calc_ge_gm, calc_rho, dipole_ffs, get_b2, hbarc
from plot import fill_between, plot_ge_gm, calc_interval
import plot

from sympy.stats import ContinuousRV, P, E
from sympy import exp, Symbol, Interval, oo

import random

from scipy.stats import shapiro  # [MOD: FOR NORMALITY CHECK]


matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 26
matplotlib.rcParams["font.family"] = "lmodern"
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
matplotlib.rcParams["xtick.labelsize"] = 20
matplotlib.rcParams["ytick.labelsize"] = 20

# Number of samples to use when generating statistical uncertainty bands
N_SAMPLES = 1000


def assign_band_curves(interval):
    ge_up = interval[1, 0]  # The top of the G_E stat band
    ge_low = interval[0, 0]  # The bottom of the G_E stat band
    gm_up = interval[1, 1]  # The top of the G_M stat band
    gm_low = interval[0, 1]  # The bottom of the G_M stat band
    return ge_up, ge_low, gm_up, gm_low


# Returns the top and bottom of the statistic 68% CI band
def calc_stat_error_simple(GE, GM, interval):
    ge_up, ge_low, gm_up, gm_low = assign_band_curves(interval)
    '''
    # alt implementation: midpoints
    sigma_ge = (ge_up - ge_low) / 2
    ge_mid = ge_up - sigma_ge / 2
    sigma_gm = (gm_up - gm_low) / 2
    gm_mid = gm_up - sigma_gm / 2
    sigma_f = (GE / GM) * np.sqrt((sigma_ge / GE) ** 2 + (sigma_gm / GM) ** 2)
    # f_stat_up = (GE / GM) + sigma_f
    # f_stat_low = (GE / GM) - sigma_f
    f_stat_up = (ge_mid / gm_mid) + sigma_f
    f_stat_low = (ge_mid / gm_mid) - sigma_f
    '''
    sigma_ge_up = ge_up - GE
    sigma_ge_low = GE - ge_low
    sigma_gm_up = gm_up - GM
    sigma_gm_low = GM - gm_low
    sigma_f_up = (GE/GM) * np.sqrt(
        (sigma_ge_up/GE)**2 + (sigma_gm_low/GM)**2
    )
    sigma_f_low = (GE / GM) * np.sqrt(
        (sigma_ge_low / GE) ** 2 + (sigma_gm_up / GM) ** 2
    )
    f_stat_up = (GE / GM) + sigma_f_up
    f_stat_low = (GE / GM) - sigma_f_low

    return f_stat_low, f_stat_up


def calc_stat_error_montecarlo(Q2_range, cs_data, order, reg_param):
    params, cov = plot.calc_params(cs_data, order, reg_param)
    params = np.random.multivariate_normal(params, cov, size=N_SAMPLES)
    out = np.array([calc_ge_gm(Q2_range, param, order) for param in params])
    print(out.shape, " should be 1000 rows each of 2 arrays each of 1400 or something elements")
    ge_gm_distribution = []
    for row in out:
        ge = row[0]
        gm = row[1]
        ratio = ge / gm
        ge_gm_distribution.append(ratio)
    ge_gm_distribution = np.asarray(ge_gm_distribution)

    stat_up = np.percentile(ge_gm_distribution, 84.1, axis=0)
    stat_down = np.percentile(ge_gm_distribution, 15.9, axis=0)

    return stat_down, stat_up


def trim_zero(Q2_range, interval):
    Q2_range = Q2_range[1:]
    new_interval = np.empty((2, 2, len(Q2_range)))
    new_interval[0, 0] = np.delete(interval[0, 0], 0)
    new_interval[1, 0] = np.delete(interval[1, 0], 0)
    new_interval[1, 1] = np.delete(interval[1, 1], 0)
    new_interval[0, 1] = np.delete(interval[0, 1], 0)
    interval = new_interval
    return Q2_range, interval


def calc_stat_error_ratio_distribution(interval, Q2_range):
    Q2_range, interval = trim_zero(Q2_range, interval)
    assert len(Q2_range) == len(interval[1,0]) and len(Q2_range) == len(interval[0,0]) and len(Q2_range) == \
           len(interval[1,1]) and len(Q2_range) == len(interval[0,1])
    ge_up, ge_low, gm_up, gm_low = assign_band_curves(interval)

    # Create arrays of means and stdevs at all Q^2
    sigma_ge = (ge_up - ge_low) / 2
    sigma_gm = (gm_up - gm_low) / 2
    mu_ge = ge_low + sigma_ge
    mu_gm = gm_low + sigma_gm

    '''
    # [TESTING] Compares the closeness of the mean and the midpoint (which is used as an approximate mean)
    print(len(sigma_ge))
    print("midpoint: ", sigma_ge + ge_low)
    params, cov = plot.calc_params(cs_data, order, reg_param)
    params = np.random.multivariate_normal(params, cov, size=N_SAMPLES)
    out = np.array([calc_ge_gm(Q2_range, param, order) for param in params])
    mu_s = np.empty((1,200))
    for q2 in range(len(Q2_range)):
        ge_s_at_q2 = np.empty((1,1000))
        for idx, sample in enumerate(out):
            ge_s = sample[0]
            ge_s_at_q2[0][idx] = ge_s[q2]
        mu_at_q2 = np.average(ge_s_at_q2[0])
        mu_s[0][q2] = mu_at_q2

    print("mu: ", mu_s)
    params, cov = plot.calc_params(cs_data, order, reg_param)
    GE, GM = calc_ge_gm(Q2_range, params, order)
    print("line: ", GE/GM)
    '''

    # Returns the ratio of two RCV Gaussian distributions (represented by their expected value and standard deviation)
    def ratio_distribution(z, x_mean, x_sigma, y_mean, y_sigma):
        x_sigma2 = x_sigma ** 2
        y_sigma2 = y_sigma ** 2
        a = np.sqrt(
            (1 / x_sigma2) * (z ** 2) + (1 / y_sigma2)
        )
        b = (x_mean / x_sigma2) * z + (y_mean / y_sigma2)
        c = (x_mean ** 2) / x_sigma2 + (y_mean ** 2) / y_sigma2
        d = np.exp(
            (
                    b ** 2 - (c * (a ** 2))
            ) / (2 * (a ** 2))
        )
        pdf = (b * d) / (a ** 3) * (1 / (np.sqrt(2 * np.pi) * x_sigma * y_sigma)) * \
              (norm.cdf(b / a) - norm.cdf(-b / a)) + (1 / ((a ** 2) * np.pi * x_sigma * y_sigma)) * np.exp(-c / 2)
        return pdf

    # print(mu_ge)
    # print(sigma_ge)
    # print(mu_gm)
    # print(sigma_gm)

    '''
    # Testing: check if generated pdf is actually a pdf (integral is 1)
    num = 35
    def func_ratio_distribution_testing(z):
        return ratio_distribution(z, mu_ge[num], sigma_ge[num], mu_gm[num], sigma_gm[num])
    # Ensure pdf is a pdf
    print("Integrating the pdf from -inf to +inf: ",
          scipy.integrate.quad(func_ratio_distribution_testing, -np.inf, np.inf))
    '''

    # Calculate the mu and sigma at a specific Q^2 using the pdf of the ratio distribution generated from GE and GM
    def calc_mu_sigma(x_mean, x_sigma, y_mean, y_sigma):  # these should be discrete values, not structures
        def func_ratio_distribution(z): return ratio_distribution(z, x_mean, x_sigma, y_mean, y_sigma)

        def mu_integrand(x): return x * func_ratio_distribution(x)

        mu = scipy.integrate.quad(mu_integrand, -np.inf, np.inf)[0]

        def sigma_integrand(x): return ((x - mu) ** 2) * func_ratio_distribution(x)

        sigma_sq = scipy.integrate.quad(sigma_integrand, -np.inf, np.inf)[0]
        sigma = np.sqrt(sigma_sq)


        # Testing the pdf-->0 results
        pdf_integral = scipy.integrate.quad(func_ratio_distribution, -np.inf, np.inf)

        print("Q2: ", Q2_range[list(mu_ge).index(x_mean)])
        print(pdf_integral)
        print("x mean, x sigma, y mean, y sigma, ", x_mean, " ", x_sigma, " ", y_mean, " ", y_sigma)
        print("mu sigma: ", mu, " ", sigma)

        return mu, sigma

    '''
    # [TESTING] ensures sigma/mu calculated above matches with rigorous scipy test
    for Q_sq_idx in range(len(Q2_range)):
        if Q_sq_idx > 50:
            def func_ratio_distribution_testing(z):
                return ratio_distribution(z, mu_ge[Q_sq_idx], sigma_ge[Q_sq_idx], mu_gm[Q_sq_idx], sigma_gm[Q_sq_idx])
            # print(ratio_distribution(GE/GM, mu_ge, sigma_ge, mu_gm, sigma_gm))

            # Printing the generated ratio pdf
            y = []
            for x in range(0,101,1):
                z = x * 0.01
                y.append(ratio_distribution(z, mu_ge[Q_sq_idx], sigma_ge[Q_sq_idx], mu_gm[Q_sq_idx], sigma_gm[Q_sq_idx]))
            # print(y)

            # Ensure pdf is a pdf
            print("Integrating the pdf from -inf to +inf: ", scipy.integrate.quad(func_ratio_distribution_testing, -np.inf, np.inf))

            # Canonical way to calculate st dev (impossible long run time)
            class ratio_pdf(st.rv_continuous):
                def _pdf(self, z, *args):  # *args redundant at this point
                    return ratio_distribution(z, mu_ge[Q_sq_idx], sigma_ge[Q_sq_idx], mu_gm[Q_sq_idx], sigma_gm[Q_sq_idx])
            ratio_cv = ratio_pdf(a=0, b=1.2)

            fancy_mu = ratio_cv.mean()
            jank_mu = calc_mu_sigma(mu_ge[Q_sq_idx], sigma_ge[Q_sq_idx], mu_gm[Q_sq_idx], sigma_gm[Q_sq_idx])[0]
            fancy_sigma = ratio_cv.std()
            jank_sigma = calc_mu_sigma(mu_ge[Q_sq_idx], sigma_ge[Q_sq_idx], mu_gm[Q_sq_idx], sigma_gm[Q_sq_idx])[1]
            print("Test st dev, Q^2: ", Q2_range[Q_sq_idx])
            print("Scipy.stats.rv_continuous numbers: ", fancy_mu, " ", fancy_sigma)
            print("My makeshift method numbers: ", jank_mu, " ", jank_sigma)
    '''

    # Testing -- checks to see if generated ratio pdf is normal (68% matches)
    nums = (50, 100)
    for num in nums:
        class ratio_pdf(st.rv_continuous):
            def _pdf(self, z, *args):  # *args redundant at this point
                return ratio_distribution(z, mu_ge[num], sigma_ge[num], mu_gm[num], sigma_gm[num])
        ratio_cv = ratio_pdf(a=0, b=1.2)

        def sample(n):
            return [ratio_cv.ppf(np.random.random()) for _ in range(n)]

        y1 = sample(N_SAMPLES)
        samples = np.linspace(0, 999, 1000)

        print(y1)
        stat, p = shapiro(y1)
        print('stat=%.3f, p=%.3f\n' % (stat, p))
        plt.plot(samples, y1)
        y1 = np.asarray(y1)
        p1, p2 = np.percentile(y1,(15.9, 84.1))  # assuming this goes by size?
        sigma_percentile = (p2-p1)/2
        mu_percentile = p1 + sigma_percentile
        print("percentile: p1: ", p1, ", p2: ", p2, ", sigma: ", sigma_percentile, ", mu: ", mu_percentile)
        print("analytical: ", calc_mu_sigma(mu_ge[num], sigma_ge[num], mu_gm[num], sigma_gm[num]))


    # This array's first row is mu's and second row is sigma's; everything is mapped from Q^2, consistent with Gramolin.
    mu_sigma_array = np.empty((2, len(Q2_range)))

    # For each Q^2 value, calculates the ratio distribution and stores the mu and sigma of each said distribution.
    for Q_sq_idx in range(len(Q2_range)):
        # print("Q2: ", Q2_range[Q_sq_idx])
        mu_ge_at_idx = mu_ge[Q_sq_idx]
        sigma_ge_at_idx = sigma_ge[Q_sq_idx]
        mu_gm_at_idx = mu_gm[Q_sq_idx]
        sigma_gm_at_idx = sigma_gm[Q_sq_idx]
        mu, sigma = calc_mu_sigma(mu_ge_at_idx, sigma_ge_at_idx, mu_gm_at_idx, sigma_gm_at_idx)
        mu_sigma_array[0][Q_sq_idx], mu_sigma_array[1][Q_sq_idx] = mu, sigma

    f_stat_up = mu_sigma_array[0] + mu_sigma_array[1]
    f_stat_low = mu_sigma_array[0] - mu_sigma_array[1]

    return f_stat_low, f_stat_up


# Returns the width of the two systematic error bands
def calc_sys_error_separate(GE, GM, f1_low, f1_up, f2_low, f2_up):
    f_sys_up = (GE / GM) * np.sqrt((f1_up / GE) ** 2 + (f2_low / GM) ** 2)  # f2_low is used here because f \alpha 1/GM
    f_sys_low = (GE / GM) * np.sqrt((f1_low / GE) ** 2 + (f2_up / GM) ** 2)  # ^the same reason for this line
    return f_sys_low, f_sys_up


def calc_ge_over_gm(Q2_range, params, order):
    """Calculate GE / GM"""
    GE, GM = models.calc_ge_gm(Q2_range, params, order)
    return GE / GM


# Modified version of calc_sys_bands(calc_func, x_range, data, order, reg_param) from the original plot.py
def calc_sys_error_original(x_range, data, order, reg_param):
    """Calculate systematic error bands for given quantity."""
    params, _ = plot.calc_params(data, order, reg_param)
    ff_ratio = calc_ge_over_gm(x_range, params, order)
    mincut_params = fit.fit_systematic_variant("cs_mincut", data, order, reg_param)[0]
    maxcut_params = fit.fit_systematic_variant("cs_maxcut", data, order, reg_param)[0]
    sysup_params = fit.fit_systematic_variant("cs_sysup", data, order, reg_param)[0]
    syslow_params = fit.fit_systematic_variant("cs_syslow", data, order, reg_param)[0]
    mincut_ff_ratio = calc_ge_over_gm(x_range, mincut_params, order)
    maxcut_ff_ratio = calc_ge_over_gm(x_range, maxcut_params, order)
    sysup_ff_ratio = calc_ge_over_gm(x_range, sysup_params, order)
    syslow_ff_ratio = calc_ge_over_gm(x_range, syslow_params, order)
    # Calculate upper and lower limits for each of the systematic variations:
    ff_ratio_cut_up = np.clip(np.max(np.stack([mincut_ff_ratio - ff_ratio, maxcut_ff_ratio - ff_ratio]), 0), 0, None)
    ff_ratio_cut_low = np.clip(np.min(np.stack([mincut_ff_ratio - ff_ratio, maxcut_ff_ratio - ff_ratio]), 0), None, 0)
    ff_ratio_sys_up = np.clip(np.max(np.stack([sysup_ff_ratio - ff_ratio, syslow_ff_ratio - ff_ratio]), 0), 0, None)
    ff_ratio_sys_low = np.clip(np.min(np.stack([sysup_ff_ratio - ff_ratio, syslow_ff_ratio - ff_ratio]), 0), None, 0)
    # Add two systematic "errors" in quadrature:
    ff_ratio_up = np.sqrt(ff_ratio_cut_up ** 2 + ff_ratio_sys_up ** 2)
    ff_ratio_low = np.sqrt(ff_ratio_cut_low ** 2 + ff_ratio_sys_low ** 2)
    return ff_ratio_low, ff_ratio_up


def plot_data_set(cs_data, order, reg_param, Q2_max, axes):
    """Plot results from the FFs extracted from the Gramolin fit for a particular data set"""  # transverse charge
    # density analysis?

    # Extract Sachs FF distribution data
    GE, GM, Q2_range, interval, f1_up, f1_low, f2_up, f2_low = plot_ge_gm(cs_data, order, reg_param, Q2_max=Q2_max)

    # Plot the best-fit line for G_E*mu/G_M
    # axes.plot(Q2_range, GE / GM, color="black", lw=1, alpha=0.7, label="Gramolin")
    axes.plot(Q2_range, GE / GM, color="black", lw=1, alpha=0.7)

    # Calculate the statistical uncertainties using a ratio distribution/monte carlo/separate
    f_stat_low, f_stat_up = calc_stat_error_simple(GE, GM, interval)  # the top and bottom of a stat band
    # f_stat_low, f_stat_up = calc_stat_error_ratio_distribution(interval, Q2_range)
    # f_stat_low, f_stat_up = calc_stat_error_montecarlo(Q2_range, cs_data, order, reg_param)
    # Calculate the systematic uncertainties using standard propagation/author's og method
    # f_sys_low, f_sys_up = calc_sys_error_separate(GE, GM, f1_low, f1_up, f2_low, f2_up)  # discrete band widths (independent)
    f_sys_low, f_sys_up = calc_sys_error_original(Q2_range, cs_data, order, reg_param)

    trim_Q2 = False
    if trim_Q2:
        Q2_range = Q2_range[1:]
        f_sys_up = f_sys_up[1:]
        f_sys_low = f_sys_low[1:]
    # Plot the statistical band
    axes.fill_between(Q2_range, f_stat_up, f_stat_low, color="#FFAAAA", lw=0, alpha=0.75)
    # Plot the systematic bands
    axes.fill_between(Q2_range, f_stat_up + f_sys_up, f_stat_up, color="red", lw=0, alpha=0.75)
    axes.fill_between(Q2_range, f_stat_low, f_stat_low - f_sys_low, color="red", lw=0, alpha=0.75)

    """Plot asymmetry data"""
    studies = ["asymdata/Crawford.dat", "asymdata/Punjabi.dat", "asymdata/Paolone.dat", "asymdata/Zhan.dat"]

    # Display choices
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
        axes.plot(study["Q2"], study["ff_ratio"], fmts[idx], color=colors[idx], label=studies[idx][9:-4], ms=8)
        # Add error bars
        axes.errorbar(study["Q2"], study["ff_ratio"], xerr=0,
                      yerr=study["stat_error"], fmt=fmts[idx], color=colors[idx], lw=2, ms=8, zorder=666, capsize=5)
        axes.errorbar(study["Q2"], study["ff_ratio"], xerr=0,
                      yerr=combine_stat_sys_error(study["stat_error"], study["sys_error"]),
                      fmt=fmts[idx], color=colors[idx], lw=2, ms=8, zorder=666, capsize=5)

    """Plot Alarcon and Weiss"""
    from scipy.interpolate import interp1d
    # Read data file into array
    # Transpose array to allow for standard indexing
    paramT = np.loadtxt('DIXEFT-Parameterization.dat')
    param = paramT.transpose()
    # Compute interpolating spline
    AEp = interp1d(param[0], param[1], kind='cubic')
    AEn = interp1d(param[0], param[2], kind='cubic')
    BE = interp1d(param[0], param[3], kind='cubic')
    BbarE = interp1d(param[0], param[4], kind='cubic')
    AMp = interp1d(param[0], param[5], kind='cubic')
    BMp = interp1d(param[0], param[6], kind='cubic')
    BbarMp = interp1d(param[0], param[7], kind='cubic')
    AMn = interp1d(param[0], param[8], kind='cubic')
    BbarMn = interp1d(param[0], param[9], kind='cubic')
    BMn = interp1d(param[0], param[10], kind='cubic')

    # Proton Electric Form Factor
    def GEp(Q2, r2Ep, r2En):
        return AEp(Q2) + r2Ep * BE(Q2) + r2En * BbarE(Q2)

    # Neutron Magnetic Form Factors
    def GEn(Q2, r2En, r2Ep):
        return AEn(Q2) + r2En * BE(Q2) + r2Ep * BbarE(Q2)

    # Proton Magnetic Form Factors
    def GMp(Q2, r2Mp, r2Mn):
        return AMp(Q2) + r2Mp * BMp(Q2) + r2Mn * BbarMp(Q2)

    # Neutron Magnetic Form Factors
    def GMn(Q2, r2Mn, r2Mp):
        return AMn(Q2) + r2Mn * BMn(Q2) + r2Mp * BbarMn(Q2)

    # Set default values of squared radii
    # All values in fm^2 units
    r2Ep = 0.842 ** 2
    r2Mp = 0.850 ** 2
    r2En = -0.116
    r2Mn = 0.864 ** 2

    Q2list = np.linspace(0, 1.0, 201)
    mup = 1 + models.kappa
    alarcon_ge = GEp(Q2list, 0.842 ** 2, r2En)
    alarcon_gm = GMp(Q2list, 0.85 ** 2, r2Mn) / mup

    # Plot proton electric form factor
    # axes.plot(Q2list, alarcon_ge/alarcon_gm, '--', label='Alarcon and Weiss', color='grey', lw=1)

    # Plot error
    re_distribution = np.random.normal(0.842, 0.002, 1000)
    rm_distribution = np.random.normal(0.850, 0.001, 1000)
    re_rm_distribution = np.stack([re_distribution, rm_distribution], axis=1)
    ge_gm_distribution = []
    for row in re_rm_distribution:
        re = row[0]
        rm = row[1]
        ge = GEp(Q2list, re ** 2, r2En)
        gm = GMp(Q2list, rm ** 2, r2Mn) / mup
        ratio = ge / gm
        ge_gm_distribution.append(ratio)
    ge_gm_distribution = np.asarray(ge_gm_distribution)
    stat_up = np.percentile(ge_gm_distribution, 84.1, axis=0)
    stat_down = np.percentile(ge_gm_distribution, 15.9, axis=0)
    axes.fill_between(Q2list, stat_up, stat_down, color='grey', lw=0, alpha=0.4, label="Alarcon and Weiss")


def draw_model_handle(ax, pos, size):
    x = pos[0]
    start_y = pos[1]
    width = size[0]
    height = size[1]
    rect1 = matplotlib.patches.Rectangle((x,start_y), width, height/4, color = "red")
    rect2 = matplotlib.patches.Rectangle((x,start_y+height/4), width, height/2, color = "#FFAAAA")
    rect3 = matplotlib.patches.Rectangle((x,start_y+0.75*height), width, height/4, color = "red")
    line = matplotlib.patches.Rectangle((x, start_y+height*0.5), width, 0.0001, color = "black")

    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(line)


def plot_subplot(data_file_name, order, reg_param, title, label, axes_coordinates, Q2_max, y_min, y_max, legend=False,
                 x_off=False, y_off=False, rectangle_settings=True, erase_x_0=True, erase_y_0=False):
    handle_on = False
    bold = True
    if rectangle_settings:
        label_pos = 0.053, 1.029
        model_param_pos = 1.06, 1.028
        if bold:
            model_param_pos = 1.01, 1.028
    else:
        label_pos = 0.043, 1.025
        model_param_pos = 0.813, 1.028

    # parameters
    cs_data = fit.read_cs_data(data_file_name)[0]

    # Axes and limits
    ax = axes_coordinates
    x_ticks = ax.xaxis.get_major_ticks()
    y_ticks = ax.yaxis.get_major_ticks()

    if erase_x_0:
        x_ticks[0].label1.set_visible(False)
    if erase_y_0:
        # tick_last = copy.deepcopy(y_ticks[len(y_ticks)-1])
        # tick_second_last = copy.deepcopy(y_ticks[len(y_ticks)-2])
        #y_ticks[0].label1.set_visible(False)
        y_ticks[len(y_ticks)-1].label1.set_visible(False)
        # y_ticks.remove(y_ticks[0])


    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)

    if x_off:
        ax.xaxis.set_visible(False)
    if y_off:
        ax.yaxis.set_visible(False)

    # ax.set_title(title, fontstyle='italic')
    axes_coordinates.annotate(label, label_pos, ha='center', va='center', fontsize=28, color='black')
    if bold:
        axes_coordinates.annotate(r'\textbf{Model: \emph{N} = %d,} $\mathbf{\lambda = %.2f}$' % (order, reg_param), model_param_pos, ha='center',
                                  va='center', fontsize=22, color='#FF0000')
    else:
        axes_coordinates.annotate(r'Model: \emph{N} = %d, $\lambda = %.2f$' % (order, reg_param),
                                  model_param_pos, ha='center',
                                  va='center', fontsize=24, color='red')

    x_handle = model_param_pos[0] - 0.43
    y_handle = model_param_pos[1] - 0.0097

    if handle_on:
        draw_model_handle(ax, (x_handle, y_handle), (0.08, 0.022))
    # axes_coordinates.text(0.01, 1.085, title, fontsize='small', verticalalignment='top', fontfamily='serif',
    #        bbox=dict(facecolor='0.7', edgecolor='none'))
    plot_data_set(cs_data, order, reg_param, Q2_max, ax)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, Q2_max)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        # labels[0] = "Model"
        # handles[0] = red_patch = mpatches.Patch(facecolor='#FFAAAA', edgecolor="red", label='The red data')
        ax.legend(handles, labels, loc='lower left', frameon=False, fontsize='small')
    # else:
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles, labels, loc='lower left', frameon=False)


def main():  # other legend options?
    rectangle_settings = True
    legends_on = False

    if rectangle_settings:
        figsize = (18,12)
        x_axis_pos = 0.5, 0.04
        y_axis_pos = 0.065, 0.5
        wspace = 0.015
        Q2_max = 1.4
    else:
        figsize = (13, 13)
        #figsize = (18,12)
        x_axis_pos = 0.5, 0.04
        y_axis_pos = 0.045, 0.5
        wspace=0.025
        Q2_max = 1.0

    # Plotting all the subplots
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
    fig.subplots_adjust(hspace=0.025, wspace=wspace)
    fig.text(x_axis_pos[0], x_axis_pos[1], r"$Q^2$ [GeV/c]$^2$", ha='center', fontsize=30)
    fig.text(y_axis_pos[0], y_axis_pos[1], r"$\mu$ $G_{E}$/$G_{M}$", va='center', rotation='vertical', fontsize=30)

    # fig.suptitle('TITLE')
    Q2_max = Q2_max
    y_max = 1.05
    y_min = 0.7
    title_type = "normal"

    '''OG data'''
    order = 5
    reg_param = 0.02
    data_file_name = "data/CrossSections.dat"
    title = r'Bernauer \emph{et al.}'
    label = r'\textbf{a)}'
    axes_coordinates = axes[0, 0]
    plot_subplot(data_file_name, order, reg_param, title, label, axes_coordinates,
                 Q2_max, y_min, y_max, legend=True, x_off=True, rectangle_settings=rectangle_settings, erase_y_0=False)

    '''Rebinned Data'''
    order = 5
    reg_param = 0.01
    data_file_name = "data/RebinnedCrossSectionsData.dat"
    title = r'Lee \emph{et al.}'
    label = r'\textbf{b)}'
    axes_coordinates = axes[0,1]
    plot_subplot(data_file_name, order, reg_param, title, label, axes_coordinates, Q2_max,
                 y_min, y_max, x_off=True, legend=legends_on, y_off=True, rectangle_settings=rectangle_settings)

    '''OG+PRad Data'''
    order = 7
    reg_param = 0.63
    data_file_name = "data/OG+PRadCrossSectionsData.dat"
    title = r'Bernauer \emph{et al.} + Xiong \emph{et al.}'
    label = r'\textbf{c)}'
    axes_coordinates = axes[1, 0]
    plot_subplot(data_file_name, order, reg_param, title, label, axes_coordinates, Q2_max, y_min, y_max,
                 legend=legends_on, rectangle_settings=rectangle_settings, erase_x_0=False, erase_y_0=False)

    '''Rebinned+PRad Data'''
    order = 6
    reg_param = 0.1
    data_file_name = "data/Rebinned+PRadCrossSectionsData.dat"
    title = r'Lee \emph{et al.} + Xiong \emph{et al.}'
    label = r'\textbf{d)}'
    axes_coordinates = axes[1,1]
    plot_subplot(data_file_name, order, reg_param, title, label, axes_coordinates, Q2_max, y_min, y_max,
                 legend=legends_on, y_off=True, rectangle_settings=rectangle_settings)

    # Final plot settings
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper left', frameon=False)
    # plt.savefig("OriginalRatio.png")
    # plt.tight_layout()
    plt.savefig("figures/figure_1.png", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
   main()


'''
# NOTES

# interval = calc_interval()  this is the alt to find just 68%, other alt is find sigmas differently
def calc_ge_over_gm(GE, GM):
    """Calculate GE / GM"""
    return GE / GM



(formmerly in the body of plot_data_set()
def func_ratio_distribution(z):
    return ratio_distribution(z, mu_ge[100], sigma_ge[100], mu_gm[100], sigma_gm[100])
#print(ratio_distribution(GE/GM, mu_ge, sigma_ge, mu_gm, sigma_gm))
y = []
for x in range(0,101,1):
    z = x * 0.01
    y.append(ratio_distribution(z, mu_ge[100], sigma_ge[100], mu_gm[100], sigma_gm[100]))
print(y)
print("Integrating the pdf from -inf to +inf: ", scipy.integrate.quad(func_ratio_distribution, -np.inf, np.inf))

class ratio_pdf(st.rv_continuous):
    def _pdf(self, z):
        return ratio_distribution(z, mu_ge[100], sigma_ge[100], mu_gm[100], sigma_gm[100])

ratio_cv = ratio_pdf(a=0, b=1.5)
print(ratio_cv.std())
'''