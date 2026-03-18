import numpy as np
import os
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed
from numpy.polynomial import Polynomial
from qnetwork.dde import EmittersInWaveguideDDE
from aux_funs import dde_scalar, dde_scalar_simple
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from two_qubit_control import EmittersInWaveguideWW as WW
from two_qubit_control import Waveguide as WG
from scipy.integrate import trapezoid
from qnetwork.tools import set_plot_style


PLOT_COLORS = {
    "swap": "#1F5A99",
    "stirap": "#1E9B92",
    "czkm": "#C76B2A",
    "theory": "#6D7F99",
    "guide": "#94A3BC",
    "q1": "#2C6FB2",
    "q2": "#D69A2D",
    "dde": "#4C628A",
    "ww": "#7E8FB5",
    "error": "#000000",
    "scatter_a": "#5E7FC6",
    "scatter_b": "#5DA59D",
    "scatter_c": "#D08A56",
}


# ------------------------------------------------------------------------------------
def dde_series_function(gamma, tau, eta, N, alpha=None):
    """Return the truncated DDE series solution as a callable of time."""
    if alpha is None:
        alpha = 0.5 * gamma

    P = Polynomial([1.0])
    polys: list[Polynomial] = []
    for _ in range(N):
        Q = P.integ()
        P = P + Q
        polys.append(Q)

    n_list = np.arange(1, N + 1)
    eta_pows = eta**n_list
    n_tau = n_list * tau

    def evaluate(t):
        """Evaluate the cached polynomial series on scalar or array inputs."""
        t = np.asarray(t, dtype=float)
        result = np.exp(-alpha * t).astype(np.complex128, copy=False)

        for Q, eta_n, nt in zip(polys, eta_pows, n_tau):
            tn = t - nt
            H = np.heaviside(tn, 0.0)
            if np.any(H):
                result += eta_n * np.exp(-alpha * tn) * Q(-gamma * tn) * H
        return result

    return evaluate


def compute_period_and_fidelity(Delta, gamma, tau=1, T_min=0.8, T_max=1.2):
    """Estimate the period, peak fidelity, and integrated link occupancy."""
    FSR = np.pi / tau
    phi = math.modf(Delta)[0] * FSR * tau
    eta_b = np.exp(1j * phi)
    eta_d = -eta_b

    g = np.sqrt(gamma / (2 * tau))
    Omega = np.sqrt(Delta**2 + 8 * g**2) / 2
    T = np.pi / Omega
    n_terms = int(T_max * T / tau)
    cb = dde_series_function(gamma, tau, eta_b, n_terms)
    cd = dde_series_function(gamma, tau, eta_d, n_terms)
    p1 = lambda t: np.abs(0.5 * (cb(t) + cd(t))) ** 2
    p2 = lambda t: np.abs(0.5 * (cb(t) - cd(t))) ** 2

    res = minimize_scalar(
        lambda t: -p2(t),
        bounds=(T_min * T, T_max * T),
        method="bounded",
        options={"xatol": 1e-9},
    )
    t_peak, F = float(res.x), float(-res.fun)

    # Integrate the probability weight living in the link up to the peak time.
    t_grid = np.linspace(0.0, t_peak, 4001)
    n_loss = np.clip(1.0 - p1(t_grid) - p2(t_grid), 0.0, None)
    N_link = float(trapezoid(n_loss, x=t_grid))

    return g / FSR, T, t_peak, F, N_link


def expt_002_swapspeed(
    Delta,
    gamma_list,
    tau=1.0,
    T_min=0,
    T_max=1.9,
    kappa=0.1,
    n_jobs=-1,
    overwrite=False,
    filename="expt_002_cache.npz",
):
    """Scan SWAP speed versus gamma, caching only the integrated link occupancy."""
    if os.path.exists(filename) and not overwrite:
        print(f"[load] {filename}")
        data = np.load(filename)
        g_list = data["g"]
        T_list = data["T"]
        t_list = data["t"]
        F_list = data["F"]
        if "N_link" in data.files:
            N_link_list = np.asarray(data["N_link"], float)
        elif "P_loss" in data.files:
            N_link_list = np.asarray(data["P_loss"], float)
        elif "P_link_loss" in data.files:
            kappa_cache = float(data["kappa"]) if "kappa" in data.files else kappa
            P_survival = np.clip(np.asarray(data["P_link_loss"], float), 1e-300, 1.0)
            N_link_list = -np.log(P_survival) / kappa_cache
        else:
            N_link_list = np.full_like(F_list, np.nan, dtype=float)
    else:
        if gamma_list is None:
            raise ValueError(
                "gamma_list is None but cache file does not exist (or force=True)."
            )

        print("[compute] running parallel scan ...")

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_period_and_fidelity)(Delta, gamma, tau, T_min, T_max)
            for gamma in gamma_list
        )
        g_list, T_list, t_list, F_list, N_link_list = (
            np.asarray(values, float) for values in zip(*results)
        )

        np.savez(
            filename,
            g=g_list,
            T=T_list,
            t=t_list,
            F=F_list,
            N_link=N_link_list,
            Delta=Delta,
            tau=tau,
            T_min=T_min,
            T_max=T_max,
        )
        print(f"[save] {filename}")

    P_link_loss_list = 1.0 - np.exp(-kappa * N_link_list)

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    gamma_tau = 2 * g_list**2

    #  swap speed
    axes[0].plot(
        gamma_tau, t_list / tau, "-", color=PLOT_COLORS["swap"], label="Swap speed"
    )
    axes[0].plot(
        gamma_tau,
        T_list / tau,
        "--",
        color=PLOT_COLORS["theory"],
        label=r"$\pi/\Omega$",
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\gamma \tau$")
    axes[0].set_ylabel("T / τ")
    axes[0].grid(True)
    axes[0].legend()

    #  infidelity
    axes[1].plot(
        gamma_tau, 1 - F_list, "-", color=PLOT_COLORS["swap"], label="Infidelity"
    )
    axes[1].plot(
        gamma_tau,
        1.5 * gamma_tau,
        "--",
        color=PLOT_COLORS["theory"],
        label=r"$\frac{3\gamma\tau}{2}$",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\gamma \tau$")
    axes[1].set_ylabel("1 − F")
    axes[1].grid(True)
    axes[1].legend()

    #  link loss estimated from 1 - exp(-kappa * \int n(t) dt)
    axes[2].plot(
        gamma_tau, P_link_loss_list, "-", color=PLOT_COLORS["swap"], label="Link loss"
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\gamma \tau$")
    axes[2].set_ylabel(r"$1-e^{-\kappa \int_0^{t_*} n(t)\,dt}$")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    plt.show()


def swap_appendix_graph(
    Delta: float = 0.0,
    gamma: float = 0.1,
    tau: float = 50.0,
    T: float = 100.0,
    dt_max: float = 0.01,
    figsize=(7.2, 7.2),
    hspace: float = 0.10,
    save: str = "fig_swap_app.pdf",
    dpi: int = 600,
):
    """Appendix figure:
    (a) SWAP dynamics (populations + ideal Rabi point)
    (b) optimal duration vs gamma from cache
    (c) optimal infidelity vs gamma from cache
    """

    initial = "10"
    tmax = T * tau
    phi = math.modf(Delta)[0] * np.pi

    # ---------- DDE simulation ----------
    dde = EmittersInWaveguideDDE(
        phi=phi, N=2, gamma=gamma, U=-1, tau=tau, dt_max=dt_max
    )
    dde.evolve(tmax)
    t_DDE, pop_DDE = dde.n_photons(initial)

    # ---------- load cache ----------
    cache = np.load("swap_qst.npz")
    g_list = np.asarray(cache["g"], float)
    T_list = np.asarray(cache["T"], float)
    t_list = np.asarray(cache["t"], float)
    F_list = np.asarray(cache["F"], float)

    gamma_tau = 2 * g_list**2 * tau

    g0 = np.sqrt(gamma / (2 * tau))
    Omega = np.sqrt(2) * g0
    T_rabi = np.pi / Omega

    # ---------- plot ----------
    with plt.rc_context():
        set_plot_style()
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = False

        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw=dict(hspace=hspace))

        def place_labels(axis, xlabel, ylabel, x=(-0.0, -0.2), y=(-0.12, 0.5)):
            """Apply the paper-style label placement used in appendix figures."""
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.xaxis.set_label_coords(0.5, x[1])
            axis.yaxis.set_label_coords(y[0], y[1])

        def panel(axis, lab, xy=(-0.2, 1.02)):
            """Add the panel tag in axes coordinates."""
            axis.text(
                *xy, lab, transform=axis.transAxes, ha="left", va="bottom", fontsize=16
            )

        # ========== (a) dynamics ==========
        axis = axes[0]
        axis.plot(
            t_DDE / tau,
            pop_DDE[:, 0],
            "--",
            lw=2.0,
            color=PLOT_COLORS["q1"],
            label=r"$Q_1$",
        )
        axis.plot(
            t_DDE / tau,
            pop_DDE[:, 1],
            "-",
            lw=2.0,
            color=PLOT_COLORS["q2"],
            label=r"$Q_2$",
        )
        axis.axvline(
            T_rabi / tau,
            ls="--",
            c=PLOT_COLORS["theory"],
            alpha=0.5,
            label=r"$\pi/\Omega$",
        )
        # axis.grid(True, which="both", alpha=0.30)
        axis.legend(frameon=False)
        place_labels(axis, r"$t/\tau$", r"$\langle \sigma^+ \sigma \rangle$")
        panel(axis, "(a)")

        # ========== (b) T_opt vs gamma ==========
        axis = axes[1]
        axis.plot(
            gamma_tau,
            t_list / tau,
            "-",
            lw=2.0,
            color=PLOT_COLORS["swap"],
            label=r"$T(\gamma_0)/\tau$",
        )
        axis.plot(
            gamma_tau,
            T_list / tau,
            "--",
            lw=2.0,
            color=PLOT_COLORS["theory"],
            label=r"$\pi/\Omega$",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        # axis.grid(True, which="both", alpha=0.30)
        axis.legend(frameon=False)
        place_labels(axis, r"$\gamma_0\tau$", r"$T/\tau$")
        panel(axis, "(b)")

        # ========== (c) infidelity ==========
        axis = axes[2]
        axis.plot(
            gamma_tau,
            1.0 - F_list,
            "-",
            lw=2.0,
            color=PLOT_COLORS["swap"],
            label=r"$1-F$",
        )
        axis.plot(
            gamma_tau,
            1.5 * gamma_tau,
            "--",
            lw=2.0,
            color=PLOT_COLORS["theory"],
            label=r"$\frac{3\gamma_0\tau}{2}$",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        # axis.grid(True, which="both", alpha=0.30)
        axis.legend(frameon=False)
        place_labels(axis, r"$\gamma_0\tau$", r"$1-F$")
        panel(axis, "(c)")

        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.10, hspace=hspace)

        if save:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()


# ==================================================================================
def gamma_pulse(gamma, T, tau):
    """Return the undelayed STIRAP-like pulse pair."""
    den = 2.0 * T * tau
    return (
        lambda t: gamma * np.sin(np.pi * t / den) ** 2,
        lambda t: gamma * np.cos(np.pi * t / den) ** 2,
    )


def gamma_pulse_delay(gamma, T, tau=1.0):
    """Return the delayed pulse pair with piecewise saturation."""
    limit = (T - 1.0) * tau
    den = 2.0 * limit
    return (
        lambda t: gamma * np.sin(np.pi * t / den) ** 2 if t <= limit else gamma,
        lambda t: gamma * np.cos(np.pi * (t - tau) / den) ** 2 if t >= tau else gamma,
    )


def Fidelity(gamma, tau, phi, T, dt_max, pulse_delay, return_link=False):
    """Compute the final STIRAP fidelity and, optionally, the link integral."""
    gamma1, gamma2 = (gamma_pulse_delay if pulse_delay else gamma_pulse)(gamma, T, tau)
    t_list, c = dde_scalar(
        t_max=T * tau,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
    )
    F = np.abs(c[-1, 1]) ** 2
    if not return_link:
        return F

    n_link = np.clip(1.0 - np.sum(np.abs(c[:, :2]) ** 2, axis=1), 0.0, None)
    N_link = float(trapezoid(n_link, x=t_list))
    return F, N_link


# ------------------------------------------------------------------------------------
def stirap_optimal_Peak(
    gamma=0.1,
    T_range=None,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    pulse_delay=True,
):
    """Optimize the pulse duration and report the associated link integral."""
    if T_range is None:
        T_range = [0.5, 10.0 / np.sqrt(gamma)]
    res = minimize_scalar(
        lambda T: -Fidelity(gamma, tau, phi, T, dt_max, pulse_delay),
        bounds=tuple(np.asarray(T_range, float)),
        method="bounded",
    )

    T_opt = float(res.x)
    F_opt = float(-res.fun)
    _, N_link_opt = Fidelity(
        gamma,
        tau,
        phi,
        T_opt,
        dt_max,
        pulse_delay,
        return_link=True,
    )
    return T_opt, F_opt, N_link_opt


# ------------------------------------------------------------------------------------
def expt_008_ScanGamma_Refined(
    gamma_list,
    T_range=None,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    pulse_delay=True,
    kappa=1.0,
    n_jobs=-1,
    plot=True,
    cache_file="expt_008_cache.npz",
    overwrite=False,
):
    """Scan the optimal STIRAP duration over gamma and cache only the link integral."""
    gamma_list = np.asarray(gamma_list, float)

    if cache_file and os.path.exists(cache_file) and not overwrite:
        print(f"[info] load cache: {cache_file}")
        cache = np.load(cache_file)
        gamma_vals = np.asarray(cache["gamma"], float)
        T_opt = np.asarray(cache["T_opt"], float)
        F_opt = np.asarray(cache["F_opt"], float)
        if "N_link" in cache.files:
            N_link = np.asarray(cache["N_link"], float)
        elif "P_link_loss" in cache.files and "kappa" in cache.files:
            N_link = -np.log(np.clip(np.asarray(cache["P_link_loss"], float), 1e-300, 1.0)) / float(
                cache["kappa"]
            )
        else:
            N_link = np.full_like(F_opt, np.nan, dtype=float)
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(stirap_optimal_Peak)(
                gamma=g,
                T_range=T_range,
                phi=phi,
                tau=tau,
                dt_max=dt_max,
                pulse_delay=pulse_delay,
            )
            for g in gamma_list
        )
        T_opt, F_opt, N_link = (np.asarray(values, float) for values in zip(*results))
        gamma_vals = gamma_list
        if cache_file:
            np.savez(
                cache_file,
                gamma=gamma_vals,
                T_opt=T_opt,
                F_opt=F_opt,
                N_link=N_link,
            )
            print(f"[info] saved cache: {cache_file}")

    if not plot:
        return

    plt.rcParams["mathtext.fontset"] = "cm"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    P_link_loss = 1.0 - np.exp(-kappa * N_link)

    axes[0].plot(gamma_vals, T_opt, "-", color=PLOT_COLORS["stirap"])
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"$T^*$")
    axes[0].set_title("Optimal T")

    axes[1].plot(gamma_vals, 1.0 - F_opt, "-", color=PLOT_COLORS["stirap"])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel(r"$1-F$")
    axes[1].set_title("Minimal infidelity")

    axes[2].plot(gamma_vals, P_link_loss, "-", color=PLOT_COLORS["stirap"])
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$\gamma$")
    axes[2].set_ylabel(r"$1-e^{-\kappa \int n(t)\,dt}$")
    axes[2].set_title("Link loss")

    fig.tight_layout()
    plt.show()


# ---------------------------------
def stirap_appendix_graph(
    gamma: float,
    T_list,
    phi: float = 0.0,
    tau: float = 1.0,
    dt_max: float = 0.01,
    pulse_delay: bool = True,
    n_jobs: int = -1,
    cache_file_nodelay: str = "stirap_qst.npz",
    save: str = "fig_stirap_app.pdf",
    dpi: int = 600,
    figsize=(7.2, 7.2),
    hspace: float = 0.10,
):
    """Generate the appendix figure summarizing the STIRAP scaling trends."""
    T_list = np.asarray(T_list, float)

    def load_cache(path: str):
        """Load cached gamma, optimal duration, and fidelity arrays."""
        cache = np.load(path)
        return (
            np.asarray(cache["gamma"], float),
            np.asarray(cache["T_opt"], float),
            np.asarray(cache["F_opt"], float),
        )

    # ---------- fixed-gamma T-scan ----------
    F_Tscan = np.asarray(
        Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(Fidelity)(gamma, tau, phi, T, dt_max, pulse_delay) for T in T_list
        ),
        float,
    )
    IF_Tscan = 1.0 - F_Tscan
    gamma_ref, T_opt_ref, F_opt_ref = load_cache(cache_file_nodelay)

    # CZKM
    cut = 940

    # ---------- plot ----------
    with plt.rc_context():
        set_plot_style()
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = False

        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw=dict(hspace=hspace))

        def place_labels(axis, xlabel, ylabel, x=(-0.0, -0.2), y=(-0.12, 0.5)):
            """Apply the paper-style label placement used in appendix figures."""
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.xaxis.set_label_coords(0.5, x[1])
            axis.yaxis.set_label_coords(y[0], y[1])

        def panel(axis, lab, xy=(-0.2, 1.02)):
            """Add the panel tag in axes coordinates."""
            axis.text(
                *xy,
                lab,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=16,
            )

        # ========== (a) ==========
        axis = axes[0]
        axis.plot(gamma * T_list, IF_Tscan, "-", lw=2.0, color=PLOT_COLORS["stirap"])
        axis.set_yscale("log")
        # axis.grid(True, which="both", alpha=0.30)
        place_labels(axis, r"$\gamma_0T$", r"$1-F$")
        panel(axis, "(a)")

        # ========== (b) ==========
        axis = axes[1]
        axis.plot(
            gamma_ref[:cut],
            T_opt_ref[:cut],
            "-",
            lw=2.0,
            color=PLOT_COLORS["stirap"],
            label=r"$T(\gamma_0)/\tau$",
        )
        axis.plot(
            gamma_ref[:cut],
            9 / np.sqrt(gamma_ref[:cut]),
            "--",
            lw=2.0,
            color=PLOT_COLORS["theory"],
            label=r"$9/\sqrt{\gamma_0\tau}$",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        # axis.grid(True, which="both", alpha=0.30)
        axis.legend(frameon=False)
        place_labels(axis, r"$\gamma_0\tau$", r"$T/\tau$")
        panel(axis, "(b)")

        # ========== (c) ==========
        axis = axes[2]
        axis.plot(
            gamma_ref[:cut],
            1.0 - F_opt_ref[:cut],
            "-",
            lw=2.0,
            color=PLOT_COLORS["stirap"],
            label=r"$1-F$",
        )
        # axis.plot(
        #     gamma_czkm[:cut], 1.0 - F_czkm[:cut], "--", lw=2.0, color=PLOT_COLORS["czkm"], label=r"sech-shape"
        # )
        # axis.plot(
        #     gamma_czkm[:cut],
        #     y_ref[:940],
        #     "--",
        #     lw=2.0,
        #     label=r"$e^{-\gamma_0(T/\tau-1)}$",
        # )
        # axis.plot(
        #     gamma_czkm[:cut],
        #     y_sec[:940],
        #     "--",
        #     lw=2.0,
        #     label=r"$e^{-\gamma_0(T/\tau-1)/2}$",
        # )
        axis.plot(
            gamma_ref,
            gamma_ref**2 / 50000,
            "--",
            lw=2.0,
            color=PLOT_COLORS["theory"],
            label=r"$\frac{(\gamma_0\tau)^2}{5\times 10^4}$",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        # axis.grid(True, which="both", alpha=0.30)
        axis.legend(frameon=False)
        place_labels(axis, r"$\gamma_0\tau$", r"$1-F$")
        panel(axis, "(c)")

        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.10, hspace=hspace)

        if save:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()


# ==================================================================
def renormalize_WW_opimized(
    gamma_guess,
    T,
    Delta_guess=0.0,
    tau=1.0,
    dt_max=0.001,
    pulse_delay=False,
    PBC: bool = False,
    n_modes: int = 50,
    n_steps: int = 101,
    plot=True,
    opt=True,
):
    """Fit WW parameters so its dynamics best match the DDE reference trace."""
    # ---------- paras ----------
    setup_kind = WG.Ring if PBC else WG.Cable
    L = (2.0 if PBC else 1.0) * tau
    positions = [0.0, L / 2.0] if PBC else [0.0, L]
    tmax = T * tau

    g1_ref, g2_ref = (gamma_pulse_delay if pulse_delay else gamma_pulse)(
        gamma_guess, T, tau
    )
    t_dde, y_dde = dde_scalar(
        t_max=tmax, gamma1=g1_ref, gamma2=g2_ref, phi=0.0, tau=tau, dt_max=dt_max
    )

    idx = np.argsort(t_dde)
    t_dde = t_dde[idx]
    y_dde = y_dde[idx]
    t_dde, uniq = np.unique(t_dde, return_index=True)
    y_dde = y_dde[uniq]

    y_dde_itp = interp1d(
        t_dde,
        y_dde,
        kind="linear",
        axis=0,
        bounds_error=False,
        fill_value=(y_dde[0], y_dde[-1]),
        assume_sorted=True,
    )

    if pulse_delay:
        limit = (T - 1.0) * tau
        den = 2.0 * limit

        def g_time_mod(t):
            """Return delayed WW pulse envelopes normalized to the static coupling."""
            g1 = np.sin(np.pi * t / den) if t <= limit else 1.0
            g2 = np.cos(np.pi * (t - tau) / den) if t >= tau else 1.0
            return np.array([[g1], [g2]], dtype=float)

    else:
        den = 2.0 * T * tau

        def g_time_mod(t):
            """Return synchronized WW pulse envelopes normalized to the static coupling."""
            g1 = np.sin(np.pi * t / den)
            g2 = np.cos(np.pi * t / den)
            return np.array([[g1], [g2]], dtype=float)

    def cost_func(x):
        """Measure the WW-to-DDE mismatch over the full evolution window."""
        Delta, gamma = x
        ww = WW(
            Delta=Delta,
            positions=positions,
            gamma=gamma,
            n_modes=n_modes,
            L=L,
            setup=setup_kind,
            g_time_modulation=g_time_mod,
        )
        t_WW, pop_WW = ww.evolve(tmax, n_steps=n_steps)
        y = y_dde_itp(t_WW)
        ref = np.abs(y) ** 2
        err = pop_WW[:, : ref.shape[1]] - ref

        num2 = trapezoid(err**2, x=t_WW, axis=0).sum()
        den2 = trapezoid(ref**2, x=t_WW, axis=0).sum() + 1e-24
        return np.sqrt(num2 / den2)

    # ---------- optimize ----------
    if opt:
        res = minimize(
            cost_func,
            x0=np.array([Delta_guess, max(gamma_guess, 1e-6)]),
            method="L-BFGS-B",
            bounds=(
                (max(Delta_guess - 0.5, 1e-6), Delta_guess + 0.5),
                (max(0.5 * gamma_guess, 1e-12), 1.5 * gamma_guess),
            ),
            options={
                "ftol": 1e-10,
                "gtol": 1e-10,
                "eps": 1e-8,
            },
        )

        Delta_opt, gamma_opt = res.x
        loss_opt = res.fun
    else:
        Delta_opt, gamma_opt = Delta_guess, gamma_guess
        loss_opt = cost_func([Delta_opt, gamma_opt])

    # ---------- evaluate optimized WW ----------
    ww_opt = WW(
        Delta=Delta_opt,
        positions=positions,
        gamma=gamma_opt,
        n_modes=n_modes,
        L=L,
        setup=setup_kind,
        g_time_modulation=g_time_mod,
    )
    t_WW, pop_opt = ww_opt.evolve(tmax, n_steps=n_steps)
    Infidelity = 1.0 - pop_opt[-1, 1]

    if plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        m = min(2, y_dde.shape[1], pop_opt.shape[1])
        for j in range(m):
            ax.plot(
                t_dde,
                np.abs(y_dde[:, j]) ** 2,
                "--k",
                lw=1.5,
                label="DDE" if j == 0 else "",
            )
            ax.plot(
                t_WW,
                pop_opt[:, j],
                "-b",
                alpha=0.9,
                label="Optimized WW" if j == 0 else "",
            )
        ax.set(
            xlabel=r"t/$\tau$",
            ylabel="population",
            title=rf"Loss={loss_opt:.3e}, Δ={Delta_guess:g}→{Delta_opt:g}, γ={gamma_guess:g}→{gamma_opt:g}, 1-pop_end={Infidelity:.3e}",
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return Infidelity


# -----------------------------------------------------
def expt_009_renormalized_WW_fidelity(
    Delta_list=(50.0, 5.0, 1.0),
    data="stirap_qst.npz",
    idx_gamma=(40, 80, 100),
    n_jobs=-1,
    savefile="expt_009_renormalized_WW_fidelity.npz",
):
    """Evaluate the renormalized WW infidelity on a small Delta-gamma grid."""
    d = np.load(data, allow_pickle=True)

    gamma_all = np.asarray(d["gamma"], dtype=float)
    T_all = np.asarray(d["T_opt"], dtype=float)

    idx = np.asarray(idx_gamma, dtype=int)
    gamma_picked = gamma_all[idx]
    T_picked = T_all[idx]
    Delta_list = np.asarray(Delta_list, dtype=float)

    def _scalar(x):
        """Convert a scalar-like array entry to a plain Python float."""
        a = np.asarray(x)
        if a.ndim == 0:
            return float(a)
        return float(a.ravel()[0])

    tasks = [(g, T, D) for g, T in zip(gamma_picked, T_picked) for D in Delta_list]

    def one_case(g, T, D):
        """Run one WW renormalization case for a fixed gamma, T, and Delta."""
        g = _scalar(g)
        T = _scalar(T)
        D = _scalar(D)
        return renormalize_WW_opimized(
            gamma_guess=g,
            T=T,
            Delta_guess=D,
            n_modes=101,
            dt_max=0.001,
            n_steps=int(T / 0.001),
            plot=False,
            pulse_delay=False,
            opt=True,
        )

    out = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(one_case)(g, T, D) for (g, T, D) in tasks
    )

    IF_out = np.asarray(out)

    np.savez(
        savefile,
        T=T_picked,
        Delta=Delta_list,
        gamma_in=gamma_picked,
        IF=IF_out,
    )
    print(f"Saved to {savefile}  (N={len(out)})")


# --------------------------------------------------------
def plot_IF_vs_T(
    data0="expt_002_cache.npz",
    data1="expt_008_cache11_nodelay.npz",
    ww_data="expt_009_renormalized_WW_fidelity.npz",
    tau=1.0,
):
    """Compare infidelity-versus-time curves across the cached protocols."""
    # ---------- load ----------
    swap_cache = np.load(data0)
    stirap_cache = np.load(data1)
    ww_cache = np.load(ww_data)

    # ---------- expt_002 : swap ----------
    T_swap = swap_cache["T"] / tau
    IF_swap = 1.0 - swap_cache["F"]

    # ---------- expt_008 : STIRAP ----------
    T_stirap = stirap_cache["T_opt"]
    F_stirap = stirap_cache["F_opt"]

    # ---------- WW optimized ----------
    T_ww = ww_cache["T"]
    D_ww = ww_cache["Delta"]
    IF_ww = ww_cache["IF"].reshape(len(T_ww), len(D_ww))

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Swap
    ax.plot(T_swap, IF_swap, "-", lw=2.8, label="Swap (expt_002)")

    # STIRAP
    ax.plot(T_stirap, 1.0 - F_stirap, "--", label="STIRAP (delay)")
    ax.plot(T_stirap, 1 / T_stirap**4.5)

    # WW scatter
    for i, D in enumerate(D_ww):
        ax.scatter(
            T_ww,
            IF_ww[:, i],
            s=30,
            linewidths=0.8,
            label=rf"WW optimized ($\Delta={D:g}$)",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T/\tau$")
    ax.set_ylabel(r"$1 - F$")
    ax.set_xlim([1, None])
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def CZKM_test(
    gamma,
    T,
    tau,
    dt_max=0.01,
    phi=0.0,
    WW_sim=True,
    plot=True,
    dde_solver="simple",
    compare_mode="dde",
):
    """Compare CZKM dynamics in one of three modes: DDE-vs-DDE or DDE-vs-WW.

    `compare_mode` supports `"dde"`, `"scalar_ww"`, and `"simple_ww"`.
    Returns the final fidelities of the two traces being compared.
    """

    t_max = T * tau
    pulse_scale = 0.5 * gamma
    tanh_scale = 0.5 * gamma
    half_tau = 0.5 * tau
    half_tmax = 0.5 * t_max
    _ = dde_solver
    compare_mode = str(compare_mode).lower()
    if compare_mode not in {"dde", "scalar_ww", "simple_ww"}:
        raise ValueError(
            "compare_mode must be one of {'dde', 'scalar_ww', 'simple_ww'}."
        )

    # pulses
    gamma1 = lambda t: pulse_scale * (
        1.0 + np.tanh(tanh_scale * (t + half_tau - half_tmax))
    )
    gamma2 = lambda t: pulse_scale * (
        1.0 + np.tanh(tanh_scale * (-t + half_tau + half_tmax))
    )

    # --- DDE evolve ---
    t_scalar, c_scalar = dde_scalar(
        t_max=t_max,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
        t_start=0,
    )
    t_simple, c_simple = dde_scalar_simple(
        t_max=t_max,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
        t_start=0,
    )

    # populations
    p_scalar = np.abs(c_scalar[:, :2]) ** 2
    p_simple = np.abs(c_simple[:, :2]) ** 2
    F_scalar = float(np.abs(c_scalar[-1, 1]) ** 2)
    F_simple = float(np.abs(c_simple[-1, 1]) ** 2)

    def prepare_interp_grid(t, y):
        """Sort and deduplicate an interpolation grid before calling interp1d."""
        order = np.argsort(t)
        t_sorted = np.asarray(t, float)[order]
        y_sorted = np.asarray(y)[order]
        t_unique, unique_idx = np.unique(t_sorted, return_index=True)
        return t_unique, y_sorted[unique_idx]

    t_simple_itp, p_simple_itp = prepare_interp_grid(t_simple, p_simple)

    if compare_mode != "dde":
        if not WW_sim:
            raise ValueError("WW_sim must be True when compare_mode uses WW.")

        L = tau
        positions = [0.0, L]
        norm = np.sqrt(gamma)

        def g_time_mod(t):
            """Map the DDE pulse rates to the normalized WW modulation amplitudes."""
            return np.array(
                [[np.sqrt(gamma1(t)) / norm], [np.sqrt(gamma2(t)) / norm]],
                dtype=float,
            )

        ww = WW(
            Delta=1000,
            positions=positions,
            gamma=gamma,
            n_modes=1001,
            L=L,
            setup=WG.Cable,
            g_time_modulation=g_time_mod,
        )
        t_ww, pop_ww = ww.evolve(t_max, n_steps=int(t_max / dt_max) + 1)
        p_ww = pop_ww[:, :2]
        F_ww = float(pop_ww[-1, 1])
        t_ww_itp, p_ww_itp = prepare_interp_grid(t_ww, p_ww)

    if compare_mode == "dde":
        compare_t = t_scalar
        compare_a = p_scalar
        compare_b = interp1d(
            t_simple_itp,
            p_simple_itp,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )(compare_t)
        label_a = "dde_scalar"
        label_b = "dde_simple"
        F_a, F_b = F_scalar, F_simple
    elif compare_mode == "scalar_ww":
        compare_t = t_scalar
        compare_a = p_scalar
        compare_b = interp1d(
            t_ww_itp,
            p_ww_itp,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )(compare_t)
        label_a = "dde_scalar"
        label_b = "WW"
        F_a, F_b = F_scalar, F_ww
    else:
        compare_t = t_simple
        compare_a = p_simple
        compare_b = interp1d(
            t_ww_itp,
            p_ww_itp,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )(compare_t)
        label_a = "dde_simple"
        label_b = "WW"
        F_a, F_b = F_simple, F_ww

    p_diff = compare_a - compare_b

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

        # left: direct difference
        ax[0].plot(
            compare_t, p_diff[:, 0], color=PLOT_COLORS["q1"], label=r"$\Delta |c_1|^2$"
        )
        ax[0].plot(
            compare_t, p_diff[:, 1], color=PLOT_COLORS["q2"], label=r"$\Delta |c_2|^2$"
        )
        ax[0].set_xlabel("t")
        ax[0].set_ylabel(r"$error$")
        ax[0].legend()

        ax[1].plot(
            compare_t,
            compare_a[:, 0],
            color=PLOT_COLORS["q1"],
            label=rf"{label_a}: $|c_1|^2$",
        )
        ax[1].plot(
            compare_t,
            compare_a[:, 1],
            color=PLOT_COLORS["q2"],
            label=rf"{label_a}: $|c_2|^2$",
        )
        ax[1].plot(
            compare_t,
            compare_b[:, 0],
            "--",
            color=PLOT_COLORS["dde"],
            label=rf"{label_b}: $|c_1|^2$",
        )
        ax[1].plot(
            compare_t,
            compare_b[:, 1],
            "--",
            color=PLOT_COLORS["ww"],
            label=rf"{label_b}: $|c_2|^2$",
        )
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("population")
        ax[1].legend()
        ax[1].set_title("Dynamics")

        fig.tight_layout()

    return F_a, F_b


def QST_CZKM(
    gamma,
    T,
    tau,
    dt_max=0.01,
    phi=0.0,
    WW_sim=True,
    plot=True,
    dde_solver="simple",
    return_link=False,
):
    """
    Smooth CZKM-like emission/absorption pulses for 2-node QST under DDE.

    gamma: overall coupling scale
    T    : dimensionless total time in units of tau (t_max = T*tau)
    tau  : delay time
    """

    t_max = T * tau
    pulse_scale = 0.5 * gamma
    tanh_scale = 0.5 * gamma
    half_tau = 0.5 * tau
    half_tmax = 0.5 * t_max

    # pulses
    gamma1 = lambda t: pulse_scale * (
        1.0 + np.tanh(tanh_scale * (t + half_tau - half_tmax))
    )
    gamma2 = lambda t: pulse_scale * (
        1.0 + np.tanh(tanh_scale * (-t + half_tau + half_tmax))
    )

    # --- DDE evolve ---
    dde = dde_scalar_simple if dde_solver == "simple" else dde_scalar
    t_list, c_dde = dde(
        t_max=t_max,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
        t_start=0,
    )

    # populations (DDE)
    p1_dde = np.abs(c_dde[:, 0]) ** 2
    p2_dde = np.abs(c_dde[:, 1]) ** 2
    # sample pulses on the same grid
    g1 = np.array([gamma1(t) for t in t_list], dtype=float)
    g2 = np.array([gamma2(t) for t in t_list], dtype=float)
    if WW_sim:
        # --- WW evolve ---
        L = tau
        positions = [0.0, L]
        norm = np.sqrt(gamma)

        def g_time_mod(t):
            """Map the DDE pulse rates to the normalized WW modulation amplitudes."""
            return np.array(
                [[np.sqrt(gamma1(t)) / norm], [np.sqrt(gamma2(t)) / norm]],
                dtype=float,
            )

        ww = WW(
            Delta=1000,
            positions=positions,
            gamma=gamma,
            n_modes=1001,
            L=L,
            setup=WG.Cable,
            g_time_modulation=g_time_mod,
        )
        t_WW, pop_WW = ww.evolve(t_max, n_steps=2001)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

        # left: pulses
        ax[0].plot(t_list, g1, color=PLOT_COLORS["q1"], label=r"$\gamma_1(t)$")
        ax[0].plot(t_list, g2, color=PLOT_COLORS["q2"], label=r"$\gamma_2(t)$")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel(r"$\gamma(t)$")
        ax[0].legend()
        ax[0].set_title("Pulses")
        # ax[0].set_xlim(0, t_max)

        ax[1].plot(t_list, p1_dde, color=PLOT_COLORS["q1"], label=r"DDE: $|c_1|^2$")
        ax[1].plot(t_list, p2_dde, color=PLOT_COLORS["q2"], label=r"DDE: $|c_2|^2$")
        if WW_sim:
            ax[1].plot(
                t_WW,
                pop_WW[:, 0],
                "--",
                color=PLOT_COLORS["dde"],
                label=r"WW: $|c_1|^2$",
            )
            ax[1].plot(
                t_WW,
                pop_WW[:, 1],
                "--",
                color=PLOT_COLORS["ww"],
                label=r"WW: $|c_2|^2$",
            )
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("population")
        ax[1].legend()
        ax[1].set_title("Dynamics")
        # ax[1].set_xlim(0, t_max)

        fig.tight_layout()

    F = float(np.abs(c_dde[-1, 1]) ** 2)
    if not return_link:
        F_ww = float(pop_WW[-1, 1]) if WW_sim else None
        return (F, F_ww) if WW_sim else F

    n_link = np.clip(1.0 - np.sum(np.abs(c_dde[:, :2]) ** 2, axis=1), 0.0, None)
    N_link = float(trapezoid(n_link, x=t_list))
    F_ww = float(pop_WW[-1, 1]) if WW_sim else None
    return (F, F_ww, N_link) if WW_sim else (F, N_link)


def CZKM_app(
    gamma_list,
    T_list,
    tau=1.0,
    dt_max=0.01,
    gamma0=None,
    T=None,
    cache_file="czkm_app.npz",
    overwrite=False,
    n_jobs=-1,
    backend="loky",
    filename="fig_czkm_app.pdf",
    show=True,
    dde_solver="simple",
    figsize=(4.3, 6),
    hspace=0.25,
):
    """Compute CZKM fidelities and plot infidelity plus one representative dynamics panel."""
    gamma_is_scalar = np.isscalar(gamma_list)
    T_is_scalar = np.isscalar(T_list)
    if gamma_is_scalar and T_is_scalar:
        gamma_arr = np.array([float(gamma_list)])
        T_arr = np.array([float(T_list)])
    else:
        gamma_arr = np.asarray(gamma_list, float).ravel()
        T_arr = np.asarray(T_list, float).ravel()
        if gamma_is_scalar:
            gamma_arr = np.full(T_arr.shape, float(gamma_list))
        if T_is_scalar:
            T_arr = np.full(gamma_arr.shape, float(T_list))

    if gamma_arr.shape != T_arr.shape:
        raise ValueError(
            f"gamma_list and T_list must match: {gamma_arr.shape} vs {T_arr.shape}"
        )

    need_compute = overwrite or (not os.path.exists(cache_file))

    if not need_compute:
        print(f"[CZKM] Loading cached data from {cache_file}")
        cache = np.load(cache_file)
        gamma_arr = cache["gamma"]
        T_arr = cache["T"]
        F_arr = np.asarray(cache["F"], float)
        N_link_arr = (
            np.asarray(cache["N_link"], float)
            if "N_link" in cache.files
            else (
                -np.log(np.clip(np.asarray(cache["P_link_loss"], float), 1e-300, 1.0))
                / float(cache["kappa"])
                if "P_link_loss" in cache.files and "kappa" in cache.files
                else np.full(gamma_arr.shape, np.nan, dtype=float)
            )
        )
    else:
        if overwrite and os.path.exists(cache_file):
            print("[CZKM] Overwriting cache.")

        else:
            print("[CZKM] Computing fidelity (parallel)...")

        def rub_one(g, T):
            """Run one CZKM transfer simulation without plotting."""
            return QST_CZKM(
                g,
                T,
                tau,
                dt_max=dt_max,
                phi=0.0,
                WW_sim=False,
                plot=False,
                dde_solver=dde_solver,
                return_link=True,
            )

        results = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
            delayed(rub_one)(g, T) for g, T in zip(gamma_arr, T_arr)
        )
        F_arr, N_link_arr = (np.asarray(values, float) for values in zip(*results))

        np.savez(
            cache_file,
            gamma=gamma_arr,
            T=T_arr,
            F=F_arr,
            N_link=N_link_arr,
        )
        print(f"[CZKM] Saved to {cache_file}")

    T_over_tau = T_arr / float(tau)
    gamma_tau = gamma_arr * tau
    F_dde = F_arr[:, 0] if F_arr.ndim > 1 else F_arr
    infidelity = 1.0 - F_dde
    theory_full = np.exp(-gamma_arr * (T_over_tau - 1.0))
    theory_half = np.exp(-gamma_arr * (T_over_tau - 1.0) / 2)

    cut = min(940, T_over_tau.size)
    idx_dyn = min(len(gamma_arr) // 2, len(gamma_arr) - 1)
    gamma0 = float(gamma_arr[idx_dyn] if gamma0 is None else gamma0)
    T = float(T_arr[idx_dyn] if T is None else T)
    t_max_dyn = T * tau
    pulse_scale = 0.5 * gamma0
    tanh_scale = 0.5 * gamma0
    half_tau = 0.5 * tau
    half_tmax = 0.5 * t_max_dyn

    gamma1 = lambda t: pulse_scale * (
        1.0 + np.tanh(tanh_scale * (t + half_tau - half_tmax))
    )
    gamma2 = lambda t: pulse_scale * (
        1.0 + np.tanh(tanh_scale * (-t + half_tau + half_tmax))
    )

    dde = dde_scalar_simple if dde_solver == "simple" else dde_scalar
    t_dyn_dde, c_dyn_dde = dde(
        t_max=t_max_dyn,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=0.0,
        tau=tau,
        dt_max=dt_max,
        t_start=0,
    )
    p_dyn_dde = np.abs(c_dyn_dde[:, :2]) ** 2

    positions = [0.0, tau]
    norm = np.sqrt(gamma0)

    ww = WW(
        Delta=100,
        positions=positions,
        gamma=gamma0,
        n_modes=201,
        L=tau,
        setup=WG.Cable,
        g_time_modulation=lambda t: np.array(
            [
                [np.sqrt(np.maximum(gamma1(t), 0.0)) / norm],
                [np.sqrt(np.maximum(gamma2(t), 0.0)) / norm],
            ],
            dtype=float,
        ),
    )
    t_dyn_ww, pop_dyn_ww = ww.evolve(
        t_max_dyn, n_steps=max(int(t_max_dyn / dt_max) + 1, 1001)
    )

    # Plot
    with plt.rc_context():
        set_plot_style()
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = False

        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw=dict(hspace=hspace))
        ax_D, ax_F = axes

        ax_F.plot(
            gamma_tau[:cut],
            infidelity[:cut],
            "-",
            color=PLOT_COLORS["czkm"],
            label="sech-shape",
        )
        ax_F.plot(
            gamma_tau[:cut],
            theory_half[:cut],
            "--",
            color=PLOT_COLORS["theory"],
            label=r"$e^{-\gamma (T-\tau)/2}$",
        )
        ax_F.plot(
            gamma_tau[:cut],
            theory_full[:cut],
            "--",
            color=PLOT_COLORS["guide"],
            label=r"$e^{-\gamma (T-\tau)}$",
        )

        ax_F.set_xlabel(r"$\gamma_0\tau$")
        ax_F.set_ylabel(r"$1 - F$")
        ax_F.xaxis.set_label_coords(0.50, -0.15)
        ax_F.yaxis.set_label_coords(-0.13, 0.50)
        ax_F.set_xscale("log")
        ax_F.set_yscale("log")
        ax_F.legend(frameon=False, loc="best")
        ax_F.text(-0.2, 0.95, "(b)", transform=ax_F.transAxes, ha="left", va="bottom")

        ax_D.plot(
            t_dyn_dde / tau,
            p_dyn_dde[:, 0],
            lw=2.0,
            color=PLOT_COLORS["q1"],
            label=r"DDE: $Q_1$",
        )
        ax_D.plot(
            t_dyn_dde / tau,
            p_dyn_dde[:, 1],
            lw=2.0,
            color=PLOT_COLORS["q2"],
            alpha=0.55,
            label=r"DDE: $Q_2$",
        )
        ax_D.plot(
            t_dyn_ww / tau,
            pop_dyn_ww[:, 0],
            "--",
            lw=2.0,
            color=PLOT_COLORS["dde"],
            label=r"WW: $Q_1$",
        )
        ax_D.plot(
            t_dyn_ww / tau,
            pop_dyn_ww[:, 1],
            "--",
            lw=2.0,
            color=PLOT_COLORS["ww"],
            alpha=0.65,
            label=r"WW: $Q_2$",
        )
        ax_D.set(xlim=(0, T), ylim=(-0.02, 1.02))
        ax_D.set_xlabel(r"$t/\tau$")
        ax_D.set_ylabel(r"$\langle \sigma^+ \sigma \rangle$")
        ax_D.xaxis.set_label_coords(0.50, -0.15)
        ax_D.yaxis.set_label_coords(-0.13, 0.50)
        ax_D.legend(frameon=False, loc="best")
        ax_D.text(-0.2, 0.95, "(a)", transform=ax_D.transAxes, ha="left", va="bottom")

        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.14, hspace=hspace)

        if filename:
            fig.savefig(filename, dpi=600, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


def fig_paper_IF_vs_T(
    data0="swap_qst.npz",
    data1="stirap_qst.npz",
    ww_data="expt_009_renormalized_WW_fidelity.npz",
    czkm_data="czkm_qst.npz",
    tau=1.0,
    figsize=(7.2, 2.2),
    save=None,
    dpi=600,
):
    """Plot the transfer infidelity comparison as a standalone figure."""

    def place_labels(axis, xlabel, ylabel, x=(-0.0, -0.12), y=(-0.12, 0.5)):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.xaxis.set_label_coords(0.5, x[1])
        axis.yaxis.set_label_coords(y[0], y[1])

    def annot(axis, txt, x, y, xt, yt, ha="left", va="top"):
        axis.annotate(
            txt,
            xy=(x, y),
            xytext=(xt, yt),
            color=PLOT_COLORS["error"],
            arrowprops=dict(
                arrowstyle="->",
                color=PLOT_COLORS["error"],
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
            ),
            ha=ha,
            va=va,
        )

    swap_cache = np.load(data0)
    stirap_cache = np.load(data1)
    ww_cache = np.load(ww_data)
    czkm_cache = np.load(czkm_data)

    T_swap = swap_cache["T"] / tau
    IF_swap = 1.0 - swap_cache["F"]
    T_stirap = stirap_cache["T_opt"]
    IF_stirap = 1.0 - stirap_cache["F_opt"]
    gamma_stirap = stirap_cache["gamma"]
    T_czkm = czkm_cache["T"] / tau
    F_czkm = np.asarray(czkm_cache["F"], float)
    IF_czkm = 1.0 - (F_czkm[:, 0] if F_czkm.ndim > 1 else F_czkm)
    T_ww = ww_cache["T"]
    D_ww = ww_cache["Delta"]
    IF_ww = ww_cache["IF"].reshape(len(T_ww), len(D_ww))

    cut_swap = len(T_swap)
    cut_stirap = min(940, len(T_stirap))
    cut_czkm = min(940, len(T_czkm))

    with plt.rc_context():
        set_plot_style()
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = False

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            T_swap[:cut_swap],
            IF_swap[:cut_swap],
            "-",
            lw=2.0,
            color=PLOT_COLORS["swap"],
        )
        ax.plot(
            T_stirap[:cut_stirap],
            IF_stirap[:cut_stirap],
            "-",
            lw=2.0,
            color=PLOT_COLORS["stirap"],
        )
        ax.plot(
            T_stirap[:cut_stirap],
            np.exp(-gamma_stirap[:cut_stirap] * (T_stirap[:cut_stirap] - 1)),
            "--",
            lw=1.6,
            color=PLOT_COLORS["theory"],
        )
        ax.plot(
            T_czkm[:cut_czkm],
            IF_czkm[:cut_czkm],
            "-",
            lw=2.0,
            color=PLOT_COLORS["czkm"],
        )

        markers = ["o", "D", "^"]
        sizes = [55, 40, 70]
        edge_widths = [0.8, 0.8, 1.0]
        colors = [
            PLOT_COLORS["scatter_a"],
            PLOT_COLORS["scatter_b"],
            PLOT_COLORS["scatter_c"],
        ]

        for j, (col, marker, size, width) in enumerate(
            zip(colors, markers, sizes, edge_widths)
        ):
            ax.scatter(
                T_ww,
                IF_ww[:, j],
                s=size,
                linewidths=width,
                color=col,
                edgecolor="white",
                marker=marker,
                label=rf"$\Delta:{D_ww[j]:g}$",
                zorder=5,
            )

        i_swap = min(870, cut_swap - 1)
        i_sti = min(518, cut_stirap - 1)
        i_czkm = min(350, cut_czkm - 1)
        annot(
            ax,
            "SWAP",
            T_swap[i_swap],
            IF_swap[i_swap],
            T_swap[i_swap] * 1.2,
            IF_swap[i_swap] * 0.05,
        )
        annot(
            ax,
            "STIRAP",
            T_stirap[i_sti],
            IF_stirap[i_sti],
            T_stirap[i_sti] * 0.7,
            IF_stirap[i_sti] * 2.5,
            ha="right",
        )
        annot(
            ax,
            "CZKM",
            T_czkm[i_czkm],
            IF_czkm[i_czkm],
            T_czkm[i_czkm] * 1.15,
            IF_czkm[i_czkm] * 0.35,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1.2)
        ax.legend(frameon=False, loc=(0.05, 0.02))
        place_labels(ax, r"$T/\tau$", r"$1-F$")

        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.22)

        if save is not None:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()


def fig_paper_loss_vs_T(
    data0="swap_qst.npz",
    data1="stirap_qst.npz",
    czkm_data="czkm_qst.npz",
    kappa=0.1,
    tau=1.0,
    fit=True,
    figsize=(7.2, 2.2),
    save=None,
    dpi=600,
):
    """Plot the link-loss comparison as a standalone figure."""

    def place_labels(axis, xlabel, ylabel, x=(-0.0, -0.12), y=(-0.12, 0.5)):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.xaxis.set_label_coords(0.5, x[1])
        axis.yaxis.set_label_coords(y[0], y[1])

    def load_link_integral(cache):
        if "N_link" in cache.files:
            return np.asarray(cache["N_link"], float)
        if "P_link_loss" in cache.files and "kappa" in cache.files:
            return -np.log(
                np.clip(np.asarray(cache["P_link_loss"], float), 1e-300, 1.0)
            ) / float(cache["kappa"])
        if "P_loss" in cache.files:
            return np.asarray(cache["P_loss"], float)
        raise KeyError(
            "Cache does not contain N_link or a recoverable legacy loss field."
        )

    def fit_log_poly(x, y, degree=2):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        y_fit = np.full_like(x, np.nan, dtype=float)
        coeffs = np.polyfit(np.log(x[valid]), np.log(y[valid]), degree)
        y_fit[valid] = np.exp(np.polyval(coeffs, np.log(x[valid])))
        return coeffs, y_fit

    swap_cache = np.load(data0)
    stirap_cache = np.load(data1)
    czkm_cache = np.load(czkm_data)

    T_swap = swap_cache["T"] / tau
    T_stirap = stirap_cache["T_opt"]
    T_czkm = czkm_cache["T"] / tau
    loss_swap = 1.0 - np.exp(-kappa * load_link_integral(swap_cache))
    loss_stirap = 1.0 - np.exp(-kappa * load_link_integral(stirap_cache))
    loss_czkm = 1.0 - np.exp(-kappa * load_link_integral(czkm_cache))

    cut_swap = len(T_swap)
    cut_stirap = min(940, len(T_stirap))
    cut_czkm = min(940, len(T_czkm))

    with plt.rc_context():
        set_plot_style()
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = False

        fig, ax = plt.subplots(figsize=figsize)
        coeff_swap = coeff_stirap = coeff_czkm = None
        fit_swap = fit_stirap = fit_czkm = None
        if fit:
            coeff_swap, fit_swap = fit_log_poly(T_swap[:cut_swap], loss_swap[:cut_swap])
            coeff_stirap, fit_stirap = fit_log_poly(
                T_stirap[:cut_stirap], loss_stirap[:cut_stirap]
            )
            coeff_czkm, fit_czkm = fit_log_poly(T_czkm[:cut_czkm], loss_czkm[:cut_czkm])

        ax.plot(
            T_swap[:cut_swap],
            loss_swap[:cut_swap],
            "-",
            lw=2.0,
            color=PLOT_COLORS["swap"],
            label="SWAP",
        )
        ax.plot(
            T_stirap[:cut_stirap],
            loss_stirap[:cut_stirap],
            "-",
            lw=2.0,
            color=PLOT_COLORS["stirap"],
            label="STIRAP",
        )
        ax.plot(
            T_czkm[:cut_czkm],
            loss_czkm[:cut_czkm],
            "-",
            lw=2.0,
            color=PLOT_COLORS["czkm"],
            label="CZKM",
        )
        if fit:
            ax.plot(
                T_swap[:cut_swap],
                fit_swap,
                "--",
                lw=1.6,
                color=PLOT_COLORS["swap"],
                alpha=0.8,
            )
            ax.plot(
                T_stirap[:cut_stirap],
                fit_stirap,
                "--",
                lw=1.6,
                color=PLOT_COLORS["stirap"],
                alpha=0.8,
            )
            ax.plot(
                T_czkm[:cut_czkm],
                fit_czkm,
                "--",
                lw=1.6,
                color=PLOT_COLORS["czkm"],
                alpha=0.8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="best")
        place_labels(ax, r"$T/\tau$", r"$\varepsilon_{\rm loss}$")

        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.22)

        if save is not None:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()

    def format_log_poly(name, coeffs):
        degree = len(coeffs) - 1
        terms = []
        for power, coef in zip(range(degree, -1, -1), coeffs):
            if power == 0:
                terms.append(f"{coef:.6g}")
            elif power == 1:
                terms.append(f"{coef:.6g}*log(T/tau)")
            else:
                terms.append(f"{coef:.6g}*(log(T/tau))^{power}")
        print(f"[{name}] log(P_loss) ~= {' + '.join(terms).replace('+ -', '- ')}")

    if fit:
        format_log_poly("SWAP", coeff_swap)
        format_log_poly("STIRAP", coeff_stirap)
        format_log_poly("CZKM", coeff_czkm)


def fig_paper_loss_vs_T_poly(
    data0="swap_qst.npz",
    data1="stirap_qst.npz",
    czkm_data="czkm_qst.npz",
    kappa=0.1,
    tau=1.0,
    degree=2,
    fit_min=None,
    fit_max=None,
    figsize=(4.3, 3),
    save=None,
    dpi=600,
):
    """Plot the link loss together with power-law fits in T/tau."""

    def place_labels(axis, xlabel, ylabel, x=(-0.0, -0.12), y=(-0.12, 0.5)):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.xaxis.set_label_coords(0.5, x[1])
        axis.yaxis.set_label_coords(y[0], y[1])

    def load_link_integral(cache):
        if "N_link" in cache.files:
            return np.asarray(cache["N_link"], float)
        if "P_link_loss" in cache.files and "kappa" in cache.files:
            return -np.log(
                np.clip(np.asarray(cache["P_link_loss"], float), 1e-300, 1.0)
            ) / float(cache["kappa"])
        if "P_loss" in cache.files:
            return np.asarray(cache["P_loss"], float)
        raise KeyError(
            "Cache does not contain N_link or a recoverable legacy loss field."
        )

    def fit_power_law(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        fit_valid = valid.copy()
        if fit_min is not None:
            fit_valid &= x >= fit_min
        if fit_max is not None:
            fit_valid &= x <= fit_max
        coeffs = np.polyfit(np.log(x[fit_valid]), np.log(y[fit_valid]), 1)
        y_fit = np.full_like(x, np.nan, dtype=float)
        y_fit[valid] = np.exp(coeffs[1]) * x[valid] ** coeffs[0]
        return coeffs, y_fit

    swap_cache = np.load(data0)
    stirap_cache = np.load(data1)
    czkm_cache = np.load(czkm_data)

    T_swap = swap_cache["T"] / tau
    T_stirap = stirap_cache["T_opt"]
    T_czkm = czkm_cache["T"] / tau
    loss_swap = 1.0 - np.exp(-kappa * load_link_integral(swap_cache))
    loss_stirap = 1.0 - np.exp(-kappa * load_link_integral(stirap_cache))
    loss_czkm = 1.0 - np.exp(-kappa * load_link_integral(czkm_cache))

    cut_swap = len(T_swap)
    cut_stirap = min(900, len(T_stirap))
    cut_czkm = min(900, len(T_czkm))

    coeff_swap, fit_swap = fit_power_law(T_swap[:cut_swap], loss_swap[:cut_swap])
    coeff_stirap, fit_stirap = fit_power_law(
        T_stirap[:cut_stirap], loss_stirap[:cut_stirap]
    )
    coeff_czkm, fit_czkm = fit_power_law(T_czkm[:cut_czkm], loss_czkm[:cut_czkm])

    with plt.rc_context():
        set_plot_style()
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.grid"] = False

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            T_swap[:cut_swap],
            loss_swap[:cut_swap],
            "-",
            lw=2.0,
            color=PLOT_COLORS["swap"],
            label="SWAP",
        )
        ax.plot(
            T_stirap[:cut_stirap],
            loss_stirap[:cut_stirap],
            "-",
            lw=2.0,
            color=PLOT_COLORS["stirap"],
            label="STIRAP",
        )
        ax.plot(
            T_czkm[:cut_czkm],
            loss_czkm[:cut_czkm],
            "-",
            lw=2.0,
            color=PLOT_COLORS["czkm"],
            label="CZKM",
        )
        ax.plot(
            T_swap[:cut_swap],
            fit_swap,
            "--",
            lw=1.6,
            color=PLOT_COLORS["swap"],
            alpha=0.8,
        )
        ax.plot(
            T_stirap[:cut_stirap],
            fit_stirap,
            "--",
            lw=1.6,
            color=PLOT_COLORS["stirap"],
            alpha=0.8,
        )
        ax.plot(
            T_czkm[:cut_czkm],
            fit_czkm,
            "--",
            lw=1.6,
            color=PLOT_COLORS["czkm"],
            alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="best")
        place_labels(ax, r"$T/\tau$", r"$\varepsilon_{\rm loss}$")

        fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.22)

        if save is not None:
            fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.show()

    def format_power_law(name, coeffs):
        alpha, log_c = coeffs
        print(f"[{name}] P_loss ~= {np.exp(log_c):.6g} * (T/tau)^({alpha:.6g})")

    format_power_law("SWAP", coeff_swap)
    format_power_law("STIRAP", coeff_stirap)
    format_power_law("CZKM", coeff_czkm)


def fitting_linkloss(
    data0="swap_qst.npz",
    data1="stirap_qst.npz",
    ww_data="expt_009_renormalized_WW_fidelity.npz",
    czkm_data="czkm_qst.npz",
    kappa=0.1,
    tau=1.0,
):
    # ----------------- helpers -----------------

    def load_link_integral(cache):
        """Load the integrated link occupancy from current or legacy cache fields."""
        if "N_link" in cache.files:
            return np.asarray(cache["N_link"], float)
        if "P_link_loss" in cache.files and "kappa" in cache.files:
            return -np.log(
                np.clip(np.asarray(cache["P_link_loss"], float), 1e-300, 1.0)
            ) / float(cache["kappa"])
        if "P_loss" in cache.files:
            return np.asarray(cache["P_loss"], float)
        raise KeyError(
            "Cache does not contain N_link or a recoverable legacy loss field."
        )

    # ===================== cached data =====================
    swap_cache = np.load(data0)
    stirap_cache = np.load(data1)
    ww_cache = np.load(ww_data)
    czkm_cache = np.load(czkm_data)

    T_swap = swap_cache["T"] / tau
    IF_swap = 1.0 - swap_cache["F"]
    N_link_swap = load_link_integral(swap_cache)

    T_stirap = stirap_cache["T_opt"]
    IF_stirap = 1.0 - stirap_cache["F_opt"]
    gamma_stirap = stirap_cache["gamma"]
    N_link_stirap = load_link_integral(stirap_cache)

    T_czkm = czkm_cache["T"] / tau
    F_czkm = np.asarray(czkm_cache["F"], float)
    IF_czkm = 1.0 - (F_czkm[:, 0] if F_czkm.ndim > 1 else F_czkm)
    N_link_czkm = load_link_integral(czkm_cache)

    T_ww = ww_cache["T"]
    D_ww = ww_cache["Delta"]
    IF_ww = ww_cache["IF"].reshape(len(T_ww), len(D_ww))

    cut_swap = len(T_swap)
    cut_stirap = min(940, len(T_stirap))
    cut_czkm = min(940, len(T_czkm))
    loss_swap = 1 - np.exp(-kappa * N_link_swap)
    loss_stirap = 1 - np.exp(-kappa * N_link_stirap)
    loss_czkm = 1 - np.exp(-kappa * N_link_czkm)
    import numpy as np

    def fit_log_series(x, y, degree=2, mask_positive=True):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if mask_positive:
            valid_mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        else:
            valid_mask = np.isfinite(x) & np.isfinite(y)

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        logx = np.log(x_valid)
        logy = np.log(y_valid)

        coeffs = np.polyfit(logx, logy, degree)

        y_fit = np.full_like(x, np.nan, dtype=float)
        y_fit[valid_mask] = np.exp(np.polyval(coeffs, np.log(x_valid)))

        return coeffs, y_fit, valid_mask
