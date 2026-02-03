import numpy as np
import os
import matplotlib.pyplot as plt
import math
import scipy
from joblib import Parallel, delayed
from numpy.polynomial import Polynomial
from qnetwork.dde import EmittersInWaveguideDDE
from qnetwork.ww import EmittersInWaveguideWW
from qnetwork.multiphoton_ww import EmittersInWaveguideMultiphotonWW, Waveguide
from aux_funs import dde_scalar
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from two_qubit_control import EmittersInWaveguideWW as WW
from two_qubit_control import Waveguide as WG
from scipy.integrate import trapezoid


def expt_001_dynamics(
    Delta: float = 0.0,
    gamma: float = 0.1,
    tau: float = 50.0,
    n_modes: int = 50,
    c: float = 1.0,
    PBC: bool = False,
    multiphoton: bool = False,
    T: float = 100.0,
    dt_max: float = 0.01,
    n_steps: int = 101,
):
    """Compare dynamics: DDE vs WW vs TC (and add simple Rabi-period markers)."""

    initial = "10"
    setup_kind = Waveguide.Ring if PBC else Waveguide.Cable
    tmax = T * tau

    def run_once(Delta_val: float):
        L = (2 if PBC else 1) * tau * c
        positions = [0.0, L / 2] if PBC else [0.0, L]

        WW_cls = (
            EmittersInWaveguideMultiphotonWW if multiphoton else EmittersInWaveguideWW
        )
        setup_WW = WW_cls(
            Delta=Delta_val,
            positions=positions,
            gamma=gamma,
            n_modes=n_modes,
            L=L,
            setup=setup_kind,
        )
        t_WW, pop_WW = setup_WW.evolve(tmax, n_steps=n_steps)

        delta = math.modf(Delta_val)[0]
        phi = delta * setup_WW.FSR * tau
        setup_dde = EmittersInWaveguideDDE(
            phi=phi, N=2, gamma=gamma, U=-1, tau=tau, dt_max=dt_max
        )
        setup_dde.evolve(tmax)
        t_DDE, pop_DDE = setup_dde.n_photons(initial)

        t_TC, pop_TC = setup_dde.evolve_TC(tmax, initial)
        return t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC

    t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC = run_once(Delta)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(rf"Dynamics ($\Delta={int(Delta)}*FSR$)")

    # WW + DDE
    ax.plot(t_WW / tau, pop_WW[:, 0], color="#0a4570", label=r"$Q_1$: WW")
    ax.plot(t_WW / tau, pop_WW[:, 1], color="#af1a2e", label=r"$Q_2$: WW")
    ax.plot(t_DDE / tau, pop_DDE[:, 0], "--", color="#055805", label=r"$Q_1$: DDE")
    ax.plot(t_DDE / tau, pop_DDE[:, 1], "--", color="#055805", label=r"$Q_2$: DDE")

    # TC markers (two qubits)
    tc_style = [
        dict(markerfacecolor="#3c91ce", markeredgecolor="#0a4570"),
        dict(markerfacecolor="#e02e46", markeredgecolor="#af1a2e"),
    ]
    for j in (0, 1):
        ax.plot(
            t_TC / tau,
            pop_TC[:, j],
            linestyle="none",
            marker="s",
            markevery=20,
            markeredgewidth=2,
            **tc_style[j],
        )

    # Rabi period lines
    g = np.sqrt(gamma / 2)
    Omega = np.sqrt(2) * g
    T_rabi = np.pi / Omega
    for k in (1, 2, 3):
        ax.axvline(k * T_rabi, linestyle="--", color="black")

    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$\langle \sigma^\dagger \sigma \rangle$")
    ax.legend()
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------
def dde_series_function(gamma, tau, eta, N, alpha=None):
    if alpha is None:
        alpha = 0.5 * gamma

    P = Polynomial([1.0])
    polys: list[tuple[int, Polynomial]] = []
    for n in range(1, N + 1):
        Q = P.integ()
        P = P + Q
        polys.append((n, Q))

    n_list = np.arange(1, N + 1)
    eta_pows = eta**n_list
    n_tau = n_list * tau

    def evaluate(t):
        t = np.asarray(t, dtype=float)
        result = np.exp(-alpha * t).astype(np.complex128, copy=False)

        for (n, Q), eta_n, nt in zip(polys, eta_pows, n_tau):
            tn = t - nt
            H = np.heaviside(tn, 0.0)
            if np.any(H):
                result += eta_n * np.exp(-alpha * tn) * Q(-gamma * tn) * H
        return result

    return evaluate


def compute_period_and_fidelity(Delta, gamma, tau=1, T_min=0.8, T_max=1.2):
    # phase
    FSR = np.pi / tau
    delta = math.modf(Delta)[0]
    phi = delta * FSR * tau
    eta_b = np.exp(1j * phi)
    eta_d = -np.exp(1j * phi)

    # p2(gamma, t)
    def make_p2(gamma, T_max):
        cb = dde_series_function(gamma, tau, eta_b, int(T_max / tau))
        cd = dde_series_function(gamma, tau, eta_d, int(T_max / tau))
        return lambda t: np.abs(0.5 * (cb(t) - cd(t))) ** 2

    g = np.sqrt(gamma / (2 * tau))
    # analytic Omega
    Omega = np.sqrt(Delta**2 + 8 * g**2) / 2
    T = np.pi / Omega
    p2 = make_p2(gamma, T_max * T)

    # peak search
    res = scipy.optimize.minimize_scalar(
        lambda t: -p2(t),
        bounds=(T_min * T, T_max * T),
        method="bounded",
        options={"xatol": 1e-9},
    )
    t_peak, F = float(res.x), float(-res.fun)
    return g / FSR, T, t_peak, F


def expt_002_swapspeed(
    Delta,
    gamma_list,
    tau=1.0,
    T_min=0,
    T_max=1.9,
    n_jobs=-1,
    overwrite=False,
    filename="expt_002_cache.npz",
):
    fpath = filename

    # ---------- load ----------
    if os.path.exists(fpath) and not overwrite:
        print(f"[load] {fpath}")
        data = np.load(fpath)
        g_list = data["g"]
        T_list = data["T"]
        t_list = data["t"]
        F_list = data["F"]

    # ---------- compute ----------
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

        g_list = np.array([r[0] for r in results])
        T_list = np.array([r[1] for r in results])
        t_list = np.array([r[2] for r in results])
        F_list = np.array([r[3] for r in results])

        np.savez(
            fpath,
            g=g_list,
            T=T_list,
            t=t_list,
            F=F_list,
            Delta=Delta,
            tau=tau,
            T_min=T_min,
            T_max=T_max,
        )
        print(f"[save] {fpath}")

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # left: swap speed
    ax[0].plot(g_list, t_list / tau, "-", label="Swap speed")
    ax[0].plot(g_list, T_list / tau, "--", label=r"$\pi/\Omega$")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("g / FSR")
    ax[0].set_ylabel("T / τ")
    ax[0].grid(True)
    ax[0].legend()

    # right: infidelity
    ax[1].plot(g_list, 1 - F_list, "-", label="Infidelity")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("g / FSR")
    ax[1].set_ylabel("1 − F")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------
def expt_003_compare_peaks(
    Delta=0.0,
    gamma_list=None,
    tau=1.0,
    T_min_max_list=None,
    n_jobs=-1,
):
    if gamma_list is None:
        gamma_list = np.linspace(0.01, 0.1, 40)
    gamma_list = np.asarray(gamma_list, float)

    if T_min_max_list is None:
        T_min_max_list = [
            (0.3, 0.8),
            (0.5, 1.0),
            (0.8, 1.5),
        ]

    fig, axs = plt.subplots(ncols=2, figsize=(12, 4), sharex=True)

    def _one(gamma, T_min, T_max):
        return compute_period_and_fidelity(
            Delta,
            gamma,
            tau,
            T_min=T_min,
            T_max=T_max,
        )

    for T_min, T_max in T_min_max_list:
        results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(_one)(gamma, T_min, T_max) for gamma in gamma_list
        )

        g_list = np.array([r[0] for r in results], float)
        t_list = np.array([r[2] / tau for r in results], float)
        err_list = np.array([1.0 - r[3] for r in results], float)

        label = rf"$T_{{\min}}={T_min},\;T_{{\max}}={T_max}$"

        axs[0].plot(g_list, t_list, lw=2, label=label)
        axs[1].plot(g_list, err_list, lw=2)

    axs[0].set_ylabel(r"$t_{\mathrm{peak}}/\tau$")
    axs[1].set_ylabel(r"$1 - F_{\max}$")
    axs[0].set_xlabel(r"$g/\mathrm{FSR}$")
    axs[1].set_xlabel(r"$g/\mathrm{FSR}$")

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

    axs[0].legend(frameon=False)
    fig.tight_layout()
    plt.show()


# ==================================================================================
def gamma_pulse(gamma, T, tau):
    den = 2.0 * tau
    return (
        lambda t: gamma * np.sin(np.pi * t / den) ** 2,
        lambda t: gamma * np.cos(np.pi * t / den) ** 2,
    )


def gamma_pulse_delay(gamma, T, tau=1.0):
    den = 2.0 * (T - 1.0) * tau
    return (
        (
            lambda t: gamma * np.sin(np.pi * t / den) ** 2
            if t <= (T - 1.0) * tau
            else gamma
        ),
        (lambda t: gamma * np.cos(np.pi * (t - tau) / den) ** 2 if t >= tau else gamma),
    )


def Fidelity(gamma, tau, phi, T, dt_max, pulse_delay):
    if pulse_delay:
        gamma1, gamma2 = gamma_pulse_delay(gamma, T, tau)
    else:
        gamma1, gamma2 = gamma_pulse(gamma, T, tau)
    _, c = dde_scalar(
        t_max=T * tau,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
    )
    F = np.abs(c[-1, 1]) ** 2
    return F


def expt_004_stirap_T_scan(
    gamma,
    T_list,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    n_jobs=-1,
    pulse_delay=True,
):
    gamma = gamma
    T_list = np.asarray(T_list, float)

    # ---- parallel scan over T ----
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(Fidelity)(gamma, tau, phi, T, dt_max, pulse_delay) for T in T_list
    )
    F = np.array(results)

    # ---- pulse plot (use first T) ----
    T_example = 10
    gamma1d, gamma2d = gamma_pulse_delay(gamma, T_example, tau)

    tg = np.linspace(0.0, T_example * tau, 800)
    g1dv = np.array([gamma1d(x) for x in tg])
    g2dv = np.array([gamma2d(x) for x in tg])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
    axL.plot(tg / tau, np.sqrt(g1dv / (2.0 * tau)) / np.pi, "--", label=r"$g_1$ ")
    axL.plot(tg / tau, np.sqrt(g2dv / (2.0 * tau)) / np.pi, "--", label=r"$g_2$")

    axL.axvline(1.0, color="black", linestyle="--", label=r"$\tau$")
    axL.axvline(T_example - 1.0, color="gray", linestyle="--", label=r"$T-\tau$")
    axL.set_xlabel(r"$t/\tau$")
    axL.set_ylabel(r"$g(t)$")
    axL.set_title(
        f"pulses (g_max/FSR={np.sqrt(gamma / 2) / np.pi:g}, T={T_example:g}τ)"
    )
    axL.legend()
    gm = np.sqrt(gamma / (2 * np.pi**2))
    axR.plot(T_list, 1 - F, label=f"g_max/FSR={gm:g}")
    axR.set_xlabel("T/tau")
    axR.set_ylabel("1 - F")
    axR.set_yscale("log")
    axR.legend()

    fig.tight_layout()
    plt.show()
    return F


# ------------------------------------------------------------------------------------
def stirap_optimal_Peak(
    gamma=0.1,
    T_range=None,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    pulse_delay=True,
):
    if T_range is None:
        T_range = [0.5, 10.0 / np.sqrt(gamma)]
    T_range = np.asarray(T_range, float)

    res = minimize_scalar(
        lambda T: -Fidelity(gamma, tau, phi, T, dt_max, pulse_delay),
        bounds=tuple(T_range),
        method="bounded",
    )

    T_opt = float(res.x)
    F_opt = float(-res.fun)
    return T_opt, F_opt


# ------------------------------------------------------------------------------------
def expt_008_ScanGamma_Refined(
    gamma_list,
    T_range=None,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    pulse_delay=True,
    n_jobs=-1,
    plot=True,
    cache_file="expt_008_cache.npz",
    overwrite=False,
):
    gamma_list = np.asarray(gamma_list, float)

    # ---------- load cache ----------
    def _plot_008(gamma, T_opt, F_opt):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

        ax1.plot(gamma, T_opt, "-")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"$\gamma$")
        ax1.set_ylabel(r"$T^*$")
        ax1.set_title("Optimal T")

        ax2.plot(gamma, 1.0 - F_opt, "-")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$\gamma$")
        ax2.set_ylabel(r"$1-F$")
        ax2.set_title("Minimal infidelity")

        fig.tight_layout()
        plt.show()

    if cache_file and os.path.exists(cache_file) and not overwrite:
        print(f"[info] load cache: {cache_file}")
        d = np.load(cache_file)
        gamma = d["gamma"]
        T_opt = d["T_opt"]
        F_opt = d["F_opt"]

        if plot:
            _plot_008(gamma, T_opt, F_opt)

        return

    # ---------- compute ----------
    def run_one_gamma(g):
        T_opt, F_opt = stirap_optimal_Peak(
            gamma=g,
            T_range=T_range,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
            pulse_delay=pulse_delay,
        )
        return T_opt, F_opt

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(run_one_gamma)(g) for g in gamma_list
    )
    T_opt = np.array([r[0] for r in results], float)
    F_opt = np.array([r[1] for r in results], float)
    # ---------- save ----------
    if cache_file:
        np.savez(
            cache_file,
            gamma=gamma_list,
            T_opt=T_opt,
            F_opt=F_opt,
        )
        print(f"[info] saved cache: {cache_file}")

    if plot:
        _plot_008(gamma_list, T_opt, F_opt)


# ------------------------------------------------------------------------------------
def renormalize_WW_opimized(
    gamma_guess,
    T,
    Delta_guess=0.0,
    tau=1.0,
    dt_max=0.01,
    pulse_delay=True,
    PBC: bool = False,
    n_modes: int = 50,
    n_steps: int = 101,
    plot=True,
    opt=True,
):
    # ---------- geometry / times ----------
    c = 1.0
    setup_kind = WG.Ring if PBC else WG.Cable
    L = (2.0 if PBC else 1.0) * tau * c
    positions = [0.0, L / 2.0] if PBC else [0.0, L]
    tmax = T * tau

    # ---------- reference DDE ----------
    g1_ref, g2_ref = (gamma_pulse_delay if pulse_delay else gamma_pulse)(
        gamma_guess, T, tau
    )
    t_dde, y_dde = dde_scalar(
        t_max=tmax, gamma1=g1_ref, gamma2=g2_ref, phi=0.0, tau=tau, dt_max=dt_max
    )
    # ---------- cleanup t-grid for interp (dedup) ----------
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

    # ---------- time modulation ----------

    if pulse_delay:
        den = 2.0 * (T - 1.0) * tau

        def g_time_mod(t):
            g1 = np.sin(np.pi * t / den) if t <= (T - 1.0) * tau else 1.0
            g2 = np.cos(np.pi * (t - tau) / den) if t >= tau else 1.0
            return np.array([[g1], [g2]], dtype=float)

    else:
        den = 2.0 * tau

        def g_time_mod(t):
            g1 = np.sin(np.pi * t / den)
            g2 = np.cos(np.pi * t / den)
            return np.array([[g1], [g2]], dtype=float)

    # ---------- loss ----------
    def cost_func(x):
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

        err = pop_WW[:, : y.shape[1]] - np.abs(y) ** 2
        ref = np.abs(y) ** 2

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
                (0.5 * gamma_guess, 1.5 * gamma_guess),
            ),
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
    Delta_list=(100.0, 5.0, 1.0),
    data="expt_008_cache11.npz",
    idx_gamma=(40, 80, 100),
    n_jobs=-1,
    savefile="expt_009_renormalized_WW_fidelity.npz",
):
    d = np.load(data, allow_pickle=True)

    gamma_all = np.asarray(d["gamma"], dtype=float)
    T_all = np.asarray(d["T_opt"], dtype=float)

    idx = np.asarray(idx_gamma, dtype=int)
    gamma_picked = gamma_all[idx]
    T_picked = T_all[idx]
    Delta_list = np.asarray(Delta_list, dtype=float)

    def _scalar(x):
        a = np.asarray(x)
        if a.ndim == 0:
            return float(a)
        return float(a.ravel()[0])

    tasks = [(g, T, D) for g, T in zip(gamma_picked, T_picked) for D in Delta_list]

    def one_case(g, T, D):
        g = _scalar(g)
        T = _scalar(T)
        D = _scalar(D)
        IF = renormalize_WW_opimized(
            gamma_guess=g,
            T=T,
            Delta_guess=D,
            n_modes=201,
            n_steps=int(T / 0.005),
            plot=False,
            pulse_delay=True,
            opt=True,
        )
        return IF

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


# ---------------------------------------------------------
def plot_IF_vs_T(
    data0="expt_002_cache.npz",
    data1="expt_008_cache11.npz",
    ww_data="expt_009_renormalized_WW_fidelity.npz",
    tau=1.0,
):
    # ---------- load ----------
    swap = np.load(data0)
    stirap = np.load(data1)
    ww = np.load(ww_data)

    # ---------- expt_002 : swap ----------
    T_swap = swap["T"] / tau
    IF_swap = 1.0 - swap["F"]

    # ---------- expt_008 : STIRAP ----------
    Td, Fd = stirap["T_opt"], stirap["F_opt"]

    # ---------- WW optimized ----------
    T_ww = ww["T"]
    D_ww = ww["Delta"]
    IF_ww = ww["IF"].reshape(len(T_ww), len(D_ww))

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(9, 6))

    # Swap (baseline)
    ax.plot(T_swap, IF_swap, "-", lw=2.8, label="Swap (expt_002)")

    # STIRAP
    ax.plot(Td, 1 - Fd, "--", label="STIRAP (delay)")
    # dark state condition
    ax.plot(T_swap, np.exp(-0.2 * T_swap))
    ax.plot(T_swap, np.exp(-0.5 * T_swap))

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


def QST_CZMK(gamma, T, tau, dt_max=0.01, phi=0.0, WW_sim=True, plot=True):
    """
    Smooth CZMK-like emission/absorption pulses for 2-node QST under DDE.

    gamma: overall coupling scale
    T    : dimensionless total time in units of tau (t_max = T*tau)
    tau  : delay time
    """

    t_max = T * tau

    # pulses (kept as your latest form)
    gamma1 = lambda t: 0.5 * gamma * (1.0 + np.tanh(0.5 * gamma * (t - 0.5 * t_max)))
    gamma2 = (
        lambda t: 0.5 * gamma * (1.0 + np.tanh(0.5 * gamma * (0.5 * t_max - t + tau)))
    )

    # --- DDE evolve ---
    t_list, c_dde = dde_scalar(
        t_max=t_max,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
    )

    # populations (DDE)
    p1_dde = np.abs(c_dde[:, 0]) ** 2
    p2_dde = np.abs(c_dde[:, 1]) ** 2
    # sample pulses on the same grid
    g1 = np.array([gamma1(t) for t in t_list], dtype=float)
    g2 = np.array([gamma2(t) for t in t_list], dtype=float)
    if WW_sim:
        # --- WW evolve ---
        c_light = 1.0
        setup_kind = WG.Cable
        L = tau * c_light
        positions = [0.0, L]

        def g_time_mod(t):
            g1 = np.sqrt(
                (0.5 * gamma * (1.0 + np.tanh(0.5 * gamma * (t - 0.5 * t_max))))
                / (2 * tau)
            )
            g2 = np.sqrt(
                (0.5 * gamma * (1.0 + np.tanh(0.5 * gamma * (0.5 * t_max - t + tau))))
                / (2 * tau)
            )
            return np.array(
                [[g1 / np.sqrt(gamma / (2 * tau))], [g2 / np.sqrt(gamma / (2 * tau))]],
                dtype=float,
            )

        ww = WW(
            Delta=100,
            positions=positions,
            gamma=gamma,
            n_modes=201,
            L=L,
            setup=setup_kind,
            g_time_modulation=g_time_mod,
        )
        t_WW, pop_WW = ww.evolve(t_max, n_steps=1001)
    if plot:
        # --- 1×2 layout: left pulses, right populations ---
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

        # left: pulses
        ax[0].plot(t_list, g1, label=r"$\gamma_1(t)$")
        ax[0].plot(t_list, g2, label=r"$\gamma_2(t)$")
        ax[0].set_xlabel("t")
        ax[0].set_ylabel(r"$\gamma(t)$")
        ax[0].legend()
        ax[0].set_title("Pulses")
        ax[0].set_xlim(0, t_max)

        # right: populations (DDE solid, WW dashed)
        ax[1].plot(t_list, p1_dde, label=r"DDE: $|c_1|^2$")
        ax[1].plot(t_list, p2_dde, label=r"DDE: $|c_2|^2$")
        if WW_sim:
            ax[1].plot(t_WW, pop_WW[:, 0], "--", label=r"WW: $|c_1|^2$")
            ax[1].plot(t_WW, pop_WW[:, 1], "--", label=r"WW: $|c_2|^2$")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("population")
        ax[1].legend()
        ax[1].set_title("Dynamics")
        ax[1].set_xlim(0, t_max)

        fig.tight_layout()

    F = float(np.abs(c_dde[-1, 1]) ** 2)
    return F


def CZMK_Scan_T(T_list=None, gamma=0.1, tau=1.0, dt_max=0.01):
    data1 = "expt_008_cache11.npz"
    stirap = np.load(data1)

    # ---------- case 1: use cached optimal (gamma_i, T_i) ----------
    if T_list is None:
        T_arr = np.asarray(stirap["T_opt"], float)
        gamma_arr = np.asarray(stirap["gamma"], float)

        F_list = np.array(
            [
                QST_CZMK(
                    g,
                    T,
                    tau,
                    dt_max=dt_max,
                    phi=0.0,
                    WW_sim=False,
                    plot=False,
                )
                for g, T in zip(gamma_arr, T_arr)
            ]
        )

        x = T_arr / tau
        label = "CZMK (opt points)"

    # ---------- case 2: fixed gamma, scan T ----------
    else:
        T_arr = np.asarray(T_list, float)

        F_list = np.array(
            [
                QST_CZMK(
                    gamma,
                    T,
                    tau,
                    dt_max=0.01,
                    phi=0.0,
                    WW_sim=False,
                    plot=False,
                )
                for T in T_arr
            ]
        )

        x = T_arr / tau
        label = rf"CZMK ($\gamma={gamma}$)"

    # ---------- plot ----------
    plt.plot(x, 1.0 - F_list, label=label)
    plt.plot(x, np.exp(-gamma * x), label=r"$e^{-\gamma_{max} T}$")
    plt.xlabel(r"$T/\tau$")
    plt.ylabel(r"$1 - F$")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def fig_paper_QST_and_IF_vs_T(
    data0="expt_002_cache.npz",
    data1="expt_008_cache11.npz",
    ww_data="expt_009_renormalized_WW_fidelity.npz",
    tau=1.0,
    gamma=0.1,
    T_dyn=200.0,
    dt_max=0.01,
    phi=0.0,
    figsize=(7.2, 3.0),
    panel_labels=True,
    save=None,
    dpi=300,
):
    """
    Two-panel figure using the global style set by qnetwork.tools.set_plot_style().

    (a) Dynamics: DDE (black solid), WW (lightskyblue dashed), plus exp(-gamma t) ref.
    (b) IF vs T (log-log): Swap/STIRAP lines + 3 WW scatter groups (3 colors) + inset zoom.

    Requires: dde_scalar, WW, WG already imported.
    """

    # =============================
    # (a) Left panel: dynamics
    # =============================
    t_max = T_dyn * tau

    gamma1 = lambda t: 0.5 * gamma * (1.0 + np.tanh(0.5 * gamma * (t - 0.5 * t_max)))
    gamma2 = (
        lambda t: 0.5 * gamma * (1.0 + np.tanh(0.5 * gamma * (0.5 * t_max - t + tau)))
    )

    # --- DDE ---
    t_list, c_dde = dde_scalar(
        t_max=t_max,
        gamma1=gamma1,
        gamma2=gamma2,
        phi=phi,
        tau=tau,
        dt_max=dt_max,
    )
    p_dde = np.abs(c_dde[:, :2]) ** 2

    # --- WW ---
    L = tau
    positions = [0.0, L]

    def g_time_mod(t):
        m1 = np.sqrt(np.maximum(gamma1(t), 0.0) / gamma)
        m2 = np.sqrt(np.maximum(gamma2(t), 0.0) / gamma)
        return np.array(
            [[m1], [m2]], dtype=float
        )  # (2,1) for broadcasting to (2,n_modes)

    ww = WW(
        Delta=100,
        positions=positions,
        gamma=gamma,
        n_modes=201,
        L=L,
        setup=WG.Cable,
        g_time_modulation=g_time_mod,
    )
    t_WW, pop_WW = ww.evolve(t_max, n_steps=1001)

    # =============================
    # (b) Right panel: IF vs T
    # =============================
    swap = np.load(data0)
    stirap = np.load(data1)
    ww_opt = np.load(ww_data)

    T_swap = swap["T"] / tau
    IF_swap = 1.0 - swap["F"]

    Td = stirap["T_opt"]
    IF_stirap = 1.0 - stirap["F_opt"]
    gamma_list = stirap["gamma"]

    T_ww = ww_opt["T"]
    D_ww = ww_opt["Delta"]
    IF_ww = ww_opt["IF"].reshape(len(T_ww), len(D_ww))

    # =============================
    # Plot (assumes set_plot_style() already called globally)
    # =============================
    fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # ---- (a) dynamics ----
    ax0 = ax[0]
    # DDE (black)
    ax0.plot(t_list / tau, p_dde[:, 0], "k", label=r"DDE: $|c_1|^2$")
    ax0.plot(t_list / tau, p_dde[:, 1], "k", alpha=0.55, label=r"DDE: $|c_2|^2$")

    # WW (lightskyblue dashed)
    ax0.plot(
        t_WW / tau,
        pop_WW[:, 0],
        linestyle="dashed",
        color="lightskyblue",
        label=r"WW: $|c_1|^2$",
    )
    ax0.plot(
        t_WW / tau,
        pop_WW[:, 1],
        linestyle="dashed",
        color="lightskyblue",
        alpha=0.65,
        label=r"WW: $|c_2|^2$",
    )

    ax0.set_xlabel(r"$t/\tau$")
    ax0.set_ylabel("Population")
    ax0.set_xlim(0, T_dyn)
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(frameon=False, loc="best")

    # ---- (b) IF vs T ----
    ax1 = ax[1]

    cut_swap = 1500
    cut_stirap = 900

    ax1.plot(T_swap[:cut_swap], IF_swap[:cut_swap], "-", color="C0", label="Swap")
    ax1.plot(Td[:cut_stirap], IF_stirap[:cut_stirap], "-", color="C1", label="STIRAP")

    ax1.plot(
        Td[:cut_stirap],
        np.exp(-gamma_list[:cut_stirap] * Td[:cut_stirap]),
        linestyle="--",
        color="0.1",
        lw=1.0,
        label=r"$e^{-\gamma_{max} T}$",
    )

    idx = [0, 1, 2]
    colors = ["C3", "C4", "C5"]
    for j, col in zip(idx, colors):
        ax1.scatter(
            T_ww,
            IF_ww[:, j],
            s=24,
            linewidths=1.0,
            label=rf"Renorm.-WW ($\Delta={D_ww[j]:g}$)",
            zorder=5,
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$T/\tau$")
    ax1.set_ylabel(r"$1-F$")
    ax1.set_xlim(1.0, None)
    ax1.legend(frameon=False, loc="best")

    # panel labels
    if panel_labels:
        ax0.text(0.02, 0.98, "(a)", transform=ax0.transAxes, va="top", ha="left")
        ax1.text(0.02, 0.98, "(b)", transform=ax1.transAxes, va="top", ha="left")

    if save:
        fig.savefig(save, dpi=dpi)

    plt.show()
    return fig, ax
