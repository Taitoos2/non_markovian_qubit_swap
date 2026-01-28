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
from scipy.linalg import expm
from aux_funs import dde_scalar
from scipy.signal import find_peaks, savgol_filter
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
    PBC: bool = True,
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


# ----------------------------------------------------------------------------
def expt_0011_dynamics(
    Delta: float = 0.0,
    gamma: float = 0.1,
    tau: float = 50.0,
    n_modes: int = 50,
    c: float = 1.0,
    PBC: bool = True,
    T: float = 100.0,
    dt_max: float = 0.01,
    n_steps: int = 101,
):
    """Compare dynamics: DDE vs WW vs TC (and add simple Rabi-period markers)."""
    from two_qubit_control import EmittersInWaveguideWW as WW
    from two_qubit_control import Waveguide as WG

    initial = "10"
    setup_kind = WG.Ring if PBC else WG.Cable
    tmax = T * tau

    def run_once(Delta_val: float):
        L = (2 if PBC else 1) * tau * c
        positions = [0.0, L / 2] if PBC else [0.0, L]

        setup_WW = WW(
            Delta=Delta_val,
            positions=positions,
            gamma=gamma,
            n_modes=n_modes,
            L=L,
            setup=setup_kind,
            g_time_modulation=lambda t: np.array(
                [[np.sin(np.pi * t / (2 * tmax))], [np.cos(np.pi * t / (2 * tmax))]]
            ),
        )
        t_WW, pop_WW = setup_WW.evolve(tmax, n_steps=n_steps)

        delta = math.modf(Delta_val)[0]
        phi = delta * setup_WW.FSR * tau
        setup_dde = EmittersInWaveguideDDE(
            phi=phi, N=2, gamma=gamma, U=-1, tau=tau, dt_max=dt_max
        )
        setup_dde.evolve(tmax)
        t_DDE, pop_DDE = setup_dde.n_photons(initial)
        return t_WW, np.abs(pop_WW) ** 2, t_DDE, pop_DDE

    t_WW, pop_WW, t_DDE, pop_DDE = run_once(Delta)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(rf"Dynamics ($\Delta={int(Delta)}*FSR$)")

    # WW + DDE
    ax.plot(t_WW / tau, pop_WW[:, 0], color="#0a4570", label=r"$Q_1$: WW")
    ax.plot(t_WW / tau, pop_WW[:, 1], color="#af1a2e", label=r"$Q_2$: WW")
    # ax.plot(t_DDE / tau, pop_DDE[:, 0], "--", color="#055805", label=r"$Q_1$: DDE")
    # ax.plot(t_DDE / tau, pop_DDE[:, 1], "--", color="#055805", label=r"$Q_2$: DDE")

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
    T_min=0.8,
    T_max=1.2,
    n_jobs=-1,
    force=False,
    filename="expt_002_cache.npz",
):
    fpath = filename

    # ---------- load ----------
    if os.path.exists(fpath) and not force:
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
    ax[0].plot(g_list, t_list / tau, "o-", label="Swap speed")
    ax[0].plot(g_list, T_list / tau, "--", label=r"$\pi/\Omega$")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("g / FSR")
    ax[0].set_ylabel("T / τ")
    ax[0].grid(True)
    ax[0].legend()

    # right: infidelity
    ax[1].plot(g_list, 1 - F_list, "o-", label="Infidelity")
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
    """
    expt_003-style plot:
    x-axis: g / FSR
    left : t_peak / tau
    right: 1 - F_max   (log scale)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from joblib import Parallel, delayed

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


# ------------------------------------------------------------------------------------
def expt_004_stirap(T=100.0, gamma=0.1, tau=1.0, dt_max=0.01, phi=0.0):
    # DDE
    phi = phi
    setup_dde = EmittersInWaveguideDDE(
        phi=phi,
        N=2,
        gamma=gamma,
        U=-1,
        tau=tau,
        dt_max=dt_max,
    )
    setup_dde.evolve(T * tau)
    initial = "10"
    t_DDE, pop_DDE = setup_dde.n_photons(initial)
    # dde scalar
    # g1 = lambda t: np.sin((np.pi * t) / (2 * T * tau)) ** 2 * gamma
    # g2 = lambda t: np.cos((np.pi * t) / (2 * T * tau)) ** 2 * gamma
    g1 = lambda t: (
        gamma * np.sin(np.pi * (t) / (2 * (T - 1) * tau)) ** 2
        if t <= (T - 1) * tau
        else gamma
    )
    g2 = lambda t: (
        gamma * np.cos(np.pi * (t - tau) / (2 * (T - 1) * tau)) ** 2
        if t >= tau
        else gamma
    )
    tlist, clist = dde_scalar(
        t_max=T * tau, gamma1=g1, gamma2=g2, phi=phi, tau=tau, dt_max=dt_max
    )
    plt.plot(t_DDE / tau, pop_DDE[:, 0], label="DDE Q1")
    plt.plot(t_DDE / tau, pop_DDE[:, 1], label="DDE Q2")
    plt.plot(
        tlist / tau, np.abs(clist[:, 0]) ** 2, label="DDE scalar Q1", linestyle="--"
    )
    plt.plot(
        tlist / tau, np.abs(clist[:, 1]) ** 2, label="DDE scalar Q2", linestyle="--"
    )
    plt.xlabel("t/tau")
    plt.ylabel("<sigma^+ sigma>")
    plt.legend()
    plt.show()
    return np.abs(clist[-1, 1]) ** 2


# ------------------------------------------------------------------------------------
def stirap_3level(T=20.0, dt=0.01, tau=1.0, gamma=0.1):
    psi = np.array([1.0, 0.0, 0.0], dtype=np.complex128)

    n_steps = int(np.ceil(T / dt))
    t = np.linspace(0.0, T, n_steps + 1)
    pop = np.empty((n_steps + 1, 3), dtype=float)

    for i in range(n_steps + 1):
        pop[i] = np.abs(psi) ** 2
        if i == n_steps:
            break

        x = (np.pi / (2.0 * (T) * tau)) * t[i]
        Ωp = np.sqrt(gamma / (2.0 * tau)) * np.sin(x)
        Ωs = np.sqrt(gamma / (2.0 * tau)) * np.cos(x)

        H = np.array([[0, Ωp, 0], [Ωp, 0, Ωs], [0, Ωs, 0]], dtype=np.complex128)
        psi = expm((-1j * dt) * H) @ psi  # U(t) psi

    return t, pop


def gamma_pulse(gamma, T, tau):
    den = 2.0 * (T - 1.0) * tau
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


def expt_005_stirap_T_scan(
    gamma,
    T_list,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    n_jobs=-1,
):
    gamma = float(gamma)
    T_list = np.asarray(T_list, float)

    def solve_one_T(T):
        gamma1, gamma2 = gamma_pulse(gamma, T, tau)
        gamma1d, gamma2d = gamma_pulse_delay(gamma, T, tau)

        _, c = dde_scalar(
            t_max=(T - 1.0) * tau,
            gamma1=gamma1,
            gamma2=gamma2,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
        )
        _, cd = dde_scalar(
            t_max=T * tau,
            gamma1=gamma1d,
            gamma2=gamma2d,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
        )

        _, pop_cav = stirap_3level(T=(T - 1.0) * tau, dt=dt_max, tau=tau, gamma=gamma)

        F = np.abs(c[-1, 1]) ** 2
        Fd = np.abs(cd[-1, 1]) ** 2
        Fc = pop_cav[-1, 2]
        return 1.0 - F, 1.0 - Fd, 1.0 - Fc

    # ---- parallel scan over T ----
    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(solve_one_T)(float(T)) for T in T_list
    )
    inf = np.array([r[0] for r in results], float)
    inf_delay = np.array([r[1] for r in results], float)
    inf_cav = np.array([r[2] for r in results], float)

    # ---- pulse plot (use first T) ----
    T0 = float(T_list[0])
    gamma1, gamma2 = gamma_pulse(gamma, T0, tau)
    gamma1d, gamma2d = gamma_pulse_delay(gamma, T0, tau)

    tg = np.linspace(0.0, T0 * tau, 800)
    g1v = np.array([gamma1(x) for x in tg])
    g2v = np.array([gamma2(x) for x in tg])
    g1dv = np.array([gamma1d(x) for x in tg])
    g2dv = np.array([gamma2d(x) for x in tg])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))

    axL.plot(tg / tau, np.sqrt(g1v / (2.0 * tau)) / np.pi, label=r"$g_1$")
    axL.plot(tg / tau, np.sqrt(g2v / (2.0 * tau)) / np.pi, label=r"$g_2$")
    axL.plot(
        tg / tau, np.sqrt(g1dv / (2.0 * tau)) / np.pi, "--", label=r"$g_1$ (delay)"
    )
    axL.plot(
        tg / tau, np.sqrt(g2dv / (2.0 * tau)) / np.pi, "--", label=r"$g_2$ (delay)"
    )

    axL.axvline(1.0, color="black", linestyle="--", label=r"$\tau$")
    axL.axvline(T0 - 1.0, color="gray", linestyle="--", label=r"$T-\tau$")
    axL.set_xlabel(r"$t/\tau$")
    axL.set_ylabel(r"$g(t)$")
    axL.set_title(f"pulses (g_max/FSR={np.sqrt(gamma / 2) / np.pi:g}, T={T0:g}τ)")
    axL.legend()

    # ---- right plot ----
    gm = np.sqrt(gamma / (2 * np.pi**2))
    axR.plot(T_list - 1.0, inf, label=f"g_max/FSR={gm:g}")
    axR.plot(T_list, inf_delay, "--", label=f"g_max/FSR={gm:g} (delay)")
    axR.plot(T_list - 1.0, inf_cav, ":", label=f"g_max/FSR={gm:g} (cavity)")

    axR.set_xlabel("T/tau")
    axR.set_ylabel("1 - F")
    axR.set_yscale("log")
    axR.legend()

    fig.tight_layout()
    plt.show()

    return inf, inf_delay, inf_cav


# ------------------------------------------------------------------------------------
def valley_T(
    y,
    x,
    log_prominence=0.6,
    distance=2,
    smooth=True,
    smooth_window=7,
    smooth_poly=3,
):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    assert y.shape == x.shape

    ymax = np.nanmax(y)
    if not (np.isfinite(ymax) and ymax > 0):
        ymax = 1.0
    eps = 1e-12 * ymax

    logy = np.log(np.maximum(y, 0.0) + eps)

    if smooth and logy.size >= 5:
        w = int(smooth_window) | 1
        w = min(w, logy.size - 1 if logy.size % 2 == 0 else logy.size)
        if w >= 5:
            logy = savgol_filter(logy, w, smooth_poly)

    idx, props = find_peaks(
        -logy,
        prominence=log_prominence,
        distance=distance,
    )

    return x[idx], props


def expt_006_FindPeak(
    gamma,
    T_list,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    n_jobs=-1,
    plot=True,
    keep=4,
    return_curves=False,
):
    gamma = float(gamma)
    T_list = np.asarray(T_list, float)

    def solve_one_T(T):
        gamma1, gamma2 = gamma_pulse(gamma, T, tau)
        gamma1d, gamma2d = gamma_pulse_delay(gamma, T, tau)

        _, c = dde_scalar(
            t_max=(T - 1.0) * tau,
            gamma1=gamma1,
            gamma2=gamma2,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
        )
        _, cd = dde_scalar(
            t_max=T * tau,
            gamma1=gamma1d,
            gamma2=gamma2d,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
        )

        return 1.0 - np.abs(c[-1, 1]) ** 2, 1.0 - np.abs(cd[-1, 1]) ** 2

    results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
        delayed(solve_one_T)(float(T)) for T in T_list
    )

    res = np.asarray(results, float)
    infidelity = res[:, 0]
    infidelity_delay = res[:, 1]

    # ---- valleys: return T directly ----
    T_peak, props = valley_T(infidelity, T_list)
    T_peak_delay, props_delay = valley_T(infidelity_delay, T_list)

    T_peak = np.sort(T_peak)[:keep]
    T_peak_delay = np.sort(T_peak_delay)[:keep]

    if len(T_peak) < keep:
        print(f"[warn] only {len(T_peak)}/{keep} valleys found (no-delay).")
    if len(T_peak_delay) < keep:
        print(f"[warn] only {len(T_peak_delay)}/{keep} valleys found (delay).")

    if plot:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        gm = np.sqrt(gamma / (2 * np.pi**2))

        ax.plot(T_list - tau, infidelity, label=f"g_max/FSR={gm:g}")
        ax.plot(T_list, infidelity_delay, "--", label=f"g_max/FSR={gm:g} (delay)")

        if len(T_peak):
            y_peak = np.interp(T_peak, T_list, infidelity)
            ax.plot(T_peak - tau, y_peak, "o", ms=5)
        if len(T_peak_delay):
            y_peak_d = np.interp(T_peak_delay, T_list, infidelity_delay)
            ax.plot(T_peak_delay, y_peak_d, "s", ms=5)

        ax.set_xlabel("T/tau")
        ax.set_ylabel("1 - F")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        plt.show()

    out = (T_peak, T_peak_delay, props, props_delay)
    if return_curves:
        out = out + (infidelity, infidelity_delay)
    return out


def expt_007_RefinePeak(
    gamma,
    T_list,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    n_jobs=-1,
    keep=4,
    halfwidth=5.0,
    plot=True,
):
    gamma = float(gamma)
    T_list = np.asarray(T_list, float)

    T_ref, T_ref_delay, props, props_delay, infidelity, infidelity_delay = (
        expt_006_FindPeak(
            gamma=gamma,
            T_list=T_list,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
            n_jobs=n_jobs,
            plot=False,
            keep=keep,
            return_curves=True,
        )
    )

    # ---- objectives (single evaluation) ----
    _cache_no = {}
    _cache_d = {}

    def _key(T):
        return np.round(T, 9)

    def inf_no_delay(T):
        k = _key(T)
        if k in _cache_no:
            return _cache_no[k]
        gamma1, gamma2 = gamma_pulse(gamma, T, tau)
        _, c = dde_scalar(
            t_max=(T - 1.0) * tau,
            gamma1=gamma1,
            gamma2=gamma2,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
        )
        a = c[-1, 1]
        val = 1 - (a.real * a.real + a.imag * a.imag)
        _cache_no[k] = val
        return val

    def inf_delay(T):
        k = _key(T)
        if k in _cache_d:
            return _cache_d[k]
        gamma1d, gamma2d = gamma_pulse_delay(gamma, T, tau)
        _, cd = dde_scalar(
            t_max=T * tau,
            gamma1=gamma1d,
            gamma2=gamma2d,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
        )
        a = cd[-1, 1]
        val = 1.0 - (a.real * a.real + a.imag * a.imag)
        _cache_d[k] = val
        return val

    def refine_one(T0, f):
        T0 = float(T0)
        lo = max(1.0 + 1e-9, T0 - halfwidth)
        hi = T0 + halfwidth
        if hi <= lo:
            return T0, np.nan
        res = minimize_scalar(
            f,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return float(res.x), float(res.fun)

    T_star, inf_star = (
        zip(*[refine_one(T0, inf_no_delay) for T0 in T_ref]) if len(T_ref) else ([], [])
    )
    T_star_delay, inf_star_delay = (
        zip(*[refine_one(T0, inf_delay) for T0 in T_ref_delay])
        if len(T_ref_delay)
        else ([], [])
    )

    T_star = np.asarray(T_star, float)
    inf_star = np.asarray(inf_star, float)
    T_star_delay = np.asarray(T_star_delay, float)
    inf_star_delay = np.asarray(inf_star_delay, float)

    if plot:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        gm = np.sqrt(gamma / (2 * np.pi**2))

        ax.plot(T_list - tau, infidelity, label=f"g_max/FSR={gm:g}")
        ax.plot(T_list, infidelity_delay, "--", label=f"g_max/FSR={gm:g} (delay)")

        if len(T_ref):
            ax.plot(
                T_ref - tau,
                np.interp(T_ref, T_list, infidelity),
                "o",
                ms=5,
                alpha=0.6,
                label="coarse valleys",
            )
        if len(T_ref_delay):
            ax.plot(
                T_ref_delay,
                np.interp(T_ref_delay, T_list, infidelity_delay),
                "s",
                ms=5,
                alpha=0.6,
                label="coarse valleys (delay)",
            )

        # refined minima
        if len(T_star):
            ax.plot(T_star - tau, inf_star, "o", ms=8, label="refined minima")
        if len(T_star_delay):
            ax.plot(
                T_star_delay, inf_star_delay, "s", ms=8, label="refined minima (delay)"
            )

        ax.set_xlabel("T (note: no-delay uses T-τ on x)")
        ax.set_ylabel("1 - F")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        plt.show()

    return (
        T_star,
        inf_star,
        T_star_delay,
        inf_star_delay,
    )


# ------------------------------------------------------------------------------------
def expt_008_ScanGamma_Refined(
    gamma_list,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    keep=1,
    halfwidth=9.0,
    n_jobs=-1,
    n_jobs_inner=1,
    plot=True,
    cache_file="expt_008_cache.npz",
    force=False,
):
    gamma_list = np.asarray(gamma_list, float)

    # ---------- plot ----------
    def do_plot():
        x = gamma_list

        fig1, ax1 = plt.subplots(figsize=(7.6, 4.3))
        for k in range(keep):
            ax1.plot(x, T_star[:, k], "-", label=f"T*[{k + 1}]")
            ax1.plot(x, T_star_delay[:, k], "--", label=f"T*[{k + 1}] (delay)")
        ax1.set(xlabel="gamma", ylabel="T* (refined)")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        fig1.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(7.6, 4.3))
        for k in range(keep):
            ax2.plot(x, inf_star[:, k], "-", label=f"inf*[{k + 1}]")
            ax2.plot(x, inf_star_delay[:, k], "--", label=f"inf*[{k + 1}] (delay)")
        ax2.set(xlabel="gamma", ylabel="min infidelity (1 - F)")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend()
        fig2.tight_layout()

        plt.show()

    # ---------- load cache ----------
    if (not force) and cache_file and os.path.exists(cache_file):
        try:
            d = np.load(cache_file, allow_pickle=True)
            T_star = d["T_star"]
            inf_star = d["inf_star"]
            T_star_delay = d["T_star_delay"]
            inf_star_delay = d["inf_star_delay"]

            if plot:
                do_plot()

            return gamma_list, T_star, inf_star, T_star_delay, inf_star_delay
        except Exception as e:
            print(f"[warn] cache load failed, recomputing. reason: {e}")

    # ---------- compute ----------
    ng = len(gamma_list)
    T_star = np.full((ng, keep), np.nan, float)
    inf_star = np.full((ng, keep), np.nan, float)
    T_star_delay = np.full((ng, keep), np.nan, float)
    inf_star_delay = np.full((ng, keep), np.nan, float)

    def run_one_gamma(gamma):
        T_list = np.linspace(2, 10 / np.sqrt(gamma), 50)
        Ts, IFs, Tsd, IFsd = expt_007_RefinePeak(
            gamma=float(gamma),
            T_list=T_list,
            phi=phi,
            tau=tau,
            dt_max=dt_max,
            n_jobs=n_jobs_inner,
            keep=keep,
            halfwidth=halfwidth,
            plot=False,
        )
        return Ts, IFs, Tsd, IFsd

    results = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
        delayed(run_one_gamma)(g) for g in gamma_list
    )

    for i, (Ts, IFs, Tsd, IFsd) in enumerate(results):
        m = min(len(Ts), keep)
        if m:
            T_star[i, :m] = Ts[:m]
            inf_star[i, :m] = IFs[:m]

        m = min(len(Tsd), keep)
        if m:
            T_star_delay[i, :m] = Tsd[:m]
            inf_star_delay[i, :m] = IFsd[:m]

    # ---------- save cache ----------
    if cache_file:
        try:
            np.savez(
                cache_file,
                T_star=T_star,
                inf_star=inf_star,
                T_star_delay=T_star_delay,
                inf_star_delay=inf_star_delay,
            )
        except Exception as e:
            print(f"[warn] cache save failed: {e}")

    if plot:
        do_plot()


# ------------------------------------------------------------------------------------
def plot_expt_002_and_008(
    data0="expt_002_cache.npz",
    data1="expt_008_cache1.npz",
    data2="expt_008_cache2.npz",
    data3="expt_008_cache3.npz",
    data4="expt_008_cache4.npz",
    tau=1.0,
):
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 16,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2.5,
        }
    )

    # ---------- load ----------
    data = np.load(data0)
    stirap1 = np.load(data1)
    stirap2 = np.load(data2)
    stirap3 = np.load(data3)
    stirap4 = np.load(data4)

    # swap (expt_002)
    gamma_list = np.logspace(np.log10(0.002), np.log10(0.3), 1000)
    T_list = data["T"]
    infidelity = 1 - data["F"]

    # stirap scans (expt_008)
    gamma_list2 = np.logspace(np.log10(0.002), np.log10(0.3), 300)

    def unpack(d):
        return d["T_star"], d["inf_star"], d["T_star_delay"], d["inf_star_delay"]

    T1, inf1, T1d, inf1d = unpack(stirap1)
    T2, inf2, T2d, inf2d = unpack(stirap2)
    T3, inf3, T3d, inf3d = unpack(stirap3)
    T4, inf4, T4d, inf4d = unpack(stirap4)

    # ---------- plot ----------
    fig, ax = plt.subplots(1, 2, figsize=(18, 10))

    # left: time
    ax[0].plot(gamma_list, T_list / tau, "-", label="Swap (expt_002)")

    ax[0].plot(gamma_list2, T1 - 1, "-", color="#0A3D62", label="stirap1 (no delay)")
    ax[0].plot(gamma_list2, T1d, "--", color="#1595F1", label="stirap1 (with delay)")

    ax[0].plot(gamma_list2, T2 - 1, "-", color="#580E09", label="stirap2 (no delay)")
    ax[0].plot(gamma_list2, T2d, "--", color="#DD3131", label="stirap2 (with delay)")

    ax[0].plot(gamma_list2, T3 - 1, "-", color="#075507", label="stirap3 (no delay)")
    ax[0].plot(gamma_list2, T3d, "--", color="#27C04E", label="stirap3 (with delay)")

    ax[0].plot(gamma_list2, T4 - 1, "-", color="#400842", label="stirap4 (no delay)")
    ax[0].plot(gamma_list2, T4d, "--", color="#9D17BE", label="stirap4 (with delay)")

    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$\gamma \tau$")
    ax[0].set_ylabel("T / τ")
    ax[0].grid(True)
    ax[0].legend()

    # right: infidelity
    ax[1].plot(gamma_list, infidelity, "-", label="Swap (expt_002)")

    ax[1].plot(gamma_list2, inf1, "-", color="#0A3D62", label="stirap1 (no delay)")
    ax[1].plot(gamma_list2, inf1d, "--", color="#1595F1", label="stirap1 (with delay)")

    ax[1].plot(gamma_list2, inf2, "-", color="#580E09", label="stirap2 (no delay)")
    ax[1].plot(gamma_list2, inf2d, "--", color="#DD3131", label="stirap2 (with delay)")

    ax[1].plot(gamma_list2, inf3, "-", color="#075507", label="stirap3 (no delay)")
    ax[1].plot(gamma_list2, inf3d, "--", color="#27C04E", label="stirap3 (with delay)")

    ax[1].plot(gamma_list2, inf4, "-", color="#400842", label="stirap4 (no delay)")
    ax[1].plot(gamma_list2, inf4d, "--", color="#9D17BE", label="stirap4 (with delay)")

    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel(r"$\gamma \tau$")
    ax[1].set_ylabel("1 − F")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------
def plot_optimal_TF(
    data0="expt_002_cache.npz",
    data1="expt_008_cache1.npz",
    data2="expt_008_cache2.npz",
    data3="expt_008_cache3.npz",
    data4="expt_008_cache4.npz",
    tau=1.0,
):
    # ---------- style ----------
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.fontsize": 16,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "lines.linewidth": 2.5,
        }
    )

    # ---------- load ----------
    s0 = np.load(data0)
    s1 = np.load(data1)
    T_swap = s0["T"]
    inf_swap = 1 - s0["F"]
    T, inf = s1["T_star"], s1["inf_star"]
    Td, infd = s1["T_star_delay"], s1["inf_star_delay"]

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        T_swap,
        inf_swap,
        "-",
        color="#1BC915",
        alpha=0.5,
        label="swap_first_peak",
    )
    ax.plot(
        T - 1,
        inf,
        "-",
        color="#1670C4",
        label="stirap_first_peak (no delay)",
    )

    ax.plot(
        Td,
        infd,
        "-",
        color="#B11B11",
        label="stirap_first_peak (with delay)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$T/\tau$")
    ax.set_ylabel(r"Infidelity $(1-F)$")
    ax.grid(True)
    ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------
def optimize_gamma_and_phi(
    gamma_guess,
    T,
    Delta_guess=0.0,
    tau=1.0,
    dt_max=0.01,
    delay=False,
    PBC: bool = True,
    c=1.0,
    n_modes: int = 50,
    n_steps: int = 101,
    plot=True,
    opt=True,
):
    # System basic params
    setup_kind = WG.Ring if PBC else WG.Cable
    L = (2 if PBC else 1) * tau * c
    positions = [0.0, L / 2] if PBC else [0.0, L]
    tmax = T * tau if delay else (T - 1.0) * tau

    # Interpolate data to target time axis (support complex)
    def _interp_to(t_src, y_src, t_tgt):
        y_src, t_tgt = np.asarray(y_src), np.asarray(t_tgt)
        out = np.empty((t_tgt.size, y_src.shape[1]), dtype=y_src.dtype)
        for j in range(y_src.shape[1]):
            if np.iscomplexobj(y_src):
                re = np.interp(t_tgt, t_src, y_src[:, j].real)
                im = np.interp(t_tgt, t_src, y_src[:, j].imag)
                out[:, j] = re + 1j * im
            else:
                out[:, j] = np.interp(t_tgt, t_src, y_src[:, j])
        return out

    # Generate gamma pulses and solve DDE for reference data
    gamma1_ref, gamma2_ref = (gamma_pulse_delay if delay else gamma_pulse)(
        gamma_guess, T, tau
    )
    t_dde, y_dde = dde_scalar(
        t_max=tmax, gamma1=gamma1_ref, gamma2=gamma2_ref, phi=0, tau=tau, dt_max=dt_max
    )
    t_dde, y_dde = np.asarray(t_dde), np.asarray(y_dde)

    # Time modulation function for WW coupling
    def g_time_mod(t):
        if delay:
            s, c = (
                np.sin(np.pi * t / (2 * (tmax - tau))),
                np.cos(np.pi * (t - tau) / (2 * (tmax - tau))),
            )
        else:
            s, c = np.sin(np.pi * t / (2 * tmax)), np.cos(np.pi * t / (2 * tmax))
        return np.array([[s], [c]])

    # Cost function: normalized trapezoid integral loss
    def cost_func(paras):
        Delta, gamma = paras
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
        y_dde_interp = _interp_to(t_dde, y_dde, t_WW)
        err = pop_WW[:, : y_dde_interp.shape[1]] - np.abs(y_dde_interp) ** 2
        num = trapezoid(np.abs(err), x=t_WW, axis=0).sum()
        den = trapezoid(np.abs(y_dde_interp) ** 2, x=t_WW, axis=0).sum() + 1e-12
        return float(num / den)

    # Parameter optimization or use initial guess
    if opt:
        res = minimize(
            cost_func,
            x0=[Delta_guess, max(gamma_guess, 1e-6)],
            method="L-BFGS-B",
            bounds=[
                (max(Delta_guess - 0.5, 1e-6), Delta_guess + 0.5),
                (0.5 * gamma_guess, 1.5 * gamma_guess),
            ],
        )
        Delta_opt, gamma_opt, loss_opt = res.x[0], res.x[1], res.fun
    else:
        Delta_opt, gamma_opt, loss_opt = (
            Delta_guess,
            gamma_guess,
            cost_func([Delta_guess, gamma_guess]),
        )

    # Plot: opt=True(DDE+ori WW+opt WW) | opt=False(DDE+ori WW)
    if plot:
        # Init original WW and evolve
        ww_ori = WW(
            Delta=Delta_guess,
            positions=positions,
            gamma=gamma_guess,
            n_modes=n_modes,
            L=L,
            setup=setup_kind,
            g_time_modulation=g_time_mod,
        )
        t_WW, pop_ori = ww_ori.evolve(tmax, n_steps=n_steps)

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 5))
        # Plot DDE and original WW
        for j in range(pop_ori.shape[1]):
            ax.plot(
                t_dde,
                np.abs(y_dde[:, j]) ** 2,
                "--k",
                lw=1.5,
                label="DDE" if j == 0 else "",
            )
            ax.plot(
                t_WW,
                pop_ori[:, j],
                "-r",
                alpha=0.8,
                label="Original WW" if j == 0 else "",
            )
        # Plot optimized WW if opt=True
        if opt:
            ww_opt = WW(
                Delta=Delta_opt,
                positions=positions,
                gamma=gamma_opt,
                n_modes=n_modes,
                L=L,
                setup=setup_kind,
                g_time_modulation=g_time_mod,
            )
            _, pop_opt = ww_opt.evolve(tmax, n_steps=n_steps)
            for j in range(pop_opt.shape[1]):
                ax.plot(
                    t_WW,
                    pop_opt[:, j],
                    "-b",
                    alpha=0.9,
                    label="Optimized WW" if j == 0 else "",
                )

        # Plot config
        print(loss_opt)
        ax.set_xlabel(r"t/$\tau$"), ax.set_ylabel("population")
        title = rf"NormLoss={float(loss_opt):.3e}, $\Delta/FSR$={Delta_guess:.6g}, $\gamma \tau$={gamma_guess:.6g}"
        if opt:
            title += rf", $\Delta_{{opt}}/FSR$={Delta_opt:.6g}, $\gamma_{{opt}} \tau$={gamma_opt:.6g}"
        ax.set_title(title), ax.legend(loc="best"), ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return loss_opt, Delta_opt, gamma_opt
