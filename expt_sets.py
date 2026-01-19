import numpy as np
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
    """This function mainly compares the dynamics of the DDE , WW model ,TC model, and pertubation result for a given parametters.
    Parameters:
    Delta: emitters frequency, we should use FSR*Delta in simulation
    gamma: decay rate
    tau: distance between two emitters
    n_modes: the number of modes in WW model
    c: light speed
    PBC: Periodic boundary condition or not
    multiphoton: one or more photon
    T: the total time, we should use T*tau in simulation
    dt_max: the maximum time step in DDE model
    n_steps: the number of steps in WW model
    """

    # run one simulation for a given Delta
    def run_once(Delta_val):
        if PBC:
            L = 2 * tau * c
            positions = [0, L / 2]
        else:
            L = tau * c
            positions = [0, L]
        initial = "10"
        # WW
        if multiphoton:
            setup_WW = EmittersInWaveguideMultiphotonWW(
                Delta=Delta_val,
                positions=positions,
                gamma=gamma,
                n_modes=n_modes,
                L=L,
                setup=Waveguide.Ring if PBC else Waveguide.Cable,
            )
        else:
            setup_WW = EmittersInWaveguideWW(
                Delta=Delta_val,
                positions=positions,
                gamma=gamma,
                L=L,
                n_modes=n_modes,
                setup=Waveguide.Ring if PBC else Waveguide.Cable,
            )
        t_WW, pop_WW = setup_WW.evolve(T * tau, n_steps=n_steps)
        # DDE
        delta = math.modf(Delta_val)[0]
        phi = delta * setup_WW.FSR * tau
        setup_dde = EmittersInWaveguideDDE(
            phi=phi,
            N=2,
            gamma=gamma,
            U=-1,
            tau=tau,
            dt_max=dt_max,
        )
        setup_dde.evolve(T * tau)
        t_DDE, pop_DDE = setup_dde.n_photons(initial)

        # DDE analytical
        # eta_b = np.exp(1j * phi)
        # eta_d = -np.exp(1j * phi)

        # cb = dde_series_function(gamma, tau, eta_b, int(t_WW / tau))
        # cd = dde_series_function(gamma, tau, eta_d, int(t_WW / tau))

        # c1 = (cb + cd) / 2.0
        # c2 = (cb - cd) / 2.0
        # P1 = np.abs(c1) ** 2
        # P2 = np.abs(c2) ** 2

        # pop_dde_ana = np.stack([P1, P2], axis=1)
        # TC
        t_TC, pop_TC = setup_dde.evolve_TC(T * tau, initial)
        return t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC

    # Δ = 0 and Δ = input
    res0 = run_once(int(Delta))
    resD = run_once(Delta)

    fig, axs = plt.subplots(ncols=2, figsize=(18, 5))

    # left: 0 detuning
    ax = axs[0]
    ax.set_title(rf"Dynamics ($\Delta={int(Delta)}*FSR$)")
    t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC = res0

    ax.plot(t_WW / tau, pop_WW[:, 0], color="#0a4570", label=r"$Q_1$: WW")
    ax.plot(t_WW / tau, pop_WW[:, 1], color="#af1a2e", label=r"$Q_2$: WW")
    ax.plot(t_DDE / tau, pop_DDE[:, 0], "--", color="#055805", label=r"$Q_1$: DDE")
    ax.plot(t_DDE / tau, pop_DDE[:, 1], "--", color="#055805", label=r"$Q_2$: DDE")
    ax.plot(
        t_TC / tau,
        pop_TC[:, 0],
        linestyle="none",
        marker="s",
        markevery=20,
        markerfacecolor="#3c91ce",
        markeredgecolor="#0a4570",
        markeredgewidth=2,
    )
    ax.plot(
        t_TC / tau,
        pop_TC[:, 1],
        linestyle="none",
        marker="s",
        markevery=20,
        markerfacecolor="#e02e46",
        markeredgecolor="#af1a2e",
        markeredgewidth=2,
    )
    g = np.sqrt(gamma / 2)
    Omega = np.sqrt(8 * g**2)
    T = 2 * np.pi / Omega
    ax.axvline(T, linestyle="--", color="black")
    ax.axvline(2 * T, linestyle="--", color="black")
    ax.axvline(3 * T, linestyle="--", color="black")
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$\langle \sigma^\dagger \sigma \rangle$")

    # right: large detuning

    # perturbation
    m_max = n_modes
    FSR = np.pi / tau
    delta = math.modf(Delta)[0] * FSR
    g = np.sqrt(gamma / (tau * 2))
    ms = np.arange(-m_max, m_max)
    delta_m = ms * FSR - delta
    eps = -(g**2) * np.sum(1.0 / delta_m)
    pm = np.cos(np.pi * ms)
    # J = -(g**2) * np.sum(pm / delta_m)
    J = (np.pi * g**2 / FSR) / np.sin(np.pi * delta / FSR)  # assmuming infinite modes
    pop_pert = np.stack([np.cos(J * t_TC) ** 2, np.sin(J * t_TC) ** 2], axis=1)
    ax = axs[1]

    ax.set_title(rf"Dynamics ($\Delta={Delta}*FSR$)")
    t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC = resD

    ax.plot(t_WW / tau, pop_WW[:, 0], color="#0a4570", label=r"$Q_1$: WW")
    ax.plot(t_WW / tau, pop_WW[:, 1], color="#af1a2e", label=r"$Q_2$: WW")
    ax.plot(t_DDE / tau, pop_DDE[:, 0], "--", color="#055805", label=r"$Q_1$: DDE")
    ax.plot(t_DDE / tau, pop_DDE[:, 1], "--", color="#055805", label=r"$Q_2$: DDE")
    ax.plot(
        t_TC / tau,
        pop_TC[:, 0],
        linestyle="none",
        marker="s",
        markevery=10,
        markerfacecolor="#3c91ce",
        markeredgecolor="#0a4570",
        markeredgewidth=2,
        label=r"$Q_1$: TC_one_mode",
    )
    ax.plot(
        t_TC / tau,
        pop_TC[:, 1],
        linestyle="none",
        marker="s",
        markevery=20,
        markerfacecolor="#e02e46",
        markeredgecolor="#af1a2e",
        markeredgewidth=2,
        label=r"$Q_2$: TC_one_mode",
    )
    ax.plot(
        t_TC / tau,
        pop_pert[:, 0],
        linestyle="--",
        marker="s",
        markevery=10,
        markerfacecolor="#2bc85d",
        markeredgecolor="#054a1b",
        markeredgewidth=2,
        label=r"$Q_1$: TC_perturbation",
    )
    ax.plot(
        t_TC / tau,
        pop_pert[:, 1],
        linestyle="--",
        marker="s",
        markevery=20,
        markerfacecolor="#c51fc2",
        markeredgecolor="#580a57",
        markeredgewidth=2,
        label=r"$Q_2$: TC_perturbation",
    )
    Omega = 2 * J
    T = np.pi / Omega
    ax.axvline(T, linestyle="--", color="black")
    ax.set_ylabel(r"$\langle \sigma^\dagger \sigma \rangle$")
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$t/\tau$")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------


def expt_002_perturbation_convergence(
    Delta=0.5 * np.pi,
    gamma=0.01,
    FSR=np.pi,
    m_list=None,
    sample_m=None,
    T=200,
    n_steps=2001,
):
    """Compare J(m_max) with analytic J_inf and plot dynamics."""

    if m_list is None:
        m_list = np.arange(5, 150, 5)
    if sample_m is None:
        sample_m = [5, 10, 20, 40, 80, 120]

    # J(m_max)
    def J_eff(m_max):
        g = np.sqrt(gamma / 2)
        ms = np.arange(-m_max, m_max)
        return -(g**2) * np.sum(np.cos(np.pi * ms) / (ms * FSR - Delta))

    # analytic infinite-sum
    def J_inf():
        g = np.sqrt(gamma / 2)
        return (np.pi * g**2 / FSR) / np.sin(np.pi * Delta / FSR)

    J_infty = J_inf()
    J_list = np.array([J_eff(m) for m in m_list])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ---- left: J(m_max) ----
    ax = axes[0]
    ax.plot(m_list, J_list, "o-", label="J(m_max)")
    ax.axhline(J_infty, color="r", linestyle="--", label="J_inf")
    ax.set_xlabel("m_max")
    ax.set_ylabel("J")
    ax.set_title("J convergence")
    ax.legend()
    ax.grid()

    # ---- right: dynamics ----
    ax = axes[1]
    t = np.linspace(0, T, n_steps)

    for m in sample_m:
        Jm = J_eff(m)
        ax.plot(t, np.cos(Jm * t) ** 2, label=f"nmodes={m}")

    ax.plot(t, np.cos(J_infty * t) ** 2, "k--", linewidth=2, label=r"nmodes=$\infty$")
    ax.set_xlabel("t")
    ax.set_ylabel("P1(t)")
    ax.set_title("Dynamics")
    ax.legend()
    ax.grid()

    plt.tight_layout()
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

    def evaluate(t):
        t = np.asarray(t, dtype=float)
        # leading term
        result = np.exp(-alpha * t).astype(np.complex128)
        for n, Q in polys:
            tn = t - n * tau
            term = (
                (eta**n)
                * np.exp(-alpha * t)
                * np.exp(alpha * n * tau)
                * Q(-gamma * tn)
                * np.heaviside(tn, 0.0)
            )
            result += term

        return result

    return evaluate


# binary peak search
def binary_peak_search(p2, Tmin, Tmax, n_scan=1000, tol=1e-9):
    # coarse scan
    ts = np.linspace(Tmin, Tmax, n_scan)
    vals = np.array([p2(t) for t in ts])
    idx = np.argmax(vals)

    # find a local interval
    if idx == 0:
        tL, tR = ts[0], ts[1]
    elif idx == n_scan - 1:
        tL, tR = ts[-2], ts[-1]
    else:
        tL, tR = ts[idx - 1], ts[idx + 1]

    # binary search inside [tL, tR]
    while (tR - tL) > tol:
        m1 = tL + (tR - tL) / 3
        m2 = tR - (tR - tL) / 3
        if p2(m1) < p2(m2):
            tL = m1
        else:
            tR = m2
    t_peak = 0.5 * (tL + tR)
    return t_peak, p2(t_peak)


def compute_period_and_fidelity(
    Delta, gamma, tau=1, T_min=0.8, T_max=1.2, optim_method="binary"
):
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
    if delta == 0:
        Omega = np.sqrt(Delta**2 + 8 * g**2) / 2
    else:
        J = (np.pi * g**2 / FSR) / np.sin(np.pi * delta / FSR)
        Omega = 2 * J

    T = np.pi / Omega
    p2 = make_p2(gamma, T_max * T)

    # peak search
    if optim_method == "binary":
        t_peak, F = binary_peak_search(p2, Tmin=T_min * T, Tmax=T_max * T)

    elif optim_method == "minimize":
        res = scipy.optimize.minimize(
            lambda t: -p2(t),
            np.array([T]),
            bounds=[(T_min * T, T_max * T)],
        )
        t_peak, F = float(res.x), float(-res.fun)

    elif optim_method == "scalar":
        res = scipy.optimize.minimize_scalar(
            lambda t: -p2(t),
            bounds=(T_min * T, T_max * T),
            method="bounded",
        )
        t_peak, F = float(res.x), float(-res.fun)

    return g / FSR, T, t_peak, F


def expt_003_swapspeed(
    Delta, gamma_list, tau=1, T_min=0.8, T_max=1.2, opti_method="binary", n_jobs=8
):
    # parallel scan
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_period_and_fidelity)(
            Delta, gamma, tau, T_min, T_max, opti_method
        )
        for gamma in gamma_list
    )

    # unpack
    g_list = np.array([r[0] for r in results])
    T_list = np.array([r[1] for r in results])
    t_list = np.array([r[2] for r in results])
    F_list = np.array([r[3] for r in results])

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # left: swap speed
    ax[0].plot(g_list, t_list / tau, "o-", color="#0a4570", label="Swap speed")
    ax[0].plot(g_list, T_list / tau, "--", label=r"$\pi/\Omega$")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("g/FSR")
    ax[0].set_ylabel("T/τ")
    ax[0].grid(True)
    ax[0].legend()

    # right: infidelity
    ax[1].plot(g_list, 1 - F_list, "o-", color="#0a4570", label="Infidelity")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("g")
    ax[1].set_ylabel("1 - F")
    ax[1].grid(True)
    ax[1].legend()

    fig.tight_layout()
    plt.show()

    return t_list, F_list


# ------------------------------------------------------------------------------------


def expt_004_opti_compare(Delta, gamma_list, tau=1):
    results = {}

    methods = ["binary", "minimize", "scalar"]

    for m in methods:
        g_list = []
        t_list = []
        F_list = []

        print(f"\nRunning optimizer: {m}")
        for gamma in gamma_list:
            g_over_FSR, T, t_peak, F = compute_period_and_fidelity(
                Delta, gamma, tau, optim_method=m
            )
            g_list.append(g_over_FSR)
            t_list.append(t_peak / tau)
            F_list.append(1 - F)

        results[m] = (np.array(g_list), np.array(t_list), np.array(F_list))

    # ---- plotting ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Left: swap time
    for m in methods:
        g, tpk, inf = results[m]
        ax[0].plot(g, tpk, "--", label=m)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("g/FSR")
    ax[0].set_ylabel("t_peak/τ")
    ax[0].legend()
    ax[0].grid(True)

    # Right: infidelity
    for m in methods:
        g, tpk, inf = results[m]
        ax[1].plot(g, inf, "--", label=m)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("g/FSR")
    ax[1].set_ylabel("1 - F")
    ax[1].legend()
    ax[1].grid(True)

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------
def expt_005_compare_peaks(
    Delta=0.0,
    gamma_list=None,
    tau=1.0,
    T_min_max_list=None,
    optim_method="binary",
):
    """
    expt_003-style plot:
    x-axis: g / FSR
    left : t_peak / tau
    right: 1 - F_max   (log scale)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if gamma_list is None:
        gamma_list = np.linspace(0.01, 0.1, 40)

    if T_min_max_list is None:
        T_min_max_list = [
            (0.3, 0.8),
            (0.5, 1.0),
            (0.8, 1.5),
        ]

    fig, axs = plt.subplots(
        ncols=2,
        figsize=(12, 4),
        sharex=True,
    )

    for T_min, T_max in T_min_max_list:
        g_list = []
        t_list = []
        err_list = []

        for gamma in gamma_list:
            g_over_FSR, T, t_peak, F = compute_period_and_fidelity(
                Delta,
                gamma,
                tau,
                T_min=T_min,
                T_max=T_max,
                optim_method=optim_method,
            )

            g_list.append(g_over_FSR)
            t_list.append(t_peak / tau)
            err_list.append(1.0 - F)

        label = rf"$T_{{\min}}={T_min},\;T_{{\max}}={T_max}$"

        # --- left: t_peak ---
        axs[0].plot(g_list, t_list, lw=2, label=label)

        # --- right: 1 - F ---
        axs[1].plot(g_list, err_list, lw=2)

    # --- axis labels ---
    axs[0].set_ylabel(r"$t_{\mathrm{peak}}/\tau$")
    axs[1].set_ylabel(r"$1 - F_{\max}$")

    axs[0].set_xlabel(r"$g/\mathrm{FSR}$")
    axs[1].set_xlabel(r"$g/\mathrm{FSR}$")

    # --- log scale (as in expt_003) ---
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    for ax in axs:
        ax.grid(alpha=0.3)

    axs[0].legend(frameon=False)

    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------------


def expt_006_stirap(T=100.0, gamma=0.1, tau=1.0, dt_max=0.01, phi=0.0):
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


def expt_008_stirap_T_scan(
    gamma_list,
    T_list,
    phi=0.0,
    tau=1.0,
    dt_max=0.01,
    n_jobs=-1,
):
    gamma_list = np.asarray(gamma_list, float)
    T_list = np.asarray(T_list, float)

    ng, nT = len(gamma_list), len(T_list)
    inf = np.zeros((ng, nT))
    inf_delay = np.zeros((ng, nT))
    inf_cav = np.zeros((ng, nT))

    # pulses
    def gamma_pulse(gamma, T):
        den = 2.0 * (T - 1.0) * tau
        return (
            lambda t: gamma * np.sin(np.pi * t / den) ** 2,
            lambda t: gamma * np.cos(np.pi * t / den) ** 2,
        )

    def gamma_pulse_delay(gamma, T):
        den = 2.0 * (T - 1.0) * tau
        return (
            (
                lambda t: gamma * np.sin(np.pi * t / den) ** 2
                if t <= (T - 1.0) * tau
                else gamma
            ),
            (
                lambda t: gamma * np.cos(np.pi * (t - tau) / den) ** 2
                if t >= tau
                else gamma
            ),
        )

    # ---- one (gamma, T) job ----
    def _solve_one_T(gamma: float, T: float):
        gamma1, gamma2 = gamma_pulse(gamma, T)
        gamma1d, gamma2d = gamma_pulse_delay(gamma, T)

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

    # compute (parallel over T for each gamma)
    for ig, gamma in enumerate(gamma_list):
        results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
            delayed(_solve_one_T)(float(gamma), float(T)) for T in T_list
        )
        # unzip
        inf[ig, :] = [r[0] for r in results]
        inf_delay[ig, :] = [r[1] for r in results]
        inf_cav[ig, :] = [r[2] for r in results]

    # ----- plot pulses (first gamma, first T) -----
    gamma0, T0 = float(gamma_list[0]), float(T_list[0])
    gamma1, gamma2 = gamma_pulse(gamma0, T0)
    gamma1d, gamma2d = gamma_pulse_delay(gamma0, T0)

    tg = np.linspace(0.0, T0 * tau, 800)
    gamma1v = np.array([gamma1(x) for x in tg])
    gamma2v = np.array([gamma2(x) for x in tg])
    gamma1dv = np.array([gamma1d(x) for x in tg])
    gamma2dv = np.array([gamma2d(x) for x in tg])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))

    axL.plot(tg / tau, np.sqrt(gamma1v / (2.0 * tau)) / np.pi, label=r"$g_1$")
    axL.plot(tg / tau, np.sqrt(gamma2v / (2.0 * tau)) / np.pi, label=r"$g_2$")
    axL.plot(
        tg / tau, np.sqrt(gamma1dv / (2.0 * tau)) / np.pi, "--", label=r"$g_1$ (delay)"
    )
    axL.plot(
        tg / tau, np.sqrt(gamma2dv / (2.0 * tau)) / np.pi, "--", label=r"$g_2$ (delay)"
    )

    # x-axis is t/tau so verticals should be 1 and (T0-1)
    axL.axvline(1.0, color="black", linestyle="--", label=r"$\tau$")
    axL.axvline(T0 - 1.0, color="gray", linestyle="--", label=r"$T-\tau$")

    axL.set_xlabel(r"$t/\tau$")
    axL.set_ylabel(r"$g(t)$")
    axL.set_title(f"pulses (g_max/FSR={np.sqrt(gamma0 / 2) / np.pi:g}, T={T0:g}τ)")
    axL.legend()

    # avoid mathtext pitfalls in legend by using plain text labels
    for ig, gamma in enumerate(gamma_list):
        gm = np.sqrt(gamma / (2 * np.pi**2))
        axR.plot(T_list, inf[ig], label=f"g_max/FSR={gm:g}")
        axR.plot(T_list, inf_delay[ig], "--", label=f"g_max/FSR={gm:g} (delay)")
        axR.plot(T_list, inf_cav[ig], ":", label=f"g_max/FSR={gm:g} (cavity)")

    axR.set_xlabel("T/tau")
    axR.set_ylabel("1 - F")
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.legend()

    fig.tight_layout()
    plt.show()
