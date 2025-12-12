import numpy as np
import matplotlib.pyplot as plt


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
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from qnetwork.dde import EmittersInWaveguideDDE
    from aux_funs import dde_series
    from qnetwork.dde import DDE_analytical
    from qnetwork.ww import EmittersInWaveguideWW
    from qnetwork.multiphoton_ww import EmittersInWaveguideMultiphotonWW, Waveguide

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
        eta_b = np.exp(1j * phi)
        eta_d = -np.exp(1j * phi)

        cb = dde_series(gamma, tau, t_DDE, eta_b)  # bright
        cd = dde_series(gamma, tau, t_DDE, eta_d)  # dark
        # cb = DDE_analytical(gamma, phi, tau, t_DDE)  # bright
        # cd = DDE_analytical(gamma, phi + np.pi, tau, t_DDE)  # dark

        c1 = (cb + cd) / 2.0
        c2 = (cb - cd) / 2.0
        P1 = np.abs(c1) ** 2
        P2 = np.abs(c2) ** 2

        pop_dde_ana = np.stack([P1, P2], axis=1)
        # TC
        t_TC, pop_TC = setup_dde.evolve_TC(T * tau, initial)
        return t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC, pop_dde_ana

    # Δ = 0 and Δ = input
    res0 = run_once(int(Delta))
    resD = run_once(Delta)

    fig, axs = plt.subplots(ncols=2, figsize=(18, 5))

    # left: 0 detuning
    ax = axs[0]
    ax.set_title(rf"Dynamics ($\Delta={int(Delta)}*FSR$)")
    t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC, pop_dde_ana = res0

    ax.plot(t_WW / tau, pop_WW[:, 0], color="#0a4570", label=r"$Q_1$: WW")
    ax.plot(t_WW / tau, pop_WW[:, 1], color="#af1a2e", label=r"$Q_2$: WW")
    ax.plot(t_DDE / tau, pop_dde_ana[:, 0], "--", color="#055805", label=r"$Q_1$: DDE")
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
    J = (np.pi * g**2 / FSR) / np.sin(np.pi * delta / FSR)
    pop_pert = np.stack([np.cos(J * t_TC) ** 2, np.sin(J * t_TC) ** 2], axis=1)
    ax = axs[1]

    ax.set_title(rf"Dynamics ($\Delta={Delta}*FSR$)")
    t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC, pop_dde_ana = resD

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
        linestyle="none",
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
        linestyle="none",
        marker="s",
        markevery=20,
        markerfacecolor="#c51fc2",
        markeredgecolor="#580a57",
        markeredgewidth=2,
        label=r"$Q_2$: TC_perturbation",
    )
    ax.set_ylabel(r"$\langle \sigma^\dagger \sigma \rangle$")
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$t/\tau$")
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


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
        ax.plot(t, np.cos(Jm * t) ** 2, label=f"m_max={m}")

    ax.plot(t, np.cos(J_infty * t) ** 2, "k--", linewidth=2, label=r"m_max=$\infty$")
    ax.set_xlabel("t")
    ax.set_ylabel("P1(t)")
    ax.set_title("Dynamics")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()


def expt_003_swapspeed(Delta, gamma_list, tau=1, p_target=0.99, T_max=350):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    from qnetwork.dde import dde_series

    phi = math.modf(Delta)[0] * np.pi

    def p2(gamma, t):
        eta_b = np.exp(1j * phi)
        eta_d = -np.exp(1j * phi)
        cb = dde_series(gamma, tau, t, eta_b)
        cd = dde_series(gamma, tau, t, eta_d)
        return np.abs(0.5 * (cb - cd)) ** 2

    def find_first_peak(gamma):
        dt = 0.02
        Tmax = T_max * tau
        t_prev, p_prev = 0.0, p2(gamma, 0.0)

        for k in range(1, int(Tmax / dt) + 1):
            t = k * dt
            p = p2(gamma, t)
            if p >= p_target > p_prev:
                lo, hi = t_prev, t
                break
            t_prev, p_prev = t, p
        else:
            lo, hi = 0.0, Tmax

        res = minimize_scalar(
            lambda t: -p2(gamma, t),
            bounds=(lo, hi),
            method="bounded",
        )
        t_peak = res.x
        return t_peak, p2(gamma, t_peak)

    t_peaks = []
    F_list = []

    for gamma in gamma_list:
        t_peak, F = find_first_peak(gamma)
        t_peaks.append(t_peak)
        F_list.append(F)

    t_peaks = np.array(t_peaks)
    F_list = np.array(F_list)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(gamma_list, t_peaks, "-o")
    ax[0].set_xlabel("gamma")
    ax[0].set_ylabel("t_peak")

    ax[1].plot(gamma_list, 1 - F_list, "-o")
    ax[1].set_xlabel("gamma")
    ax[1].set_ylabel("1 - F")

    plt.tight_layout()
    plt.show()

    return t_peaks, F_list


def expt_003_swapspeed(Delta, gamma_list, tau=1, p_target=0.99, T_max=350, dt0=0.02):
    import numpy as np, math
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    from qnetwork.dde import dde_series

    phi = math.modf(Delta)[0] * np.pi
    eta_b, eta_d = np.exp(1j * phi), -np.exp(1j * phi)

    # ---------------------------------------------------------
    #  p2 with cache
    # ---------------------------------------------------------
    p2_cache = {}

    def p2(gamma, t):
        key = (gamma, t)
        if key in p2_cache:
            return p2_cache[key]

        cb = dde_series(gamma, tau, t, eta_b)[0]
        cd = dde_series(gamma, tau, t, eta_d)[0]
        val = np.abs(0.5 * (cb - cd)) ** 2
        p2_cache[key] = val
        return val

    # ---------------------------------------------------------
    #  search first peak: reduce calls to p2!
    # ---------------------------------------------------------
    def find_first_peak(gamma):
        t_prev, p_prev = 0.0, p2(gamma, 0.0)
        dt = dt0
        t = dt
        Tmax = T_max * tau

        # --- adaptive coarse scan: speed-up ---
        while t < Tmax:
            p = p2(gamma, t)
            if p >= p_target > p_prev:
                lo, hi = t_prev, t
                break
            t_prev, p_prev = t, p
            t += dt
            # accelerate scanning in large T region
            if t > 50:
                dt *= 1.05
        else:
            lo, hi = 0.0, Tmax

        # --- refine peak ---
        res = minimize_scalar(
            lambda x: -p2(gamma, x), bounds=(lo, hi), method="bounded"
        )
        return res.x, p2(gamma, res.x)

    # ---------------------------------------------------------
    #  scan over gamma
    # ---------------------------------------------------------
    t_peaks, F_list = [], []

    for gamma in gamma_list:
        p2_cache.clear()  # avoid cache explosion
        t_peak, F = find_first_peak(gamma)
        t_peaks.append(t_peak)
        F_list.append(F)

    t_peaks = np.array(t_peaks)
    F_list = np.array(F_list)

    # ---------------------------------------------------------
    #  plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(gamma_list, t_peaks, "-o")
    ax[0].set_xlabel("gamma")
    ax[0].set_ylabel("t_peak")

    ax[1].plot(gamma_list, 1 - F_list, "-o")
    ax[1].set_xlabel("gamma")
    ax[1].set_ylabel("1 - F")

    plt.tight_layout()
    plt.show()

    return t_peaks, F_list
