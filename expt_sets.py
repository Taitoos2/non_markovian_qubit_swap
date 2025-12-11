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
        # TC
        t_TC, pop_TC = setup_dde.evolve_TC(T * tau, initial)
        return t_WW, pop_WW, t_DDE, pop_DDE, t_TC, pop_TC

    # Δ = 0 and Δ = input
    res0 = run_once(int(Delta))
    resD = run_once(Delta)

    fig, axs = plt.subplots(ncols=2, figsize=(18, 5))

    # left: Δ = 0
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
    ax.set_xlabel(r"$t/\tau$")
    ax.set_ylabel(r"$\langle \sigma^\dagger \sigma \rangle$")

    # right: Δ = input
    # perturbation
    m_max = 50
    FSR = np.pi / tau
    delta = math.modf(Delta)[0] * FSR
    print(delta)
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
