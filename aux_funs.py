def dde_series(gamma, tau, t_list, eta, alpha=None, prec=60):
    """
    High-precision analytical series solution of the DDE:

        dc/dt = -alpha c(t) - gamma sum_{n>=1} eta^n c(t - n*tau) Θ(t-nτ)

    Parameters
    ----------
    gamma : float
    tau   : float
    t_list: array-like
        Times at which c(t) is evaluated.
    eta   : complex
        Phase factor: bright = exp(i phi), dark = -exp(i phi)
    alpha : complex or None
        If None, alpha = gamma/2
    prec  : int
        Decimal digits precision for mpmath (default 60).

    Returns
    -------
    result : list of mp.mpf or mp.mpc
        Complex amplitudes c(t).
    """
    import mpmath as mp
    import numpy as np

    # Set precision
    mp.mp.dps = prec

    # Convert inputs to mpf/mpc
    t_list = [mp.mpf(t) for t in t_list]
    gamma = mp.mpf(gamma)
    tau = mp.mpf(tau)
    eta = mp.mpc(eta)

    if alpha is None:
        alpha = gamma / 2
    else:
        alpha = mp.mpc(alpha)

    # Leading term: exp(-alpha t)
    base = [mp.e ** (-alpha * t) for t in t_list]
    result = base[:]  # deep copy

    # Maximum delay index
    N = int(mp.floor(t_list[-1] / tau))

    # Polynomial recursion: P_0(x)=1
    P = [mp.mpf("1")]
    polys = []  # store Q_n

    for n in range(1, N + 1):
        # Q_n(x) = ∫ P_{n-1}(x) dx
        Q = [mp.mpf("0")] + [P[k] / (k + 1) for k in range(len(P))]
        polys.append(Q)

        # P_n(x) = P_{n-1}(x) + Q_n(x)
        P = [(P[k] if k < len(P) else mp.mpf("0")) + Q[k] for k in range(len(Q))]

    # ---- Sum the series ----
    for n in range(1, N + 1):
        Qn = polys[n - 1]

        for i, t in enumerate(t_list):
            tn = t - n * tau
            if tn < 0:
                continue

            x = -gamma * tn

            # Horner evaluation of Q_n(x)
            val = mp.mpf("0")
            for c in reversed(Qn):
                val = val * x + c

            term = (eta**n) * mp.e ** (-alpha * t) * mp.e ** (alpha * n * tau) * val
            result[i] += term

    return np.array(result, dtype=complex)
