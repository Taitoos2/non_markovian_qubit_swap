import numpy as np
from qnetwork.multiphoton_ww import EmittersInWaveguideMultiphotonWW
from numpy.polynomial import Polynomial
from typing import Optional
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.integrate import RK45


def dde_series(gamma, tau, t, eta, alpha=None, _poly_cache={}):
    import numpy as np
    from numpy.polynomial import Polynomial

    t = np.asarray(t, float)
    if t.ndim == 0:
        t = t.reshape(1)

    if alpha is None:
        alpha = 0.5 * gamma

    result = np.zeros_like(t, dtype=complex)

    N = int(t[-1] // tau)

    key = (tau, N)
    if key not in _poly_cache:
        P = Polynomial([1.0])
        polys = [(P.copy(), None)]
        for _ in range(1, N + 1):
            Q = P.integ()
            P = P + Q
            polys.append((P.copy(), Q.copy()))
        _poly_cache[key] = polys
    polys = _poly_cache[key]

    # leading term
    result += np.exp(-alpha * t)

    for n in range(1, N + 1):
        Pn, Qn = polys[n]
        tn = t - n * tau
        mask = tn >= 0
        if not np.any(mask):
            continue

        x = -gamma * tn[mask]

        term = (eta**n) * np.exp(-alpha * t[mask]) * np.exp(alpha * n * tau) * Qn(x)

        result[mask] += term

    return result


# ---------------------------------------------------------------------


def DDE_analytical(gamma, phi, tau, t):
    """returns the analitycal solution for the DDE of a single emitter in a cavity"""

    alpha = 1j * phi / tau + 0.5 * gamma
    result = np.exp(-alpha * t) * np.ones(len(t), dtype=complex)
    poli = Polynomial([1])
    N = int(t[-1] / tau)

    for n in range(1, N + 1):
        dummie = poli.integ()
        result += (
            np.exp(-alpha * t)
            * np.exp(n * alpha * tau)
            * dummie(-gamma * (t - n * tau))
            * np.heaviside(t - n * tau, 1)
        )
        poli += dummie
    return result


def DDE_polynomial_series(N):
    """returns the polynomial series used in the analytical solution"""
    poli = Polynomial([1])
    series = [poli]
    for n in range(1, N + 1):
        dummie = poli.integ()
        series.append(dummie)
        poli += dummie
    return series


def DDE_evaluate_series(gamma, phi, tau, t, series):
    """evaluation of the polynomial series in a specific time t"""
    result = 0.0
    alpha = 1j * phi / tau + 0.5 * gamma
    for k, poli in enumerate(series):
        n = k
        result += (
            np.exp(-alpha * t)
            * np.exp(n * alpha * tau)
            * poli(-gamma * (t - n * tau))
            * np.heaviside(t - n * tau, 1)
        )

    return result


def run_ww_simulation(
    t_max: Optional[float] = None,
    gamma: float = 0.1,
    Delta: float = 10.0,
    L: float = 1,
    c: float = 1,
    n_steps: int = 201,
    n_modes=20,
):
    """run the dynamic of A SINGLE QUBIT in a cavity, using the Wigner-Weisskopf integrator"""
    tau = 2 * L / c
    if t_max is None:
        t_max = 25 * tau
    setup = EmittersInWaveguideMultiphotonWW(
        gamma=gamma,
        Delta=Delta,
        L=L,
        c=c,
        positions=[0.0],
        n_modes=n_modes,
        n_excitations=list(range(2)),
    )
    t, e = setup.evolve(t_max, n_steps=n_steps, initial_state="1")
    return t, e[:, 0]


# --------------------------------------------------------------------------------


def two_qubits_analytical(
    t_max: float = 20,
    n_steps: int = 201,
    gamma: float = 0.1,
    phi: float = 2 * np.pi,
    tau: float = 2,
    initial: np.ndarray = np.asarray([0, 1, 0, 0]),
):
    t = np.linspace(0, t_max, n_steps)
    b = np.asarray([[0, 0], [1, 0]])
    b1 = np.kron(b, np.eye(2))
    b2 = np.kron(np.eye(2), b)

    c_plus = DDE_analytical(gamma=gamma, phi=phi, tau=tau, t=t)
    c_minus = DDE_analytical(gamma=gamma, phi=phi + np.pi, tau=tau, t=t) * np.exp(
        1j * np.pi / tau * t
    )

    sum = 0.5 * (c_plus + c_minus)
    dif = 0.5 * (c_plus - c_minus)

    pop1 = (np.abs(sum) ** 2) * np.dot(initial, b1.T @ b1 @ initial).astype(complex)
    pop1 += np.conjugate(sum) * dif * np.dot(initial, b1.T @ b2 @ initial)
    pop1 += np.conjugate(dif) * sum * np.dot(initial, b2.T @ b1 @ initial)
    pop1 += (np.abs(dif) ** 2) * np.dot(initial, b2.T @ b2 @ initial)

    pop2 = (np.abs(dif) ** 2) * np.dot(initial, b1.T @ b1 @ initial).astype(complex)
    pop2 += np.conjugate(dif) * sum * np.dot(initial, b1.T @ b2 @ initial)
    pop2 += np.conjugate(sum) * dif * np.dot(initial, b2.T @ b1 @ initial)
    pop2 += (np.abs(sum) ** 2) * np.dot(initial, b2.T @ b2 @ initial)
    return t, [np.abs(pop1), np.abs(pop2)]


def two_qubits_analytical_Hong(
    t_max: float = 20,
    n_steps: int = 201,
    gamma: float = 0.1,
    phi: float = 2 * np.pi,
    L: float = 1,
    c: float = 1,
    initial: np.ndarray = np.asarray([0, 1, 0, 0]),
):
    tau = L / c
    t = np.linspace(0, t_max, n_steps)
    b = np.asarray([[0, 0], [1, 0]])
    b1 = np.kron(b, np.eye(2))
    b2 = np.kron(np.eye(2), b)

    c_plus = dde_series(gamma=gamma, tau=tau, eta=np.exp(1j * phi), t=t)
    c_minus = dde_series(gamma=gamma, tau=tau, eta=-np.exp(1j * phi), t=t)

    sum = 0.5 * (c_plus + c_minus)
    dif = 0.5 * (c_plus - c_minus)

    pop1 = (np.abs(sum) ** 2) * np.dot(initial, b1.T @ b1 @ initial).astype(complex)
    pop1 += np.conjugate(sum) * dif * np.dot(initial, b1.T @ b2 @ initial)
    pop1 += np.conjugate(dif) * sum * np.dot(initial, b2.T @ b1 @ initial)
    pop1 += (np.abs(dif) ** 2) * np.dot(initial, b2.T @ b2 @ initial)

    pop2 = (np.abs(dif) ** 2) * np.dot(initial, b1.T @ b1 @ initial).astype(complex)
    pop2 += np.conjugate(dif) * sum * np.dot(initial, b1.T @ b2 @ initial)
    pop2 += np.conjugate(sum) * dif * np.dot(initial, b2.T @ b1 @ initial)
    pop2 += (np.abs(sum) ** 2) * np.dot(initial, b2.T @ b2 @ initial)
    return t, [np.abs(pop1), np.abs(pop2)]


## -------------------------------------
def paralelizar(parameter_list, f, ncores: int = 10):
    resultados = Parallel(n_jobs=ncores, backend="loky")(
        delayed(f)(param) for param in parameter_list
    )
    return resultados


# -----------------------------------------------------
def dde_scalar(
    t_max: float,
    gamma1,  # float | Callable[[float], float]
    gamma2,  # float | Callable[[float], float]
    phi: float = 0.0,
    tau: float = 1.0,
    N: int = 2,
    dt_max: float = 1e-2,
    buffer_size: int = 1000,
):
    phase = np.exp(1j * phi)
    shape = (N,)

    next_index = 1
    t_list = np.zeros(buffer_size, dtype=float)

    c_list = np.zeros((buffer_size, N), dtype=np.complex128)
    c_list[0] = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    # a_out_list: (time, 2 directions, N emitters)
    a_out_list = np.zeros((buffer_size, 2, N), dtype=np.complex128)

    right = list(range(1, N)) + [0]
    left = [N - 1] + list(range(N - 1))

    zero_current = np.zeros((2, N), dtype=np.complex128)

    def zero_current_interpolator(t: np.floating) -> np.ndarray:
        return zero_current

    interpolator = zero_current_interpolator

    def gamma_of_t(t: float) -> np.ndarray:
        g1 = gamma1(t) if callable(gamma1) else gamma1
        g2 = gamma2(t) if callable(gamma2) else gamma2
        return np.array([g1, g2], dtype=float)

    def derivative(t, y):
        c = y.reshape(shape)  # (N,)
        a_out_past = interpolator(t - tau)  # (2, N)

        a_in = phase * a_out_past[0, right] + phase * a_out_past[1, left]  # (N,)

        gamma = gamma_of_t(float(t))
        gamma_half = 0.5 * gamma

        dcdt = -gamma_half * c - 1j * np.sqrt(gamma_half) * a_in
        return dcdt.reshape(-1)

    def add_solution(t: float | np.floating, c: np.ndarray):
        nonlocal next_index, buffer_size, t_list, c_list, a_out_list, interpolator
        i = next_index
        L = buffer_size

        if i >= L:
            buffer_size = L = int(L * 1.5)
            t_list = np.resize(t_list, L)
            c_list = np.resize(c_list, (L, N))
            a_out_list = np.resize(a_out_list, (L, 2, N))

        t_list[i] = float(t)
        c_list[i] = c

        a_out_past = interpolator(np.float64(t - tau))  # (2, N)

        gamma_now = gamma_of_t(float(t))
        emitted = (-1j * np.sqrt(0.5 * gamma_now)) * c  # (N,)

        a_out_list[i, 0, :] = phase * a_out_past[0, right] + emitted
        a_out_list[i, 1, :] = phase * a_out_past[1, left] + emitted

        next_index = i + 1

    add_solution(np.float64(0), c_list[0])

    integrator = RK45(
        derivative,
        t_list[0],
        c_list[0].reshape(-1),
        t_bound=float(t_max),
        max_step=float(dt_max) * float(tau),
        vectorized=False,
    )

    while integrator.t < t_max:
        if integrator.t > tau:
            interpolator = interp1d(
                t_list[:next_index],
                a_out_list[:next_index],
                assume_sorted=True,
                axis=0,
                copy=False,
            )
        integrator.step()
        add_solution(integrator.t, integrator.y.reshape(shape))


    i = next_index
    t_list = t_list[:i]
    c_list = c_list[:i]

    return t_list, c_list


from joblib import Parallel, delayed 
from numpy.fft import fft, fftshift
from scipy.interpolate import interp1d

def paralelizar(parameter_list,f,ncores: int = 12):
	resultados = Parallel(n_jobs=ncores, backend='loky')(
		delayed(f)(param) for param in parameter_list
	)
	return resultados


def fast_f_t(x : np.ndarray,y:np.ndarray, M:int = 500):
		t_interp = np.linspace(0, x[-1], M)  
		dt = t_interp[1] - t_interp[0]
		y = np.interp(t_interp,x, y )
		y -= np.mean(y)
		k = np.fft.fftfreq(M, d=dt)
		yw = np.fft.fft(y)
		return 2*np.pi*fftshift(k), fftshift(yw)*dt


def average_fft(x, y, Ms):
	spectra = []
	freqs_list = []

	for M in Ms:
		omega, A = fast_f_t(x, y, M)
		freqs_list.append(omega)
		spectra.append(A)

	omega_min = max(freqs[0] for freqs in freqs_list)     # límite inferior común
	omega_max = min(freqs[-1] for freqs in freqs_list)    # límite superior común
	N_common = max(len(f) for f in freqs_list)            # densidad similar a la mayor
	omega_common = np.linspace(omega_min, omega_max, N_common)


	spectra_interp = []
	for omega, A in zip(freqs_list, spectra):
		f_interp = interp1d(omega, A, kind='linear', bounds_error=False, fill_value=0.0)
		spectra_interp.append(f_interp(omega_common))

	A_avg = np.mean(spectra_interp, axis=0)
	return omega_common, A_avg

