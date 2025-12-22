import numpy as np
from qnetwork.multiphoton_ww import EmittersInWaveguideMultiphotonWW
from numpy.polynomial import Polynomial
from typing import Optional

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


def dde_series_highprecision(gamma, tau, t_list, eta, alpha=None, prec=60):
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

# ---------------------------------------------------------------------

def DDE_analytical(gamma,phi,tau,t):
	''' returns the analitycal solution for the DDE of a single emitter in a cavity'''
	
	alpha = 1j*phi/tau + 0.5*gamma 
	result =  np.exp(-alpha*t)*np.ones(len(t),dtype=complex)
	poli = Polynomial([1])
	N = int(t[-1]/tau)
	
	for n in range(1,N+1):
		dummie = poli.integ() 
		result += np.exp(-alpha*t)*np.exp(n*alpha*tau)*dummie(-gamma*(t-n*tau))*np.heaviside(t-n*tau,1)
		poli += dummie 
	return result 

def DDE_polynomial_series(N):
	''' returns the polynomial series used in the analytical solution'''
	poli = Polynomial([1])
	series=[poli]
	for n in range(1,N+1):
		dummie = poli.integ() 
		series.append(dummie)
		poli += dummie 
	return series

def DDE_evaluate_series(gamma,phi,tau,t,series):
	''' evaluation of the polynomial series in a specific time t '''
	result = 0.0
	alpha = 1j*phi/tau + 0.5*gamma 
	for k,poli in enumerate(series):
		n=k
		result+= np.exp(-alpha*t)*np.exp(n*alpha*tau)*poli(-gamma*(t-n*tau))*np.heaviside(t-n*tau,1)

	return result 


def run_ww_simulation(t_max: Optional[float] = None , gamma :float = 0.1, Delta: float = 10.0 , L:float = 1, c: float = 1, n_steps: int = 201,n_modes=20):
	''' run the dynamic of A SINGLE QUBIT in a cavity, using the Wigner-Weisskopf integrator'''
	tau=2*L/c
	if t_max is None:
		t_max = 25*tau 
	setup=EmittersInWaveguideMultiphotonWW(gamma=gamma,Delta=Delta,L=L,c=c,positions=[0.0], n_modes=n_modes, n_excitations=list(range(2)))
	t,e = setup.evolve(t_max,n_steps=n_steps,initial_state="1")
	return t,e[:,0]

# --------------------------------------------------------------------------------

def two_qubits_analytical(t_max: float = 20,
						n_steps: int = 201,
						gamma: float = 0.1,
						phi: float = 2*np.pi, 
						tau : float = 2,
						initial: np.ndarray = np.asarray([0,1,0,0])):
	 
	t = np.linspace(0,t_max,n_steps)
	b = np.asarray([[0,0],[1,0]])
	b1 = np.kron(b,np.eye(2))
	b2 = np.kron(np.eye(2),b)

	c_plus = DDE_analytical(gamma=gamma,phi=phi,tau=tau,t=t)
	c_minus = DDE_analytical(gamma=gamma,phi=phi + np.pi,tau=tau,t=t) * np.exp(1j*np.pi/tau*t)

	sum = 0.5*(c_plus+c_minus)
	dif = 0.5*(c_plus-c_minus)

	pop1 = (np.abs(sum)**2)*np.dot(initial,b1.T@b1@initial).astype(complex)
	pop1+= np.conjugate(sum)*dif*np.dot(initial,b1.T@b2@initial)
	pop1+= np.conjugate(dif)*sum*np.dot(initial,b2.T@b1@initial)
	pop1+= (np.abs(dif)**2)*np.dot(initial,b2.T@b2@initial)

	pop2 = (np.abs(dif)**2)*np.dot(initial,b1.T@b1@initial).astype(complex)
	pop2+= np.conjugate(dif)*sum*np.dot(initial,b1.T@b2@initial)
	pop2+= np.conjugate(sum)*dif*np.dot(initial,b2.T@b1@initial)
	pop2+= (np.abs(sum)**2)*np.dot(initial,b2.T@b2@initial)
	return t,[np.abs(pop1),np.abs(pop2)]


def two_qubits_analytical_Hong(t_max: float = 20,
						  n_steps: int = 201,
						  gamma: float = 0.1,
						  phi: float = 2*np.pi, 
						  L:float =1,
						  c: float = 1,
						  initial: np.ndarray = np.asarray([0,1,0,0])):
	tau = L/c
	t = np.linspace(0,t_max,n_steps)
	b = np.asarray([[0,0],[1,0]])
	b1 = np.kron(b,np.eye(2))
	b2 = np.kron(np.eye(2),b)

	c_plus = dde_series(gamma=gamma,tau=tau,eta=np.exp(1j*phi),t=t)
	c_minus = dde_series(gamma=gamma,tau=tau,eta=-np.exp(1j*phi),t=t)

	sum = 0.5*(c_plus+c_minus)
	dif = 0.5*(c_plus-c_minus)

	pop1 = (np.abs(sum)**2)*np.dot(initial,b1.T@b1@initial).astype(complex)
	pop1+= np.conjugate(sum)*dif*np.dot(initial,b1.T@b2@initial)
	pop1+= np.conjugate(dif)*sum*np.dot(initial,b2.T@b1@initial)
	pop1+= (np.abs(dif)**2)*np.dot(initial,b2.T@b2@initial)

	pop2 = (np.abs(dif)**2)*np.dot(initial,b1.T@b1@initial).astype(complex)
	pop2+= np.conjugate(dif)*sum*np.dot(initial,b1.T@b2@initial)
	pop2+= np.conjugate(sum)*dif*np.dot(initial,b2.T@b1@initial)
	pop2+= (np.abs(sum)**2)*np.dot(initial,b2.T@b2@initial)
	return t,[np.abs(pop1),np.abs(pop2)]

## -------------------------------------

from joblib import Parallel, delayed 

def paralelizar(parameter_list,f,ncores: int = 80):
	resultados = Parallel(n_jobs=ncores, backend='loky')(
		delayed(f)(param) for param in parameter_list
	)
	return resultados
