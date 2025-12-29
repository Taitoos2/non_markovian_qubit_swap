import numpy as np
import matplotlib.pyplot as plt
from aux_funs import run_ww_simulation,DDE_analytical,paralelizar
from scipy.optimize import minimize
from typing import Optional
def parameter_correction(gamma:float,
						Delta:float =1,
						L:float =1,
						c:float = 1, 
						n_modes:int = 100,
						n_steps:int = 401):
	''' given Delta and gamma, returns the renormalized set of parameters that best 
	recover Rabi-like behavior '''
	tau=2*L/c
	T = np.pi/(np.sqrt(gamma/tau))

	def min_estimation(Delta):
		_,e = run_ww_simulation(t_max=0.75*T,gamma = gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		return np.min(e)
	res = minimize(min_estimation,
				x0=Delta,
				bounds=[(Delta-0.6,Delta+0.6)],
				method="Nelder-Mead",)
				# options={"xatol": 1e-8,"fatol": 1e-8})
	
	t_op,e_op = run_ww_simulation(t_max=3*T,gamma=gamma,Delta=res.x[0],L=L,c=c,n_modes=n_modes,n_steps=n_steps)
	
	def dde_correction_estimation(gamma):
		e_dde = np.abs(DDE_analytical(gamma=gamma,phi=2*np.pi,tau=tau,t=t_op))**2
		error = np.trapezoid(np.abs(e_op-e_dde),t_op)
		return np.sum(error)
	res2 = minimize(dde_correction_estimation,x0=gamma,bounds=[(0.5*gamma,1.5*gamma)])	

	lamb_shift = res.x[0]-Delta
	gamma_correction = res2.x[0] - gamma 
	return [lamb_shift,gamma_correction]

def exp005( gamma :float = 0.1,
			Delta_0: float =1 ,
			L:float = 1,
			c: float = 1,
			n_steps: int = 1001,
			n_modes=150,
			fancy_bool:bool=True):
	''' the point of this experiment is to test wether corrections to the frequency of the qubit can help restablish 
	Rabi oscillations in the lower energies of the FSR of the cavity. '''
	tau=2*L/c
	T = np.pi/(np.sqrt(gamma/tau))

	lamb_shift,gamma_correction = parameter_correction(gamma=gamma,Delta=Delta_0,L=L,c=c,n_modes=n_modes,n_steps=n_steps)

	t_op,e_op = run_ww_simulation(t_max=3*T,L=L,c=c,n_steps=n_steps,Delta=Delta_0+lamb_shift,gamma=gamma)
	t,e = run_ww_simulation(t_max=3*T,gamma=gamma,Delta=1,L=L,c=c,n_steps=n_steps)

	print('min. excitation level: '+str(np.min(e_op)))
	fig,ax = plt.subplots()
	ax.plot(t,e,label=r"$\Delta = 1.0 $ F.S.R ")
	ax.plot(t_op,e_op,label=rf"$\Delta = {Delta_0+lamb_shift:.4f}$ F.S.R")
	ax.set_title(rf"$\gamma = {gamma:.1f} , \tau = 2 $")
	ax.grid()
	#plt.xlim(0,5)
	ax.legend()
	fig.tight_layout()
	
	plt.show()

def exp006(Delta_0:Optional[float] = 1,
		   gamma_0: Optional[float] = None,
		   Delta_list: Optional[list] = None,
		   gamma_list: Optional[list] = list(np.linspace(0.01,0.5,30)),
		   n_modes: int = 150,
		   n_steps: int = 2001,
		   L: float = 1,
		   c: float = 1):
	
	''' This experiment paralelizes the calculation of the lamb shift and the gamma correction. you can do this 
	as a function of the qubits freq. Delta or the Free space coupling gamma. 	'''
	
	if Delta_0 is not None and gamma_0 is None:
		x = np.asarray(gamma_list)
		def renormalized_parameters(gamma:float):
			return parameter_correction(gamma=gamma,Delta=Delta_0,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		tit=rf"$\Delta = {Delta_0:.0f}$"
		xlab=rf"free space $\gamma $"

	else: 
		x = np.asarray(Delta_list)
		def renormalized_parameters(Delta:float):
			return parameter_correction(gamma=gamma_0,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		tit=rf"$\gamma = {gamma_0:.2f}$"
		xlab=rf" $\Delta $"
	
	data = paralelizar(x,renormalized_parameters)
	lamb_shift=[]
	gamma_shift=[]

	for data_set in data:
		lamb_shift.append(data_set[0])
		gamma_shift.append(data_set[1])

	lamb_shift = np.asarray(lamb_shift)
	gamma_shift = np.asarray(gamma_shift)


	fig,axs = plt.subplots(1,2,figsize=(12,5))
	fig.suptitle(tit)
	
	axs[0].set_title(r'$\Delta_{LS} / $fsr')
	axs[0].plot(x,lamb_shift,'-o',label='data')
	axs[0].set_xlabel(xlab)
	axs[0].set_yscale('log')
	axs[0].set_xscale('log')


	axs[1].set_title(r"$\delta \gamma/ \gamma $")
	axs[1].plot(x,gamma_shift,'-o',label='data')
	axs[1].set_xlabel(xlab)
	axs[1].set_yscale('log')
	axs[1].set_xscale('log')

	fig.tight_layout()
	plt.show()
	return lamb_shift,gamma_shift

#exp006(Delta_0=1,n_steps=1001)