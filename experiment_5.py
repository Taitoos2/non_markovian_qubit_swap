import numpy as np
import matplotlib.pyplot as plt 
from aux_funs import run_ww_simulation,DDE_analytical
from scipy.optimize import minimize

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

	def min_estimation(Delta):
		t,e = run_ww_simulation(t_max=T,gamma = gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		return np.min(e)
	

	res = minimize(min_estimation,x0=1,bounds=[(0.5,1.5)])
	print('lamb shift: '+ str(res.x[0])+' F.S.R.')
	#print('new minima: '+str(min_estimation(res.x)))
	t_op,e_op = run_ww_simulation(t_max=6*T,gamma=gamma,Delta=res.x[0],L=L,c=c,n_steps=n_steps)

	def dde_correction_estimation(gamma):
		e_dde = np.abs(DDE_analytical(gamma=gamma,phi=2*np.pi,tau=tau,t=t_op))**2
		error = np.trapezoid(np.abs(e_op-e_dde),t_op)
		return np.sum(error)


	res2 = minimize(dde_correction_estimation,x0=gamma,bounds=[(0.25*gamma,2*gamma)])	
	print('correction to gamma: '+str(res2.x[0]-gamma))
	print(dde_correction_estimation(res2.x[0]))

	t,e = run_ww_simulation(t_max=6*T,gamma=gamma,Delta=1,L=L,c=c,n_steps=n_steps)
	e_dde = np.abs(DDE_analytical(gamma=res2.x[0],phi=2*np.pi,tau=tau,t=t_op))**2
	plt.plot(t,e,label=r"$\Delta = 1.0 $ F.S.R ")
	plt.plot(t_op,e_op,label=rf"$\Delta = {res.x[0]:.4f}$ F.S.R")
	plt.plot(t_op,e_dde,'r--',label='DDE optimized')
	plt.title(rf"$\gamma = {gamma:.1f} , \tau = 2 $")
	#plt.xlim(0,5)
	plt.legend()
	plt.grid()
	plt.show()
