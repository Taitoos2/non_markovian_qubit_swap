import numpy as np
import matplotlib.pyplot as plt
from aux_funs import run_ww_simulation,DDE_analytical,paralelizar
from experiment_2 import parameter_correction
from scipy.optimize import minimize
from typing import Optional




def exp003(Delta_0:Optional[float] = 1,
		   gamma_0: Optional[float] = None,
		   Delta_list: Optional[list] = None,
		   gamma_list: Optional[list] = list(np.linspace(0.01,0.5,30)),
		   n_modes: int = 150,
		   n_steps: int = 2001,
		   L: float = 1,
		   c: float = 1,
		   scale_log_x: bool=True,
		   scale_log_y: bool=True,
		   method_m: str = 'Powell'):
	
	''' This experiment paralelizes the calculation of the lamb shift and the gamma correction. you can do this 
	as a function of the qubits freq. Delta or the Free space coupling gamma. 	'''
	
	if Delta_0 is not None and gamma_0 is None:
		x = np.asarray(gamma_list)
		def renormalized_parameters(gamma:float):
			return parameter_correction(gamma=gamma,Delta=Delta_0,L=L,c=c,n_modes=n_modes,n_steps=n_steps,method_m=method_m)
		tit=rf"$\Delta = {Delta_0:.0f}$"
		xlab=rf"free space $\gamma $"

	else: 
		x = np.asarray(Delta_list)
		def renormalized_parameters(Delta:float):
			return parameter_correction(gamma=gamma_0,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps,method_m=method_m)
		tit=rf"$\gamma = {gamma_0:.2f}$"
		xlab=rf" $\Delta $"
	
	data = paralelizar(x,renormalized_parameters)
	lamb_shift=[]
	gamma_shift=[]

	for data_set in data:
		lamb_shift.append(data_set[0])
		gamma_shift.append(data_set[1])

	fsr = np.pi*c/L
	lamb_shift = np.asarray(lamb_shift)/fsr
	gamma_shift = np.asarray(gamma_shift)/fsr


	fig,axs = plt.subplots(1,2,figsize=(12,5))
	fig.suptitle(tit)
	
	axs[0].set_title(r'$\Delta_{LS} / $fsr')
	axs[0].plot(x,lamb_shift,'-o',label='data')
	axs[0].set_xlabel(xlab)

	axs[1].set_title(r"$\delta \gamma/ fsr $")
	axs[1].plot(x,gamma_shift,'-o',label='data')
	axs[1].set_xlabel(xlab)
	

	if scale_log_x:
		axs[0].set_xscale('log')
		axs[1].set_xscale('log')
	if scale_log_y:
		axs[0].set_yscale('log')
		axs[1].set_yscale('log')


	fig.tight_layout()
	plt.show()
	return lamb_shift,gamma_shift

# --------------------------------------------------------------

#Delta_list_1 = list(range(1,60))
#ls1,gs1=exp006(Delta_0=None,gamma_0=0.1,Delta_list=Delta_list_1 ,gamma_list=None,n_modes=100,n_steps=1001,sclae_log_y=False)