import numpy as np 
import matplotlib.pyplot as plt 
from aux_funs import run_ww_simulation,DDE_analytical
from qnetwork.tools import set_plot_style
from typing import Optional


def exp002( t_max: Optional[float] = None ,
			gamma_list : list = [0.05,0.1,0.2],
			Delta_list: list = list(range(1,54,4)),
			L:float = 1,
			c: float = 1,
			n_steps: int = 201,
			n_modes=100,
			fancy_bool:bool=True):
	
	''' The goal of this experiment is to show that coupling the qubit to higher frequencies of the FSR results in 
	a protection of the Rabi-like oscillations. 
	note: I am aware I am computing many times the dde solution, which would not be necessary if the time vector can be fixed beforehand. 
	I will check this later '''
	
	phi = 2*np.pi 
	tau = 2*L/c
	if t_max is None:
		t_max = 25*tau 
	fig,ax = plt.subplots(figsize=(7,5))

	for gamma in gamma_list:
		error_list=[]
		for Delta in Delta_list:
			phi = 2*np.pi*Delta
			t,e_ww = run_ww_simulation(t_max=t_max,gamma=gamma,Delta=Delta,L=L,c=c,n_steps=n_steps,n_modes=n_modes)
			J = DDE_analytical(gamma=gamma,phi=phi,tau=tau,t=t)
			e_dde = np.abs(J)**2
			error = np.sqrt(np.trapezoid(y=np.abs(e_ww-e_dde)**2,x=t)) 
			norm = np.sqrt(np.trapezoid(y=np.abs(e_dde)**2,x=t))
			error_list.append(error/norm)
		
		ax.plot(Delta_list,error_list,'-v',label=rf"$\gamma = {gamma:.2f} $")

	ax.set_xlabel(r"$\Delta / \omega_{0} $")
	ax.set_title(r" $L^{2}$ error")
	ax.legend()
	ax.grid()
	fig.tight_layout()
	plt.show()

exp002(t_max=20)