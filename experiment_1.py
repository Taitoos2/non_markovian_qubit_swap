import numpy as np 
import matplotlib.pyplot as plt 
from aux_funs import run_ww_simulation,DDE_analytical
from qnetwork.tools import set_plot_style
from typing import Optional


def exp001( t_max: Optional[float] = None ,
			gamma :float = 0.1,
			Delta_list: list =[2,4,6],
			L:float = 1,
			c: float = 1,
			n_steps: int = 201,
			n_modes=50,
			fancy_bool:bool=True):
	''' The goal of this experiment is to show that coupling the qubit to higher frequencies of the FSR results in 
	a protection of the Rabi-like oscillations.

	IMPORTANT the comparison only makes sense with deltas spaced by an integrer number, otherwise you are 
	coupled differently to the cavity '''

	tau = 2*L/c
	if t_max is None:
		t_max = 25*tau 

	t_dde = np.linspace(0,t_max,n_steps)
	phi = 2*np.pi*Delta_list[0]
	J = DDE_analytical(gamma=gamma,phi=phi,tau=tau,t=t_dde)
	e_dde = np.abs(J)**2

	if fancy_bool:
		set_plot_style()

	fig,ax = plt.subplots(figsize=(8,6))
	
	for i,Delta in enumerate(Delta_list):
		t,e_ww = run_ww_simulation(t_max=t_max,gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		ax.plot(gamma*t,e_ww,label=rf"$\Delta = {Delta:.1f} \omega_{0}$")

	ax.plot(gamma*t_dde,e_dde,'v', markevery=int(n_steps/100),label='DDE')
	ax.set_xlabel(r"$\gamma t$")
	ax.set_title(r"$\langle \sigma^{+} \sigma^{-} \rangle$")
	# ax.legend()
	
	fig.tight_layout()
	plt.show()
	
exp001(t_max=20,gamma=1,Delta_list=[1,5,20])