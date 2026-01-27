import numpy as np 
import matplotlib.pyplot as plt 
from aux_funs import run_ww_simulation,DDE_analytical
from typing import Optional

plt.rcParams['mathtext.fontset'] = 'cm'


CB_color_cycle = ['darkblue','cornflowerblue','lightskyblue']
linestyle_cycle=['dotted','dashdot','dashed']
marker_cycle = ['v','o','*','d']


def exp001(gamma :float = 0.1,
			Delta_list: list =[2,4,6],
			L:float = 1,
			c: float = 1,
			n_steps: int = 401,
			n_modes=70,
			n_points: int =100):
	''' The goal of this experiment is to show that coupling the qubit to higher frequencies of the FSR results in 
	a protection of the Rabi-like oscillations.

	IMPORTANT the comparison only makes sense with deltas spaced by an integrer number, otherwise you are 
	coupled differently to the cavity '''

	tau = 2*L/c
	T=np.pi / (np.sqrt(gamma/tau))
	t_max= 2.1*T

	t = np.linspace(0,t_max,n_steps)
	e_dde = np.abs(DDE_analytical(gamma=gamma,phi=0,tau=tau,t=t))**2
	fig,axs = plt.subplots(figsize=(8,5))
	axs.set_xlabel(r"$t/\tau$")
	


	
	axs.plot(t/tau,e_dde,color='k',label="DDE")

	for i,Delta in enumerate(Delta_list):
		_,e = run_ww_simulation(t_max=t_max,gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		axs.plot(t/tau,e,label=rf"$\omega_e = {Delta:.0f} \omega_{0} $",
		   linestyle = linestyle_cycle[i],
		   color = CB_color_cycle[i],
		   markevery=int(n_steps/n_points))
	
	
	axs.legend()
	#fig.tight_layout()
	fig.savefig('figure1.pdf')
	axs.set_xlabel(r"$t/\tau$")
	plt.show()


from qnetwork.tools import set_plot_style
set_plot_style()
exp001(gamma=0.5,Delta_list=[1,5,100],n_points=201)