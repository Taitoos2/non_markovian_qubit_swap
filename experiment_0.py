import numpy as np 
import matplotlib.pyplot as plt 
from aux_funs import run_ww_simulation, DDE_analytical
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from typing import Optional

plt.rcParams['mathtext.fontset'] = 'cm'

def exp000(t_max: Optional[float] = None ,
			gamma :float = 0.1,
			Delta: float = 10,
			L:float = 1,
			c: float = 1,
			n_steps: int = 201,
			n_modes=100,
			periods_derivative: int = 6 ):
	''' The goal of this experiment is to show the piece-wise differentiable nature of the dynamics as we include more modes'''
	tau = 2*L/c
	if t_max is None:
		t_max = 12*tau 

	phi = 2*np.pi*Delta 
	t,e_ww = run_ww_simulation(t_max=t_max,gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
	J = DDE_analytical(gamma=gamma,phi=phi,tau=tau,t=t)
	e_dde = np.abs(J)**2
	
	dt = t[1]-t[0]
	d_ww_dt = np.gradient(e_ww,dt)
	d_dde_dt = np.gradient(e_dde,dt)



	fig,axs = plt.subplots(2,1,figsize=(7,5),constrained_layout=True, gridspec_kw={"height_ratios": [5, 4]})

	axs[0].plot(t/tau,np.exp(-gamma*t),'k-.',label=r"$ e^{-\gamma t} $",alpha=0.35)
	axs[0].plot(t/tau,e_dde,'k',label='DDE')
	axs[0].plot(t/tau,e_ww,label='Wigner-Weisskopf',color ='lightskyblue',linestyle='dashed')
	axs[0].set_xlabel(r"$ t / \tau  $")

	axins = inset_axes(axs[0],width="20%",height="45%",loc="upper right", bbox_to_anchor=(0.055, 0.04, 0.95, 0.95),bbox_transform=axs[0].transAxes)
	idx = np.argmin(np.abs(t - 3*tau))
	e_dde_3 = e_dde[idx]
	axins.plot(t/tau,np.exp(-gamma*t),'k-.',label=r"$ e^{-\gamma t} $",alpha=0.35)
	axins.plot(t/tau,e_dde,'k',label='DDE')
	axins.plot(t/tau,e_ww,label='WW',color ='lightskyblue',linestyle='dashed')
	
	axins.set_yscale('log')
	axins.set_xlim(0.1,3.1)
	axins.set_ylim(e_dde_3,1)

	axins.set_yscale('log')


	
	axs[1].plot(t/tau,d_dde_dt,label='DDE ',color='k')
	axs[1].plot(t/tau,d_ww_dt,color ='lightskyblue',linestyle='dashed',label='WW ')
	axs[1].set_xlabel(r"$ t / \tau  $")
	axs[1].legend()
	axs[1].set_xlim(0,t[-1]/tau+(t[1]-t[0]))

	for i in range(1,int(t[-1]/tau)+1):
		axs[1].axvline(i,color='k',linestyle='--',alpha=0.35)
	axs[1].set_xlim(0,periods_derivative)

	#fig.tight_layout()  # already included in set_plot_style ()
	fig.savefig('figure0.pdf')
	
	plt.show()
	
from qnetwork.tools import set_plot_style
set_plot_style()
exp000(Delta=50,gamma=0.1,periods_derivative=6,n_steps=2001,n_modes=100)