import numpy as np 
import matplotlib.pyplot as plt 
from aux_funs import run_ww_simulation, DDE_analytical
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qnetwork.tools import set_plot_style
from typing import Optional

def exp000(t_max: Optional[float] = None ,
			gamma :float = 0.1,
			Delta: float = 10,
			L:float = 1,
			c: float = 1,
			n_steps: int = 201,
			n_modes=50,
			fancy_bool:bool=True,
			periods_derivative: int = 6 ):
	''' The goal of this experiment is to show the piece-wise differentiable nature of the dynamics as we include more modes'''
	tau = 2*L/c
	if t_max is None:
		t_max = 25*tau 

	phi = 2*np.pi*Delta 
	t,e_ww = run_ww_simulation(t_max=t_max,gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
	J = DDE_analytical(gamma=gamma,phi=phi,tau=tau,t=t)
	e_dde = np.abs(J)**2
	
	dt = t[1]-t[0]
	d_ww_dt = np.gradient(e_ww,dt)
	d_dde_dt = np.gradient(e_dde,dt)
	if fancy_bool:
		set_plot_style()

	fig,axs = plt.subplots(1,2,figsize=(12,5))

	axs[0].plot(t/tau,e_ww,label='Wigner-Weisskopf')
	axs[0].plot(t/tau,e_dde,'r--',label='DDE')
	axs[0].plot(t/tau,np.exp(-gamma*t),'k-.',label=r"$ e^{-\gamma t} $")
	axs[0].set_xlabel(r"$ t / \tau  $")
	axs[0].set_title(r"$\langle \sigma^{+}\sigma^{-} \rangle $")

	axins = inset_axes(axs[0],width="30%",height="30%",loc="upper right")
	idx = np.argmin(np.abs(t - 3*tau))
	e_dde_3 = e_dde[idx]
	axins.plot(t/tau,e_ww,label='WW')
	axins.plot(t/tau,e_dde,'r--',label='DDE')
	axins.plot(t/tau,np.exp(-gamma*t),'k-.',label=r"$ e^{-\gamma t} $")
	axins.set_yscale('log')
	axins.set_xlim(0.1,3.1)
	axins.set_ylim(e_dde_3,1)
	#axins.legend()
	axins.set_yscale('log')
	for i in range(1,4):
		axins.axvline(i,color='k',linestyle='--',alpha=0.35)

	axs[1].plot(t/tau,d_ww_dt)
	axs[1].plot(t/tau,d_dde_dt,'r--')
	axs[1].set_title(r"$ \frac{d }{dt} \langle \sigma^{+}\sigma^{-} \rangle  $")
	axs[1].set_xlabel(r"$ t / \tau  $")
	axs[1].set_xlim(0,t[-1]/tau+(t[1]-t[0]))

	for i in range(1,int(t[-1]/tau)+1):
		axs[1].axvline(i,color='k',linestyle='--',alpha=0.35)
	axs[1].set_xlim(0,periods_derivative)

	fig.tight_layout()
	
	plt.show()
	
#exp000(Delta=50,gamma=0.1,periods_derivative=6,n_steps=2001,n_modes=100)