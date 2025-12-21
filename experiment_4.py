import numpy as np
from qnetwork.multiphoton_ww import EmittersInWaveguideMultiphotonWW,Waveguide
from aux_funs import two_qubits_analytical
import matplotlib.pyplot as plt 
from experiment_3 import exp003

def exp004(gamma: float=0.1,
		   phi_list: list = [np.pi,10*np.pi],
		   L: float = 2, 
		   c:float = 1,
		   t_max:float=40,
		   n_steps:int = 201):
	
	fig,axs=plt.subplots(1,len(phi_list),figsize=(6*len(phi_list),6))

	for i,phi in enumerate(phi_list):
		Delta = phi/np.pi 
		data_dde,data_ww = exp003(gamma=gamma,phi=phi,L=L,c=c,t_max=t_max,n_steps=n_steps,plot_bool=False)
		t_dde,pop_dde =data_dde
		t_ww,pop_ww = data_ww
		axs[i].plot(t_ww,pop_ww[0],label='ww qubit 1')
		axs[i].plot(t_ww,pop_ww[1],label='ww qubit 2')

		if i == len(phi_list)-1:
			axs[i].plot(t_dde,pop_dde[0],'o',markevery=5,label='dde qubit 1')
			axs[i].plot(t_dde,pop_dde[1],'o',markevery=5,label='dde qubit 2')
			axs[i].legend()

		axs[i].set_xlabel(r"$ \gamma \tau $")
		axs[i].set_title(rf"$\Delta ={Delta :.0f}$ F.S.R ")
	fig.tight_layout()
	plt.show()
	
