import numpy as np
from qnetwork.multiphoton_ww import EmittersInWaveguideMultiphotonWW,Waveguide
from aux_funs import two_qubits_analytical
import matplotlib.pyplot as plt 

def exp003(gamma: float=0.1,
		   phi:float = 2*np.pi,
		   L: float = 2, 
		   c:float = 1,
		   t_max:float=40,
		   n_steps:int = 201,
		   plot_bool: bool= True,
		   ):
	''' I am going to assume a cavity (Cable Waveguide). '''


	positions = [0,L]
	Delta = phi/(np.pi) 
	tau=L/c

	setup_ww=EmittersInWaveguideMultiphotonWW(positions=positions,gamma=gamma,Delta=Delta,n_excitations=list(range(2)),L=L,c=c,setup=Waveguide.Cable)
	t_ww,pop_ww=setup_ww.evolve(T=t_max,n_steps=n_steps)
	pop_ww = np.asarray(pop_ww)
	pop_ww= [pop_ww[:,0],pop_ww[:,1]]

	t_dde,pop_dde = two_qubits_analytical(gamma=gamma,phi=phi,tau=tau,t_max=t_max)
	if plot_bool:
		fig,ax = plt.subplots(figsize=(8,6))
		ax.plot(t_dde,pop_dde[0],label='DDE qubit 1')
		ax.plot(t_dde,pop_dde[1],label='DDE qubit 2')
		ax.plot(t_ww,pop_ww[0],'v',markevery=20,label='WW qubit 1')
		ax.plot(t_ww,pop_ww[1],'v',markevery=20,label='WW qubit 2')
		ax.set_xlabel(r"$\gamma t$")
		ax.set_title(r"$ \langle \sigma^{+} \sigma^{-}\rangle $")
		ax.legend()
		
		fig.tight_layout()
		plt.show()
	return [t_dde,pop_dde],[t_ww,pop_ww]