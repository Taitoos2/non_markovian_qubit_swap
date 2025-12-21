import numpy as np
import matplotlib.pyplot as plt
from aux_funs import run_ww_simulation,DDE_analytical
from scipy.optimize import minimize


def parameter_correction(gamma:float,
							Delta:float =1,
							L:float =1,
							c:float = 1, 
							n_modes:int = 100,
							n_steps:int = 401):
	''' given Delta and gamma, returns the renormalized set of parameters that best 
	recover Rabi-like behavior '''
	tau=2*L/c
	fsr = 2*np.pi / tau 
	T = np.pi/(np.sqrt(gamma/tau))

	def min_estimation(Delta):
		_,e = run_ww_simulation(t_max=T,gamma = gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		return np.min(e)
	res = minimize(min_estimation,x0=Delta,bounds=[(Delta-0.5,Delta+0.5)])
	t_op,e_op = run_ww_simulation(t_max=6*T,gamma=gamma,Delta=res.x[0],L=L,c=c,n_modes=n_modes,n_steps=n_steps)
	
	def dde_correction_estimation(gamma):
		e_dde = np.abs(DDE_analytical(gamma=gamma,phi=2*np.pi,tau=tau,t=t_op))**2
		error = np.trapezoid(np.abs(e_op-e_dde),t_op)
		return np.sum(error)
	res2 = minimize(dde_correction_estimation,x0=gamma,bounds=[(0.5*gamma,1.5*gamma)])	

	lamb_shift = res.x[0]-Delta
	gamma_correction = res2.x[0] - gamma 
	return [lamb_shift/fsr,gamma_correction/gamma]


def exp006(Delta:float = 1,
		   gamma_list: list =list(np.linspace(0.01,0.5,30)),
		   n_modes: int = 150,
		   n_steps: int = 2001,
		   L: float = 1,
		   c: float = 1):
	
	lamb_shift=[]
	gamma_shift=[]
	def renormalized_parameters(gamma:float):
		return parameter_correction(gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)

	for gamma in gamma_list:
		data = renormalized_parameters(gamma)
		lamb_shift.append(data[0])
		gamma_shift.append(data[1])
		# print(gamma)

	
	lamb_shift = np.asarray(lamb_shift)
	gamma_shift = np.asarray(gamma_shift)
	gamma_list=np.asarray(gamma_list)

	m1,b1 = np.polyfit(gamma_list,100*lamb_shift,1)
	m2,b2 = np.polyfit(gamma_list,100*gamma_shift,1)

	fig,axs = plt.subplots(1,2,figsize=(12,5))
	fig.suptitle(rf"$\Delta = {Delta:.0f}$")
	
	axs[0].set_title(r'$\Delta_{LS} / $fsr')
	axs[0].plot(gamma_list,100*lamb_shift,'-o',label='data')
	axs[0].plot(gamma_list,m1*gamma_list+b1,label='linear fit')
	axs[0].set_xlabel(r"Free space $\gamma $ ")
	axs[0].set_ylabel(r" % " )
	axs[0].legend()


	axs[1].set_title(r"$\delta \gamma/ \gamma $")
	axs[1].plot(gamma_list,100*gamma_shift,'-o',label='data')
	axs[1].plot(gamma_list,m2*gamma_list+b2,label='linear fit')
	axs[1].set_xlabel(r"Free space $\gamma $ ")
	axs[1].set_ylabel(r" % " )
	axs[1].legend()

	fig.tight_layout()
	plt.show()
	return lamb_shift,gamma_shift


def exp007( Delta_list : list  = list(range(5,46,2)),
		   gamma: float = 0.3,
		   n_modes: int = 150,
		   n_steps: int = 2001,
		   L: float = 1,
		   c: float = 1):
	
	lamb_shift=[]
	gamma_shift=[]
	def renormalized_parameters(Delta:float):
		return parameter_correction(gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)

	for Delta in Delta_list:
		data = renormalized_parameters(Delta)
		lamb_shift.append(data[0])
		gamma_shift.append(data[1])
		# print(gamma)

	
	lamb_shift = np.asarray(lamb_shift)
	gamma_shift = np.asarray(gamma_shift)
	Delta_list=np.asarray(Delta_list)

	x = np.log(Delta_list)
	y1 = np.long(100*lamb_shift)
	y2= np.log(100*gamma_shift)

	n1,logm1 = np.polyfit(x=x,y=y1,deg=1)
	n2,logm2 = np.polyfit(x=x,y=y2,deg=1)

	m1=np.exp(logm1)
	m2=np.exp(logm2)

	fig,axs = plt.subplots(1,2,figsize=(12,5))
	fig.suptitle(rf"$\gamma = {gamma:.3f}$")
	
	axs[0].set_title(r'$\Delta_{LS} / $fsr')
	axs[0].set_yscale('log')
	axs[0].set_xscale('log')
	axs[0].plot(Delta_list,100*lamb_shift,'-o',label='data')
	axs[0].plot(Delta_list,m1*Delta_list**n1,label=rf"$y = {n1:.2f}x + {m1:.0f}$")
	axs[0].set_xlabel(r" n-th resonant mode ")
	axs[0].set_ylabel(r" % " )
	axs[0].legend()


	axs[1].set_title(r"$\delta \gamma/ \gamma $")
	axs[1].plot(Delta_list,100*gamma_shift,'-o',label='data')
	axs[1].set_yscale('log')
	axs[1].set_xscale('log')
	axs[1].plot(Delta_list,m2*Delta_list**n2,label=rf"$y = {n2:.2f}x + {m2:.0f}$")
	axs[1].set_xlabel(r" n-th resonant mode ")
	axs[1].set_ylabel(r" % " )
	axs[1].legend()

	fig.tight_layout()
	plt.show()
	return lamb_shift,gamma_shift