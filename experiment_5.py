import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter

from scipy.interpolate import interp1d
from aux_funs import average_fft,DDE_analytical,paralelizar
plt.rcParams['mathtext.fontset'] = 'cm'

def exp005(gamma:float=0.1,
		   tau: float = 1,
		   phi_range: list = list(np.linspace(-4*np.pi,4*np.pi,900)),
		   t_max: float = 64,
		   n_points: int = 12000,
		   Ms: list = list(np.arange(12000,12021,1))
		   ):
	
	t = np.linspace(0,t_max,n_points)

	def a_in(J_0,phi):
		N = int(t[-1]/tau)
		Delta = phi/tau
		J = np.exp(1j*Delta*t)*J_0

		a = np.sqrt(gamma)*J
		J_inter = interp1d(x=t,y=J,bounds_error=False, fill_value=0)
		for n in range(1,N):
			a+= np.sqrt(gamma)*np.exp(1j*n*phi)*J_inter(t-n*tau)*np.heaviside(t-n*tau,0)
		return a
			
	def sample_phi(phi):
		J = DDE_analytical(gamma=gamma,phi=phi,tau=tau,t=t)
		a = a_in(J_0=J,phi=phi)
		w,u = average_fft(x=t,y=a,Ms=Ms)
		I = np.abs(u)**2
		return [w,I]
	
	data = paralelizar(phi_range,sample_phi)
	w = np.real(np.asarray(data)[0,0,:])
	u = np.abs(np.asarray(data)[:,1,:])

	w_ref = np.linspace(-3*np.pi,3*np.pi,1000)
	u_shift = []

	for n,phi in enumerate(phi_range):
		interp = interp1d(x=w-phi/tau,y=u[n,:])
		u_shift.append(interp(w_ref))

	u_map = np.asarray(u_shift)
	w_min = w_ref[0]
	w_max = w_ref[-1]
	
	w_res,I_res = sample_phi(0)

	fig,ax =plt.subplots(figsize=(8, 4))

	im = ax.imshow(
		u_map.T,
		cmap='Blues',
		origin='lower',
		extent=[w_max/np.pi, w_min/np.pi,
				phi_range[0]/np.pi, phi_range[-1]/np.pi],
		aspect='auto',
		norm=PowerNorm(gamma=0.46),
					)
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label(r"Arbitrary Units ")
	cbar.formatter = FuncFormatter(lambda x, pos: f"{x/1000:.1f}")
	cbar.update_ticks()

	ax.set_ylabel(r"$\nu \tau / \pi$")
	ax.set_xlabel(r"$\phi / \pi$")
	ax.axvline(x=0,linestyle='dashdot',color='gray',alpha=0.5)
	ax.invert_xaxis()

	axins = inset_axes(
	ax,
	width="35%", height="35%",
	loc="upper left",
	bbox_to_anchor=(0.05, 0.08, 0.9, 0.9),
	bbox_transform=ax.transAxes
						)

	axins.plot(w_res/np.pi,I_res/1000)
	axins.set_xlim(-1,1)
	axins.set_xlabel(r"$\nu \tau / \pi $")

	fig.savefig('figure2_2.pdf')
	plt.show()


from qnetwork.tools import set_plot_style
set_plot_style()
exp005(gamma=1,phi_range =list(np.linspace(-3*np.pi,3*np.pi,2500)))