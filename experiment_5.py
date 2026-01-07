import numpy as np 
from numpy.fft import fft, fftfreq,fftshift
from aux_funs import paralelizar,run_ww_simulation
from scipy.interpolate import interp1d
from scipy.optimize import minimize


def fast_ft(x,y):
	''' regular spacing  is assumed''' 
	N =len(y)
	dt = x[1]-x[0]
	u = fftshift(fft(y-np.mean(y))) 
	w = fftshift(fftfreq(N,dt)) 
	return 2*np.pi*w,u

def L2_error(x,y,y_ref): 
	dif = np.abs(y_ref-y)
	
	return  np.abs(np.trapezoid(y=np.sqrt(dif**2),x=x)/ np.trapezoid(y=np.abs(y),x=x))


def estimate_frequency_fft(NT_max:int=30,
		   n_sample: int = 51,
		   gamma:float=0.1,
		   Delta:float=10,
		   L:float=1,
		   c:float=1,
		   n_modes:int=40,
		   n_steps:int=1001,):
	''' Determine an approximated frequency of the dynamics using fast fourier transform techniques + interpolation.
	We estimate the dynamics as [cos(w_fft*t)]**2
	with an initial estimation of the freq es np.sqrt(gamma/tau), we simulate a few periods 
	and change slightly the final time, to explore frequencies around the maximum value. We then keep   '''

	T = 2*np.pi*np.sqrt(2*L/(gamma*c))
	t_max = NT_max*T 

	t_max_list = np.linspace(t_max-T/4,t_max+T/4,n_sample)

	def fft_sample(t_sample):
		t,e= run_ww_simulation(t_max=t_sample,gamma=gamma,Delta=Delta,L=L,c=c,n_modes=n_modes,n_steps=n_steps)
		w,u= fast_ft(t,e) 
		u_m = np.max(np.abs(u))  
		w_m = w[np.argmax(u)]
		return np.abs(w_m),np.abs(u_m) # make the freqs positive. could also be done taking np.real(e) 
	
	data = np.asarray(paralelizar(t_max_list,fft_sample)).T
	x= data[0]
	y = data[1]
	x_interp = np.linspace(np.min(x),np.max(x),1000)
	f = interp1d(x=x,y=y,kind='cubic')
	y_interp = f(x_interp)


	w_fft = x_interp[np.argmax(y_interp)]

	return 0.5*w_fft


def estimate_frequency_L2(gamma:float=0.1,
						Delta:float = 12,
						L:float=1,
						c:float=1,
						n_modes:int=40,
						n_steps:int=1001):
	''' Also estimating the frequency, but now minimizing the L2 error with a scipy tool.  '''

	T = 2*np.pi*np.sqrt(2*L/(gamma*c))
	t_max = 30*T
	
	t,e = run_ww_simulation(t_max=t_max,gamma=gamma,Delta=Delta,n_modes=n_modes,n_steps=n_steps,L=L,c=c)
	
	def sample_function(freq):
		y = np.cos(freq*t)**2
		return L2_error(t,y,e)
	
	w = minimize(fun=sample_function,x0=np.sqrt(gamma/(2*L/c)),method='TNC')
	print('estimated frequency: '+str(np.round(w.x,5)))
	return w.x

# ----------------------------------------------------------------------------------------------------


def exp005( gamma_list: list = list(np.linspace(0.01,0.1,20)),
			Delta_list: list = list(range(1,20)),
			L: float =1,
			c:float=1,
			n_modes: int = 50,
			n_steps:int = 1001,
			estimation_method: str = 'L2'): 
	
	''' The idea is to show the behavior of the frequency as we increase some parameter. '''
	cD = len(Delta_list)==1
	cG = len(gamma_list)==1
	assert  cD or cG,"at least one parameter must be fixed --> one list must have a single element"
	
	if cD:
		'Delta is fixed '
		Delta = Delta_list[0]
		xlab=r"$\gamma $ "
		
	if cG: 
		'gamma is fixed'
		gamma=gamma_list[0]
		xlab=r"$\Delta$"

	# TO BE FINISHED !!!
	
	return 0 
	
		