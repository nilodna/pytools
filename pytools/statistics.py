"""
	. Nonte: some functions (principal_ang, Tdecorr, Neff) were borrowed from https://github.com/apaloczy/ap_tools
 
 
"""
import numpy as np



def autocorr(x, biased=True):
	"""
	USAGE
	-----
	Rxx = autocorr(x, biased=True)
	Computes the biased autocorrelation function Rxx for sequence x,
	if biased==True (default). "biased" means that the k-th value of Rxx is
	always normalized by the total number of data points N, instead of the number
	of data points actually available to be summed at lag k, i.e., (N-k). The
	biased autocorrelation function will therefore always converge to 0
	as the lag approaches N*dt, where dt is the sampling interval.
	If biased==False, compute the unbiased autocorrelation function
	(i.e., normalize by (N-k)).
	References
	----------
	e.g., Thomson and Emery (2014),
	Data analysis methods in physical oceanography,
	p. 429, equation 5.4.
	Gille lecture notes on data analysis, available
	at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf
	"""
	x = np.asanyarray(x)

	N = x.size # Sample size.
	Cxx = np.zeros(N)

	# Calculate the mean of the sequence to write in the more intuitive
	# summation notation in the for loop below (less efficient).
	xb = x.mean()

	## Summing for lags 0 through N (the size of the sequence).
	for k in range(N):
		Cxx_aux = 0.
		for i in range(N-k):
			Cxx_aux = Cxx_aux + (x[i] - xb)*(x[i+k] - xb)

		# If biased==True, Calculate BIASED autocovariance function,
		# i.e., the value of Cxx at k-th lag is normalized by the
		# TOTAL amount of data points used (N) at all lags. This weights
		# down the contribution of the less reliable points at greater lags.
		#
		# Otherwise, the value of Cxx at the k-th lag is normalized
		# by (N-k), i.e., an UNBIASED autocovariance function.
		if biased:
			norm_fac = N
		else:
			norm_fac = N - abs(k)
		Cxx[k] = Cxx_aux/norm_fac

	# Normalize the (biased or unbiased) autocovariance
	# function Cxx by the variance of the sequence to compute
	# the (biased or unbiased) autocorrelation function Rxx.
	Rxx = Cxx/np.var(x)

	return Rxx
    
#########################################################################################################
    
def principal_ang(x, y):
	"""
	USAGE
	-----
	ang = principal_ang(x, y)
	Calculates the angle that the principal axes of two random variables
	'x' and 'y' make with the original x-axis. For example, if 'x' and 'y'
	are two orthogonal components of a vector quantity, the returned angle
	is the direction of maximum variance.
	References
	----------
	TODO
	"""
	x, y = map(np.asanyarray, (x,y))
	assert x.size==y.size

	N = x.size
	x = x - x.mean()
	y = y - y.mean()
	covxy = np.sum(x*y)/N
	varx = np.sum(x**2)/N
	vary = np.sum(y**2)/N
	th_princ = 0.5*np.arctan(2*covxy/(varx-vary))

	return th_princ

#########################################################################################################

def Tdecorr(Rxx, M=None, dtau=1., verbose=False):
    """
    USAGE
    -----
    Td = Tdecorr(Rxx)
    Computes the integral scale Td (AKA decorrelation scale, independence scale)
    for a data sequence with autocorrelation function Rxx. 'M' is the number of
    lags to incorporate in the summation (defaults to all lags) and 'dtau' is the
    lag time step (defaults to 1).
    The formal definition of the integral scale is the total area under the
    autocorrelation curve Rxx(tau):
    /+inf
    Td = 2 * |     Rxx(tau) dtau
    /0
    In practice, however, Td may become unrealistic if all of Rxx is summed
    (e.g., often goes to zero for data dominated by periodic signals); a
    different approach is to instead change M in the summation and use the
    maximum value of the integral Td(t):
    /t
    Td(t) = 2 * |     Rxx(tau) dtau
    /0
    References
    ----------
    e.g., Thomson and Emery (2014),
    Data analysis methods in physical oceanography,
    p. 274, equation 3.137a.
    Gille lecture notes on data analysis, available
    at http://www-pord.ucsd.edu/~sgille/mae127/lecture10.pdf
    """
    Rxx = np.asanyarray(Rxx)
    C0 = Rxx[0]
    N = Rxx.size # Sequence size.

    # Number of lags 'M' to incorporate in the summation.
    # Sum over all of the sequence if M is not chosen.
    if not M:
        M = N

    # Integrate the autocorrelation function.
    Td = np.zeros(M)
    for m in range(M):
        Tdaux = 0.
        for k in range(m-1):
            Rm = (Rxx[k] + Rxx[k+1])/2. # Midpoint value of the autocorrelation function.
            Tdaux = Tdaux + Rm*dtau # Riemann-summing Rxx.

        Td[m] = Tdaux

    # Normalize the integral function by the autocorrelation at zero lag
    # and double it to include the contribution of the side with
    # negative lags (C is symmetric about zero).
    Td = (2./C0)*Td

    if verbose:
        print("")
        print("Theoretical integral scale --> 2 * int 0...+inf [Rxx(tau)] dtau: %.2f."%Td[-1])
        print("")
        print("Maximum value of the cumulative sum: %.2f."%Td.max())

    return Td

#########################################################################################################

def Neff(Tdecorr, N, dt=1.):
	"""
	USAGE
	-----
	neff = Neff(Tdecorr, N, dt=1.)
	Computes the number of effective degrees of freedom 'neff' in a
	sequence with integral scale 'Tdecorr' and 'N' data points
	separated by a sampling interval 'dt'.
	Neff = (N*dt)/Tdecorr = (Sequence length)/(Integral scale)
	References
	----------
	e.g., Thomson and Emery (2014),
	Data analysis methods in physical oceanography,
	p. 274, equation 3.138.
	"""
	neff = (N*dt)/Tdecorr # Effective degrees of freedom.

	print("")
	print("Neff = %.2f"%neff)

	return neff

#########################################################################################################

