"""
	. Note: some functions (principal_ang, Tdecorr, Neff) were borrowed from https://github.com/apaloczy/ap_tools
 
 
"""
import numpy as np
import scipy.stats as stats
import math

#########################################################################################################


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


def medianCI(sample, confidence=0.95):
    """Get the confidence interval for the median of a sample

    Based on the work of Campbell and Gardner (1988), the confidence interval is
    calculated by the following equation:

        r = n/2 - N * np.sqrt(n)/2

        s = 1 + n/2 + N * np.sqrt(n)/2

    Parameters
    ----------
    sample : numpy.ndarray, list
        sample array
    confidence : float, optional
        Confidence to compute the interval, by default 0.95

    Returns
    -------
    [type]
        [description]
        
    References
    ----------
    
    """
    # method based on Campbell and Gardner (1988)
    
    # find N = (1-alpha/2)
    alpha = 1 - confidence
    N = stats.norm.ppf(1 - alpha/2)
    
    # sample size (n)
    n = len(sample)
    
    # compute the position of the lower limit value
    r = n/2 - N * (np.sqrt(n)/2)
    r = round(r)
    
    # compute the position of the upper limit value
    s = 1 + n/2 + N * (np.sqrt(n)/2)
    s = round(s)
    
    # construct CI tuple
    CI = (sample[r], sample[s])
    
    return CI, N * (np.sqrt(n)/2)

#########################################################################################################


def ci_for_proportion(df, var1, var2, group, conf=.95, method=None):
    """Function to calculate the confidence interval for a very specific dataframe.
    
    If the method chosen is 'clopper_pearson', then compute the confidence interval by:
        [[equation]]
    Otherwise, compute the confidence interval with a general equation:
        [[equation]]
    
    based on: https://towardsdatascience.com/a-complete-guide-to-confidence-interval-and-examples-in-python-ff417c5cb593
    """       
    # get z-score based on the confidence level given (conf)
    alpha = 1 - conf
    zscore = stats.norm.ppf(1-alpha/2)
    
    # create contingency table (2x2)
    df_crosstab = pd.crosstab(df[var1], df[var2]).T
    
    N = df_crosstab[group].sum()
    s = df_crosstab[group][1] # s of success
    
    percentage = s / N
    standard_error = np.sqrt(percentage * (1 - percentage)/N)
    
    if method == 'clopper_pearson':
        b = stats.beta.ppf
        lo = b(alpha / 2, s, N - s + 1)
        hi = b(1 - alpha / 2, s + 1, N - s)
        
        data = [percentage,                    # percentage
                0.0 if math.isnan(lo) else lo, # lower CI
                1.0 if math.isnan(hi) else hi] # upper CI
    else:
        # apply a broader method
        # if N < 30, then the sample must have a normal distribution. Otherwise, a normal distribution is not necessary
        if N < 30:
            print('Sample size lower than 30, therefore a normal distribution is requested. \nIf your sample doens\'t have a normal distribution, be careful with these results.')

        ci = zscore*standard_error

        lower_ci = percentage - ci
        upper_ci = percentage + ci
        
        data = [percentage,
                lower_ci,
                upper_ci,
                standard_error,
                ci            
        ]

    return data

#########################################################################################################


def non_param_unpaired_CI(sample1, sample2, conf):
    n1 = len(sample1)  
    n2 = len(sample2)  
    alpha = 1-conf      
    N = stats.norm.ppf(1 - alpha/2) 

    # The confidence interval for the difference between the two population
    # medians is derived through the n x m differences.
    diffs = sorted([i-j for i in sample1 for j in sample2])

    # the Kth smallest to the Kth largest of the n x m differences then determine 
    # the confidence interval, where K is:
    k = np.math.ceil(n1*n2/2 - (N * (n1*n2*(n1+n2+1)/12)**0.5))

    CI = (round(diffs[k-1],3), round(diffs[len(diffs)-k],3))
    return CI

#########################################################################################################


def spec_error(E,sn,ci=.95):
    """ 
    Computes confidence interval for one-dimensional spectral estimate E.

    Parameters
    ===========
    - sn is the number of spectral realizations;
            it can be either an scalar or an array of size(E)
    - ci = .95 for 95 % confidence interval

    Output
    ==========
    lower (El) and upper (Eu) bounds on E 
    
    Usage
    ======
    >> data = 
    >> f, ddof, spec = block_avgz(b1s, 1/freq/60, 12)
    >> spec_lower, spec_upper = spec_error(spec, ddof/2, ci=0.95)
    >> plt.plot(spec)
    >> plt.fill_between(f, spec_lower, y2=spec_upper, alpha=.2)
    
    Source
    =======
    https://nbviewer.org/github/apaloczy/InnerShelfReynoldsStresses/blob/master/plot_figs/fig02/fig02.ipynb
    """

    from pyspec.spectrum import yNlu

    dbin = .005
    yN = np.arange(0,2.+dbin,dbin)

    El, Eu = np.empty_like(E), np.empty_like(E)

    yNl,yNu = yNlu(sn,yN=yN,ci=ci)
    El = E/yNl
    Eu = E/yNu

    return El, Eu

############################################################################################################


def block_avgz(A, dts, Nblks):
    """
    compute spectral analysis for each vertical level and then return the depth-averaged spectrum,
    along with a degrees of freedom (dof) and frequency (f).
    
    """
    from pyspec.spectrum import block_avg
    
    nz, nt = A.shape
    for k in range(nz):
        a = A[k, :]
        fg = np.isfinite(a)
        if not fg.any():
            if k==0:
                if not a.size%2:
                    siz = a.size/2/Nblks + 1
                else:
                    siz = a.size/2/Nblks
                S = np.ones(int(siz))*np.nan
            else:
                S = np.vstack((S, Saa.f*np.nan))
            continue
        a[~fg] = np.nanmean(a)
        Saa, nblks = block_avg(a, dts, N=Nblks);
        if k==0:
            S = Saa.spec
        else:
            S = np.vstack((S, Saa.spec[np.newaxis, ...]))
    f = Saa.f

    dof = 2*nblks
    return f, dof, np.nanmean(S, axis=0)

############################################################################################################