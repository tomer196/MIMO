from torch import arange, zeros, cat, stack, max, zeros_like, complex64
from torch.fft import ifft, fft

def A(c, M, N):
    v = N * ifft(c)
    return v[:M]

def AT(y, M, N):
    return fft(cat((y, zeros(N-M).to(y.device)),0))

def soft(x, T):
    """
    % y = soft(x, T)
    %
    % SOFT THRESHOLDING
    % for real or complex data.
    %
    % INPUT
    %   x : data (scalar or multidimensional array)
    %   T : threshold (scalar or multidimensional array)
    %
    % OUTPUT
    %   y : output of soft thresholding
    %
    % If x and T are both multidimensional, then they must be of the same size.
    """
    return max(1 - T/x.abs(), zeros_like(x).real) * x

def bp_missing(y, F, Ft, p, s, mu, Nit, lam=1):
    """
    % c = bp_missing(y, F, Ft, p, s, mu, Nit)
    %
    % MISSING data estimation using BP
    % Minimize || c ||_1 such that y(s) = (F c)(s)
    % where F Ft = p I
    % The index vector s indicates known data.
    %
    % INPUT
    %   y     : data
    %   F, Ft : function handles for F and its conj transpose
    %   p     : Parseval constant
    %   s     : index vector for known data
    %   mu    : ADMM parameter
    %   Nit   : Number of iterations
    %
    % OUTPUT
    %   c     : coefficients of estimated data
    %
    % [c, cost] = bp_missing(...) returns cost function history
    %
    % bp_missing(..., lambda) minimizes || lambda .* c ||_1

    % Ivan Selesnick
    % NYU-Poly
    % selesi@poly.edu

    % The algorithm is a variant of SALSA (Afonso, Bioucas-Dias, Figueiredo,
    % IEEE Trans Image Proc, 2010, p. 2345)
    """

    # Initialization
    c = Ft(y)/p
    d = zeros(c.shape).to(c.device)

    for i in range(Nit):
        u = soft(c + d, 0.5*lam/mu) - d
        d = (1/p) * Ft(y - s*F(u))
        c = d + u

    return c

def salsa_reconstruction(smat_all, rx_binary):
    # Define oversampled DFT
    M = smat_all.shape[1]  # M: length of signal
    Nfft = 2**9  # N: length of Fourier coefficient vector
    mu = 15  # mu: augmented Lagrangian parameter
    Nit = 100  # Nit: number of iteration

    Afun = lambda x: A(x, M, Nfft)  # Afun: oversampled DFT
    ATfun = lambda x: AT(x, M, Nfft)

    results = []
    for i in range(smat_all.shape[0]):
        smat = smat_all[i]
        smat_max = 2 * smat.abs().max()
        rangeAzMap_salsa = zeros(Nfft, smat.shape[1], dtype=complex64)
        smat_salsa = zeros_like(smat)
        for j in range(smat.shape[1]):
            sig = smat[:, j]
            y = sig / smat_max

            rangeAzMap_salsa[:, j] = bp_missing(y, Afun, ATfun, Nfft, rx_binary.to(y.device), mu, Nit)

            smat_salsa[:, j] = Afun(rangeAzMap_salsa[:, j])  # estimated signal
            # print(f'i-{i}, j-{j}')
        results.append(smat_salsa)

    return stack(results, 0)