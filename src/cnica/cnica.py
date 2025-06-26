from __future__ import annotations
from typing import Tuple, Literal
from numpy.typing import NDArray

import numpy as np
import warnings
from scipy.optimize import minimize, nnls
from sklearn.decomposition import NMF
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq

class CNICA:
    """ Coupled Nonnegative Independent Component Analysis (CNICA)

    Finds two non-negative matrices, i.e., matrices with all non-negative 
    elements, (C, S), whose product approximates the non-negative matrix D.
    Moreover, the components of S are linearly transformed so that they and
    their first and second derivatives are maximally statistically 
    independent to second order. This linear transformation is performed so
    that C and S remain non-negative and their product remains constant. 

    This proceedure amounts to a two step minimization. First, 

    .. math::
    L(C, S) = |D - C^T S|^2

    is minimized. After this minimization has been performed,

    .. math::
    L(A) = t_0 MI(A S) + t_1 MI(A dS / dw) + t_2 MI(A d^2 / dw^2)
    :label: MI

    is minimized, where :math:`t_i` is the ith trade off parameter for S. 
    The function :math:`MI` is the mutual information of these functions.
    The final output is :math:`A S -> S` and :math:`A^{-T} C -> C`.

    To deal with noise, two methods are employed. First, derivatives are
    smooth with a low-pass filter. Second, the minimization of :math:`L(A)`
    is initially performed with data near 0 clipped for S and C. Data less
    that :math:`gamma S` and :math:`gamma C` are clipped and set to 
    :math:`gamma S` and :math:`gamma C`, respectively. When function is
    clipped, the NMF and MI are both minimized again with 
    :math:`gamma = 0`.

    Attributes
    ----------
    S_ : ndarray
        Right factorization matrix (spectral matrix).
    C_ : ndarray
        Left factorization matrix (concentration matrix).
    n_components_ : int
        The number of components
    reconstruction_err_ : float
        Mean squared error between :math:`D` and the model :math:`C^T S`.
    mutual_information_ : float
        Mutual information of the fitted functions :eq:`MI`.    

    """

    def __init__(self, 
               n_components: int =2, 
               nmf_solver: Literal['cd', 'mu'] ='cd',
               nmf_tol: float =1e-5,
               nmf_max_iter: int =100000,
               mi_C_trade_offs: Tuple[float,float,float] = (0.0, 0.0, 0.0),
               mi_S_trade_offs: Tuple[float,float,float] = (1.0, 1.0, 0.0),
               mi_tol: float =1e-8,
               mi_max_iter: int = 1000,
               tc: float | None = None,
               wc: float | None = None,
               gamma: float =0.01,
               refit: bool =True,
               scaled_poissonian_noise: bool =True) -> None:
        """
        Constructs CNICA object.

        Parameters
        ----------
        n_components : int, default=2
            Number of components
        nmf_solver : {'cd', 'mu'}, default='cd'
            Numerical solver to use for nmf step. 'cd' is Coordinate Descent
            and 'mu' is Multiplicative Update.
        nmf_tol : float, default=1e-4
            Tolerance of the stopping condition for nmf.
        nmf_max_iter : int, default=100000
            Maximum number of iterations before timing out.
        mi_C_trade_offs : tuple, default=(0.0, 0.0, 0.0)
            These values are the trade off parameters for mutual information of
            C, its first derivative, and its second derivative.
        mi_S_trade_offs : tuple, default=(1.0, 1.0, 0.0)
            These values are the trade off parameters for mutual information of
            S, its first derivative, and its second derivative in :eq:`MI`. 
        mi_tol : float, default=1e-8
            Tolerance of the stopping condition for mi.            
        mi_max_iter : int, default=1000
            Maximum number of iterations for mutual information minimization.
        tc : float, default=5.0
            Critical time for C low pass filter in number of time points. 
            Inverse of the critical frequency.
        wc : float, default=5.0
            Critical wavenumber for S low pass filter in number of wavenumbers. 
            Inverse of the critical frequency.
        refit : bool
            Determines whether to refit clipped function after mutual 
            information is minimized.
        gamma : float, default=0.01
            Clipping parameter. Minimum value for the minimum of the positive 
            function divided by the maximum. This is a "fudge" factor meant to 
            deal with issues with noise near 0.
        """
        self.n_components_ = n_components
        self._nmf_solver = nmf_solver
        self._nmf_tol = nmf_tol
        self._nmf_max_iter = nmf_max_iter
        self._mi_S_trade_offs = mi_S_trade_offs
        self._mi_C_trade_offs = mi_C_trade_offs
        self._mi_tol = mi_tol
        self._mi_max_iter = mi_max_iter
        self._tc = tc
        self._wc = wc
        self._gamma = gamma
        self._refit = refit
        self._scaled_poissonian_noise = scaled_poissonian_noise


    def fit(self, D: NDArray[np.float64], y: None = None, init: None = None):
        """ Fits data matrix using coupled non-negative ICA
     
        Uses coupled non-negative independent component analysis to fit D. 

        Parameters
        ----------
        D : ndarray
            Data matrix with dimensions D.shape = (T, W), where T is the number
            of time points and W is the number of wave numbers.
        y : Ignored
            Not used, present for scikit-learn API consistency.
        init : tuple
            Tuple of initial guess for left and right factorization matrices,
            (C, S). Matrices must be numpy arrays with C.shape = (L, T)
            and S.shape = (L, W). If None, NMF is used to obtain initial guess.
        """
        # Initial positive guess
        if init is None:
            if np.any(D < 0.0):
                D = D.copy()
                D[D < 0] = 0.0
            nmf_res0 = NMF(n_components=self.n_components_, 
                           init=None, 
                           solver=self._nmf_solver,
                           beta_loss='frobenius',
                           tol=self._nmf_tol, 
                           max_iter=self._nmf_max_iter)
            C0 = nmf_res0.fit_transform(D).T
            S0 = nmf_res0.components_
            self._nmf_res0 = nmf_res0
            if self._scaled_poissonian_noise:
                wnmf = WNMF(tol=self._nmf_tol,
                            max_iter=self._nmf_max_iter)
                C0, S0 = wnmf.fit_transform(D, C0, S0)
                self._wnmf_res0 = wnmf
        else:
            C0, S0 = init
        C0, S0 = self.normalize(C0, S0)
        self.C0_, self.S0_ = C0, S0

        # Minimize mutual information of initial guess
        mi_C0, mi_S0, mi_res0 = self._minimize_mutual_information(
            C0, 
            S0, 
            self._mi_C_trade_offs,
            self._mi_S_trade_offs,
            self._mi_tol, 
            self._mi_max_iter,
            self._tc,
            self._wc, 
            self._gamma)
        self._mi_res0 = mi_res0
        self._mi_C0_, self._mi_S0_ = mi_C0, mi_S0

        if self._refit:
            # Performs non-negative matrix factorization to minimize errors from 
            # zeroing components during first MI minimization.
            if self._scaled_poissonian_noise:
                wnmf = WNMF(tol=self._nmf_tol,
                            max_iter=self._nmf_max_iter)
                nmf_C, nmf_S = wnmf.fit_transform(D, mi_C0, mi_S0)                 
                self._wnmf_res = wnmf
            else:
                nmf_res = NMF(n_components=self.n_components_, 
                              init='custom', 
                              beta_loss='frobenius',
                              tol=self._nmf_tol, 
                              max_iter=self._nmf_max_iter)
                nmf_C = nmf_res.fit_transform(D, 
                                              W=np.ascontiguousarray(mi_C0.T), 
                                              H=mi_S0).T
                nmf_S = nmf_res.components_
                self._nmf_res = nmf_res
            nmf_C, nmf_S = self.normalize(nmf_C, nmf_S)
            self._nmf_C_, self._nmf_S_ = nmf_C, nmf_S

            # Minimize mutual information of updated guess with no 
            C, S, mi_res = self._minimize_mutual_information(
                nmf_C, 
                nmf_S, 
                self._mi_C_trade_offs,
                self._mi_S_trade_offs,
                self._mi_tol,
                self._mi_max_iter,
                self._tc,
                self._wc,       
                0.0)
            self._mi_res = mi_res
        else:
            C, S = mi_C0, mi_S0

        # Orders C and S by curvature of S for convenience
        C, S = self.order(C, S)

        # Stores C and S as attributes
        self.C_ = C
        self.S_ = S

        # Gets reconstruction error
        self.reconstruction_err_ = self.get_reconstruction_error(D, C, S)

        # Gets mutual information
        self.mutual_information_ = self.get_mutual_information(C, S)

    def fit_transform(self, 
                      D: NDArray[np.float64], 
                      y: None = None, 
                      init: Tuple[NDArray[np.float64],NDArray[np.float64]] | None = None, 
                      refit: bool =True) -> Tuple[NDArray[np.float64],NDArray[np.float64]]:
        """ Fits CNICA model and returns left and right factors

        Parameters
        ----------
        D : ndarray
            Data matrix with dimensions D.shape = (T, W), where T is the number
            of time points and W is the number of wave numbers.
        y : Ignored
            Not used, present for scikit-learn API consistency.
        init : tuple
            Tuple of initial guess for left and right factorization matrices,
            (C, S). Matrices must be numpy arrays with C.shape = (L, T)
            and S.shape = (L, W). If None, NMF is used to obtain initial guess.
        refit : bool
            Determines whether to refit clipped function after mutual 
            information is minimized.

        Returns
        -------
        (C, S) : tuple
            Tuple where first element is left factor with dimensions 
            C.shape = (n_components, T) and second element is right factor 
            S.shape = (n_components, W).

        """
        self.fit(D, y=y, init=init)
        return (self.C_, self.S_)

    def get_mutual_information(self, C: NDArray[np.float64], S: NDArray[np.float64]) -> float:
        """ Function to approximate  mutual information

        This is a function to approximate the mutual information above but 
        includes to 2nd order (Gaussian). See , Eq. 43 of Lee, Girolami, 
        Bell, and Sejnowski, Comp. and Math. with App. (2000).

        Parameters
        ----------
        C : ndarray
            Concentrations matrix with dimensions C.shape = (n_components, T), 
            where n_components is the number of spectra and T is the number of time points.
        S : ndarray
            Spectra matrix with dimensions S.shape = (r, W), where r is the
            number of spectra and W is the number of wavenumbers.
     
        Returns
        -------
        float
            Mutual information approximation
        """
        # Unpacks trade_offs
        t_C, t_gC, t_g2C = self._mi_C_trade_offs
        t_S, t_gS, t_g2S = self._mi_S_trade_offs

        # Obtains covariance of functions and derivatives
        tc, wc = self._tc, self._wc
        if tc is None:
            tc = self._optimal_low_pass_cutoff(C, min_cutoff=2, max_cutoff=20)
        if wc is None:
            wc = self._optimal_low_pass_cutoff(S, min_cutoff=5, max_cutoff=250)
        smooth_C = self._low_pass_filter(C, tc)
        smooth_S = self._low_pass_filter(S, wc)
        gC = self._low_pass_filter(C, tc, deriv=1)
        gS = self._low_pass_filter(S, wc, deriv=1)
        cov_C = np.cov(smooth_C)
        cov_S = np.cov(smooth_S)
        cov_gC = np.cov(gC)
        cov_gS = np.cov(gS)
        cov_g2C = np.cov(self._low_pass_filter(C, tc, deriv=2))
        cov_g2S = np.cov(self._low_pass_filter(S, wc, deriv=2))

        # Compute mutual entropy approximation
        mi_c = 0.5 * t_C * (np.sum(np.log(np.diag(cov_C))) -
            np.log(np.linalg.det(cov_C)))
        mi_gc = 0.5 * t_gC * (np.sum(np.log(np.diag(cov_gC))) -
            np.log(np.linalg.det(cov_gC)))
        mi_g2c = 0.5 * t_g2C * (np.sum(np.log(np.diag(cov_g2C))) -
            np.log(np.linalg.det(cov_g2C)))
        mi_s = 0.5 * t_S * (np.sum(np.log(np.diag(cov_S))) -
            np.log(np.linalg.det(cov_S)))
        mi_gs = 0.5 * t_gS * (np.sum(np.log(np.diag(cov_gS))) -
            np.log(np.linalg.det(cov_gS)))
        mi_g2s = 0.5 * t_g2S * (np.sum(np.log(np.diag(cov_g2S))) -
            np.log(np.linalg.det(cov_g2S)))

        # Sums terms
        mi = mi_c + mi_gc + mi_g2c
        mi += mi_s + mi_gs + mi_g2s

        return mi

    def _minimize_mutual_information(
        self, 
        C: NDArray[np.float64], 
        S: NDArray[np.float64], 
        C_trade_offs,
        S_trade_offs, 
        tol: float,
        max_iter: int,
        tc: float | None, 
        wc: float | None, 
        gamma: float):
        """ Fits data matrix using coupled non-negative ICA
     
        Uses coupled non-negative independent component analysis to fit D. 

        """

        # Initializes left and right factors and smoothed derivatives
        C, S = self.normalize(C, S)
        clipped_C, clipped_S = C.copy(), S.copy()
        if tc is None:
            tc = self._optimal_low_pass_cutoff(C, min_cutoff=2, max_cutoff=20)
        if wc is None:
            wc = self._optimal_low_pass_cutoff(S, min_cutoff=5, max_cutoff=250)
        smooth_C = self._low_pass_filter(C, tc)
        smooth_S = self._low_pass_filter(S, wc)
        grad_C = self._low_pass_filter(C, tc, deriv=1)
        grad_S = self._low_pass_filter(S, wc, deriv=1)
        clipped_C[C < gamma * np.max(C)] = gamma * np.max(C)
        clipped_S[S < gamma * np.max(S)] = gamma * np.max(S)

        # Obtains covariance of gradients
        cov_C = np.cov(C)
        cov_S = np.cov(S)
        cov_grad_C = np.cov(grad_C)
        cov_grad_S = np.cov(grad_S)
        cov_grad2_C = np.cov(self._low_pass_filter(C, tc, deriv=2))
        cov_grad2_S = np.cov(self._low_pass_filter(S, wc, deriv=2))

        # Finds mean of C and S
        mu_C = np.mean(C, axis=1)
        mu_S = np.mean(S, axis=1)

        # Finds M to minimize mutual information given positive functions
        t_C, t_gC, t_g2C = C_trade_offs
        t_S, t_gS, t_g2S = S_trade_offs
        cons = [ 
            {'type': 'eq', 'fun': self._determinant_constraint},
            {'type': 'ineq', 'fun': self._MC_positive_constaint,
                'args': (clipped_C,)},
            {'type': 'ineq', 'fun': self._MS_positive_constaint,
                'args': (clipped_S,)}
        ]  
        M0 = np.eye(self.n_components_).flatten()
        mi_res = minimize(
            self._linearly_transformed_mutual_information,
            M0, 
            args=(cov_C, cov_grad_C, cov_grad2_C,
                  cov_S, cov_grad_S, cov_grad2_S,
                  mu_C, mu_S,
                  t_C, t_gC, t_g2C, 
                  t_S, t_gS, t_g2S),
            constraints=cons,
            method='SLSQP',
            tol=tol,
            options={'maxiter': max_iter})

        # Gets positive C and S
        M = mi_res['x'].reshape((self.n_components_, self.n_components_)) 
        mi_C = np.linalg.pinv(M.T) @ C
        mi_S = M @ S
        mi_C[mi_C < 0] = 0
        mi_S[mi_S < 0] = 0
        mi_C, mi_S = self.normalize(mi_C, mi_S)
        
        return (mi_C, mi_S, mi_res)

    @staticmethod
    def _low_pass_filter(signal: NDArray[np.float64], min_length, order: int = 2, deriv: int = 0):
        """ Butterworth digital and analog filter design

s         """
        signal = np.atleast_2d(signal)
        L, N = signal.shape
        if isinstance(min_length, float) or isinstance(min_length, int):
            min_length = np.ones(L) * min_length


        # Low-pass filter using filtfilt (row-wise)
        smoothed = np.zeros_like(signal)
        for l in range(L):
            b, a = butter(order, 1.0 / min_length[l], btype='low')
            if len(signal[l]) > 9:
                smoothed[l] = filtfilt(b, a, signal[l])
            else:
                smoothed[l] = signal[l]

        if deriv == 0:
            out = smoothed
        elif deriv in (1, 2):
            # Mirror padding
            pad_width = N // 2
            left_pad = smoothed[:, pad_width-1::-1]
            right_pad = smoothed[:, -2:-pad_width-2:-1]
            padded = np.concatenate([left_pad, smoothed, right_pad], axis=1)

            # Frequency axis
            M = padded.shape[1]
            freqs = fftfreq(M, d=1.0)
            omega = 2 * np.pi * freqs

            # FFT and derivative
            fft_butter = fft(padded, axis=1)
            if deriv == 1:
                fft_deriv = 1j * omega * fft_butter
            else:
                fft_deriv = -omega ** 2 * fft_butter
            ifft_out = np.real(ifft(fft_deriv, axis=1))
            out = ifft_out[:, pad_width:pad_width + N]
        else:
            raise ValueError("Only deriv=0, 1, or 2 are supported")
        return out.squeeze()

    def _optimal_low_pass_cutoff(self, data, min_cutoff=5, max_cutoff=250, 
                                 n_test: int =20):

        n_components = data.shape[0]
        W = data.shape[1]

        mses = np.zeros((n_test, n_components))
        cutoff_lengths = np.logspace(np.log10(min_cutoff), np.log10(max_cutoff), 
                                     n_test)
        diff = np.zeros_like(data)
        autocorrelation = np.zeros_like(diff)

        for idx, cutoff_length in enumerate(cutoff_lengths):

            # Computes difference
            diff = data - self._low_pass_filter(data, cutoff_length)

            # Computes autocorrelation functions
            for w in range(W):
                autocorrelation[:, w] = np.sum(diff[:, :W - w] * diff[:, w:], 
                                               axis=1)
            autocorrelation = (autocorrelation.T / np.sum(diff ** 2, axis=1)).T

            # Stores mean squared error from 0
            mses[idx] = np.mean(autocorrelation[:, 1:] ** 2, axis=1)

        # Obtains optimal
        optimal_cutoff = cutoff_lengths[np.argmin(mses, axis=0)]

        return optimal_cutoff

    @staticmethod
    def _linearly_transformed_mutual_information(
        M_flat: NDArray[np.float64], 
        cov_C: NDArray[np.float64], 
        cov_gC: NDArray[np.float64],
        cov_g2C: NDArray[np.float64],
        cov_S: NDArray[np.float64], 
        cov_gS: NDArray[np.float64], 
        cov_g2S: NDArray[np.float64],
        mu_C: NDArray[np.float64],
        mu_S: NDArray[np.float64],
        t_C: float, t_gC: float, t_g2C: float, t_S: float, t_gS: float, t_g2S: float):
        """ Function to approximate  mutual information

        Performs mutual information approximation for C, S, and their 
        derivatives under some linear transformation M that maintains their
        product, C^T @ S. Mutual information is approximated to second order 
        (Gaussian) in each case, using the formula suggested in Eq. 43 of Lee, 
        Girolami, Bell, and Sejnowski, Comp. and Math. with App. (2000). 

        For efficiency, this function uses the identity cov(M @ S) = 
        M @ cov(S) @ M.T. Moreover, it assumes det(M) = 1 in which case 
        det(cov(M @ S)) = det(M @ cov(S) @ M.T) = det(M)^2 cov(S) = cov(S) is
        a constant. Thus, the function is only computing the mutual entropy 
        up to a constant.

        Parameters
        ----------
        M_flat : ndarray
            Matrix that linearly transforms C and S. The shape of M_flat may be
            M_flat.shape = (n_components, n_components) or M_flat.shape = 
            n_component * n_component, where n_component is the
            number of components in decomposition of D. It is assumed that
            det(M) = 1.
        cov_C : ndarray
            Covariance of concentrations matrix; 
            cov_C.shape = (n_components, n_components).
        cov_gC : ndarray
            Covariance of first derivative of concentrations matrix; 
            cov_grad_C.shape = (n_components, n_components).
        cov_g2C : ndarray
            Covariance of second derivative of concentrations matrix; 
            cov_grad2_C.shape = (n_components, n_components).
        cov_S : ndarray
            Covariance of spectra matrix; 
            cov_S.shape = (n_components, n_components).
        cov_gS : ndarray
            Covariance of first derivative of spectra matrix; 
            cov_grad_S.shape = (n_components, n_components).
        cov_g2S : ndarray
            Covariance of second derivative of spectra matrix; 
            cov_grad2_S.shape = (n_components, n_components). 
        t_C : float
            Trade off parameter for concentration mutual information.
        t_gC : float
            Trade off parameter for first derivative of concentration mutual 
            information.
        t_g2c : float
            Trade off parameter for second derivative of concentration mutual 
            information.
        t_S : float
            Trade off parameter for spectral mutual information.
        t_gS : float
            Trade off parameter for first derivative of spectral mutual 
            information.
        t_g2S : float
            Trade off parameter for second derivative of spectral mutual 
            information.
     
        Returns
        -------
        float
            Mutual information approximation
        """
        # Pre-computation
        n_components = cov_S.shape[0]
        M = M_flat.reshape((n_components, n_components))
        inv_MT = np.linalg.pinv(M.T)

        # Compute correlation matricies
        cov_MC = inv_MT @ cov_C @ inv_MT.T
        cov_MgC = inv_MT @ cov_gC @ inv_MT.T
        cov_Mg2C = inv_MT @ cov_g2C @ inv_MT.T
        cov_MS = M @ cov_S @ M.T
        cov_MgS = M @ cov_gS @ M.T
        cov_Mg2S = M @ cov_g2S @ M.T 

        # Compute mutual entropy approximation
        # Assumes det(M) is constant which yields constant  
        # det(cov_MC) and det(cov_MS) terms.  
        mi = t_C * np.sum(np.log(np.diag(cov_MC)))
        mi += t_gC * np.sum(np.log(np.diag(cov_MgC)))
        mi += t_g2C * np.sum(np.log(np.diag(cov_Mg2C)))
        mi += t_S * np.sum(np.log(np.diag(cov_MS)))
        mi += t_gS * np.sum(np.log(np.diag(cov_MgS)))
        mi += t_g2S * np.sum(np.log(np.diag(cov_Mg2S)))

        # Term to ensure stability --> 0 when constraints obeyed
        mi += (1.0 / np.linalg.det(M) - 1.0)**2

        # Regularization term
        #mi += 10 * np.mean((M @ mu_S) / np.sqrt(np.diag(cov_MS))  + np.sqrt(np.diag(cov_MC)))

        return mi


    @staticmethod
    def _determinant_constraint(M_flat: NDArray[np.float64]):
        """ Function to constrain det(M) = 1

        This function is 0 if and only if the determinant of M is 1.

        Parameters
        ----------
        M_flat : ndarray
            Matrix that linearly transforms C and S. The shape of M_flat may be
            M_flat.shape = (r, r) or M_flat.shape = r * r, where r is the
            rank of C and S.

        Returns
        -------
        float
            0 iff det(M) = 1
        """
        r = int(np.sqrt(len(M_flat)))
        M = M_flat.reshape((r, r))
        return np.linalg.det(M) - 1

    @staticmethod
    def _MC_positive_constaint(M_flat, C):
        """ Function to constrain entries of C to be positive

        This function gives an array, where all entries are >= 0 if and only if
        (M^{-T} @ C)_{i,j} >= 0 for all i and j.

        Parameters
        ----------
        M_flat : ndarray
            Matrix that linearly transforms C and S. The shape of M_flat may be
            M_flat.shape = (r, r) or M_flat.shape = r * r, where r is the
            rank of C and S.
        C : ndarray
            Concentrations matrix with dimensions C.shape = (r, T), where r is
            the number of spectra and T is the number of time points.

        Returns
        -------
        ndarray
            values are >= 0 iff entries of (M^{-T} @ C) are >=0.
        """
        r = int(np.sqrt(len(M_flat)))
        M = M_flat.reshape((r, r))
        return (np.linalg.pinv(M.T) @ C).flatten()

    @staticmethod
    def _MS_positive_constaint(M_flat: NDArray[np.float64], S: NDArray[np.float64]) -> np.float64:
        """ Function to constrain entries of S to be positive

        This function gives an array, where all entries are >= 0 if and only if
        (M @ S)_{i,j} >= 0 for all i and j.

        Parameters
        ----------
        M_flat : ndarray
            Matrix that linearly transforms C and S. The shape of M_flat may be
            M_flat.shape = (r, r) or M_flat.shape = r * r, where r is the
            rank of C and S.
        S : ndarray
            Spectra matrix with dimensions S.shape = (r, W), where r is the
            number of spectra and W is the number of wavenumbers.

        Returns
        -------
        float
            values are >= 0 iff entries of (M @ S) are >=0.
        """
        r = int(np.sqrt(len(M_flat)))
        M = M_flat.reshape((r, r))
        return (M @ S).flatten()

    @staticmethod
    def normalize(C: NDArray[np.float64], S: NDArray[np.float64]):
        """ Normalizes C and S so variances of each row are equal

        """
        if C.shape[0] == 1:
            diag = np.cov(C) / np.cov(S)
            lam = diag**0.25
            norm_C = (1.0 / lam) * C
            norm_S = lam * S
        else:
            try:
                diag = np.diag(np.cov(C)) / np.diag(np.cov(S))
            except:
                diag = np.diag(np.cov(C)) / (np.diag(np.cov(S)) + 1e-20)
            lam = np.diag(diag**0.25)
            norm_C = np.linalg.inv(lam.T) @ C
            norm_S = lam @ S
        return (norm_C, norm_S)

    def order(self, C, S):
        """ Orders rows of C and S by maximum curvature in S
        """
        wc = self._wc
        if wc is None:
            wc = self._optimal_low_pass_cutoff(S, min_cutoff=5, max_cutoff=250)
        smooth_S = self._low_pass_filter(S, wc)
        gS = np.gradient(smooth_S, axis=1)
        g2S = np.gradient(gS, axis=1)
        idx = np.argsort(np.mean(g2S ** 2, axis=1))[::-1]
        return (C[idx], S[idx])
       
    def get_reconstruction_error(self, D: NDArray[np.float64], C: NDArray[np.float64], S: NDArray[np.float64]) -> np.float64:
        """ 
        """
        if self._scaled_poissonian_noise:
            if self._refit:
                v = self._wnmf_res.v_
            else:
                v = self._wnmf_res0.v_

            M = C.T @ S
            V = np.maximum(v * M, 1e-20)
            error = np.mean((M - D)**2 / V + np.log(V))

        else:
            M = C.T @ S
            error = np.mean((D - M) ** 2)

        return error


class WNMF:
    """ Weighted Non-negative Matrix Factorization (WNMF)

    Finds two non-negative matrices whose product approximates D. Uses scaled
    Poisonian process to maximize probability of the model.

    """

    def __init__(self,
                 max_iter=10000,
                 tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, D: NDArray[np.float64], C: NDArray[np.float64], S: NDArray[np.float64]):

        # Initializes data
        m, n = D.shape
        D, C, S = D.copy(), C.copy(), S.copy()
        D[D < 0.0] = 0.0 
        C[C<=0] = np.min(C.flatten()[C.flatten() > 0]) 
        S[S<=0] = np.min(S.flatten()[S.flatten() > 0]) 

        # Loops through iterations
        M = C.T @ S 
        for iter_ in range(self.max_iter):
            if iter_ == 0:
                f_old = np.inf
            else:
                f_old = f 

            # Updates arrays
            v = self.update_v(D, M, C, S)
            if iter_ == 0:
                V = np.maximum(v * M, 1e-20)
                f = np.mean((M - D)**2 / V + np.log(V))
                self.initial_f = f
            self.update_C(D, M, C, S, v) 
            self.update_S(D, M, C, S, v)

            # Convergence check
            V[:] = np.maximum(v * M, 1e-20)
            f = np.mean((M - D)**2 / V + np.log(V))

            # Breaks for small changes in objective function
            if abs(f - f_old) < self.tol:
                break

        self.f_ = f
        self.n_iter_ = iter_
        self.v_ = v
        self.C_ = C
        self.S_ = S

    def fit_transform(self, D, C, S):
        self.fit(D, C, S)
        return (self.C_, self.S_)

    @staticmethod
    def update_C(D: NDArray[np.float64], M: NDArray[np.float64], 
                 C: NDArray[np.float64], S: NDArray[np.float64], v: float) -> None:
        """
        Updates the matrix C and M using the multiplicative update rule.

        Parameters
        ----------
        D : ndarray of shape (m, n)
            The observed data matrix.
        M : ndarray of shape (m, n)
            The model matrix
        C : ndarray of shape (m, r)
            The current estimate of the factor matrix C.
        S : ndarray of shape (r, n)
            The current estimate of the factor matrix S.
        v : float
            A positive constant scaling the contribution of C^T S.

        Returns
        -------
        C : ndarray of shape (m, r)
            The updated matrix C.
        """
        # Pre-computation of model and variance
        #M = C.T @ S
        #V = np.maximum(v * M, 1e-20)

        # Computes multiplicative updates
        #neg_C = S @ ((2 * M / V + v / V).T)
        #pos_C = S @ ((2 * D / V + v * ((D - M) / V)**2).T)
        Mp = np.maximum(M, 1e-20)
        neg_C = S @ ((2 + v / Mp).T)
        pos_C = S @ ((2 * D / Mp + ((D - M) / Mp)**2).T)
        C *= pos_C / np.maximum(neg_C, 1e-20)  # max avoids division by 0
        #C[:] = C * pos_C / np.maximum(neg_C, 1e-20)

        # New
        #C *= (S @ (D/M).T) / np.maximum(S @ (1 + v / M.T), 1e-20) 
        M[:] = C.T @ S

        #return C

    @staticmethod
    def update_S(D: NDArray[np.float64], M: NDArray[np.float64], 
                 C: NDArray[np.float64], S: NDArray[np.float64], v: float) -> None:
        """
        Updates the matrix S and M using the multiplicative update rule.

        Parameters
        ----------
        D : ndarray of shape (m, n)
            The observed data matrix.
        M : ndarray of shape (m, n)
            The model matrix
        C : ndarray of shape (m, r)
            The current estimate of the factor matrix C.
        S : ndarray of shape (r, n)
            The current estimate of the factor matrix S.
        v : float
            A positive constant scaling the contribution of C^T S.

        Returns
        -------
        S : ndarray of shape (r, n)
            The updated matrix S.
        """
        # Pre-computation of model and variance
        #M = C.T @ S
        #V = np.maximum(v * M, 1e-20)

        # Computes multiplicative updates original
        #neg_S = C @ (2 * M / V + v / V)
        #pos_S = C @ (2 * D / V + v * ((D - M) / V)**2)
        Mp = np.maximum(M, 1e-20)
        neg_S = C @ (2 + v / Mp)
        pos_S = C @ (2 * D / Mp + ((D - M) / Mp)**2)
        S *= pos_S / np.maximum(neg_S, 1e-20) # max avoids division by 0

        # New
        #S *= (C @ (D/M)) / np.maximum(C @ (1 + v / M), 1e-20)
        M[:] = C.T @ S

        #return S

    @staticmethod
    def update_v(D: NDArray[np.float64], M: NDArray[np.float64], C: NDArray[np.float64], S: NDArray[np.float64]):
        # Pre-computation of model and variance
        #M = C.T @ S
        #V = np.maximum(v * M, 1e-20)

        # Multiplicative updates original
        #pos_A = M * (D - M)**2 / V**2
        #neg_A = M / V
        #v *= np.sum(pos_A) / max(np.sum(neg_A), 1e-20)

        # New
        v = np.sum((D - M)**2 / np.maximum(M, 1e-20)) / D.size

        return v

    @staticmethod
    def estimate_v(D: NDArray[np.float64], M: NDArray[np.float64]):
        # Initial estimate of v by nnls fitting
        var = (D - M)**2
        M_flat = M.flatten()
        var_flat = var.flatten()
        coefs = np.array([M_flat]).T
        v, _ = nnls(coefs, var_flat)

        return v[0]
