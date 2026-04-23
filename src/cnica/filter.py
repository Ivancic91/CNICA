import numpy as np
from numpy.typing import NDArray
from scipy.signal import remez, correlate # type: ignore[import-untyped]
from typing import Any, List, Optional, cast
import warnings


class Filter:
    """
    A generalized FIR filter class designed for optimal (Parks-McClellan) 
    implementation of low-pass filters and first/second-order derivatives.

    The filter parameters are initialized to sensible defaults and are scaled 
    relative to the sampling frequency (fs).
    
    Tuning is performed in an UNSUPERVISED manner: the optimal cutoff frequency 
    (wc) is chosen by minimizing the Mean Squared Residual Autocorrelation (MSRAC) 
    of the *rejected* high-frequency component (x - y_filtered). This ensures 
    the filter leaves only white noise in the rejected band.

    Attributes:
        fs (float): The sampling frequency in Hz.
        m: Controls the number of taps. For order 0/1, num_taps = 2*m. 
            For order 2, num_taps = m (since it's squared, resulting length is 2*m - 1).
        max_lag: The number of positive lags (k=1 to max_lag) to include in the MSRAC metric.
        a (float): A constant multiplier used to estimate the transition bandwidth.
        epsilon (float): A small frequency offset used to prevent division by zero 
                         or setting cutoffs exactly at DC or Nyquist.
        taps (np.ndarray): The calculated filter coefficients (impulse response).
        candidate_wc (np.ndarray): tested cutoff frequencies.
        best_wc (float): optimal cutoff frequency
        trim: If False (default), returns full-length output using reflection padding.
                  If True, removes m samples from each end to eliminate edge artifacts.
                  Output length will be (len(x) - 2 * m) when trim=True.
    """
    candidate_wc: NDArray[np.float64]
    
    def __init__(self,
                 fs: float = 1.0,
                 m: int = 24,
                 max_lag: int = 10,
                 a: float = 0.8,
                 epsilon: float = 1e-6,
                 trim: bool = False):
        self.fs = fs
        self.m = m
        self.max_lag = max_lag
        self.a = a
        self.epsilon = epsilon
        self.taps: Optional[NDArray[np.float64]] = None
        start_freq = 1e-3
        end_freq = 0.5 * self.fs - 0.01
        num_points = 30
        self.candidate_wc = np.logspace(np.log10(start_freq), 
                                        np.log10(end_freq), 
                                        num=num_points)
        self.best_wc: Optional[float] = None
        self.trim = trim
    
    def transform(self,
                  x: NDArray[np.float64], 
                  order: int = 0,
                  wc: Optional[float] = None) -> NDArray[np.floating[Any]]:
        """
        Designs the FIR filter taps and applies the filter to the input signal x.

        The filter type is determined by the 'order' parameter:
        - order 0: Low-pass filter (LPF).
        - order 1: First derivative (Differentiator).
        - order 2: Second derivative (Convolution of two first-order differentiators).

        Args:
            x: The 1D input signal array.
            order: The derivative order (0 for LPF, 1 for d/dt, 2 for d^2/dt^2).
            wc: The nominal center frequency for the transition band, in Hz. If
                no wc is given, then internally tuned best_wc is used.

        Returns:
            The filtered output signal y, with edge effects removed to match 
            the original signal length (self.trim = True) or full length with 
            reflection padding (self.trim = False).
            
        Raises:
            ValueError: If an unsupported 'order' is provided.
        """
        if len(x) == 0:
            raise ValueError("Input signal x cannot be empty.")
    
        if len(x) <= 2 * self.m and self.trim:
            raise ValueError(
                f"Input signal too short ({len(x)}) for edge trimming. "
                f"Need at least {2*self.m + 1} samples or reduce m."
            )

        # --- 1. Filter Length and Transition Width Estimation ---
        
        # Calculate N, the total number of taps.
        if order in [0, 1]:
            num_taps = 2 * self.m
        elif order == 2:
            # The final convolved filter length will be 2*m - 1
            num_taps = self.m
        else:
            raise ValueError("Unsupported filter order. Must be 0, 1, or 2.")
            
        # Get wc
        if wc is None:
            wc = self.tune(x)
            self.best_wc = wc
            
        # Approximates the transition width based on a heuristic (e.g., Kaiser's formula)
        dw = self.a * (2 * np.pi) / num_taps / self.fs
        
        # --- 2. Design Frequency Bands ---
        
        # Define the lower and upper bounds for the transition band, 
        # ensuring they stay within (epsilon, fs - epsilon).
        w1 = max(wc - 0.5 * dw, self.epsilon)
        w2 = min(wc + 0.5 * dw, 0.5 * self.fs - self.epsilon)
        
        # Bands array for remez is normalized to the Nyquist frequency (0.5 * fs).
        bands: List[float] = [0, w1, w2, 0.5 * self.fs]
        desired = [1, 0]
        
        # --- 3. Obtain Filter Taps via Remez ---
        taps: NDArray[np.float64]
        self.fs: float
        num_taps: int
        if order == 0:
            taps = np.asarray(
                cast(NDArray[np.float64],
                remez(num_taps, bands, desired, fs=self.fs)),
                dtype=np.float64
            )
        elif order == 1:
            taps = np.asarray(
                cast(NDArray[np.float64], 
                remez(num_taps, bands, desired, 
                fs=self.fs, type="differentiator")),
                dtype=np.float64
            )
            taps *= (2 * np.pi) * self.fs  # Scale for correct amplitude
        elif order == 2:
            # Convolve two first-order differentiators.
            first_deriv_taps = np.asarray(
                cast(NDArray[np.float64], 
                remez(self.m, bands, desired,
                fs=self.fs, type="differentiator")),
                dtype=np.float64
            )            
            first_deriv_taps *= (2 * np.pi) * self.fs  # Scale for correct amplitude
            taps = np.asarray(
                cast(NDArray[np.float64], 
                np.convolve(first_deriv_taps, first_deriv_taps, mode="full")),
                dtype=np.float64
            )
        else:
            raise ValueError("Unsupported filter order. Must be 0, 1, or 2.")
            
        # Store taps for inspection
        self.taps = taps
        
        # --- 4. Signal Processing and Convolution ---
        
        # Pad signal with reflection for safe edge handling
        filter_len = len(taps)
        pad_len = (filter_len - 1) // 2
        x_pad = np.pad(x, pad_len, mode='reflect')
        
        # Perform convolution
        y = np.convolve(taps, x_pad, mode='same')

        # --- 5. Get central array ---
        l = pad_len
        if self.trim:
            l += self.m

        # Return the central portion of the result, matching the original 
        # input signal length and removing the padded regions.
        return y[l:-l]


    def _msrac(self, residuals: NDArray[np.float64], max_lag: int) -> float:
        """
        Computes the Mean Squared Residual Autocorrelation (MSRAC) metric 'f'.

        f = <c[k]^2>_{k > 0}, where c[k] is the normalized autocorrelation of 
        the residuals at lag k. This metric is minimized when residuals are white noise.

        Args:
            residuals: The error signal (y_target - y_filtered).
            max_lag: The maximum lag k to include in the metric (k = 1 to max_lag).

        Returns:
            The MSRAC metric f.
        """
        if max_lag <= 0:
            raise ValueError("max_lag must be greater than 0")

        if len(residuals) < max_lag + 1:
            raise ValueError("Residual signal too short for given max_lag.")

        # Calculate unnormalized autocorrelation
        acf : NDArray[np.float64] = np.asarray(
            cast(NDArray[np.float64], correlate(residuals, residuals, mode='full')),
            dtype=np.float64
        )

        # Ensure we don't exceed available lags
        zero_lag_index = len(residuals) - 1
        actual_max_lag = min(max_lag, len(acf) - zero_lag_index - 1)
        if actual_max_lag < max_lag:
            warnings.warn(f"max_lag reduced from {max_lag} to {actual_max_lag} due to signal length.")

        positive_lags = acf[zero_lag_index + 1 : zero_lag_index + 1 + actual_max_lag]

        # Compute the metric f = <c[k]^2>_{k > 0}
        f_metric = np.mean(positive_lags ** 2) / (acf[zero_lag_index] ** 2)

        return f_metric # pyright: ignore[reportUnknownVariableType]

    def tune(
        self,
        x: NDArray[np.float64]
    ) -> float:
        """
        Determines the optimal cutoff frequency (wc) by finding the value that 
        minimizes the Mean Squared Residual Autocorrelation (MSRAC) metric 'f'.

        Args:
            x: The input signal.

        Returns:
            The wc value from candidate_wc that minimizes residual correlation.
        """
        min_f_metric = np.inf
        best_wc = self.candidate_wc[0]

        max_lag = self.max_lag
        if len(x) < 2 * max_lag:
            max_lag = len(x) // 2
            warnings.warn(
                f"Input signal length ({len(x)}) must be at least "
                f"2 * max_lag ({2 * self.max_lag}). Adjusting max_lag to {max_lag}."
            )

        for wc in self.candidate_wc:
            filtered_x = self.transform(x=x, wc=wc)
            if self.trim:
                residuals = x[self.m:-self.m] - filtered_x
            else:
                residuals = x - filtered_x
            f_metric = self._msrac(residuals, max_lag)

            if f_metric < min_f_metric:
                min_f_metric = f_metric
                best_wc = wc

        return best_wc