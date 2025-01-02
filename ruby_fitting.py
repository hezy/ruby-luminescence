import numpy as np
from scipy.optimize import curve_fit

# Ruby calibration constants from Mao et al. (1986)
# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/JB091iB05p04673
# Physical constants and equations parameters
# Ruby pressure scale constants from Mao et al. (1986)
A_MAO = 1904.0    # Pressure coefficient
B_MAO = 7.665     # Pressure exponent
λ0 = 693.6887     # Reference wavelength (nm)

# Fitting parameters for ruby R1 and R2 lines
# Parameters based on typical ruby fluorescence spectrum
R1_R2_SEPARATION = 1.58  # nm, separation between R1 and R2 lines
R1_WIDTH = 0.696        # nm, typical FWHM of R1 line
R2_WIDTH = 0.62         # nm, typical FWHM of R2 line

# Fitting bounds
λ_MIN = 650  # nm, minimum wavelength for peaks
λ_MAX = 720  # nm, maximum wavelength for peaks
W_MIN = 0.01 # nm, minimum peak width
W_MAX = 30   # nm, maximum peak width
η_MIN = 0.01 # minimum Gaussian/Lorentzian mixing ratio
η_MAX = 1.0  # maximum Gaussian/Lorentzian mixing ratio


def gaussian(x, w):
    """Normalized Gaussian with FWHM = w.
    
    G(x) = exp(-4ln(2)x²/w²)
    
    Args:
        x: Position relative to peak center
        w: Full width at half maximum (FWHM)
    """
    σ = w / np.sqrt(2 * np.log(2))  # Convert FWHM to standard deviation
    return np.exp(-x**2 / (2*σ**2))


def lorentzian(x, w):
    """Normalized Lorentzian with FWHM = w.
    
    L(x) = 1 / (1 + 4x²/w²)
    
    Args:
        x: Position relative to peak center
        w: Full width at half maximum (FWHM)
    """
    return 1 / (1 + np.square(2*x/w))


def pseudo_voigt(x, x0, w, η, a):
    """Pseudo-Voigt profile - linear combination of Gaussian and Lorentzian.
    
    V(x) = a[η*G(x) + (1-η)*L(x)]
    
    Args:
        x0: Peak center position
        w: Full width at half maximum (FWHM)
        η: Mixing parameter (0 = pure Lorentzian, 1 = pure Gaussian)
        a: Peak amplitude
    """
    x_shifted = x - x0
    return a * (η * gaussian(x_shifted, w) + (1-η) * lorentzian(x_shifted, w))


def background(x, b0, b1, b2):
    """Quadratic background function.
    
    B(x) = b0 + b1*x + b2*x²
    """
    return b0 + b1*x + b2*x**2


def ruby_spectrum(x, x1, w1, η1, a1, x2, w2, η2, a2, b0, b1, b2):
    """Complete ruby spectrum model with two pseudo-Voigt peaks and background.
    
    Total spectrum = B(x) + V1(x) + V2(x)
    
    Args:
        x1, x2: Peak positions for R2 and R1 lines
        w1, w2: Peak widths (FWHM)
        η1, η2: Gaussian/Lorentzian mixing ratios
        a1, a2: Peak amplitudes
        b0, b1, b2: Background polynomial coefficients
    """
    return (background(x, b0, b1, b2) +
            pseudo_voigt(x, x1, w1, η1, a1) +
            pseudo_voigt(x, x2, w2, η2, a2))


def setup_initial_guess(spectrum):
    """Create initial parameters guess from spectrum data.
    
    Strategy:
    1. Find the highest peak (usually R1)
    2. Set R2 position relative to R1
    3. Use typical peak widths and mixing ratios
    """
    max_idx = np.argmax(spectrum.iloc[:, 1])
    λ_max = spectrum.iloc[max_idx, 0]    # R1 position
    I_max = spectrum.iloc[max_idx, 1]    # R1 intensity
    
    return [
        λ_max - R1_R2_SEPARATION,  # x1 (R2 position)
        R2_WIDTH,                  # w1 (R2 width)
        0.5,                       # η1 (R2 mixing)
        0.6 * I_max,              # a1 (R2 amplitude)
        λ_max,                    # x2 (R1 position)
        R1_WIDTH,                 # w2 (R1 width)
        0.5,                      # η2 (R1 mixing)
        I_max,                    # a2 (R1 amplitude)
        0, 0, 0                   # background
    ]


def setup_fit_bounds():
    """Create parameter bounds for curve fitting.
    
    Returns:
        tuple: (lower_bounds, upper_bounds)
    """
    lower_bounds = [
        λ_MIN, W_MIN, η_MIN, 0.01,      # R2 parameters
        λ_MIN, W_MIN, η_MIN, 0.01,      # R1 parameters
        -np.inf, -np.inf, -np.inf       # background
    ]
    
    upper_bounds = [
        λ_MAX, W_MAX, η_MAX, np.inf,    # R2 parameters
        λ_MAX, W_MAX, η_MAX, np.inf,    # R1 parameters
        np.inf, np.inf, np.inf          # background
    ]
    
    return lower_bounds, upper_bounds


def fit_ruby_spectrum(spectrum):
    """Fit the ruby spectrum model to data.
    
    Args:
        spectrum: DataFrame with wavelength and intensity columns
        
    Returns:
        tuple: (optimal_parameters, covariance_matrix)
    """
    x = spectrum.iloc[:, 0]
    y = spectrum.iloc[:, 1]
    
    p0 = setup_initial_guess(spectrum)
    bounds = setup_fit_bounds()
    
    return curve_fit(ruby_spectrum, x, y, p0=p0, bounds=bounds)


def pressure(λ):
    """Calculate pressure from ruby R1 peak wavelength.
    
    Uses the calibration from Mao et al. (1986):
    P = (A/B) * [((λ/λ0))^B - 1]
    
    Args:
        λ (float): Wavelength in nm
        
    Returns:
        float: Pressure in GPa
    """
    return (A_MAO / B_MAO) * (((λ/λ0))**B_MAO - 1)
