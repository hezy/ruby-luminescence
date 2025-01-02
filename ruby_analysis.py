#!/usr/bin/env -S ./venv/bin/python3
"""Ruby pressure calibration script.

Analyzes ruby fluorescence spectra to determine pressure using the
Mao et al. (1986) calibration scale.
"""

import glob
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all the constants and functions from ruby_fitting.py
from ruby_fitting import (
    fit_ruby_spectrum, ruby_spectrum, pressure,
    background, pseudo_voigt
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_spectrum_file(file_path):
    """Process a single ruby spectrum file.
    
    Args:
        file_path: Path to the spectrum file
        
    Returns:
        dict: Results including pressure, wavelength, and fit parameters
    """
    logging.info(f"Processing {file_path}")
    
    try:
        # Read spectrum data
        spectrum = pd.read_csv(
            file_path, 
            sep='\t',
            header=None,
            names=['wavelength', 'intensity']
        )
        
        # Perform the fit
        popt, pcov = fit_ruby_spectrum(spectrum)
        
        # Extract uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate pressure from R1 line position (popt[4] is x2, the R1 position)
        P = pressure(popt[4])
        δP = abs(pressure(popt[4] + perr[4]) - P)  # Pressure uncertainty
        
        # Create figure
        create_figure(spectrum, popt, perr, P, δP, file_path)
        
        return {
            'pressure': P,
            'pressure_error': δP,
            'R1_wavelength': popt[4],
            'R1_wavelength_error': perr[4],
            'fit_parameters': popt,
            'parameter_errors': perr
        }
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        raise

def create_figure(spectrum, popt, perr, pressure_gpa, pressure_err, file_path):
    """Create and save analysis figure."""
    x = spectrum['wavelength']
    y = spectrum['intensity']
    
    # Calculate components
    y_fit = ruby_spectrum(x, *popt)
    y_back = background(x, *popt[8:11])
    y_r2 = y_back + pseudo_voigt(x, *popt[0:4])
    y_r1 = y_back + pseudo_voigt(x, *popt[4:8])
    residuals = y_fit - y
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(12, 8))
    
    # Main plot
    ax1.plot(x, y, '.', label='Data', alpha=0.5)
    ax1.plot(x, y_fit, '-', label='Fit')
    ax1.plot(x, y_back, '--', label='Background')
    ax1.plot(x, y_r2, '-', label='R2 line')
    ax1.plot(x, y_r1, '-', label='R1 line')
    
    # Add peak positions
    ax1.axvline(popt[0], color='gray', linestyle=':')  # R2
    ax1.axvline(popt[4], color='gray', linestyle=':')  # R1
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Intensity (arb. units)')
    ax1.legend()
    
    # Title with results
    title = "Ruby Fluorescence Analysis\n"
    title += f"λ(R1) = {popt[4]:.4f} ± {perr[4]:.4f} nm\n"
    title += f"P = {pressure_gpa:.2f} ± {pressure_err:.2f} GPa"
    ax1.set_title(title)
    
    # Residuals plot
    ax2.plot(x, residuals, '-')
    ax2.axhline(y=0, color='k', linestyle=':')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Residuals')
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = Path(file_path).with_suffix('.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Main function to process all spectrum files."""
    # Find all spectrum files
    spectrum_files = glob.glob('*.txt')
    
    if not spectrum_files:
        logging.warning("No spectrum files found in current directory")
        return
    
    # Process each file
    results = []
    for file_path in spectrum_files:
        try:
            result = process_spectrum_file(file_path)
            results.append({
                'file': file_path,
                **result
            })
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {str(e)}")
            continue
    
    # Create summary
    if results:
        summary = pd.DataFrame(results)
        summary.to_csv('ruby_analysis_summary.csv', index=False)
        logging.info(f"Processed {len(results)} files successfully")
        
if __name__ == '__main__':
    main()
