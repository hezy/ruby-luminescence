### A Pluto.jl notebook ###
# v0.19.19

using Plots
using SpecialFunctions

x = range(start=-3, stop=3, step=0.01)

function Gaussian(x, fwhm)
	#g1 = 2.0 * √(log(2)/π)
	#g2 = 4.0 * log(2)
    #return @. (g1 / fwhm) * exp((-g2 * x^2) / (fwhm^2)) 
    σ = fwhm/(2√(2log(2)))
    return @. 1/√(2π)/σ * exp(-x^2/2σ^2)
end

function Lorentzian(x, fwhm)
	#l1 = 2/π
	#l2 = 4.0
    #return @. (l1 / fwhm) / ((1 + l2 * x^2) / fwhm^2 ) 
    γ = fwhm / 2
    return @. (γ/pi) / (x^2 + γ^2)
end

function pseudo_Voigt(x, fwhm, n)
	return n * Lorentzian(x, fwhm) + (1 - n) * Gaussian(x, fwhm)
end

function Voigt(x, fwhm_L, fwhm_G)
    γ = fwhm_L/2
    σ = fwhm_G/(2√(2log(2)))
    z = @. -im * (x + im * γ) / (√2 * σ)
    return @. real(erfcx(z)) / (√(2pi) * σ)
end


y1 = Lorentzian(x, 1)
y2 = Gaussian(x, 1)
y3 = pseudo_Voigt(x, 1, 0.5)
y4 = Voigt(x, 0.45, 0.72)

p = plot(x,[y1 y2 y3 y4], label=["Lorentzian" "Gaussian" "Pseudo Voigt" "Voigt"])
#p = plot(x,[y1 y2 y3], label=["Lorentzian" "Gaussian" "Pseudo Voigt"])
title!("peak functions")
xlabel!(raw"x")
ylabel!(raw"y")

#display(p)

#savefig(plot,"file.png")
