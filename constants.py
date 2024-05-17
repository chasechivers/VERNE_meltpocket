from dataclasses import dataclass

@dataclass
class constants:
	styr  = 3.154e7  # s/yr, seconds in a year

	g      = 1.32  # m/s2, Europa surface gravity

	# Thermal properties
	rho_i  = 917.     # kg/m^3, pure ice density
	rho_w  = 1000.    # kg/m^3 pure water density
	cp_i   = 2.11e3   # J/kg/K, pure ice specific heat
	cp_w   = 4.19e3   # J/kg/K, pure water specific heat
	ki     = 2.3      # W/m/K, pure ice thermal conductivity
	kw     = 0.56     # W/m/K, pure water thermal conductivity
	ac     = 567.     # W/m, ice thermal conductivity constant, ki = ac/T (Klinger, 1980)
	Tm     = 273.15   # K, pure ice melting temperature at 1 atm
	Lf     = 333.6e3  # J/kg, latent heat of fusion of ice
	expans = 1.6e-4   # 1/K, thermal expansivity of ice

	# Bulk properties of the vehicle
	# assuming mass = 480 kg
	#  cylinder of dimensions Height, Diameter = 4.5 m, 0.34 m
	#  volume = 0.406 m^3
	# These don't necessarily matter if we assume a constant temperature
	rho_v = 480 / 0.406  # kg/m^3, assumed vehicle density
	kv    = 15.0  # W/m/K, bulk thermal conductivity of vehicle, assumed titanium
	cp_v  = 544.3  # J/kg/K, bulk specific heat of vehicle, assumed titanium

	C_rho  = 0.       # kg/m^3/ppt, linear density-salinity relationship for water (density = rho_w + C_rho * S)
	Ci_rho = 0.       # kg/m^3/ppt, linear density-salinity relationship for ice (density = rho_i + Ci_rho * S)
	rho_s  = 0.       # kg/m^3, salt density, assigned only when salinity is used

	# Radiation properties
	emiss  = 0.97     # pure ice emissivity
	stfblt = 5.67e-8  # W/m2K4 Stefan-Boltzman constant

	# Constants for viscosity dependent tidal heating
	#   from Mitri & Showman (2005)
	act_nrg = 26.        # activation energy for diffusive regime
	Qs      = 60e3       # J/mol, activation energy of ice (Goldsby & Kohlstadt, 2001)
	Rg      = 8.3144598  # J/K/mol, gas constant
	eps0    = 1e-5       # maximum tidal flexing strain
	omega   = 2.5e-5     # 1/s, tidal flexing frequency
	visc0i  = 1e13       # Pa s, minimum reference ice viscosity at T=Tm
	visc0w  = 1.3e-3     # Pa s, dynamic viscosity of water at 0 K

	# Mechanical properties of ice
	G = 3.52e9    # Pa, shear modulus/rigidity (Moore & Schubert, 2000)
	E = 2.66 * G  # Pa, Young's Modulus