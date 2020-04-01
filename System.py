# Author: Chase Chivers
# Last updated: 10/28/19
# Modular build for 2d heat diffusion problem
#   applied to a descent probe in Europa's ice shell

import numpy as np
from scipy import optimize
from HeatSolver import HeatSolver

class IceSystem(HeatSolver):
	"""
	Class with methods to set up initial conditions for two-dimensional, two-phase thermal diffusion model that
	includes temperature-dependent conductivity and salinity. Includes the HeatSolver class used to solve the heat
	equation utilizing an enthalpy method (Huber et al., 2008) to account for latent heat from phase change as well
	as a parameterization for a saline system.
	"""
	def __init__(self, Lx, Lz, dx, dz, kT=True, cpT=False, use_X_symmetry=False, Z0=0):
		"""
		Initialize the system.
		Parameters:
			Lx : float
				length of horizontal spatial domain, m
			Lz : float
				thickness of shell, length of vertical spatial domain, m
			dx : float
				horizontal spatial step size, m
			dz : float
				vertical spatial step size, m
			cpT : bool
			    choose whether to use temperature-depedent specific heat,
			    default = False.
			    True: temperature-dependent, cp_i ~ 185 + 7*T (Hesse et al., 2019)
			kT : bool
			    choose whether to use temperature-dependent thermal conductivity,
			    default = True, temperature-dependent, k=ac/T (Petrenko, Klinger, etc.)
			use_X_symmetry : bool
				assume the system is symmetric about the center of the intrusion
				* NOTE: Must use Reflecting boundary condition for sides if using this
			issalt : bool
				declare whether salinity will be used in this system, necessary for declaring fit functions and
				melting temperature calculations
		Usage:
			Ice Shell is 40 km thick and 40 km wide at a spatial discretization of 50 m.
				model = IceSystem(40e3, 40e3, 50, 50)

			See README
		"""

		self.Lx, self.Lz = Lx, Lz
		self.dx, self.dz = dx, dz
		self.nx, self.nz = int(Lx / dx + 1), int(Lz / dz + 1)
		self.Z = np.linspace(Z0, Z0+Lz, self.nz, dtype=float) # z domain starts at zero, z is positive down
		if use_X_symmetry:
			self.symmetric = True
			self.Lx = self.Lx / 2
			self.nx = int(self.Lx / self.dx + 1)
			self.X = np.array([i * dx for i in range(self.nx)], dtype=float)
		elif use_X_symmetry is False:
			self.X = np.array([-Lx/2 + i*dx for i in range(self.nx)], dtype=float)  # x domain centered on 0
		self.X, self.Z = np.meshgrid(self.X, self.Z)  # create spatial grid
		self.T = np.ones((self.nz, self.nx), dtype=float)  # initialize domain at one temperature
		self.S = np.zeros((self.nz, self.nx), dtype=float)  # initialize domain with no salt
		self.phi = np.zeros((self.nz, self.nx), dtype=float)  # initialize domain as ice
		self.vehicle = np.zeros((self.nz, self.nx), dtype=float)
		self.kT, self.cpT = kT, cpT  # k(T), cp_i(T) I/O
		self.issalt = False  # salt I/O

	class constants:
		"""
		No-methods class used for defining constants in a simulation. May be changed inside here or as an
		instance during simulation runs.
		"""
		styr = 3.154e7  # s/yr, seconds in a year

		g = 1.32  # m/s2, ropa surface gravity

		# Thermal properties
		rho_i = 917.  # kg/m3, pure ice density
		rho_w = 1000.  # kg/m3 pure water density
		# assuming icefin specs: https://auvac.org/configurations/view/270
		# mass = 93.9 kg
		# H x W x L = 3 x 0.23 x 0.23 m
		# assuming its a cylinder
		# rho_v = mass/(pi * (W/2)**2 * H) = 753.3 kg/m3
		# seems pretty low....
		# using aluminum instead
		rho_v = 753.3 #3000.  # kg/m3, assumed vehicle density, aluminum-ish
		cp_i = 2.11e3  # J/kgK, pure ice specific heat
		cp_w = 4.19e3  # J/kgK, pure water specific heat
		cp_v = 910.01  # J/kgK
		ki = 2.3  # W/mK, pure ice thermal conductivity
		kw = 0.56  # W/mK, pure water thermal conductivity
		kv = 205.
		ac = 567  # W/m, ice thermal conductivity constant, ki = ac/T (Klinger, 1980)
		Tm = 273.15  # K, pure ice melting temperature at 1 atm
		Lf = 333.6e3  # J/kg, latent heat of fusion of ice
		expans = 1.6e-4  # 1/K, thermal expansivity of ice

		rho_s = 0.  # kg/m3, salt density, assigned only when salinity is used

		# Radiation properties
		emiss = 0.97  # pure ice emissivity
		stfblt = 5.67e-8  # W/m2K4 Stefan-Boltzman constant

		# Constants for viscosity dependent tidal heating
		#   from Mitri & Showman (2005)
		act_nrg = 26.  # activation energy for diffusive regime
		Qs = 60e3  # J/mol, activation energy of ice (Goldsby & Kohlstadt, 2001)
		Rg = 8.3144598  # J/K*mol, gas constant
		eps0 = 1e-5  # maximum tidal flexing strain
		omega = 2.5e-5  # 1/s, tidal flexing frequency
		visc0i = 1e13  # Pa s, minimum reference ice viscosity at T=Tm
		visc0w = 1.3e-3  # Pa s, dynamic viscosity of water at 0 K

		# Mechanical properties of ice
		G = 3.52e9  # Pa, shear modulus/rigidity (Moore & Schubert, 2000)
		E = 2.66 * G  # Pa, Young's Modulus

	def save_initials(self):
		""" Save initial values to compare with simulation results. """
		self.T_initial = self.T.copy()
		self.Tm_initial = self.Tm.copy()
		self.phi_initial = self.phi.copy()
		self.S_initial = self.S.copy()

		if self.kT:
			self.k_initial = self.phi_initial * self.constants.kw + (1 - self.phi_initial) * \
			                 self.constants.ac / self.T_initial
		else:
			self.k_initial = self.phi_initial * self.constants.kw + (1 - self.phi_initial) * self.constants.ki

	def init_volume_averages(self):
		"""
		Initialize volume averaged values over the domain. In practice, this is automatically called by any future
		function that are changing physical parameters such as liquid fraction, salinity or temperature.
		"""

		if self.kT:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
			self.k[self.vehicle==1] = self.constants.kv
		else:
			self.k = (1 - self.phi) * self.constants.ac / self.T + self.phi * self.constants.kw

		if self.cpT == "GM89":
			"Use temperature-dependent specific heat for pure ice from Grimm & McSween 1989"
			self.cp_i = 185. + 7.037 * self.T
		elif self.cpT == "CG10":
			"Use temperature-dependent specific heat for pure ice from Choukroun & Grasset 2010"
			self.cp_i = 74.11 + 7.56 * self.T
		else:
			self.cp_i = self.constants.cp_i

		# this is very unimportant overall
		if self.issalt:
			self.rhoc = (1 - self.phi) * (self.constants.rho_i + self.Ci_rho * self.S) * self.cp_i \
			            + self.phi * (self.constants.rho_w + self.C_rho * self.S) * self.constants.cp_w
			self.rhoc[self.vehicle == 1] = self.constants.rho_v * self.constants.cp_v

		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w
			self.rhoc[self.vehicle == 1] = self.constants.rho_v * self.constants.cp_v

		self.save_initials()

	def init_T(self, Tsurf, Tbot, profile='non-linear', real_Lz=0):
		"""
		Initialize temperature profile
			Parameters:
				Tsurf : float
					surface temperature
				Tbot : float
					temperature at bottom of domain
				profile : string
					-> defaults to 'non-linear'
					prescribed temperature profile
					'non-linear' -- expected equilibrium thermal gradient with k(T)
					'linear'     -- equilibirium thermal gradient for constant k
					'stefan'     -- sets up the freezing stefan problem temperature profile
									in this instance, Tbot should be the melting temperature
				real_Lz : float
					used if you want to simulate some portion of a much larger shell, so this parameter is used to
					make the temperature profile that of the much larger shell than the one being simulated.
					For example, a 40 km conductive shell (real_Lz = 40e3) discretized at 10 m can be computationally
					expensive. However, if we assume that any temperature anomaly at shallow depths (~1-5km) won't
					reach to 40km within the model time, we can reduce the computational domain to ~5km to speed up
					the simulation. This will take the Tbot as the Tbot of a 40km and find the temperature at 5km to
					account for the reduced domain size.
					Usage case down below
			Returns:
				T : (nz,nx) grid
					grid of temperature values

			Usage :
				Default usage:
					model.init_T(Tsurf=75, Tbot=273.15)

				Linear profile:
					model.init_T(Tsurf = 50, Tbot = 273.15, profile='linear')

				Cheated domain:
					realLz = 50e3
					modelLz = 5e3
					model = IceSystem(Lz=modelLz, ...)
					model.init_T(Tsurf=110,Tbot=273.15,real_lz=realLz)
		"""
		# set melting temperature to default
		self.Tm = self.constants.Tm * self.T.copy()

		if isinstance(profile, str):
			if profile == 'non-linear':
				if real_Lz > 0:
					self.real_Lz = real_Lz
					Tbot = Tsurf * (Tbot / Tsurf) ** (self.Lz / real_Lz)
				self.T = Tsurf * (Tbot / Tsurf) ** (abs(self.Z / self.Lz))

			elif profile == 'linear':
				if real_Lz > 0:
					self.real_Lz = real_Lz
					Tbot = (Tbot - Tsurf) * (self.Lz / real_Lz) + Tsurf
				self.T = (Tbot - Tsurf) * abs(self.Z / self.Lz) + Tsurf

			print('init_T(Tsurf = {}, Tbot = {})'.format(Tsurf, Tbot))
			print('\t Temperature profile initialized to {}'.format(profile))

		else:
			self.T = profile
			print('init_T: custom profile implemented')

		# save boundaries for dirichlet or other
		# left and right boundaries
		self.TtopBC = self.T[0, :]
		self.TbotBC = self.T[-1, :]
		self.Tedge = self.T[:, 0] = self.T[:, -1]
		self.Tsurf = Tsurf
		self.Tbot = Tbot
		self.init_volume_averages()

	def set_vehicle_geom(self, depth, length, radius, num_RTG, geometry, rtg_size=0.6, noseshape=None, **kwargs):
		print('setting geometry')
		#center = self.vh / 2 + de
		r = np.where(abs(self.X[0, :]) <= radius / 2)[0]
		z = np.intersect1d(np.where(self.Z[:, 0] <= length + depth), np.where(self.Z[:, 0] >= depth))
		tmp = np.zeros(self.T.shape)
		tmp[z[0]:z[-1], r[0]:r[-1] + 1] = 1
		if noseshape == "conical":
			nl = kwargs['noselength']
			nose_depth = length + depth
			cone_area = np.intersect1d(np.where(self.Z[:, 0] <= nose_depth + nl),
			                      np.where(self.Z[:, 0] >= nose_depth - self.dz))
			Xcone = self.X[cone_area[0]:cone_area[-1], r[0]:r[-1] + 1]
			Zcone = self.Z[cone_area[0]:cone_area[-1], r[0]:r[-1] + 1]
			cone = np.where((Zcone-self.Z[0,0])/(Zcone.max()-self.Z[0,0]) <= 1 - Xcone*radius)
			self.nosecone = (cone[0] + cone_area[0], cone[1])
			tmp[self.nosecone] = 1


		elif noseshape is None:  # assumed cylindrical
			pass

		self.vehicle_geom = np.where(tmp == 1)

		if num_RTG in [3,4]:
			# current design dictates TWO FRONT RTGS!
			# find where the RTGs are
			# two RTGS in the front
			front_rtg_loc = np.intersect1d(np.where(self.Z[:, 0] <= length + depth),
			                       np.where(self.Z[:, 0] >= length + depth - 2 * rtg_size))
			tmp = np.zeros(self.T.shape)
			tmp[front_rtg_loc.min():front_rtg_loc.max(), r.min():r.max() + 1] = 1
			frontrtgs = np.where(tmp == 1)
			# md.headgeom = np.where(tmp == 1)

			# only 1 rear RTG
			if num_RTG == 3:
				# one rear RTG
				rear_rtg_loc = np.intersect1d(np.where(self.Z[:, 0] <= depth + rtg_size),
				                       np.where(self.Z[:,0] >= depth))
			elif num_RTG == 4:
				# two rear RTGs
				rear_rtg_loc = np.intersect1d(np.where(self.Z[:, 0] <= depth + 2*rtg_size),
				                      np.where(self.Z[:, 0] >= depth))

			tmp = np.zeros(self.T.shape)
			tmp[rear_rtg_loc.min():rear_rtg_loc.max(), r.min():r.max() + 1] = 1
			rearrtgs = np.where(tmp==1)
			self.RTGS = (np.append(frontrtgs[0], rearrtgs[0]),
			             np.append(frontrtgs[1], rearrtgs[1]))
		else:
			pass

		return self.vehicle_geom

	def init_vehicle(self, depth, length, radius, T, num_RTG=0, temp_regulation=['Constant T'],
	                 rtg_size=0.6,power=1e3, noseshape=None, **kwargs):
		# Probably want to be able to make custom temperature profile/geometry
		## e.g. High front RTG temperature = ??, 300 K rest of body
		##      or some step function, polynomial, w/e
		self.vehicle_h = length
		self.vehicle_w = radius
		self.vehicle_d = depth
		self.vehicle_T = T
		# Assume vehicle geometry is a rectangle for now
		if noseshape=="conical":
			if "noselength" not in kwargs:
				raise Exception("ERROR: must chooser length of conical nose; \n\ti.e. model.set_vehicle_geom( "
			                "..., noseshape='conical', noselength=noselength)")
			else:
				print('using conical nose')
				self.vehicle_geom = self.set_vehicle_geom(depth, length, radius, num_RTG,rtg_size,
				                                          noseshape=noseshape, noselength=kwargs['noselength'])
		else:
			self.vehicle_geom = self.set_vehicle_geom(depth, length, radius, num_RTG, geometry, rtg_size, noseshape)
		self.vehicle[self.vehicle_geom] = 1
		self.T[self.vehicle_geom] = T
		self.vehicle_heat_scheme = temp_regulation
		if 'Constant Flux' in temp_regulation or 'RTG Flux' in temp_regulation:
			self.power = power
		self.init_volume_averages()

	# define a bunch of useful functions for salty systems, unused otherwise
	# non-linear fit, for larger dT
	def shallow_fit(self, dT, a, b, c, d):
		return a + b * (dT + c) * (1 - np.exp(-d / dT)) / (1 + dT)

	# linear fit, for small dT
	def linear_fit(self, dT, a, b):
		return a + b * dT

	# FREEZCHEM quadratic fit for liquidus curve
	def Tm_func(self, S, a, b, c):
		return a * S ** 2 + b * S + c

	def init_salinity(self, S=None, composition='MgSO4', concentration=12.3, rejection_cutoff=0.25, shell=False,
	                  in_situ=False, T_match=True):
		"""
		Initialize salinity properties for simulations.
		Parameters:
			S : (nz,nx) grid
				Necessary for a custom background salinity or other configurations, e.g. a super saline layer
				-> though this could be done outside of this command so....
			composition : string
				Choose which composition the liquid should be.
				Options: 'MgSO4', 'NaCl'
			concentration : float
				Initial intrusion concentration and/or ocean concentration; if using the shell option (below),
				this assumes that the shell was frozen out of an ocean with this concentration and composition
			rejection_cutoff : float > 0
				Liquid fraction (phi) below which no more salt will be accepted into the remaining liquid or
				interstitial liquid. Note: should be greater than 0
			shell : bool
				Option to include background salinity in the shell given the chosen composition and concentration.
				This will automatically adjust the temperature profile to account for a salty ocean near the melting
				temperature. If assuming something else, such as a slightly cooler convecting layer between the
				brittle shell and the ocean, this can be adjusted afterward by calling init_T()
			in_situ : bool
				Note: shell option must be True.
				Assumes the intrusion is from an event that melted the shell in-situ, thus have the same
				concentration and composition as the shell at that depth.
			T_match : bool
				Option to adjust the temperature profile to make the bottom be at the melting temperature of an ocean
				with the same composition and concentration. This is mostly used if making the assumption that the
				brittle layer simulated here is directly above the ocean.

		Usage:
			Pure shell, saline intrusion: Intrusion with 34 ppt NaCl salinity
				model.init_intrusion(composition='NaCl',concentration=12.3)

			Saline shell, in-situ melting: Ocean began with 100 ppt MgSO4 and intrusion has been created by in-situ
			melting
				model.init_intrusion(composition='MgSO4', concentration=100., shell=True, in_situ=True)
		"""

		self.issalt = True  # turn on salinity for solvers
		self.rejection_cutoff = rejection_cutoff  # minimum liquid fraction of cell to accept rejected salt

		# composition and concentration coefficients for fits from Buffo et al. (2019)
		# others have been calculated by additional runs using the model from Buffo et al. (2019)

		# dict structure {composition: [a,b,c]}
		# Liquidus curves derived from Liquius 1.0 (Buffo et al. 2019 and FREEZCHEM) for MgSO4 and NaCl
		self.Tm_consts = {'MgSO4': [-1.333489497e-5, -0.01612951864, 273.055175687],
		                  'NaCl': [-9.1969758e-5, -0.03942059, 272.63617665]
		                  }

		# dict structure {composition: {concentration: [a,b,c,d]}}
		self.shallow_consts = {'MgSO4': {0: [0., 0., 0., 0.],
		                                 12.3: [12.21, -8.3, 1.836, 20.2],
		                                 100: [22.19, -11.98, 1.942, 21.91],
		                                 282: [30.998, -11.5209, 2.0136, 21.1628]},
		                       'NaCl': {0: [0., 0., 0., 0.],
		                                10: [7.662, -4.936, 2.106, 24.8],
		                                34: [11.1, -4.242, 1.91, 22.55],
		                                100: [0., 0., 0., 0.],
		                                260: [0., 0., 0., 0.]}
		                       }

		# dict structure {composition: {concentration: [a,b]}}
		self.linear_consts = {'MgSO4': {0: [0., 0.],
		                                12.3: [1.0375, 0.40205],
		                                100: [5.4145, 0.69992],
		                                282: [14.737, 0.62319]},
		                      'NaCl': {0: [0., 0.],
		                               10: [0.6442, 0.2279],
		                               34: [1.9231, 0.33668],
		                               100: [0., 0.],
		                               260: [0., 0.]}
		                      }

		# dict structure {composition: {concentration: [a,b,c]}}
		self.depth_consts = {'MgSO4': {12.3: [1.0271, -74.0332, -4.2241],
		                               100: [5.38, -135.096, -8.2515],
		                               282: [14.681, -117.429, -5.4962]},
		                     'NaCl': {10: [0., 0., 0.],
		                              34: [1.8523, -72.4049, -10.6679],
		                              100: [0., 0., 0.],
		                              260: [0., 0., 0.]}
		                     }

		# create dictionary of root to switch between shallow and linear fits
		#  dict structure {chosen composition: {concentration: root}}
		self.linear_shallow_roots = {composition: {}}
		for key in self.linear_consts[composition]:
			root_func = lambda x: self.shallow_fit(x, *self.shallow_consts[composition][key]) \
			                                  - self.linear_fit(x, *self.linear_consts[composition][key])
			self.linear_shallow_roots[composition][key] = optimize.root(root_func, 3)['x'][0]

		self.composition = composition
		self.concentration = concentration

		if self.composition == 'MgSO4':
			# Liquidus curve derived from Liquius 1.0 (Buffo et al. 2019 and FREEZCHEM) for MgSO4
			# changing from lambda notation to def notation for better pickling?

			# def self.Tm_func = lambda S: (-(1.333489497 * 1e-5) * S ** 2) - 0.01612951864 * S + 273.055175687
			# density changes for water w/ concentration of salt below
			self.C_rho = 1.145
			self.Ci_rho = 7.02441855e-01

			self.saturation_point = 282.  # ppt, saturation concentration of MgSO4 in water
			self.constants.rho_s = 2660.  # kg/m^3, density of MgSO4

		elif self.composition == 'NaCl':
			# Liquidus curve derived from Liquius 1.0 (Buffo et al. 2019 and FREEZCHEM) for NaCl
			# linear fit for density change due to salinity S
			self.C_rho = 0.8644
			self.Ci_rho = 6.94487270e-01

			self.saturation_point = 260.  # ppt, saturation concentration of NaCl in water
			self.constants.rho_s = 2160.  # kg/m^3, density of NaCl

		# save array of concentrations for chosen composition for entraining salt in ice
		self.concentrations = np.sort([key for key in self.shallow_consts[composition]])

		# method for a salinity/depth profile via Buffo et al. 2020
		s_depth = lambda z, a, b, c: a + b / (c - z)
		self.S = s_depth(self.Z, *self.depth_consts[composition][concentration])

		# update initial melting temperature
		self.Tm = self.Tm_func(self.S, *self.Tm_consts[composition])
		# update temperature profile to reflect ocean salinity
		self.Tbot = self.Tm_func(s_depth(self.Lz, *self.depth_consts[composition][concentration]),
		                         *self.Tm_consts[composition])
		print("-- Adjusting temperature profile")
		self.init_T(Tsurf=self.Tsurf, Tbot=self.Tbot, real_Lz=self.real_Lz)

		# update volume average with included salt
		self.init_volume_averages()
		# begin tracking mass
		self.total_salt = [self.S.sum()]
		# begin tracking amount of salt removed from system
		self.removed_salt = [0]

		self.save_initials()

	def entrain_salt(self, dT, S, composition):
		"""
		Calculate the amount of salt entrained in newly frozen ice that is dependent on the thermal gradient across
		the ice (Buffo et al., 2019).
		Parameters:
			dT : float, array
				temperature gradient across cell, or array of temperature gradients
			S : float, array
				salinity (ppt) of newly frozen cell, or array of salinities
			composition : string
				salt composition
				options: 'MgSO4', 'NaCl'
		Returns:
			amount of salt entrained in ice, ppt
			or array of salt entrained in ice, ppt

		Usage:
			See HeatSolver.update_salinity() function.
		"""
		if composition != 'MgSO4':
			raise Exception('Run tests on other compositions')

		if isinstance(dT, (int, float)):  # if dT (and therefore S) is a single value
			if S in self.shallow_consts[composition]:
				# determine whether to use linear or shallow fit
				switch_dT = self.linear_shallow_roots[composition][S]
				if dT > switch_dT:
					return self.shallow_fit(dT, *self.shallow_consts[composition][S])
				elif dT <= switch_dT:
					return self.linear_fit(dT, *self.linear_consts[composition][S])

			else:  # salinity not in SlushFund runs
				# find which two known concentrations current S fits between
				c_min = self.concentrations[S > self.concentrations].max()
				c_max = self.concentrations[S < self.concentrations].min()

				# linearly interpolate between the two concentrations at gradient dT
				m, b = np.polyfit([c_max, c_min], [self.entrain_salt(dT, c_max, composition),
				                                   self.entrain_salt(dT, c_min, composition)], 1)

				# return concentration of entrained salt
				return m * S + b

		else:  # recursively call this function to fill an array of the same length as input array
			return np.array([self.entrain_salt(t, s, composition) for t, s in zip(dT, S)])
