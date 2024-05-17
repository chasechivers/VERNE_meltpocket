import numpy as np
from scipy import optimize
from HeatSolver import HeatSolver
import constants
import SalinityConstants
from SalinityFuncs import SalinityFuncs
from ThermophysicalProperties import ThermophysicalProperties


class VERNESim(HeatSolver, ThermophysicalProperties, SalinityFuncs):
	def __init__(self, w, D, *, Z0=0,
	             dz=None, dx=None, nz=None, nx=None,
	             kT=True,
	             use_X_symmetry=True, verbose=False,
	             coordinates="zx"):
		super().__init__()
		# Create constants for simulations
		self.constants = constants.constants()
		self.model_time, self.run_time = 0, 0
		# generally a stable time step for most simulations used here
		self.dt = 10
		# save coordinate system used
		self.coords = coordinates
		self.w = w
		self.D = D
		self.kT = kT  # k(T) I/O
		self.verbose = verbose
		self.issalt = False
		self.symmetric = use_X_symmetry

		if nx == None and nz == None:
			self.dx, self.dz = dx, dz
			self.nx, self.nz = int(self.w / self.dx + 1), int(self.D / self.dz + 1)
			if self.verbose:
				print(f" dx = {dx}m => nx = {self.nx}\n dz = {dz}m => nz = {self.nz}")
		elif dx == None and dz == None:
			self.nx, self.nz = nx, nz
			self.dx, self.dz = self.w / (self.nx - 1), self.D / (self.nz - 1)
			if self.verbose:
				print(f"  nx = {nx} => dx = {self.dx}m\n  nz = {nz}m=> dz = {self.dz}m")
		else:
			raise Exception("Error: Must choose either BOTH nx AND nz, or dx AND dz")
		# Create spatial grid
		self.symmetric = use_X_symmetry
		self._init_grids(Z0=Z0)


	def _init_grids(self, Z0=0):
		"""Class method to initialize grids given the choices in initiating the object"""
		# Create spatial grid
		if self.symmetric:  # assuming horizontally symmetric heat loss about x=0
			self.w = self.w / 2
			self.nx = int(self.w / self.dx + 1)
			if self.verbose:
				print(f"  using horizontal symmetry, creating {self.nz, self.nx} grid")

			self.X, self.Z = np.meshgrid(np.array([i * self.dx for i in range(self.nx)], dtype=float),
			                             np.array([Z0 + j * self.dz for j in range(self.nz)], dtype=float))

		elif self.symmetric is False:  # x domain centered on 0
			self.X, self.Z = np.meshgrid(np.array([-self.w / 2 + i * self.dx for i in range(self.nx)], dtype=float),
			                             np.array([Z0 + j * self.dz for j in range(self.nz)], dtype=float))

		# Create scalar field grids
		self.T = self.constants.Tm * np.ones((self.nz, self.nx), dtype=float)  # initialize domain at one temperature
		self.S = np.zeros((self.nz, self.nx), dtype=float)  # initialize domain with no salt
		self.phi = np.zeros((self.nz, self.nx), dtype=float)  # initialize domain as ice
		self.vehicle = np.zeros((self.nz, self.nx), dtype=int) # initialize domain with no vehicle
		if self.coords in ['zr', 'rz', 'cylindrical']:
			self._rph = 0.5 * (self.X[1:-1, 2:] + self.X[1:-1, 1:-1])
			self._rmh = 0.5 * (self.X[1:-1, 1:-1] + self.X[1:-1, :-2])

	def volume_average(self, phi, V, water_prop, ice_prop, vehicle_prop):
		"""
		Volume average water and ice properties based on the volumetric liquid fraction. Generally used for
		thermophysical properties. Used in several places during simulations so collected into one function for
		readability.
		:param phi: float, arr
			Volumetric liquid fraction
		:param V: float, arr
			Vehicle
		:param water_prop: float, arr
			A physical property of water, e.g. thermal conductivity
		:param ice_prop: float, arr
			A physical property of ice, e.g. thermal conductivity
		:return:
			Volume averaged physical property
		"""
		return V * vehicle_prop + (1 - V) * (phi * water_prop + (1 - phi) * ice_prop)

	def k(self, phi=None, V=None, T=None, S=None):
		"""
		Volume averaged thermal conductivity. Used in several places during simulations.
		:param phi: float, arr
			Volumetric liquid fraction
		:param T: float, arr
			Temperature, K
		:param S: float, arr
			Salinity, ppt
		:return:
			Volume averaged thermal conductivity
		"""
		if phi is None:
			phi = self.phi
		if V is None:
			V = self.vehicle
		if T is None:
			T = self.T
		if S is None:
			S = self.S
		return self.volume_average(phi, V, self.kw(T, S), self.ki(T, S), self.kv(T))

	def rhoc(self, phi=None, V=None, T=None, S=None):
		"""
		Volume averaged volumetric heat capacity. Used in several places during simulations.
		:param phi: float, arr
			Volumetric liquid fraction
		:param T: float, arr
			Temperature, K
		:param S: float, arr
			Salinity, ppt
		:return:
			Volume averaged volumetric
		"""
		if phi is None:
			phi = self.phi
		if V is None:
			V = self.vehicle
		if T is None:
			T = self.T
		if S is None:
			S = self.S
		return self.volume_average(phi, V, self.rhow(T, S) * self.cpw(T, S),
		                           self.rhoi(T, S) * self.cpi(T, S),
		                           self.rhov(T) * self.cpv(T))

	def _save_initials(self):
		"""
		Saves the initial values for each independent variable that evolves over time. Generally internal use only.
		"""
		self.T_initial = self.T.copy()
		self.phi_initial = self.phi.copy()
		self.S_initial = self.S.copy()
		self.V_initital = self.vehicle.copy()
		if self.verbose: print(f"Initial conditions saved.")

	def init_T(self, Tsurf, Tbot, profile="non-linear", real_D=0, **kwargs):
		"""Initialize an equilibrium thermal profile of the brittle portion of an ice shell of D thickness, with surface
		temperature Tsurf and basal temperature Tbot.

		:param Tsurf: float, int
			Surface temperature of brittle ice shell, K
		:param Tbot: float, int
			Temperature at bottom of brittle ice shell, K
		:param profile: str, (arr)
			The kind of profile wanted.
			Options
				"non-linear" (Default): Tsurf*(Tbot/Tsurf)**(Z/D) - equilibrium profile assumes a that ice thermal
					conducitivity is inversely proportional to the temperature (k_i ~ 1/T)
				"linear": (Tbot - Tsurf) * Z/D + Tsurf - equilibrium profile assumes constant ice thermal conductivity
				"stefan": T(z>=dz) = Tm, T(z=0) = Tsurf - assumes domain is all liquid except for first row of grid
				profile = np.ndarray([]) generated by user, this will be the background profile.
		:param real_D: float
			Used only if using a portion of a much larger ice shell.

			For instance, an ice shell of > 50 km may be too large to simulate, but if thermal evolution of interest
			happens in the upper 1 km, we set real_D = 50e3 and it will give an accurate profile for the first 1 km.
		:return: arr
			Temperature distribution in ice shell, K

		Usage:

		"""
		self.Tsurf = Tsurf
		self.Tbot = Tbot
		self.Tm = self.constants.Tm * np.ones(self.T.shape, dtype=float)

		if isinstance(profile, str):
			if profile in ["non-linear", "conductive"]:
				if real_D > 0:
					self.real_D = real_D
					self.Tbot = Tsurf * (Tbot / Tsurf) ** (self.Z[-1, -1] / real_D)
					self.T = Tsurf * (self.Tbot / Tsurf) ** (self.Z / self.real_D)
				else:
					self.T = Tsurf * (self.Tbot / Tsurf) ** (self.Z / self.D)

			if profile in ["two layer", "convective"]:
				if self.verbose: print(f"  setting {profile} profile ")
				if ("brittle_layer" or "layer") not in kwargs:
					raise Exception("Must choose thickness of brittle layer\n init_T(...brittle_layer=5e3, "
					                "real_D=15e3)")
				elif real_D == 0:
					raise Exception("Must choose total shell thickness\n init_T(...brittle_layer=5e3, "
					                "real_D=15e3")

				else:
					self.real_D = real_D
					d = kwargs["brittle_layer"] if "brittle_layer" in kwargs else kwargs["layer"]
					self._d = d
					if self.verbose: print(f"   with total shell thickness {real_D}m and brittle layer {d}m")
					self.T = Tsurf * (Tbot / Tsurf) ** abs(self.Z / d)
					self.T[self.Z > d] = Tbot

			elif profile == "linear":
				if real_D > 0:
					self.Tbot = (Tbot - Tsurf) * (self.D / real_D) + Tsurf
				self.T = (Tbot - Tsurf) * abs(self.Z / self.D) + Tsurf

			if self.verbose:
				print(f"init_T(Tsurf = {self.Tsurf:0.03f}, Tbot={self.Tbot:0.03f}")
				print(f"\t Temperature profile initialized to {profile}")
		else:
			self.T = profile
			if self.verbose: print("init_T: Custom profile implemented")

		self.Tedge = self.T[:, 0]
		self._save_initials()
		self._set_dt(self.CFL)

	def _set_vehicle_geom(self, depth, length, radius, rtg_size, **kwargs):
		if self.verbose: print("  setting vehicle geometry")
		if "nr" in self.__dict__:
			r = np.logical_and(self.R <= radius, self.R >= 0 if self.symmetric else -radius)
		else:
			r = np.logical_and(self.X <= radius, self.X >= 0 if self.symmetric else -radius)
		z = np.logical_and(self.Z <= self.vehicle_h + self.vehicle_d, self.Z >= self.vehicle_d)
		tmp = np.multiply(r, z)

		vehicle_geom = np.where(tmp == 1)
		return vehicle_geom

	def init_vehicle(self, depth, length, radius, *, constant_T=300, melt_mod=1, mechanical_descent=False,
	                 **kwargs):
		self.vehicle_h = length
		self.vehicle_r = radius
		self.vehicle_d = depth
		self.vehicle_T = constant_T
		self.melt_mod = melt_mod
		self.mechanical_descent = mechanical_descent
		if mechanical_descent:
			if "descent_speed" not in kwargs:
				raise Exception("Mechanical descent speed not chosen. Ensure it is in m/s\n  init_vehicle(... "
				                "mechanical_descent=True, descent_speed=3)")
			else:
				self.descent_speed = kwargs["descent_speed"]
		self.vehicle_geom = self._set_vehicle_geom(depth, length, radius, None)
		self.vehicle[self.vehicle_geom] = 1
		self.T[self.vehicle_geom] = self.vehicle_T
		self._save_initials()
		self._set_dt(self.CFL)

	def _set_dt(self, CFL=1 / 24):
		"""Applies the Neumann analysis so that the simulation is stable
		       ∆t = CFL * min(∆x, ∆z)^2 / max(diffusivity)
		where diffusivity = conductivity / density / specific heat. We want to minimize ∆t, thus
		      max(diffusivity) = min(density * specific heat) / max(conductivity)
		CFL is the Courant-Fredrichs-Lewy condition, a factor to help with stabilization
		"""
		if self.kT:
			self.dt = CFL * min([self.dz, self.dx]) ** 2 * self.rhoc().min() / (
				self._conductivity_average(self.k().max(), self.k()[self.k() < self.k().max()].max()))
		else:
			self.dt = CFL * min([self.dz, self.dx]) ** 2 * self.rhoc().min() / self.k().max()

	def _get_salinity_consts(self, composition):
		"""
		Grabs constants for equations relating salinity to other functions throughout a simulation.
		:param composition: str
			Composition of major salt
			Options "NaCl", "MgSO4"
		:return:
		"""
		shallow_consts = SalinityConstants.shallow_consts[composition]
		linear_consts = SalinityConstants.linear_consts[composition]
		Tm_consts = SalinityConstants.Tm_consts[composition]
		depth_consts = SalinityConstants.depth_consts[composition]

		if composition == "NaCl":
			return shallow_consts, linear_consts, Tm_consts, depth_consts, SalinityConstants.new_fit_consts[
				composition]
		else:
			return shallow_consts, linear_consts, Tm_consts, depth_consts

	def _get_salinity_roots(self):
		"""
		Only used when the two old entrained salt vs temperature gradient functions are used (shallow_fit,
		linear_fit) to determine at what temperature gradient to use either fit.
		:param composition: str
			Composition of major salt in system
		:return:
			Dictionary with structure {concentration: root}
		"""
		linear_shallow_roots = {}
		for concentration in self.linear_consts:
			func = lambda x: self.shallow_fit(x, *self.shallow_consts[concentration]) \
			                 - self.linear_fit(x, *self.linear_consts[concentration])
			linear_shallow_roots[concentration] = optimize.root(func, 3)['x']
		return linear_shallow_roots

	def _set_salinity_values(self, composition):
		if composition == "MgSO4":
			self.C_rho = 1.145
			self.Ci_rho = 7.02441855e-01

			self.saturation_point = 174.  # ppt, saturation concentration of MgSO4 in water (Pillay et al., 2005)
			self.constants.rho_s = 2660.  # kg/m^3, density of anhydrous MgSO4

			## Only used if heat_from_precipitation is used
			# Assuming that the bulk of MgSO4 precipitated is hydrates and epsomite (MgSO4 * 7H2O), then the heat
			# created from precipitating salt out of solution is equal to the enthalpy of formation of epsomite (
			# Grevel et al., 2012).
			self.enthalpy_of_formation = 13750.0e3  # J/kg for epsomite

		elif composition == "NaCl":
			self.C_rho = 0.8644
			self.Ci_rho = 6.94487270e-01

			self.saturation_point = 232.  # ppt, saturation concentration of NaCl in water
			self.constants.rho_s = 2160.  # kg/m^3, density of NaCl

			self.enthalpy_of_formation = 0.  # J/kg

	def init_salinity(self, composition, concentration, *, rejection_cutoff=0.25, shell=True,
	                  in_situ=False, T_match=True, salinate=True, **kwargs):

		self.issalt = True

		if in_situ == True:
			shell = True

		self.composition = composition
		self.concentration = concentration
		self.rejection_cutoff = rejection_cutoff
		self.salinate = salinate

		self.heat_from_precipitation = True if "heat_from_precipitation" in kwargs and kwargs[
			"heat_from_precipitation"] == 1 else False

		if composition == "NaCl":
			self.shallow_consts, self.linear_consts, self.Tm_consts, self.depth_consts, self.new_fit_consts = \
				self._get_salinity_consts(composition)
		else:
			self.shallow_consts, self.linear_consts, self.Tm_consts, self.depth_consts = \
				self._get_salinity_consts(composition)
			self.linear_shallow_roots = self._get_salinity_roots()

		self._set_salinity_values(composition)
		self.concentrations = np.sort([conc for conc in (self.shallow_consts if composition == "MgSO4" else
		                                                 self.new_fit_consts)])

		if T_match:
			self.Tbot = self.Tm_func(concentration, *self.Tm_consts)
			if self.verbose:
				print(f"\t Readjusting temperature profile to fit {concentration} ppt {composition} ocean (assuming "
				      f"non-linear profile")
			if "real_D" in self.__dict__.keys():
				if "_d" in self.__dict__.keys():
					pass
				else:
					self.init_T(Tsurf=self.Tsurf, Tbot=self.Tbot, real_D=self.real_D)
			else:
				self.init_T(Tsurf=self.Tsurf, Tbot=self.Tbot, )

		if shell:
			if self.verbose: print("  Adding salts into background ice")
			self.S = self.salinity_with_depth(self.Z, *self.depth_consts[self.concentration])
			# because the above equation from Buffo et al., 2020 is not well known for depths < 10 m, it will predict
			# salinities over the parent liquid concentration. Here, I force it to entrain all salt where
			# it wants to concentrate more than physically possible.
			self.S[self.S > concentration] = concentration

		else:
			self.S[self.geom] = self.concentration
		try:
			self.T[self.geom] = self.Tm_func(self.S[self.geom], *self.Tm_consts)
		except AttributeError:
			pass
		self.Tm = self.Tm_func(self.S, *self.Tm_consts)

		# self.entrain_salt = self._get_interpolator() if use_interpolator else self._entrain

		self.total_salt = [self.S.sum()]
		self.removed_salt = []

		self._save_initials()
		self._set_dt(self.CFL)
