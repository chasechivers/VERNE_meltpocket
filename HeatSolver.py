# Author: Chase Chivers
# Last updated: 1/10/20

import numpy as np
import time as _timer_
from utility_funcs import *
import string, random, os
# import matplotlib.pyplot as plt
from scipy import optimize


class HeatSolver:
	"""
	Solves two-phase thermal diffusivity problem with a temperature-dependent thermal conductivity of ice in
	two-dimensions. Sources and sinks include latent heat of fusion and tidal heating
	Options:
		tidalheat -- binary; turns on/off viscosity-dependent tidal heating from Mitri & Showman (2005), default = 0
		Ttol -- convergence tolerance for temperature, default = 0.1 K
		phitol -- convergence tolerance for liquid fraction, default = 0.01
		latentheat -- 1 : use Huber et al. (2008) enthalpy method (other options coming soon?)
		freezestop -- binary; stop when intrusion is frozen, default = 0

	Usage:
		Assuming, model = IceSystem(...)
		- Turn on tidal heating component
			model.tidalheat = True

		- Change tolerances
			model.Ttol = 0.001
			model.phitol = 0.0001
			model.Stol = 0.0001
	"""
	# off and on options
	tidalheat = 0  # off/on tidalheating component
	Ttol = 0.1  # temperature tolerance
	phitol = 0.01  # liquid fraction tolerance
	Stol = 1  # salinity tolerance
	latentheat = 1  # choose enthalpy method to use
	freezestop = 0  # stop simulation upon total solidification of intrusion
	model_time = 0
	run_time = 0

	class outputs:
		"""Class structure to help define and calculate desired outputs of a simulation."""

		def __init__(self):
			self.outputs.tmp_data_directory = ''
			self.outputs.tmp_data_file_name = ''
			self.outputs.transient_results = dict()
			self.outputs.output_frequency = 0

		def choose(self, all=False, T=False, phi=False, k=False, S=False, Q=False, V=False,
		           output_frequency=1000, output_list=[]):
			"""
			Choose which outputs to track with time. Each variable is updated at the chosen output frequency and is
			returned in the dictionary object outputs.transient_results.
			Parameters:
				output_frequency : integer
					the frequency to report a transient result. Default is every 1000 time steps
				output_list : list
					list of strings for the outputs below. Generally used for simulation that had stopped (i.e. hit a
					wall time) without desired
				all : bool
					turns on all outputs listed below
				T, phi, k, S, Q : bool
					tracks and returns an array of temperature, liquid fraction, volume averaged thermal conductivity,
					salinity, and source/sink grids, respectively
				h : bool
					tracks the height (thickness) of the liquid intrusion over time into a 1d array
				r : bool
					tracks the radius of the liquid portion over time into a 1d array
				freeze_fronts : bool
					tracks the propagating freeze front at the top and bottom of the intrusion into a 1d array
				percent_frozen : bool
					tracks and returns a 1d list of the percent of the original intrusion that is now ice
			Usage:
				Output all options every 50 years
					model.outputs.choose(model, all=True, output_frequency=int(50 * model.constants.styr/model.dt))

				Output only temperature grids and salinity at every time step
					model.outputs.choose(model, T=True, S=True, output_frequency=1)
					--or--
					model.outputs.choose(model, output_list=['T','S'], output_frequency=1);
			"""
			to_output = {'time': True, 'T': T, 'phi': phi, 'k': k, 'S': S, 'Q': Q, 'V': V}
			if all:
				to_output = {key: True for key, value in to_output.items()}

			if len(output_list) != 0:
				to_output = {key: True for key in output_list}
				to_output['time'] = True

			self.outputs.transient_results = {key: [] for key in to_output if to_output[key] is True}
			self.outputs.outputs = self.outputs.transient_results.copy()

			self.outputs.output_frequency = output_frequency
			self.outputs.tmp_data_directory = './tmp/'
			self.outputs.tmp_data_file_name = 'tmp_data_runID' + ''.join(random.choice(string.digits) for _ in range(4))

		def calculate_outputs(self, n):
			"""
			Calculates the output and appends it to the list for chosen outputs. See outputs.choose() for description
			of values calculated here.
			Parameters:
				n : integer
					nth time step during simulation
			Returns:
				ans : dictionary object
					dictionary object with chosen outputs as 1d numpy arrays
			"""
			ans = {}
			for key in self.outputs.outputs:
				if key == 'time':
					ans[key] = self.model_time
				if key == 'T':
					ans[key] = self.T.copy()
				if key == 'S':
					ans[key] = self.S.copy()
				if key == 'phi':
					ans[key] = self.phi.copy()
				if key == 'k':
					ans[key] = self.k.copy()
				if key == 'Q':
					ans[key] = self.Q.copy()
				if key == 'V':
					ans[key] = self.vehicle.copy()
			return ans

		def get_results(self, n, extra=False):
			"""Calls outputs.calculate_outputs() then saves dictionary of results to file"""
			if n % self.outputs.output_frequency == 0:
				get = self.outputs.calculate_outputs(self, n)
				save_data(get, self.outputs.tmp_data_file_name + '_n={}'.format(n), self.outputs.tmp_data_directory)
			if extra:
				get = self.outputs.calculate_outputs(self, n)
				save_data(get, self.outputs.tmp_data_file_name + "_n={}".format(n), self.outputs.tmp_data_directory)

		def get_all_data(self, del_files=True):
			"""Concatenates all saved outputs from outputs.get_results() and puts into a single dictionary object."""
			cwd = os.getcwd()  # find working directory
			os.chdir(self.outputs.tmp_data_directory)  # change to directory where data is being stored

			# make a list of all results files in directory
			data_list = nat_sort([data for data in os.listdir() if data.endswith('.pkl') and \
			                      self.outputs.tmp_data_file_name in data])
			# copy dictionary of desired results
			ans = self.outputs.transient_results.copy()
			# iterate over file list
			for file in data_list:
				tmp_dict = load_data(file)  # load file
				for key in self.outputs.outputs:  # iterate over desired outputs
					ans[key].append(tmp_dict[key])  # add output from result n to final file
				del tmp_dict
				if del_files: os.remove(file)

			# make everything a numpy array for easier manipulation
			ans = {key: np.asarray(value) for key, value in ans.items()}

			# go back to working directory
			os.chdir(cwd)
			return ans

	def set_boundaryconditions(self, top=True, bottom=True, sides=True, **kwargs):
		"""
			Set boundary conditions for heat solver. A bunch of options are available in case they want to be tested
			or used.
			top : bool, string
				top boundary conditions.
				default: True = Dirichlet, Ttop = Tsurf chosen earlier
				'Flux': surface loses heat to a "ghost cell" of ice
						temperature of ghost cell is based on the equilibrium temperature profile at the depth one
						spatial size "above" the domain,
						i.e.: T_(ghost_cell) = Tsurf * (Tbot/Tsurf) ** (-dz/Lz)
								for Tsurf = 110 K, Tbot = 273.15, & Lz = 5 km => T_(ghost_cell) = 109.8 K
				'Radiative': surface loses heat beyond the "background" surface temperature through blackbody
							radiation to a vacuum using Stefan-Boltzmann Law.
							Note: The smaller the time step the better this will predict the surface warming. Larger
							time steps make the surface gain heat too fast. This is especially important if upper
							domain is cold and simulation is using temperature-dependent thermal conductivity.

			bottom: bool, string
				bottom boundary condition
				default, True: Dirichlet, Tbottom = Tbot chosen earlier
				'Flux' : bottom loses heat to a "ghost cell" of ice at a constant temperature
						 temperature of ghost cell is based on equilibrium temperature profile at depth one spatial
						 size below the domain
						 i.e. T_(ghost_cell) = Tsurf * (Tbot/Tsurf) ** ((Lz+dz)/Lz)
						    for Tsurf = 110 K, Tbot = 273.15, & Lz = 5 km => T_(ghost_cell) = 273.647 K
						-> this can be helpful for using a "cheater" vertical domain so as not to have to simulate a
						whole shell
				'FluxI', 'FluxW': bottom loses heat to a "ghost cell" of ice ('FluxI') or water ('FluxW') at a chosen
								constant temperature
								ex: model.set_boundaryconditions(bottom='FluxI', botT=260);
			sides: bool, string
				Left and right boundary conditions
				True :  Dirichlet boundary condition;
						Tleft = Tright =  Tedge (see init_T)
						* NOTE: must set up domain such that anomaly is far enough away to not interact with the
						edges of domain
				'NoFlux : a 'no flux' boundary condition
							-> boundaries are forced to be the same temperature as the adjacent cells in the domain
				'RFlux' : 'NoFlux' boundary condition on the left, with a flux boundary at T(z,x=Lx,t) = Tedge(z)
						that is dL far away. Most useful when using the symmetry about x option.
						dL value must be chosen when using this option:
						ex: model.set_boundaryconditions(sides='RFlux', dL=500e3)
		"""

		self.topBC = top
		if top == 'Radiative':
			# self.Std_Flux_in = (self.k_initial[0, 1:-1] + self.k_initial[1, 1:-1]) \
			#                   * (self.T_initial[1, 1:-1] - self.T_initial[0,1:-1])
			self.std_set = 0
		if top == "Flux":
			self.T_top_out = kwargs['Top_out'] * np.ones(self.T[0, :].shape)

		self.botBC = bottom
		if bottom == 'FluxI' or bottom == 'FluxW':
			try:
				self.botT = kwargs['botT']
			except:
				raise Exception('Bottom boundary condition temperature not chosen\n\t->ex: '
				                'model.set_boundaryconditions(bottom=\'FluxI\', botT=260);')
		self.sidesBC = sides
		if sides == 'RFlux':
			try:
				self.dL = kwargs['dL']
			except:
				raise Exception('Length for right side flux not chosen\n\t->model.set_boundaryconditions('
				                'sides=\'RFlux\', dL=500e3)')

	def update_salinity(self, phi_last, grad_mean='arithmetic'):
		# find indices where ice has just formed
		z_newice, x_newice = np.where((phi_last > 0) * (self.phi == 0))
		# get indices of cells that can accept rejected salts
		water = np.where(self.phi >= self.rejection_cutoff)
		# calculate "volume" of cells that can accept rejected salt
		vol = water[1].shape[0]
		# start catalogue of TOTAL salt removed from system after water is saturated
		self.removed_salt.append(0)

		if len(z_newice) > 0 and vol != 0:  # if new ice has formed...
			Sn = self.S.copy()  # new S matrix
			for i in range(len(z_newice)):  # iterate over cells where ice has just formed
				# save location of cell
				loc = (z_newice[i], x_newice[i])
				# get starting salinity in cell
				S_old = self.S[loc]
				# calculate centered thermal gradients across each cell
				if self.symmetric and loc[1] in [0, self.nx-1]:
					grad_x = 0
				else:
					grad_x = abs(self.T[loc[0], loc[1] - 1] - self.T[loc[0], loc[1] + 1]) / 2 / self.dx
				grad_z = (self.T[loc[0] - 1, loc[1]] - self.T[loc[0] + 1, loc[1]]) / 2 / self.dz

				# brine drainage paramterization:
				## if grad_z > 0 (water above cell) => no drainage, salt stays
				if grad_z > 0:
					self.S[loc] = S_old

				## if grad_z < 0 (watler below cell) => brien drains, reject salt
				elif grad_z < 0:
					# determine "mean" temperature gradient across cell
					if grad_mean == "arithmetic":
						grad = (abs(grad_z) + abs(grad_x)) / 2
					elif grad_mean == "geometric":
						grad = np.sqrt(grad_x * abs(grad_z))
					elif grad_mean == "harmonic":
						grad = 2. / (1. / grad_x + 1. / abs(grad_z))
					elif grad_mean == "max":
						grad = max(grad_x, abs(grad_z))
					elif grad_mean == "across":
						grad = np.hypot(grad_x, grad_z)

					self.S[loc] = self.entrain_salt(grad, S_old, self.composition)

			# assume salt is well mixed into remaining liquid solution in time step dt
			self.S[water] = self.S[water] + (Sn.sum() - self.S.sum()) / vol

			# remove salt from system if liquid is above the saturation point
			self.removed_salt[-1] += (self.S[self.S >= self.saturation_point] - self.saturation_point).sum()

			# ensure liquid hits only the saturation concentration
			self.S[self.S > self.saturation_point] = self.saturation_point

		# check salt conservation
		total_S_new = self.S.sum() + np.asarray(self.removed_salt).sum()
		if abs(total_S_new - self.total_salt[0]) <= self.Stol:
			self.total_salt.append(total_S_new)
		else:
			self.total_salt.append(total_S_new)
			raise Exception('Mass not being conserved')

	def remix_salt_melted_ice(self, phi_last):
		loc = np.where((phi_last < 1) & (self.phi==1))
		#print(loc)
		if len(loc[0]) > 0:
			water = np.where(self.phi >= self.rejection_cutoff)
			vol = water[1].shape[0]
			self.S[water] = self.S[water].sum() / vol
		else:
			pass

	def adjust_domain(self):
		"""adjust the domain as the vehicle passes further into the shell"""
		# create new "blank" matrices for each variable that must be adjusted
		# only Z direction must be adjusted
		Znew = np.zeros(self.Z.shape)
		Znew[:-1, :] = self.Z[1:, :].copy()
		Znew[-1, :] = Znew[-2, :] + self.dz
		self.Z = Znew.copy()
		del Znew

		# now adjust T
		Tnew = np.zeros(self.T.shape)
		Tnew[:-1, :] = self.T[1:, :].copy()
		# However, must make the "newly added" bottom ice at the right temperature!
		Tbot = self.Tsurf * (self.Tbot / self.Tsurf) ** ((self.Lz + self.dz) / self.real_Lz)
		Tnew[-1, :] = Tbot
		Ttop = self.T[0, :]
		self.T = Tnew

		# adjust boundary conditions
		Tbot = self.Tsurf * (self.Tbot / self.Tsurf) ** ((self.Lz + self.dz) / self.real_Lz)
		if self.botBC == True:
			self.TbotBC = Tnew[-1, :]

		# likely always going to start just below the surface so any movement will make us "lose" the radiative
		# boundary condition, meaning it woill always be a flux boundary condition that fluxes to the layer above it
		# with the previous tempearture profile (Ttop)
		self.set_boundaryconditions(top="Flux", bottom=True, sides=self.sidesBC, Top_out=Ttop)

		# it should still be the same for phi
		phinew[:-1, :] = self.phi[1:, :].copy()
		self.phi = phinew.copy()

		# and the new vehicle position
		vnew[:-1, :] = self.vehicle[1:, :].copy()
		self.vehicle = vnew.copy()
		# BUT! I have to make sure the vehcile geometry is adjusted?
		self.vehicle_geom = np.where(self.vehicle == 1)

		# adjust volume averages
		self.update_volume_averages()
		return 0

	def update_vehicle_position(self):
		# vehicle melts through layer dz at front, move the vehicle down by dz
		# widx = int(self.vehicle_w/self.dx)
		# hidx = int(self.vehicle_h/self.dz)+1
		widx = self.vehicle_geom[1].max()  # index of vehicle width
		hidx = self.vehicle_geom[0].max()  # index for vehicle front
		bidx = self.vehicle_geom[0].min()  # index for vehicle back
		Zidxs, Xidxs = self.vehicle_geom[0][-widx:] + 1, self.vehicle_geom[1][-widx:]
		if Xidxs[0] == 1:  # likely using X-symmetry
			Xidxs = self.vehicle_geom[1][:widx + 1]
			Zidxs = np.array([int(Zidxs[0]) for i in Xidxs], dtype=int)

		if self.phi[Zidxs, Xidxs].sum() == len(Xidxs):
			# save the temperature profile of vehicle
			Tsave = self.T[Zidxs, Xidxs]
			phisave = self.phi[Zidxs, Xidxs]
			if self.issalt: Ssave = self.S[Zidxs, Xidxs]
			# move vehicle down
			self.vehicle[Zidxs, Xidxs] = 1
			self.vehicle[bidx, Xidxs] = 0
			# reassign T to "behind nose"
			self.T[hidx, Xidxs] = Tsave
			# reassign water to "behind  nose"
			self.phi[hidx, Xidxs] = phisave
			# move salt??
			if self.issalt: self.S[hidx, Xidxs] = Ssave
			"""
			OLD WAY OF DOING THINGS!!
			# reassign T to new vehicle position
			#self.T[bidx, Xidxs] = Tsave
			
			# move water at nose to behind vehicle
			#self.phi[bidx, Xidxs] = phisave
			# move salt??
			#if self.issalt: self.S[bidx, Xidxs] = Ssave
			"""
			# save geometry of vehicle in current position
			self.vehicle_geom = np.where(self.vehicle == 1)
			# adjust the domain further into the ice shell
			# self.adjust_domain()
			# make sure everything else catches up like heat and thermal props
			self.update_volume_averages()
			self.vehicle_temp_scheme()
			# calculate some stuff about moving down
			dnose = np.where(self.vehicle == 1)[0][-1] * self.dz
			print('Vehicle moved down:\n\t t = {}s, dnose = {} m, v = {} km/yr'.format(
				self.model_time, dnose, (dnose - self.vehicle_d - self.vehicle_h) * 3.154e4 / self.model_time))
			#print('Vehicle moved down:\n\t t = {}s, dnose = {} m, v = {} km/yr'.format(
			#		self.model_time, dnose, (dnose - self.vehicle_d - self.vehicle_h) * 3.154e4 / self.model_time))

			return 1
		else:
			return 0

	def TmPfunc(self):
		if self.issalt:
			rho = self.phi * (self.constants.rho_w + self.C_rho * self.S) \
			      + (1 - self.phi) * (self.constants.rho_i + self.Ci_rho * self.S)
		else:
			rho = self.phi * self.constants.rho_w + (1 - self.phi) * self.constants.rho_i
			rho[self.vehicle == 1] = self.constants.rho_v
		pressure = np.zeros(rho.shape, dtype=float)
		if self.Z[0, 0] > 0:
			pressure[0, :] = self.constants.rho_i * self.constants.g + self.Z[0, 0]
		else:
			pressure[0, :] = rho[0, :] * self.constants.g * self.dz / 2.
		for i in range(1, self.nz):
			pressure[i] = pressure[i - 1] + rho[i] * self.constants.g * self.dz
		return - 0.0132 * pressure * 1e-5 - 0.1577 * np.log(pressure * 1e-5) + 0.1516 * np.sqrt(pressure * 1e-5)

	def update_liquid_fraction(self, phi_last):
		"""Application of Huber et al., 2008 enthalpy method. Determines volume fraction of liquid/solid in a cell."""
		# update melting temperature for enthalpy if salt is included in simulation
		if self.issalt:
			self.Tm = self.Tm_func(self.S, *self.Tm_consts[self.composition])
		# calculate new enthalpy of solid ice
		else:
			self.Tm = self.Tm + (self.TmPfunc() if self.use_pressure else 0)
		Hs = self.cp_i * self.Tm  # update entalpy of solid ice
		Hl = Hs + self.constants.Lf
		H = self.cp_i * self.T + self.constants.Lf * phi_last  # calculate the enthalpy in each cell
		# update liquid fraction
		self.phi[H >= Hs] = (H[H >= Hs] - Hs[H >= Hs]) / self.constants.Lf
		self.phi[H <= Hl] = (H[H <= Hl] - Hs[H <= Hl]) / self.constants.Lf
		# all ice
		self.phi[H < Hs] = 0.
		# all water
		self.phi[H > Hl] = 1

	def update_volume_averages(self):
		"""Updates volume averaged thermal properties."""
		if self.kT:
			self.k = (1 - self.phi) * self.constants.ac / self.T + self.phi * self.constants.kw
		else:
			self.k = (1 - self.phi) * self.constants.ki + self.phi * self.constants.kw
		self.k[self.vehicle == 1] = self.constants.kv

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
			self.rhoc = (1 - self.phi) * (self.constants.rho_i + self.Ci_rho * self.S) * self.constants.cp_i \
			            + self.phi * (self.constants.rho_w + self.C_rho * self.S) * self.constants.cp_w
			self.rhoc[self.vehicle == 1] = self.constants.rho_v * self.constants.cp_v

		else:
			self.rhoc = (1 - self.phi) * self.constants.rho_i * self.constants.cp_i \
			            + self.phi * self.constants.rho_w * self.constants.cp_w
			self.rhoc[self.vehicle == 1] = self.constants.rho_v * self.constants.cp_v

	def vehicle_temp_scheme(self):
		if 'Constant T' in self.vehicle_heat_scheme:
			self.T[self.vehicle_geom] = self.vehicle_T
		elif 'Constant Flux' in self.vehicle_heat_scheme:
			# c = self.dt / (self.dx * self.dz * self.rhoc[self.vehicle_geom])
			Qv = self.power  # self.k[self.vehicle_geom] * (self.vehicle_T - self.T[self.vehicle_geom]) * C
			self.Q[self.vehicle_geom] += Qv
		elif 'RTG T' in self.vehicle_heat_scheme:
			self.T[self.RTGS] = self.vehicle_T
		elif 'RTG Flux' in self.vehicle_heat_scheme:
			# c = self.dt / (self.dx * self.dz * self.rhoc[self.RTGS])
			Qv = self.power
			self.Q[self.RTGS] += Qv

	def update_sources_sinks(self, phi_last, T_last):
		"""Updates external heat or heat-sinks during simulation."""
		self.latent_heat = self.constants.rho_i * self.constants.Lf * (
				self.phi[1:-1, 1:-1] - phi_last[1:-1, 1:-1]) / self.dt

		self.tidal_heat = 0
		if self.tidalheat:
			# ICE effective viscosity follows an Arrenhius law
			#   viscosity = reference viscosity * exp[C/Tm * (Tm/T - 1)]
			# if cell is water, just use reference viscosity for pure ice at 0 K
			self.visc = (1 - phi_last[1:-1, 1:-1]) * self.constants.visc0i \
			            * np.exp(self.constants.Qs * (self.Tm[1:-1, 1:-1] / T_last[1:-1, 1:-1] - 1) / \
			                     (self.constants.Rg * self.Tm[1:-1, 1:-1])) \
			            + phi_last[1:-1, 1:-1] * self.constants.visc0w
			self.tidal_heat = (self.constants.eps0 ** 2 * self.constants.omega ** 2 * self.visc) / (
					2 + 2 * self.constants.omega ** 2 * self.visc ** 2 / (self.constants.G ** 2))

		total = self.tidal_heat - self.latent_heat
		total[self.vehicle[1:-1, 1:-1] == 1] = 0

		return total

	def apply_boundary_conditions(self, T_last, k_last, rhoc_last):
		"""Applies chosen boundary conditions during simulation run."""
		# apply chosen boundary conditions at bottom of domain
		if self.botBC == True:
			self.T[-1, 1:-1] = self.TbotBC[1:-1]

		elif self.botBC == 'Flux':
			Tbot = self.Tsurf * (self.Tbot / self.Tsurf) ** (self.Lz / self.real_Lz)
			T_bot_out = self.Tsurf * (Tbot / self.Tsurf) ** ((self.Lz + self.dz) / self.Lz)
			c = self.dt / (2 * rhoc_last[-1, 1:-1])

			Tbotx = c / self.dx ** 2 * ((k_last[-1, 1:-1] + k_last[-1, 2:]) * (T_last[-1, 2:] - T_last[-1, 1:-1]) \
			                            - (k_last[-1, 1:-1] + k_last[-1, :-2]) * (T_last[-1, 1:-1] - T_last[-1, :-2]))
			Tbotz = c / self.dz ** 2 * (
					(k_last[-1, 1:-1] + self.constants.ac / T_bot_out) * (T_bot_out - T_last[-1, :-1]) \
					- (k_last[-1, 1:-1] + k_last[-2, 1:-1]) * (T_last[-1, 1:-1] - T_last[-2, 1:-1]))
			self.T[-1, 1:-1] = T_last[-1, 1:-1] + Tbotx + Tbotz + self.Q[-1, :] * 2 * c

		elif self.botBC == 'Ocean':
			c = self.dt / 2 / rhoc_last[-1, 1:-1]
			Tbotx = c / self.dx ** 2 * ((k_last[-1, 1:-1] + k_last[-1, 2:]) * (T_last[-1, 2:] - T_last[-1, 1:-1]) \
			                            - (k_last[-1, 1:-1] + k_last[-1, :-2]) * (T_last[-1, 1:-1] - T_last[-1, :-2]))
			Tbotz = c * ((k_last[-1, 1:-1] + self.constants.kw) * (273.15 - T_last[-1, 1:-1]) \
			             - (k_last[-1, 1:-1] + k_last[-2, 1:-1]) * (T_last[-1, 1:-1] - T_last[-2, 1:-1])) / self.dz ** 2
			self.T[-1, 1:-1] = T_last[-1, 1:-1] = Tbotx + Tbotz + self.Q[-1, :] * 2 * c

		# apply chosen boundary conditions at top of domain
		if self.topBC == True:
			self.T[0, 1:-1] = self.TtopBC[1:-1]

		elif self.topBC == 'Flux':
			T_top_out = self.T_top_out[1:-1]
			if self.cpT is True:
				Cbc = rhoc_last[0, 1:-1] / (self.constants.rho_i * (185. + 2 * 7.037 * T_top_out))
			else:
				Cbc = 1
			c = self.dt / (2 * rhoc_last[0, 1:-1])
			Ttopx = c / self.dx ** 2 * ((k_last[0, 1:-1] + k_last[0, 2:]) * (T_last[0, 2:] - T_last[0, 1:-1]) \
			                            - (k_last[0, 1:-1] + k_last[0, :-2]) * (T_last[0, 1:-1] - T_last[0, :-2]))
			Ttopz = c / self.dz ** 2 * ((k_last[0, 1:-1] + k_last[1, 1:-1]) * (T_last[1, 1:-1] - T_last[0, 1:-1]) \
			                            - (k_last[0, 1:-1] + Cbc * self.constants.ac / T_top_out) * (
					                            T_last[0, 1:-1] - T_top_out))
			self.T[0, 1:-1] = T_last[0, 1:-1] + Ttopx + Ttopz + self.Q[0, :] * 2 * c

		elif self.topBC == 'Radiative':
			c = self.dt / (2 * rhoc_last[0, :])
			rad = self.dz * self.constants.stfblt * self.constants.emiss * (T_last[0, :] ** 4 - self.Tsurf ** 4)
			Ttopz = c / self.dz ** 2 * ((k_last[0, :] + k_last[1, :]) * (T_last[1, :] - T_last[0, :]) \
			                            - (self.k_initial[0, :] + self.k_initial[1, :]) * (
						                            self.T_initial[1, :] - self.Tsurf))
			Ttopx = 1 / self.dx ** 2 * ((k_last[0, 1:-1] + k_last[0, 2:]) * (T_last[0, 2:] - T_last[0, 1:-1]) \
			                            - (k_last[0, 1:-1] + k_last[0, :-2]) * (T_last[0, 1:-1] - T_last[0, :-2]))

			self.T[0, :] = T_last[0, :] + Ttopz - rad * c / self.dz ** 2
			self.T[0, 1:-1] += (Ttopx + self.Q[0, :] * 2) * self.dt / (2 * rhoc_last[0, 1:-1])
		# else:
		#	self.T[0, 1:-1] = self.Tsurf

		# apply chosen boundary conditions at sides of domain
		if self.sidesBC == True:
			self.T[:, 0] = self.Tedge.copy()
			self.T[:, self.nx - 1] = self.Tedge.copy()

		elif self.sidesBC == 'NoFlux':
			self.T[:, 0] = self.T[:, 1].copy()
			self.T[:, -1] = self.T[:, -2].copy()

		elif self.sidesBC == 'RFlux':
			# left side of domain uses 'NoFlux'
			self.T[:, 0] = T_last[:, 1].copy()
			# right side of domain
			c = self.dt / (2 * rhoc_last[1:-1, -1])
			TRX = c * ((k_last[1:-1, -1] + self.constants.ac / self.Tedge[1:-1]) * \
			           (self.Tedge[1:-1] - T_last[1:-1, -1]) / self.dx \
			           - (k_last[1:-1, -1] + k_last[1:-1, -2]) * (T_last[1:-1, -1] - T_last[1:-1, -2]) / self.dL)

			TRZ = c * ((k_last[1:-1, -1] + k_last[2:, -1]) * (T_last[2:, -1] - T_last[1:-1, -1]) \
			           - (k_last[1:-1, -1] + k_last[:-2, -1]) * (T_last[1:-1, -1] - T_last[:-2, -1])) / self.dz ** 2

			self.T[1:-1, -1] = T_last[1:-1, -1] + TRX + TRZ + self.Q[:, -1] * 2 * c

	def get_gradients(self, T_last):
		# constant in front of x-terms
		Cx = self.dt / (2 * self.rhoc[1:-1, 1:-1] * self.dx ** 2)
		# temperature terms in x direction
		Tx = Cx * ((self.k[1:-1, 1:-1] + self.k[1:-1, 2:]) * (T_last[1:-1, 2:] - T_last[1:-1, 1:-1]) \
		           - (self.k[1:-1, 1:-1] + self.k[1:-1, :-2]) * (T_last[1:-1, 1:-1] - T_last[1:-1, :-2]))
		# constant in front of z-terms
		Cz = self.dt / (2 * self.rhoc[1:-1, 1:-1] * self.dz ** 2)
		# temperature terms in z direction
		Tz = Cz * ((self.k[1:-1, 1:-1] + self.k[2:, 1:-1]) * (T_last[2:, 1:-1] - T_last[1:-1, 1:-1]) \
		           - (self.k[1:-1, 1:-1] + self.k[:-2, 1:-1]) * (T_last[1:-1, 1:-1] - T_last[:-2, 1:-1]))
		return Tx, Tz

	def print_all_options(self, nt):
		"""Prints options chosen for simulation."""

		def stringIO(bin):
			if bin:
				return 'on'
			else:
				return 'off'

		def stringBC(BC):
			if isinstance(BC, str):
				return BC
			elif BC:
				return 'Dirichlet'

		print('Starting simulation with\n-------------------------')
		print('\t total model time:  {}s, {}yr'.format(nt * self.dt, (nt * self.dt) / self.constants.styr))
		print('\t   dt = {} s'.format(self.dt))
		print('\t Ice shell thickness: {} m'.format(self.Lz))
		print('\t Lateral domain size: {} m'.format(self.Lx))
		print('\t    dz = {} m;  dx = {} m'.format(self.dz, self.dx))
		print('\t surface temperature: {} K'.format(self.Tsurf))
		print('\t bottom temperature:  {} K'.format(self.Tbot))
		print('\t boundary conditions:')
		print('\t    top:     {}'.format(stringBC(self.topBC)))
		print('\t    bottom:  {}'.format(stringBC(self.botBC)))
		print('\t    sides:   {}'.format(stringBC(self.sidesBC)))
		print('\t sources/sinks:')
		print('\t    tidal heating:  {}'.format(stringIO(self.tidalheat)))
		print('\t    latent heat:    {}'.format(stringIO(self.latentheat)))
		print('\t tolerances:')
		print('\t    temperature:     {}'.format(self.Ttol))
		print('\t    liquid fraction: {}'.format(self.phitol))
		if self.issalt:
			print('\t    salinity:        {}'.format(self.Stol))
		print('\t thermal properties:')
		print('\t    ki(T):    {}'.format(stringIO(self.kT)))
		print('\t    ci(T):    {}'.format(stringIO(self.cpT)))
		print('\t vehicle:')
		try:
			self.vehicle_geom
			print(f'\t    width:             {self.vehicle_w} m')
			print(f'\t    length:            {self.vehicle_h} m')
			print(f'\t    initial depth:     {self.vehicle_d} m')
			print(f'\t    heat scheme:       {self.vehicle_heat_scheme}')
		except:
			pass
		print('\t    salinity: {}'.format(stringIO(self.issalt)))
		if self.issalt:
			print(f'\t       composition:    {self.composition}')
			print(f'\t       concentration:  {self.concentration}ppt')
		print('-------------------------')
		try:
			print('Requested outputs: {}'.format(list(self.outputs.transient_results.keys())))
		except AttributeError:
			print('no outputs requested')

	def solve_heat(self, nt, dt, print_opts=True, n0=0, move=1, use_pressure=0):
		"""
		Iteratively solve heat two-dimension heat diffusion problem with temperature-dependent conductivity of ice.
		Parameters:
			nt : int
				number of time steps to take
			dt : float
				time step, s
			print_opts: bool
				whether to call print_opts() function above to print all chosen options
			n0 : float
				use if not starting simulation from nt=0, generally used for restarting a simulation (see
				restart_simulation.py)
		Usage:
			Run simulation for 1000 time steps with dt = 300 s
				model.solve_heat(nt=1000, dt=300)
		"""
		self.use_pressure = use_pressure
		self.dt = dt
		start_time = _timer_.clock()
		self.num_iter = []
		if print_opts: self.print_all_options(nt)

		for n in range(n0, n0 + nt):
			TErr, phiErr = np.inf, np.inf
			T_last, phi_last = self.T.copy(), self.phi.copy()
			k_last, rhoc_last = self.k.copy(), self.rhoc.copy()
			iter_k = 0
			while (TErr > self.Ttol and phiErr > self.phitol):
				Tx, Tz = self.get_gradients(T_last)
				self.update_liquid_fraction(phi_last=phi_last)
				if self.issalt:
					self.remix_salt_melted_ice(phi_last=phi_last)
					self.update_salinity(phi_last=phi_last)
				self.update_volume_averages()
				self.Q = self.update_sources_sinks(phi_last=phi_last, T_last=T_last)
				self.vehicle_temp_scheme()
				self.T[1:-1, 1:-1] = T_last[1:-1, 1:-1] + Tx + Tz + self.Q * self.dt / rhoc_last[1:-1, 1:-1]
				self.apply_boundary_conditions(T_last, k_last, rhoc_last)

				if 'Constant T' in self.vehicle_heat_scheme:
					self.T[self.vehicle_geom] = self.vehicle_T
				elif 'RTG T' in self.vehicle_heat_scheme:
					self.T[self.RTGS] = self.vehicle_T

				TErr = (abs(self.T - T_last)).max()
				phiErr = (abs(self.phi - phi_last)).max()

				# kill statement when parameters won't allow solution to converge
				if iter_k > 2000:
					raise Exception('solution not converging')

				iter_k += 1
				T_last, phi_last = self.T.copy(), self.phi.copy()
				k_last, rhoc_last = self.k.copy(), self.rhoc.copy()

			# outputs here
			if move:
				moved = self.update_vehicle_position()
				if moved:
					try:
						self.outputs.get_results(self, n=n, extra=1)
					except AttributeError:
						pass
			self.num_iter.append(iter_k)
			self.model_time += self.dt

			"""
			if n % 10000 == -1:
				outdir = '/Users/chasechivers/Dropbox (GaTech)/VERNE/'
				CMAP = 'cividis'
				fig, (ax1, ax2) = plt.subplots(num=1, nrows=1, ncols=2, clear=True, sharex='all', sharey='all')
				fig.suptitle('t = {:0.02f}s'.format(self.model_time))
				#tmp = self.T.copy()
				#tmp[self.vehicle==1] = np.nan
				Tcm = ax1.pcolormesh(self.X, -self.Z, self.T, vmin=self.T_initial.min(),
				                     vmax=self.T[self.vehicle==0].max(), cmap=CMAP)
				ax1.set_title('$T$, K')
				ax1.set_ylabel('Depth, m')
				ax1.set_xlabel('$x$, m')
				fig.colorbar(Tcm, ax=ax1)

				#tmp = self.phi.copy()
				#tmp[self.vehicle == 1] = np.nan
				phicm = ax2.pcolormesh(self.X, -self.Z, self.phi-self.vehicle, vmin=0, vmax=1, cmap=CMAP)
				ax2.set_title('$\phi$')
				ax2.set_xlabel('$x$, m')
				fig.colorbar(phicm, ax=ax2)
				fig.tight_layout()
				#short_string = '{}{}_n={}.png'.format(outdir, 'verne_firsttry', str(n).zfill(len(str(nt))))
				#fig.savefig(short_string, format='png', dpi=100.)
				plt.pause(0.001)
			"""

			try:  # save outputs
				self.outputs.get_results(self, n=n)
			# this makes the runs incredibly slow and is really not super useful, but here if needed
			#   save_data(self, 'model_runID{}.pkl'.format(self.outputs.tmp_data_file_name.split('runID')[1]),
			#          self.outputs.tmp_data_directory, final=0)
			except AttributeError:  # no outputs chosen
				pass
			if self.vehicle_geom[0].max() == self.nz - 3:
				print("reached end of domain")
				self.run_time = _timer_.clock() - start_time
				return self.model_time

		# del T_last, phi_last, Tx, Tz, iter_k, TErr, phiErr

		self.run_time += _timer_.clock() - start_time
