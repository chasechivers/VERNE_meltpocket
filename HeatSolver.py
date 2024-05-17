from Outputs import Outputs
import time as _timer_
import numpy as np

class HeatSolver:
	def __init__(self):
		"""Set default simulation parameters"""
		self.TIDAL_HEAT = True  # tidal heating source
		self.STOP_AT_BOTTOM = True

		# the Courant-Fredrichs-Lewy condition when choosing a time step: dt = CFL * min(dx,dz)^2 / max(diffusivity)
		#  where diffusivity = conductivity / density / specific heat,
		#       thus max(diffusivity) = min(density * specific heat) / max(conductivity)
		#  this should be much smaller for colder surface temperatures (i.e. CFL ~ 1/50 for Tsurf ~ 50 K)
		self.CFL = 1 / 24.
		self.ADAPT_DT = True

		self.TTOL = 0.1  # temperature tolerance
		self.PHITOL = 0.01  # liquid fraction tolerance
		self.STOL = 1  # salinity tolerance

		self.saturation_time = 0
		self.num_iter = []
		self.ITER_KILL = 2000


	def set_outputs(self, output_frequency: int, tmp_dir: str, outlist=["T", "phi", "vehicle"]):
		self.outputs = Outputs(OF=output_frequency, tmpdir=tmp_dir)
		self.outputs.choose(outlist, self.issalt)

	def set_boundaryconditions(self, top=True, bottom=True, sides=True, **kwargs):
		"""
		Choose the boundary conditions
		:param top:
		:param bottom:
		:param sides:
		:param kwargs:
		:return:
		"""
		self.topBC = top
		self.botBC = bottom
		self.sidesBC = sides

		if bottom is True:
			self.TbotBC = self.T_initial[-1, :].copy()

		if top is True:
			self.TtopBC = self.T_initial[0, :].copy()

		if top == "Flux":
			try:
				self.T_top_ghost = kwargs["T_top"]
			except:
				raise Exception("Temperature for ghost cells at top is required\n-> e.g. model.set_boundaryconditions("
				                "top='Flux', T_top=200)")

		if sides == True:
			if self.symmetric: print("WARNING! Constant vertical boundary conditions NOT RECOMMENDED for "
			                         "horizontally-symmetric assumptions. Please use 'RFlux' or 'LNoFlux'")

		if sides == "RFlux":
			try:
				self.dL = kwargs["dL"]
			except:
				raise Exception("Length for right side flux not chosen\n\t->model.set_boundaryconditions("
				                "sides='RFlux', dL=500e3)")
		if sides == "LNoFlux":
			pass

	def _update_sources_sinks(self, phi_last, T, Tm):
		latent_heat = self.constants.rho_i * self.constants.Lf * (self.phi - phi_last) / self.dt
		return - latent_heat

	def _apply_boundaryconditions(self, T, k, rhoc):
		if self.botBC == True:
			self.T[-1, :] = self.T_initial[-1, :].copy()
		elif self.botBC == "NoFlux":
			self.T[-1, :] = self.T[-2, :].copy()

		if self.topBC == True:
			self.T[0, :] = self.T_initial[0, :].copy()

		elif self.topBC == "NoFlux":
			self.T[0, :] = self.T[1, :].copy()

		elif self.topBC == "Flux":
			C = self.dt / rhoc[0, :]
			# T_1,j - T_-1,j / 2 dz = c => T_-1,j  = T_1,j - 2*dz*c
			Ttop_z = self._conductivity_average(k[0, :], k[1, :]) * (T[1, :] - T[0, :]) \
					- k[0, :] * (T[0,:] - (T[1,:] - 2 * self.dz * self.T_top_ghost))
			         #- self._conductivity_average(k[0, :], self.ki(self.T_top_ghost, 0)) * \(T[0, :] - self.T_top_ghost)
			Ttop_z *= C / self.dz ** 2

			Ttop_x = self._conductivity_average(k[0, 2:], k[0, 1:-1]) * (T[0, 2:] - T[0, 1:-1]) \
			         - self._conductivity_average(k[0, 1:-1], k[0, :-2]) * (T[0, 1:-1] - T[0, :-2])
			Ttop_x *= C[1:-1] / self.dx ** 2

			self.T[0, 1:-1] = T[0, 1:-1] + Ttop_z[1:-1] + Ttop_x + self.Q[0, 1:-1] * self.dt / rhoc[0, 1:-1]

		if self.sidesBC == True:
			self.T[:, 0] = self.T_initial[:, 0].copy()
			self.T[:, -1] = self.T_initial[:, -1].copy()

		elif self.sidesBC == "NoFlux":
			self.T[:, 0] = self.T[:, 1].copy()
			self.T[:, -1] = self.T[:, -2].copy()

		elif self.sidesBC == "LNoFlux":
			self.T[:, 0] = T[:, 1].copy()
			self.T[:, -1] = self.T_initial[:, -1].copy()

		elif self.sidesBC == "RFlux":
			self.T[:, 0] = T[:, 1].copy()

			C = self.dt / rhoc
			Tright_x = self._conductivity_average(k[1:-1, -1],self.ki(self.T_initial[1:-1, -1], self.S_initial[1:-1, -1])) \
			           * (self.T_initial[1:-1, -1] - T[1:-1, -1]) * self.dx / self.dL \
			           - self._conductivity_average(k[1:-1, -1], k[1:-1, -2]) * (T[1:-1, -1] - T[1:-1, -2])
			# Tright_x *= C[1:-1, -1] / self.dx**2

			Tright_z = self._conductivity_average(k[2:, -1], k[1:-1, -1]) * (T[2:, -1] - T[1:-1, -1]) \
			           - self._conductivity_average(k[1:-1, -1], k[:-2, -1]) * (T[1:-1, -1] - T[:-2, -1])

			# Tright_z *= C[1:-1, -1] / self.dz ** 2

			self.T[1:-1, -1] = T[1:-1, -1] + C[1:-1, -1] * (Tright_z / self.dz ** 2 + Tright_x / self.dx ** 2 \
			                                                + self.Q[1:-1, -1])

	def _vehicle_temperature_scheme(self):
		self.T[self.vehicle_geom] = self.vehicle_T

	def _remix_salt_melted_ice(self, phi_last, water, volume):
		#loc = np.where((phi_last < 1) & (self.phi == 1))
		#print(loc)
		#if len(loc[0]) > 0:
		self.S[water] = self.S[water].sum() / volume
		#else:
		#	pass

	def _update_salinity(self, phi_last):
		# find indices where ice has just formed
		z_newice, x_newice = np.where((phi_last > 0) & (self.phi == 0))
		# get indices of cells that can accept rejected salts
		water = np.where(self.phi >= self.rejection_cutoff)
		# calculate "volume" of cells that can accept rejected salt
		volume = water[1].shape[0]
		self._remix_salt_melted_ice(phi_last, water, volume)
		# start catalogue of TOTAL salt removed from system after water is saturated
		self.removed_salt.append(0)

		# if new ice has formed entrain solute into ice from parent liquid
		if len(z_newice) > 0 and volume != 0:
			S_new = self.S.copy()  # copy salinity field into new grid
			for i in range(len(z_newice)):  # iterate over newly frozen cells
				# save location i,j of newly frozen ice
				loc = (z_newice[i], x_newice[i])
				# get starting salinity at location ij
				Sij_old = self.S[loc]
				# calculate centered thermal gradients across cell ij
				## centered: (T_(i,j-1) - T_(i,j+1))/2dx
				if self.symmetric and loc[1] in [0, self.nx - 1]:
					grad_x = 0
				else:
					grad_x = abs(self.T[loc[0], loc[1] - 1] - self.T[loc[0], loc[1] + 1]) / 2. / self.dx

				grad_z = (self.T[loc[0] - 1, loc[1]] - self.T[loc[0] + 1, loc[1]]) / 2. / self.dz

				# brine drainage parameterization, assumes solution is more dense than pure water
				## if thermal gradient in z-direction is > 0 (bottom-up freezing) => no drainage, solute stays
				if grad_z > 0:
					S_new[loc] = Sij_old
				## if thermal gradient in z-direction is < 0 (top-down freezing) => brine drains, reject salt
				elif grad_z < 0:
					# determine "mean" temperature gradient across cell
					## use arithmetic mean
					grad = (abs(grad_z) + abs(grad_x)) / 2

					S_new[loc] = self.entrain_salt(grad, Sij_old)

			if self.salinate:
				S_new[water] = S_new[water] + abs(self.S.sum() - S_new.sum()) / volume

				self.removed_salt[-1] += (S_new[S_new >= self.saturation_point] - self.saturation_point).sum()

			self.S = S_new

		# check for salt conservation at each time step
		if self.salinate:
			total_S_new = self.S.sum() + sum(self.removed_salt)
			self.total_salt.append(total_S_new)
			if abs(total_S_new - self.total_salt[0]) <= self.STOL:
				pass  # -ed the test!!
			else:
				raise Exception("Salt mass not being conserved")

	def _move_vehicle_down_dz(self, w_idx, h_idx, b_idx, Z_idxs, X_idxs):
		# Ice in front of vehicle has melted enough
		#T_save = self.T[md.vehicle_geo].copy()
		#phi_save = self.phi[Z_idxs + 1, X_idxs].copy()

		# Move vehicle down one vertical step size dz
		self.vehicle[Z_idxs + 1, X_idxs] = 1  # melt in front of vehicle
		# Behind the vehicle
		self.vehicle[b_idx, X_idxs] = 0

		self.vehicle_geom = np.where(self.vehicle == 1)
		#self.T[self.vehicle_geom] = T_save
		#self.phi[self.vehicle_geom] = phi_save

		self._vehicle_temperature_scheme()

	def _update_vehicle_position(self):
		ans = False
		w_idx = self.vehicle_geom[1].max()  # index of vehicle width
		h_idx = self.vehicle_geom[0].max()  # index for vehicle front
		b_idx = self.vehicle_geom[0].min()  # index for vehicle back

		if self.symmetric:
			X_idxs = np.array([i for i in range(0, w_idx + 1)], dtype=int)
		else:
			raise Exception("update_vehicle_position not written for non-symmetric")
		Z_idxs = np.array([h_idx for i in range(len(X_idxs))], dtype=int)

		# Check for thermal descent first
		if self.phi[Z_idxs + 1, X_idxs].sum() >= len(X_idxs) * self.melt_mod:
			self._move_vehicle_down_dz(w_idx, h_idx, b_idx, Z_idxs, X_idxs)
			if self.verbose:
				dnose = self.Z[Z_idxs[-1] + 1, X_idxs[0]]
				init_nose = self.vehicle_d + self.vehicle_h
				v = (dnose - init_nose) * 3.154e4 / self.model_time
				print(f"-> Vehicle moved down:\n   t = {self.model_time / 3600:0.03f} hr, v = {v:0.03f} km/yr")
			ans = True

		# Check for mechanical descent
		if self.mechanical_descent and ans == False:
			if self.model_time % (self.dz / self.descent_speed) == 0 and self.model_time > 0:
				self._move_vehicle_down_dz(w_idx, h_idx, b_idx, Z_idxs, X_idxs)

				if self.verbose:
					dnose = self.Z[Z_idxs[-1] + 1, X_idxs[0]]
					init_nose = self.vehicle_d + self.vehicle_h
					v = (dnose - init_nose) * 3.154e4 / self.model_time
					print(f"-> Vehicle moved down:\n   t = {self.model_time / 3600:0.03f} hr, v = {v:0.03f} km/yr")

				ans = True

		# adjust "new" material behind the vehicle?
		if ans == True:
			pass

		return ans

	def _update_liquid_fraction(self, phi_last):
		"""
		Updates the phase at each time step
		:param phi_last:
		:return:
		"""
		if self.issalt:
			self.Tm = self.Tm_func(self.S, *self.Tm_consts)

		# calculate enthalpies
		Hs = self.cpi(self.T, self.S) * self.Tm
		Hl = Hs + self.constants.Lf * ((1 - self.p) if "p" in self.__dict__ else 1)
		H = self.cpi(self.T, self.S) * self.T + self.constants.Lf * phi_last

		self.phi[H >= Hs] = (H[H >= Hs] - Hs[H >= Hs]) / self.constants.Lf
		self.phi[H <= Hl] = (H[H <= Hl] - Hs[H <= Hl]) / self.constants.Lf
		# all ice
		self.phi[H < Hs] = 0.
		# all water
		self.phi[H > Hl] = 1.

	def _get_gradients(self, T, k, rhoc, Tll=None):
		"""
		Calculates the flux-conservative thermal gradients (∇q = ∇(k∇T)) for a multi-phase system with different
		thermophysical properties and a temperature-dependent thermal conductivity. Internal use only

		:param T: (nx,nz) array
			(k-1) iteration or current time step of temperature grid
		:param k: (nx,nz) array
			Phase and temperature-dependent thermal conductivity grid
		:param rhoc:
			Phase and temperature-dependent denisty * specific heat grid
		:return:

		"""

		if self.coords in ["cartesian", "zx", "xz"]:
			dqx = self._conductivity_average(k[1:-1, 2:], k[1:-1, 1:-1]) * (T[1:-1, 2:] - T[1:-1, 1:-1]) / self.dx ** 2 \
			      - self._conductivity_average(k[1:-1, 1:-1], k[1:-1, :-2]) * (T[1:-1, 1:-1] - T[1:-1, :-2]) / self.dx ** 2


		elif self.coords in ["cylindrical", "zr", "rz"]:
			# Must account for curvature in a cylindrical system
			#  see pg. 252 in Langtangen & Linge (2017) "Finite Difference Computing with PDEs"
			dqx = (self._rph * self._conductivity_average(k[1:-1, 2:], k[1:-1, 1:-1]) * (T[1:-1, 2:] - T[1:-1, 1:-1])
			       - self._rmh * self._conductivity_average(k[1:-1, 1:-1], k[1:-1, :-2]) * (T[1:-1, 1:-1] - T[1:-1,:-2])
			       ) / self.X[1:-1, 1:-1] / self.dx ** 2

		dqz = self._conductivity_average(k[2:, 1:-1], k[1:-1, 1:-1]) * (T[2:, 1:-1] - T[1:-1, 1:-1]) / self.dz**2 \
		     - self._conductivity_average(k[1:-1, 1:-1], k[:-2, 1:-1]) * (T[1:-1, 1:-1] - T[:-2, 1:-1]) / self.dz**2

		return dqz + dqx

	def _error_check(self):
		"""
		Checks whether simulation is within assumptions.
		"""
		if np.isnan(self.T).any() or np.isnan(self.S).any() or np.isnan(self.phi).any() or np.isnan(self.vehicle).any():
			raise Exception("Something went wrong... Check time step? may be too large")
		if (self.T[self.T <= 0].shape[0] or self.T[self.T > self.T_initial.max()].shape[0]) > 0:
			raise Exception("Unphysical temperatures. Check time step, may be too large")
		if (self.phi[self.phi < 0].shape[0] or self.phi[self.phi > 1].shape[0]) > 0:
			raise Exception("Unphysical phase: 0 <= phi <= 1")
		if (self.vehicle[self.vehicle < 0] or self.vehicle[self.vehicle > 1].shape[0]) > 0:
			raise Exception("Something went wrong with vehicle stuff")
		if self.issalt:
			if (self.S[self.S < 0].shape[0] or self.S[self.S > self.saturation_point].shape[0]) > 0:
				raise Exception("Unphysical salt content. Check time step, may be too large")

	def time_step(self, move, stop_idx):
		TErr, phiErr = np.inf, np.inf
		T_last, phi_last = self.T.copy(), self.phi.copy()
		rhoc_last = self.rhoc(phi_last, self.vehicle, T_last, self.S)
		k_last = self.k(phi_last, self.vehicle, T_last, self.S)
		iter_k = 0
		if self.coords in ["cylindrical", "polar", "zr", "rz"]:
			Tll = T_last.copy()
		while (TErr > self.TTOL and phiErr > self.PHITOL):
			# calculate thermal flux gradient
			## ∇q = ∇k * (∇T)
			dflux = self._get_gradients(T_last, k_last, rhoc_last,
			                            None if self.coords in ["cartesian", "zx", "xz"] else Tll)
			self._update_liquid_fraction(phi_last=phi_last)
			if self.issalt:
				self._update_salinity(phi_last=phi_last)

			rhoc = self.rhoc(self.phi, self.vehicle, self.T, self.S)

			k = self.k(self.phi, self.vehicle, self.T, self.S)

			self.Q = self._update_sources_sinks(phi_last=phi_last, T=T_last, Tm=self.Tm)
			#self._vehicle_temperature_scheme()
			self.T[1:-1, 1:-1] = T_last[1:-1, 1:-1] + self.dt * dflux / rhoc_last[1:-1,1:-1]
			self.T += self.Q * self.dt / rhoc_last
			self._vehicle_temperature_scheme()

			self._apply_boundaryconditions(T_last, k_last, rhoc_last)

			TErr = (abs(self.T - T_last)).max()
			phiErr = (abs(self.phi - phi_last)).max()

			# kill statement when parameters won't allow solution to converge
			if iter_k > self.ITER_KILL:
				raise Exception(f"Solution not converging,\n\t iterations = {iter_k}\n\t T error = {TErr}\n\t phi "
				                f"error = {phiErr}")
			self._error_check()

			iter_k += 1
			T_last, phi_last = self.T.copy(), self.phi.copy()
			# k_last, rhoc_last = self.k.copy(), self.rhoc.copy()
			rhoc_last = rhoc.copy()
			k_last = k.copy()

		self.num_iter.append(iter_k)
		# outputs here
		if move:
			# TO DO: make a way to choose different stop conditions like steady-state descent for x number of timesteps
			if self.vehicle_geom[0].max() == self.nz - stop_idx:
				#print("Reached end of domain, stopping simulation")
				#print(f"  Simulation time: {self.model_time / 3600:0.03f} hrs")
				#print(f"  Run time       : {self.run_time / 60:0.03f} min")
				return
			moved = self._update_vehicle_position()
			# move_output_counter += 1
			if moved:
				try:
					self.outputs.get_results(self, n=int(self.model_time/self.dt), extra=True)
				except AttributeError:
					pass
		else:
			try:
				self.outputs.get_results(self, n=int(self.model_time / self.dt))
			except AttributeError:  # no outputs chosen
				pass
		# try:
		#	if self.Z.shape[0] > 1000 and move_output_counter % large_out == 0:
		#		self.outputs.get_results(self, n=n, extra=1)
		#	elif self.Z.shape[0] < 1000:
		#		self.outputs.get_results(self, n=n, extra=1)
		# except AttributeError:
		#		pass

		self.model_time += self.dt

	def solve_heat(self, nt, *, dt=None, move=True, stop_idx=10, save_progress=10):

		if dt is None and self.ADAPT_DT:
			self._set_dt(self.CFL)
		else:
			pass

		n0 = 0 if self.model_time == 0 else int(self.model_time / self.dt)
		for n in range(n0, n0 + nt):
			start_time = _timer_.time()
			if self.ADAPT_DT:
				self._set_dt(self.CFL)
			self.time_step(move=move, stop_idx=stop_idx)

			if self.STOP_AT_BOTTOM:
				if self.vehicle_geom[0].max() >= self.nz - stop_idx:
					print("Reached end of domain, stopping simulation")
					print(f"  Simulation time: {self.model_time / 3600:0.03f} hrs")
					print(f"  Run time       : {self.run_time / 60:0.03f} min")
					return
			self.run_time += _timer_.time() - start_time  # - self.run_time