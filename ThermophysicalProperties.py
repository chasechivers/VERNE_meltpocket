class ThermophysicalProperties:
	"""
	Class instance to hold all of the thermophysical properties of the phases and the vehicle used in a simulation.
	"""
	def ki(self, T, S):
		ans = self.constants.ac / T if self.kT else self.constants.ki
		return ans

	def dkidT(self, T, S):
		ans = -self.constants.ac / T**2 if self.kT else 0
		return ans

	def kw(self, T, S):
		return self.constants.kw

	def dkwdT(self, T, S):
		return 0.

	def kv(self, T):
		return self.constants.kv

	def dkvdT(self, T):
		return 0.

	def _conductivity_average(self, k1, k2):
		"""
		Averages the thermal conductivity across adjacent cells for flux conservation. Generally for internal use
		during simulation.
		:param k1: arr, float
			Thermal conductivity
		:param k2: arr, float
			Thermal conductivity
		:return:
			Mean of thermal conductivity
		"""
		# Harmonic mean
		#return 2 * k1 * k2 / (k1 + k2)
		# Arithmetic
		return 0.5 * (k1 + k2)

	def cpi(self, T, S):
		return self.constants.cp_i

	def cpw(self, T, S):
		"""
		if "composition" in self.__dict__:
			if self.composition == "NaCl":
				return 4190 - 7.114 * S + 0.017 * S ** 2
			elif self.composition == "MgSO4":
				return -5.528301886792441 * S + 4228.773584905658
		else:
			return self.constants.cp_wp
		"""
		return self.constants.cp_w

	def cpv(self, T):
		return self.constants.cp_v

	def rhoi(self, T, S):
		return self.constants.rho_i + self.constants.Ci_rho * S

	def rhow(self, T, S):
		return self.constants.rho_w + self.constants.C_rho * S

	def rhov(self, T):
		return self.constants.rho_v

	def Tm_func(self, S, a, b, c):
		"""
		Melting temperature of water based on the concentration of major salt. Assumes second order polynomial fit to
		FREEZCHEM derived melting temperature (Tm) as a function of salinity (in ppt).
		Note: This may be modified to any fit (3rd+ order polynomial, exponential, etc.), just make sure to modify
		constants in SalinityConstants.py for concentration as well as the return (i.e. instead write: return a*np.exp(
		-b*S) + c)
		:param S: float, arr
			A float/int or array of salinities in ppt
		:param a: float
			Polynomial coefficient
		:param b: float
			Polynomial constants
		:param c: float
			Polynomial constants
		:return:
			Melting temperature at salinity S
		"""
		return a * S ** 2 + b * S + c
