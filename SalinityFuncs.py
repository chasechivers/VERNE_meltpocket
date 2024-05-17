import numpy as np
import SalinityConstants

# Define a bunch of useful functions for salty systems, unused otherwise

class SalinityFuncs:

	def __init__(self):
		pass

	# Constitutive equation fits that relate temperature gradient at time of freezing (i.e. freeze rate) to entrained
	# salt content. All are from Buffo et al, 2020a unless specified.
	def shallow_fit(self, dT, a, b, c, d):
		"""
		Constitutive equation for relating thermal gradient at time of freezing to entrained bulk salinity in
		the ice. Formulation derived in Buffo et al., 2020a. Constants a,b,c,d are given for a particular composition in SalinityConstants.py

		This specific formulation is for high thermal gradients (defined by the intersection of shallow_fit and
		linear_fit).
		:param dT: float, arr
			Thermal gradient at time of freezing, K/m
		:param a: float, int
		:param b: float, int
		:param c: float, int
		:param d: float, int
		:return:
			Bulk salinity, ppt
		"""
		return a + b * (dT + c) * (1 - np.exp(-d / dT)) / (1 + dT)


	def linear_fit(self, dT, a, b):
		"""
		Constitutive equation for relating thermal gradient at time of freezing to entrained bulk salinity in
		the ice. Formulation derived in Buffo et al., 2020a. Constants a,b,c,d are given for a particular composition in SalinityConstants.py

		This specific formulation is for small thermal gradients (defined by the intersection of shallow_fit and
		linear_fit).
		:param dT: float, arr
			Thermal gradient at time of freezing, K/m
		:param a: float, int
		:param b: float, int
		:param c: float, int
		:param d: float, int
		:return:
			Bulk salinity, ppt
		"""
		return a + b * dT


	def new_fit(self, dT, a, b, c, d, f, g, h):
		"""
		Constitutive equation for relating thermal gradient at time of freezing to entrained bulk salinity in
		the ice. Formulation derived in Buffo et al., in review. Constants a,b,c,d,f,g,h are given for a particular
		composition in SalinityConstants.py

		Thus far this is only used for sodium chloride (NaCl) and will work for all thermal gradients
		:param dT: float, arr
			Thermal gradient at time of freezing, K/m
		:param a: float, int
		:param b: float, int
		:param c: float, int
		:param d: float, int
		:param f: float, int
		:param g: float, int
		:param h: float, int
		:return:
			Bulk salinity, ppt

		Usage:
			S = model.entrain_salt(dT, *model.new_fit_consts[concentration])
		"""
		return a + b * (dT + c) * (1 - g * np.exp(-h / dT)) / (d + f * dT)


	def salinity_with_depth(self, z, a, b, c):
		"""
		Method for a salinity with depth profile via Buffo et al. 2020. Note that z is depth from surface where 0 is
		the surface and +D is the total shell thickness
		:param z: float, arr
			Positive depths from surface in meters
		:param a: float
		:param b: float
		:param c: float
		:return:
			Bulk salinity in ice with depth, K

		Usage:
			S = model.salinity_with_depth(model.Z, *model.depth_consts)
		"""
		return a + b / (c - z)

	def entrain_salt(self, dT, S, dTMAX=100):
		if isinstance(dT, (int, float)):
			if dT >= dTMAX:
				return S
			else:
				if S in self.concentrations:
					if self.composition in SalinityConstants.new_fit_consts.keys():
						ans = self.new_fit(dT, *self.new_fit_consts[S])
						return S if ans > S else ans
					elif self.composition in (SalinityConstants.linear_consts.keys() and SalinityConstants.shallow_consts.keys()):
						ans = self.linear_fit(dT, *self.linear_consts[S]) if dT <= self.linear_shallow_roots[S] \
							else self.shallow_fit(dT, *self.shallow_consts[S])
						return S if ans > S else ans
					else:
						raise Exception("Salt not included")

				else:
					# interpolation & extrapolation steps
					## interpolation: choose the sa
					c_min = self.concentrations[S > self.concentrations].max() if S <= self.concentrations.max() else \
						self.concentrations[-2]
					c_max = self.concentrations[S < self.concentrations].min() if S <= self.concentrations.max() else \
						self.concentrations[-1]

					# linearly interpolate between the two concentrations at gradient dT
					m, b = np.polyfit([c_max, c_min],
					                  [self.entrain_salt(dT, c_max), self.entrain_salt(dT, c_min)],
					                  1)

					# return concentration of entrained salt
					ans = m * S + b
					return S if ans > S else ans
		else:
			return np.array([self.entrain_salt(t, s) for t, s in zip(dT, S)], dtype=float)