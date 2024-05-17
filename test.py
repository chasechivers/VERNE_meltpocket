from VERNESim import VERNESim as Model
import numpy as np
import utility_funcs as uf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

D = 1
w = 1
Tc = 100
Th = 600
Tw = 274
Ti = 273.14
coords = 'xz'


def ffa(t, l, k):
	return 2 * l * (k * t) ** 0.5


def approx_lam(md, Ti, Tw):
	"""Approximation for trascendental equation to solve for the lambda inthe Stefan solution
	Source: http://www.math.utk.edu/~vasili/475/Handouts/4.PhChgbk.2.1.pdf"""
	Sti = md.cpi(md.T, 0) * (md.constants.Tm - Ti) / md.constants.Lf
	Stw = md.cpw(md.T, 0) * (Tw - md.constants.Tm) / md.constants.Lf
	print("   Sti =", Sti)
	print("   Stw =", Stw)
	if Sti < Stw:
		lam = 0.706 * Stw ** 0.5 * (1 - 0.21 * (0.5642 * Stw) ** (0.93 - 0.15 * Stw))
	else:
		lam = 0.706 * Sti ** 0.5 * (1 - 0.21 * (0.5642 * Stw) ** (0.93 - 0.15 * Sti))
	print("   lambda =", lam)
	return lam


def stefan(md, Ti, Tw):
	from scipy import optimize
	from scipy.special import erf, erfc
	Ki = md.ki(md.T, 0) / (md.rhoi(md.T, 0) * md.cpi(md.T, 0))
	Kw = md.kw(md.T, 0) / (md.rhow(md.T, 0) * md.cpw(md.T, 0))
	v = np.sqrt(Ki / Kw)
	Sti = md.cpi(md.T, 0) * (md.constants.Tm - Ti) / md.constants.Lf
	Stw = md.cpw(md.T, 0) * (Tw - md.constants.Tm) / md.constants.Lf
	print("   Sti =", Sti)
	print("   Stw =", Stw)
	if Sti < Stw:
		St = Stw
	elif Stw < Sti:
		St = Sti
	func = lambda x: np.sqrt(np.pi) * x * np.exp(x ** 2) * erf(x) - St
	# func = lambda x: St / (np.exp(x ** 2) * erf(x)) - x * np.sqrt(np.pi)
	# - v * md.kw(md.T, 0) * md.cpi(md.T, 0) * Stw \
	# / (md.ki(md.T, 0) * md.cpw(md.T, 0) * erfc(v * x) * np.exp(x ** 2 * v ** 2))

	lam = optimize.root(func, np.array([1]))['x'][0]
	print("   lambda =", lam)
	return lam


def topdown_melt_test():
	print("=" * 30)
	print("Topdown melt test")
	nx = 6
	nz = 201

	md = Model(w=w, D=D, nx=nx, nz=nz, coordinates="xz")
	md.ki = lambda T, S: md.constants.ki
	md.dkidT = lambda T, S: 0
	md.rhoi = lambda T, S: md.constants.rho_i
	md.cpi = lambda T, S: md.constants.cp_i
	md.kv = lambda T: md.constants.ki

	md.dkvdT = lambda T: 0
	md.rhov = lambda T: md.constants.rho_i
	md.cpv = lambda T: md.constants.cp_i
	md.Tm = md.constants.Tm * np.ones(md.T.shape)
	K = md.ki(md.T, 0) / (md.rhoi(md.T, 0) * md.cpi(md.T, 0))
	dt = min([md.dx, md.dz]) ** 2 / K / 12

	md.T[:1, :] = Th
	md.T[1:, :] = Ti
	md.phi[:1, :] = 1.
	md.phi[1:, :] = 0.
	md.init_vehicle(depth=0, length=0, radius=0, constat_T=Th)
	md._save_initials()
	md.set_boundaryconditions(sides="NoFlux")
	md.STOP_AT_BOTTOM = False
	print(" Root finding solution")
	stf = stefan(md, Tc, Tw)
	print(" Approximate solution")
	stf = 0.5 * (stf + approx_lam(md, Tc, Tw))
	Nt = int((D / 2 / stf) ** 2 / K / dt) // 10

	md.set_outputs(10, './tmp/', outlist=['phi', 'T'])
	md.outputs.tmp_data_directory = './tmp/'

	print("   starting", Nt, "steps")
	pt = np.zeros(Nt)
	for n in range(Nt):
		md.solve_heat(nt=1, dt=dt, move=False, save_progress=False)
		pt[n] = md.Z[np.where(md.phi == 0)[0].min(), 0] - md.Z[1, 0]
	md.outputs.results = md.outputs.get_all_data(del_files=True)
	print("   solved in", md.run_time, "s")
	md.pt = pt
	md.stf = stf
	t = np.linspace(0, md.model_time, len(md.pt))
	C, _ = curve_fit(ffa, t, md.pt)
	print(f"  Numerical solution:\n   lambda = {C[0]}\n   K = {C[1]}")
	print("=" * 30)
	return md


def topdown_freeze_test():
	print("=" * 30, "\n Left freeze test")
	nx = 6
	nz = 201

	md = Model(w=w, D=D, nx=nx, nz=nz, coordinates="xz")
	md.ki = lambda T, S: md.constants.ki
	md.dkidT = lambda T, S: 0
	md.rhoi = lambda T, S: md.constants.rho_i
	md.cpi = lambda T, S: md.constants.cp_i
	md.kv = lambda T: md.constants.ki
	md.dkvdT = lambda T: 0
	md.rhov = lambda T: md.constants.rho_i
	md.cpv = lambda T: md.constants.cp_i
	md.Tm = md.constants.Tm * np.ones(md.T.shape)
	K = md.ki(md.T, 0) / (md.rhoi(md.T, 0) * md.cpi(md.T, 0))
	dt = min([md.dx, md.dz]) ** 2 / K / 12

	md.T[:1, :] = Tc
	md.T[1:, :] = Tw
	md.phi[:1, :] = 0.
	md.phi[1:, :] = 1.
	md.init_vehicle(depth=0, length=0, radius=0, constant_T=Tc)
	md._save_initials()
	md.set_boundaryconditions(sides="NoFlux")
	md.STOP_AT_BOTTOM = False

	print(" Root finding solution")
	stf = stefan(md, Tc, Tw)
	print(" Approximate solution")
	stf = 0.5 * (stf + approx_lam(md, Tc, Tw))
	Nt = int((D / 2 / stf) ** 2 / K / dt) // 10

	md.set_outputs(10, './tmp/', outlist=['phi', 'T'])
	md.outputs.tmp_data_directory = './tmp/'

	print("  starting", Nt, "steps")
	pt = np.zeros(Nt)
	for n in range(Nt):
		md.solve_heat(nt=1, dt=dt)
		pt[n] = md.Z[np.where(md.phi == 0)[0].max(), 1]
	print("  solved in", md.run_time, "s")
	md.outputs.results = md.outputs.get_all_data(del_files=True)
	md.pt = pt
	md.stf = stf
	t = np.linspace(0, md.model_time, len(md.pt))
	C, _ = curve_fit(ffa, t, md.pt)
	print(f"  Numerical solution:\n   lambda = {C[0]}\n   K = {C[1]}")

	print("=" * 30)
	return md


def left_freeze_test():
	print("=" * 30, "\n Left freeze test")
	nx = 401
	nz = 3

	md = Model(w=w, D=D, nx=nx, nz=nz, coordinates="zr")
	md.ki = lambda T, S: md.constants.ki
	md.dkidT = lambda T, S: 0
	md.rhoi = lambda T, S: md.constants.rho_i
	md.cpi = lambda T, S: md.constants.cp_i
	md.kv = lambda T: md.constants.ki
	md.dkvdT = lambda T: 0
	md.rhov = lambda T: md.constants.rho_i
	md.cpv = lambda T: md.constants.cp_i
	md.Tm = md.constants.Tm * np.ones(md.T.shape)
	K = md.ki(md.T, 0) / (md.rhoi(md.T, 0) * md.cpi(md.T, 0))
	dt = min([md.dx, md.dz]) ** 2 / K / 12

	md.T[:, :1] = Tc
	md.T[:, 1:] = Tw
	md.phi[:, :1] = 0.
	md.phi[:, 1:] = 1.
	md.init_vehicle(depth=0, length=0, radius=0, constant_T=Tc)
	md._save_initials()

	md.set_boundaryconditions(top="NoFlux", bottom="NoFlux", sides=True)
	md.STOP_AT_BOTTOM = False

	print(" Root finding solution")
	stf = stefan(md, Tc, Tw)
	print(" Approximate solution")
	stf = 0.5 * (stf + approx_lam(md, Tc, Tw))
	Nt = int((D / 2 / stf) ** 2 / K / dt) // 10

	md.set_outputs(50, './tmp/', outlist=['phi', 'T'])
	md.outputs.tmp_data_directory = './tmp/'

	print("   starting", Nt, "steps")
	pt = np.zeros(Nt)
	for n in range(Nt):
		md.solve_heat(nt=1, dt=dt, save_progress=False, move=False)
		pt[n] = md.X[0, np.where(md.phi == 0)[1].max()]
	print("   solved in", md.run_time, "s")
	md.outputs.results = md.outputs.get_all_data(del_files=True)
	md.pt = pt
	md.stf = stf
	t = np.linspace(0, md.model_time, len(md.pt))
	C, _ = curve_fit(ffa, t, md.pt)
	print(f"  Numerical solution:\n   lambda = {C[0]}\n   K = {C[1]}")
	print("=" * 30)
	return md

def left_melt_test():
	print("=" * 30, "\n Left melt test")
	nx = 401
	nz = 3

	md = Model(w=w, D=D, nx=nx, nz=nz, coordinates="zr")
	md.ki = lambda T, S: md.constants.ki
	md.dkidT = lambda T, S: 0
	md.rhoi = lambda T, S: md.constants.rho_i
	md.cpi = lambda T, S: md.constants.cp_i
	md.kv = lambda T: md.constants.ki
	md.dkvdT = lambda T: 0
	md.rhov = lambda T: md.constants.rho_i
	md.cpv = lambda T: md.constants.cp_i
	md.Tm = md.constants.Tm * np.ones(md.T.shape)
	K = md.ki(md.T, 0) / (md.rhoi(md.T, 0) * md.cpi(md.T, 0))
	dt = min([md.dx, md.dz]) ** 2 / K / 12

	md.T[:, :1] = Th
	md.T[:, 1:] = Ti
	md.phi[:, :1] = 1.
	md.phi[:, 1:] = 0.
	md.init_vehicle(depth=0, length=0, radius=0, constant_T=Th)
	md._save_initials()

	md.set_boundaryconditions(top="NoFlux", bottom="NoFlux", sides=True)
	md.STOP_AT_BOTTOM = False

	print(" Root finding solution")
	stf = stefan(md, Ti, Th)
	print(" Approximate solution")
	stf = 0.5 * (stf + approx_lam(md, Ti, Th))
	Nt = int((D / 2 / stf) ** 2 / K / dt) // 10

	md.set_outputs(100, './tmp/', outlist=['phi', 'T'])
	md.outputs.tmp_data_directory = './tmp/'

	print("   starting", Nt, "steps")
	pt = np.zeros(Nt)
	for n in range(Nt):
		md.solve_heat(nt=1, dt=dt, save_progress=False, move=False)
		pt[n] = md.X[0, np.where(md.phi == 1)[1].max()]
	print("   solved in", md.run_time, "s")
	md.outputs.results = md.outputs.get_all_data(del_files=True)
	md.pt = pt
	md.stf = stf
	t = np.linspace(0, md.model_time, len(md.pt))
	C, _ = curve_fit(ffa, t, md.pt)
	print(f"  Numerical solution:\n   lambda = {C[0]}\n   K = {C[1]}")
	print("=" * 30)
	return md

#md = topdown_freeze_test()
#md = left_freeze_test()
#md = topdown_melt_test()
md = left_melt_test()
Ki = md.ki(md.T, 0) / (md.rhoi(md.T, 0) * md.cpi(md.T, 0))
Kw = md.kw(md.T, 0) / (md.rhow(md.T, 0) * md.cpw(md.T, 0))
t = np.linspace(0, md.model_time, len(md.pt))

plt.fill_between(t, ffa(t, 1.1 * md.stf, Kw), ffa(t, 0.9 * md.stf, Kw), alpha=0.6)
plt.plot(t, md.pt)
