from System import IceSystem
from utility_funcs import save_data, load_data
import numpy as np

# choose a directory to output results files
outdir = "./results/"

rs_dir = "./results/"
rs_file = rs_dir+"tmp_data_runID9536MgSO4282.0_1L_chunk=10_Z0=9470.0_n=4924800.pkl"

def scrape_from_filename(filename):
	if "/" in filename: filename = filename.split("/")[-1]
	directory, _, runIDsalt, layers, chunk_num, Z0, n = filename.split("_")
	directory = directory[:-3]

	runID = runIDsalt[:9]


	if "MgSO4" or "NaCl" in runIDsalt:
		if "MgSO4" in runIDsalt:
			salt_idx = runIDsalt.find("4")
		else:
			salt_idx = runIDsalt.find("l")
		salt = runIDsalt[9:salt_idx + 1]
		concentration = runIDsalt[salt_idx + 1:]

		chunk_num = chunk_num[chunk_num.find("=")+1:]
		Z0 = Z0[Z0.find("=") + 1:]
		n = n[n.find("=")+1:-4]

		return runID, salt, float(concentration), layers, int(chunk_num), float(Z0), int(n)
	else:

		chunk_num = chunk_num[chunk_num.find("=") + 1:]
		Z0 = Z0[Z0.find("=") + 1:]
		n = n[n.find("=") + 1:-4]

		return runID, layers, int(chunk_num), float(Z0), int(n)

print("-> Loading old results file")
_tmp = load_data(rs_file)

runID, composition, concentration, layers, chunk_num, Z0, n0 = scrape_from_filename(rs_file)
print("-> Finished scraping variables from results file")
print("-> Setting initial conditions")
Tsurf = 110
Tbot = 260.
D = 15e3
brittle_layer = 5e3
profile = "non-linear" if layers == "1L" else "two layer"

vehicle_temp = _tmp['T'].max()
rtg_size = 0.6  # m, nominal RTG length
number_of_rtgs = 3  # number of rtgs on vehicle, mainly affects length of vehicle
vehicle_length = 2 + number_of_rtgs * rtg_size  # m, nominal vehicle length (rtgs + other module lengths)
vehicle_width = 0.3  # m, nominal VERNE width

dx = 0.01  # m, horizontal step size
dz = 0.1  # m, vertical step size
dt = _tmp['time'] / n0
dt = dt if dt > 10 else 10.

# Results output choices
name = f"{runID}{composition}{concentration}_{'2L' if profile == 'two layer' else '1L'}"  # uniquely name output files
output_frequency = 24 * 60 * 60  # s, time frequency to output requested results
# choose matrices to output: T - temperature, phi - liquid fraction, V - vehicle position, S - salinity
output_list = [k for k in _tmp.keys() if k != "time"]
print("-> Set outputs to", output_list)

Lz, Lx = _tmp['T'].shape
Lz = (Lz - 1) * dz
Lx = (Lx - 1) * 2 * dx

print("-> Start initializing simulation")
# Set up simulation
md = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, use_X_symmetry=True, Z0=Z0)#, cpT=True)

# initialize temperature profile
md.init_T(Tsurf=Tsurf, Tbot=Tbot, real_Lz=brittle_layer if profile == "two layer" else D ,profile=profile,
          layer=brittle_layer)

# initialize ice shell salinity
md.init_salinity(composition=composition, concentration=concentration, shell=True,
                 T_match=False if profile == "two layer" else True)

# initialize vehicle
md.init_vehicle(depth=2 * dz + Z0, length=vehicle_length, radius=vehicle_width, T=vehicle_temp)

print(f"-> Set t={_tmp['time']/3600:0.03f} hr results to current condition")
md.T = _tmp['T']
md.phi = _tmp['phi']
md.vehicle = _tmp['V']
md.vehicle_geom = np.where(md.vehicle==1)
md.S = _tmp['S']
md.update_volume_averages()
md.total_salt[0] = md.S.sum()

# determine the temperature of ice of "ghost cells" just above simulated depth
if profile == "two layer":
	if md.Z[0,0] > brittle_layer:
		print("below  brittle layer")
		TTOPOUT = 260.

	else:
		TTOPOUT = Tsurf * (260. / Tsurf) ** ((md.Z[0,0] - md.dz) / brittle_layer)
else:
	TTOPOUT = Tsurf * (Tbot / Tsurf) ** ((Z0 - md.dz) / D)

print(TTOPOUT)
md.set_boundaryconditions(top="Flux", Top_out=TTOPOUT, sides="RFlux", dL=100.)
print(md.T_top_out[0])

# number of iterations to run simulation (i.e. total time = NT * dt)
NT = int(5e12)  # so large that it will generally run until it reaches the bottom boundary

print("-> Set outputs")
# choose outputs
md.outputs.choose(md, output_list=output_list, output_frequency=output_frequency)

# uniquely name temporary and output files
md.outputs.tmp_data_file_name = f"tmp_data_{name}_chunk={chunk_num}_Z0={Z0}"

md.model_time = _tmp['time']
print(md.outputs.outputs)
md.solve_heat(nt=int(5e12), dt=dt, n0=n0)
print(md.run_time)

# Save your results
print("Saving data to", outdir)
print("   saving model")
save_name = md.outputs.tmp_data_file_name.split('tmp_data_')[1]
save_data(md, 'md_{}'.format(save_name), outdir)
print("   saving results")
results = md.outputs.get_all_data(md)
save_data(results, "rs_{}".format(save_name), outdir)

# Save some data from this simulation for "salt conservation" for the next
water = np.where(md.phi >= md.rejection_cutoff)
old_water_volume = water[0].shape[0]
Ssum = md.S[water].sum()
print("Ssum =", Ssum)
