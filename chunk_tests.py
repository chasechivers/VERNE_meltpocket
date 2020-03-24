"""
Break up ice shell of D thickness into chunk_number of chunk_size boxes where the vehicle is just placed inside cold
to see how large the melt pocket is in each individual chunk over time_span OR when it reaches the edge of the chunk
domain.
	- Melt velocity
	- Melt pocket dimensions
		- side walls
		- behind vehicle

assuming we want to get "average temperature" by using the middle of each 1 km for chunk_number = 15
# Box 1: depth = [0, chunk_size]
# Box 2: depth = [1.45km, 1.55km]
# Box 3: depth = [2.45km, 2.55km]
...
# Box chunk_num-1: depth = [14.45km, 14.
# Box chunk_number: depth = [D-chunk_size, D]
##
"""
import numpy as np
from System import IceSystem
from utility_funcs import save_data

out_dir = "./results/"

Tsurf = 100.  # K
Tbot = 273.15  # K
D = 15e3  # m
chunk_size = 50.  # m
chunk_size_deep = 150. # m
chunk_number = 15  # N
chunk_spacing = D/chunk_number

Lx = 2.  # m, lateral domain size
dx = 0.01  # m, lateral step size
dz = 0.1  # m, vertical step size

number_of_rtgs = 3  # number of rtgs in vehicle
rtg_size = 0.6  # m, nominal RTG length
initial_depth = 2 * dz  # m, initial depth in ice below Z0 to start vehicle
vehicle_temp = 287.  # K, temperature of vehicle
thermal_power_per_rtg = 5e3/number_of_rtgs
h = 2 + number_of_rtgs * rtg_size  # m, nominal VERNE length
w = 0.3  # m, nominal width of VERNEs

# make array of chunks
Z0s = np.zeros(chunk_number+1)
for i in range(1, chunk_number-1):
	Z0s[i] = (i+1)*chunk_spacing + chunk_spacing/2 - chunk_size/2
Z0s[-1] = D - chunk_size_deep
cn=1
for it, Z0 in enumerate(Z0s[2:]):
	#tmp_file_name =
	print('\n============\nstarting at depth =', Z0, 'm')
	if Z0 > 13e3:
		print('\tusing larger chunk size for deeper simulations')
		chunk_size = chunk_size_deep
	md = IceSystem(Lx=Lx, Lz=chunk_size, dx=dx, dz=dz, use_X_symmetry=True, Z0=Z0)
	md.init_T(Tsurf=Tsurf, Tbot=Tbot, real_Lz=D)
	md.init_vehicle(depth=initial_depth+md.Z[0,0], length=h, radius=w, T=vehicle_temp, num_RTG=number_of_rtgs, temp_regulation="RTG Flux", power=thermal_power_per_rtg)
	
	if "Constant T" in md.vehicle_heat_scheme[0]:
		dt = 0.5
	else:
		dt = 4. * min(md.dx, md.dz)**2 * md.rhoc.min() / ( 3 * md.k.max()**2 )
	out_freq = 24 * 60 * 60.
	md.outputs.choose(md, output_list=["T", "phi", "V"], output_frequency=out_freq)
	
	md.outputs.tmp_data_file_name += "nc_vq_chunk={}_Z0={}".format(cn+it,Z0)

	if it == 0:
		md.set_boundaryconditions(top="Radiative", sides="RFlux", dL=100.)
	else:
		TTOPOUT = Tsurf*(Tbot/Tsurf)**((Z0-md.dz)/D)
		md.set_boundaryconditions(top="Flux", Top_out=TTOPOUT, sides="RFlux", dL=100.)
	
	NT = 5000000000000
	md.solve_heat(nt=NT, dt=dt)
	print(md.run_time)

	print("saving data to", out_dir)
	print("   saving model")
	save_name = md.outputs.tmp_data_file_name.split('tmp_data_')[1]
	save_data(md, 'md_{}'.format(save_name), out_dir)
	print("   saving results")
	results = md.outputs.get_all_data(md)
	save_data(results, "rs_{}".format(save_name), out_dir)
