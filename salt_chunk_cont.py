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
chunk_size_shallow = 20.
chunk_size = 50.  # m
chunk_size_deep = 150.  # m
chunk_number = 15  # N
chunk_spacing = D / chunk_number

Lx = 2.  # m, lateral domain size
dx = 0.01  # m, lateral step size
dz = 0.1  # m, vertical step size

number_of_rtgs = 3  # number of rtgs in vehicle
rtg_size = 0.6  # m, nominal RTG length
initial_depth = 2 * dz  # m, initial depth in ice below Z0 to start vehicle
vehicle_temp = 300.  # K, temperature of vehicle
thermal_power_per_rtg = 5e3 / number_of_rtgs
h = 2 + number_of_rtgs * rtg_size  # m, nominal VERNE length
w = 0.3  # m, nominal width of VERNEs

# make array of chunks
Z0s = np.zeros(chunk_number + 1)
for i in range(1, chunk_number - 1):
	Z0s[i] = (i + 1) * chunk_spacing + chunk_spacing / 2 - chunk_size / 2
Z0s[-1] = D - chunk_size_deep
cn = 4

for it, Z0 in enumerate(Z0s[4:]):
	# tmp_file_name =
	print('\n============\nstarting at depth =', Z0, 'm')
	if Z0 > 13e3:
		print('\tusing larger chunk size for deeper simulations')
		chunk_size = chunk_size_deep
	elif Z0 <= 3e3:
		print("\t using smaller chunk size for shallow simulations")
		chunk_size = chunk_size_shallow
	md = IceSystem(Lx=Lx, Lz=chunk_size, dx=dx, dz=dz, use_X_symmetry=True, Z0=Z0)
	md.init_T(Tsurf=Tsurf, Tbot=Tbot, real_Lz=D)
	# initialize salinity
	md.init_salinity(composition='MgSO4', concentration=12.3)
	# initialize vehicle
	md.init_vehicle(depth=initial_depth + md.Z[0, 0], length=h, radius=w, T=vehicle_temp, num_RTG=number_of_rtgs)
	
	print("...adding salt from previous chunk")
	if it == 0:
		# just use the result from the surface run, Z0 = 0.0
		#old_ppt = 6.337952495908039
		#old_water_volume = 0.911918710120458
		# chunk = 3. restarted to make sure this actually worked :)
		old_ppt = 1.3672947703924723
		old_water_volume = 1.0757593963699914
	else:
		old_water_volume = phiold.sum()/(np.ones(phiold.shape).sum()) * Lxold * Lzold
	
	print(f"...previous water pocket: {old_ppt}ppt")
	Sadd = old_ppt * md.vehicle_h * md.vehicle_w / old_water_volume
	print(f"...adding {Sadd}ppt to new chunk water")
	md.S[md.vehicle_geom] += Sadd
	print(f"...totalling: {md.S[md.vehicle_geom].mean()}ppt")
	print(f"...updating salt conservation stuff")
	md.total_salt[0] = md.S.sum()

	if "Constant T" in md.vehicle_heat_scheme[0]:
		dt = 1.0
	else:
		dt = 4. * min(md.dx, md.dz) ** 2 * md.rhoc.min() / (3 * md.k.max() ** 2)
	out_freq = 24 * 60 * 60.
	md.outputs.choose(md, output_list=["T", "phi", "V", "S"], output_frequency=out_freq)

	md.outputs.tmp_data_file_name += "salt12_chunk={}_Z0={}".format(cn + it, Z0)

	if md.Z[0,0] == 0.:
		md.set_boundaryconditions(top="Radiative", sides="RFlux", dL=100.)
	else:
		TTOPOUT = Tsurf * (Tbot / Tsurf) ** ((Z0 - md.dz) / D)
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
	Sold,phiold,old_ppt,Lxold,Lzold = md.S.copy(), md.phi.copy(), md.S[md.vehicle_geom].mean(), md.Lx, md.Lz
