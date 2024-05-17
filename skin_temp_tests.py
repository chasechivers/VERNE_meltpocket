from VERNESim import VERNESim as VERNE
import numpy as np
import utility_funcs as uf
import matplotlib.pyplot as plt
import time as _timer_

use_salt = True
coords = "zr"

# Constants that won't change per simulation
## Ice shell physical parameters
Tsurf = 110.  # K, surface temperature of the ice shell
D = 15e3  # m, total ice shell thickness
# assume a conductive shell, as it will go over the same range of temperatures as
# the convective shell
profile = "non-linear"

composition = "NaCl"  # "MgSO4"
concentration = 35.  # 12.3

## vehicle geometry
vehicle_length = 4.5  # m, nominal vehicle length (3 RTGs + other module lengths)
vehicle_width = 0.34  # m, nominal VERNE width

## model numerical parameters
Lx = 4  # m, horizontal domain size
dx = 0.01  # m, horizontal step size
dz = 0.1  # m, vertical step size
Lz = 20 * dz + vehicle_length + 20 * dz #2.5 * vehicle_length  # m, vertical domain size
dt = 0.5  # s, time step for simulation

# define a set of initial starting depths of the vehicle
# number_of_depths = 15  # N, number of individual domains to use
# spacing = D / number_of_depths  # spaces between each simulation
# Z0s = [dz]
# for i in range(1, number_of_depths):
#    Z0s.append((i + 0.5) * spacing - Lz / 2)
# Z0s[-1] = D - Lz

T0s = np.arange(Tsurf, 273.15 + 5, 5)[::2]#250, 5)[::2]#
Z0s = -D * np.log(Tsurf / T0s) / np.log(273.15 / Tsurf)
Z0s[0] = dz
if Z0s[-1] > D:
	Z0s[-1] = (Z0s[-2] + D) / 2.

# define a set of skin temperatures to test
Tsk_max = 299.  # K, we already know this will melt for all depths, so start below this
dT = 1.  # K, skin temperature increment
skin_temperatures = np.arange(Tsk_max - dT, 273.15, - dT)
#skin_temperatures = np.arange(273.15, Tsk_max + dT, dT)

# If given long enough, some Tsk >~ Tm will create melt
# However, we don't have forever to wait!
# Below is a minimum time for the side melt to occur before stopping the simulation
minimum_velocity = 1.73549962072841e-5  # m/s, minimum descent velocity from pure mechanical drilling
time_constraint = 2 * dz / minimum_velocity  # s

# create dictionary to store things we care about tracking across simulations
results = {T0: {} for T0 in T0s}
save_name = f"skin_temp_tests_D={D/1e3:0.01f}_km_{coords}_dx={dx}{'_salt' if use_salt else ''}"
count = 1
total_time = 0
for jt, Z0 in enumerate(Z0s):
	for it, Tsk in enumerate(skin_temperatures):
		start_time = _timer_.time()
		print("=" * 72)
		print(f"{count}/{len(Z0s)*len(skin_temperatures)}\t Z0 = {Z0:0.03f}m\t Tsk = {Tsk:0.03f} K")
		print("=" * 72)
		count += 1
		# initialize model
		md = VERNE(w=Lx, D=Lz, dx=dx, dz=dz, use_X_symmetry=True, Z0=Z0, coordinates=coords)
		# initialize temperature profile
		md.init_T(Tsurf=Tsurf, Tbot=273.15, real_D=D, profile=profile)
		print(f"Z0 = {md.Z[0,0]/1e3:0.03f}km\n  T0 = {md.T[-1,0]:0.03f} K\n  Tb = {md.T[-1,-1]:0.03f} K")
		# initialize shell salt content
		#  actually not needed for this! pure ice will give an upper bound!
		if use_salt is True:
			md.init_salinity(composition=composition,
			                 concentration=concentration,
			                 shell=True,  # ensure shell has salt gradient
			                 T_match=True  # match the temperature profile to the ocean's melting temperature
			                 )
			if Tsk == 273.15:
				print(f"   Modifying skin temperature to be the highest Tm in the domain\n   Tsk = {Tsk} K -> "
				      f"{md.Tm.max():0.03f} K")
				Tsk = md.Tm.max()
		print(f"   Tm_avg = {md.Tm[:,-1].mean():0.03f} K")
		# initialize vehicle geometry and position
		md.init_vehicle(depth=20 * dz + Z0,
		                length=vehicle_length,
		                radius=vehicle_width / 2.,
		                constant_T=Tsk)

		# Boundary conditions:
		#   top:    Flux (Neumann) to temperature of ice at dz above top grid
		#   bottom: Dirichlet (constant)
		#   right:  Flux (Neumann) to background temperature profile dL away
		#   left:   No flux (Robin) boundary condition to reflect symmetry
		#T_top_out =
		#tgradtop = (md.T[1,- 1] - (Tsurf * (273.15 / Tsurf) ** ((Z0 - md.dz) / D)))/2/md.dz
		md.set_boundaryconditions(sides="NoFlux") #, top="Flux", T_top=tgradtop) #T_top=T_top_out)# sides="RFlux",
		# dL=100)

		# set outputs
		## in this case, we don't want any since we just want the final results
		output_frequency = int(1e100 / dt)
		md.set_outputs(output_frequency, './tmp/', outlist=[])
		md.outputs.tmp_data_directory = './tmp/'

		# number of iterations to run simulation
		# here, will be defined by our time_constraint above
		NT = int(time_constraint / dt)

		# begin time stepping
		n = 0

		xidx = md.vehicle_geom[1].max() + 1
		zmin = md.vehicle_geom[0].min()
		zmax = md.vehicle_geom[0].max() + 1
		all_water = len(md.phi[zmin:zmax, xidx])
		# our stop condition is either the simulation reaches time_constraint
		# or all the ice in contact with the side of the vehicle is melted
		while md.phi[zmin:zmax, xidx].sum() < all_water and n < NT:
			md.solve_heat(nt=1, dt=dt, move=False)
			# if n % 5 == 0:
			#	print(n , md.phi[zmin:zmax, xidx].sum())
			n += 1
		print(f"TIMEOUT \n {md.phi[zmin:zmax, xidx].sum()/all_water*100:0.03f}% melted" if n >= NT
		      else f"MELTED @ {n * dt / 3600:0.03f} hr")
		rs = md.outputs.get_all_data(md)
		results[T0s[jt]][Tsk] = {}
		results[T0s[jt]][Tsk]["melted"] = md.phi[zmin:zmax, xidx].sum()
		results[T0s[jt]][Tsk]["time"] = md.model_time
		uf.save_data(data=results, file_name=save_name, outputdir="./results/", final=0)
		print(f"   solved in {_timer_.time() - start_time:0.03f}s")
		total_time += _timer_.time() - start_time
		print(f"total time taken {total_time:0.03f} s")


uf.save_data(data=results, file_name=save_name, outputdir="./results/", final=0)

"""
fig, ax = plt.subplots()
stime = np.zeros(((len(T0s), len(skin_temperatures))))
smelt = np.zeros(((len(T0s), len(skin_temperatures))))

for it, k in enumerate(results):
	stime[it, :] = np.array([v['time'] for k, v in results[k].items()])
	smelt[it, :] = np.array([v['melted'] for k, v in results[k].items()])

stimec = stime.copy()
stimec[smelt < 45] = np.nan
ax.set_title(md.coords)
p = ax.pcolormesh(skin_temperatures, T0s, stimec/3600,
                  shading="auto", #norm=colors.LogNorm(),
                  cmap="cividis", rasterized=True)#,
                  #vmin=cmin/3600, vmax=cmax/3600)
fig.colorbar(p, ax=ax, label="Time to melt, hrs", orientation="horizontal")
ax.set(xlabel="Skin temperature, K", ylabel="Ambient ice temperature, K")
ax.invert_yaxis()
"""