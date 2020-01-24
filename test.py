from System import IceSystem
import numpy as np
import matplotlib.pyplot as plt

Lz = 5  # m, vertical domain size
Lx = 4. # m, horizontal domain size

dx = 0.01  # m, horizontal step size
dz = 0.1  # m, vertial step size

Tsurf = 110.00  # K, surface temperature
Tbot = 273.15  # K, ocean temperature

number_of_rtgs = 4
rtg_size = 0.6 # m, nominal RTG length
initital_depth = 2*dz  # m, initial depth in ice to start simulation
h = 2 + number_of_rtgs * rtg_size  # m, nominal VERNE length
w = 0.3  # m, nomimal total width VERNE
Lz += 0.20*h  # adjust the domain size

# initialize the ice shell/model
md = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, use_X_symmetry=True)

# initialize the temperature profile
md.init_T(Tsurf=Tsurf, Tbot=Tbot, real_Lz=10e3)

# initialize vehicle geometry, depth, temperature, thermal stuff
md.init_vehicle(depth=2*dv, length=h, radius=w, T=673.15, num_RTG=4)

# set the boundary conditions
md.set_boundaryconditions(sides='NoFlux', top='Radiative')


dtmax = 0.5 * min(dx, dz) ** 2 / (3 * md.k.max() ** 2 / (md.rhoc.min()))
print('dt =', dtmax, 's')
# start simulation
md.solve_heat(nt=2*60000, dt=.005)#, n0=int(md.model_time/md.dt), print_opts=0)
print(md.run_time)

# visualize temperature profile
plt.figure()
plt.pcolormesh(md.X, -md.Z, md.T, vmin=md.T[md.vehicle==0].min(), vmax=md.T[md.vehicle==0].max())
plt.colorbar()
plt.show()