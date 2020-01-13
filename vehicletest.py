from System import IceSystem
import numpy as np
import matplotlib.pyplot as plt
import plotSettings
import seaborn as sns
plotSettings.paper()

Lz = 5
Lx = 4.

dx = 0.01
dz = 0.1

Tsurf = 110.0010004994
Tbot = 273.15

nrtg = 2
rtgz = 1.
dv = 2*dz
h = 2 + nrtg * rtgz
w = 0.3


md = IceSystem(Lx=Lx, Lz=Lz, dx=dx, dz=dz, use_X_symmetry=True)
print(md.X.shape)
md.vh = h
md.vw = w
md.vd = dv
md.init_T(Tsurf=Tsurf, Tbot=Tbot, real_Lz=10e3)
r = np.where(abs(md.X[0,:]) <= w/2)[0]
z = np.intersect1d(np.where(md.Z[:, 0] <= h + dv), np.where(md.Z[:, 0] >= dv))
zrtg = np.intersect1d(np.where(md.Z[:,0] <= h + dv), np.where(md.Z[:, 0] >= h + dv - rtgz))
tmp = np.zeros(md.T.shape)
tmp[z.min():z.max(), r.min():r.max() + 1] = 1
md.geom = np.where(tmp == 1)
tmp = np.zeros(md.T.shape)
tmp[zrtg.min():zrtg.max(), r.min():r.max() + 1] = 1
md.headgeom = np.where(tmp == 1)
#del r, z, tmp
md.vehicle[md.geom] = 1
md.T[md.geom] = 273.15 + 100.
md.Tv = 273.15 + 100.
md.init_volume_averages()

md.set_boundaryconditions(sides='NoFlux', top='Flux')

dtmax = 0.5 * min(dx, dz) ** 2 / (3 * md.k.max() ** 2 / (md.rhoc.min()))
print('dt =', dtmax, 's')
md.solve_heat(nt=500000, dt=.001)#dt=dtmax)

'''
plt.figure()
plt.axis('equal')
plt.pcolormesh(md.X, -md.Z, md.T)
plt.colorbar()
plt.show()

plt.figure()
plt.axis('equal')
PHI = md.phi.copy()
PHI[md.vehicle == 1] = 0
plt.pcolormesh(md.X, -md.Z, PHI, vmin=0, vmax=1)
plt.colorbar()
plt.show()

plt.figure()
plt.axis('equal')
plt.pcolormesh(md.X, -md.Z, md.k/md.rhoc)
plt.colorbar()
plt.show()
'''