from VERNESim import VERNESim as Model
from utility_funcs import save_data, load_data
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
#mpl.use("TkAgg")

# choose a directory to output results files
outdir = "./results/"

# Ice shell physical parameters
Tsurf = 110.  # K, surface temperature
Tbot = 273.15  # K, basal ice shell temperature
D = 15e3  # m, total ice shell thickness
brittle_layer = 5e3  # m, brittle layer thickness for two-layer ice shell (i.e. conductive and convective layer)
profile = "non-linear"#"two layer" #

composition = "NaCl"#"MgSO4"  # composition of the salt in the ice
concentration = 35#12.3  # ppt (g salt/kg water/ice), salinity of parent ocean

# VERNE vehicle properties
vehicle_temp = 300  # K, constant/initial temperature of vehicle
#rtg_size = 0.6  # m, nominal RTG length
#number_of_rtgs = 3  # number of rtgs on vehicle, mainly affects length of vehicle
vehicle_length = 4.5#2 + number_of_rtgs * rtg_size  # m, nominal vehicle length (rtgs + other module lengths)
vehicle_width = 0.34  # m, nominal VERNE width

# Numerical simulation choices
Lx = 3.5  # m, horizontal domain size
Lz = 25. #vehicle_length * 5 + 1 #15  # m, vertical domain size
dx = 0.01#0.01  #   # m, horizontal step size
dz = 0.1  #0.5   # m, vertical step size

Z0 = D - 2*Lz  # m, initial vehicle depth (this is at the surface, but can start anywhere)

# Set up simulation
md = Model(w=Lx, D=Lz, dx=dx, dz=dz, Z0=Z0, verbose=True, coordinates="zr")

# initialize temperature profile
md.init_T(Tsurf=Tsurf, Tbot=Tbot, real_D=brittle_layer if profile == "two layer" else D, profile=profile,
          brittle_layer=brittle_layer)

# initialize shell salinity
md.init_salinity(composition=composition, concentration=concentration, shell=True,
                 T_match=False if profile == "two layer" else True)

# initialize vehicle
md.init_vehicle(depth=md.Z[0,0] + 10 * md.dz, length=vehicle_length, radius=vehicle_width / 2,
                constant_T=300.0)#,
                #melt_mod=0.25, mechanical_descent=True, descent_speed=15e3/(3 * 3.154e7)

md.set_boundaryconditions(sides="NoFlux")
# number of iterations to run simulation (i.e. total time = NT * dt)
Nt = 33003 # so large that it will generally run until it reaches the bottom boundary
# choose outputs
output_frequency = int(7 * 24 * 60 * 60 / md.dt)  # s, time frequency to output
# requested results
md.set_outputs(output_frequency, "./tmp")
md.verbose = True
md.ADAPT_DT = False
md.dt = 1
md.solve_heat(Nt*10, dt=md.dt)

rs = md.outputs.get_all_data(del_files=False)