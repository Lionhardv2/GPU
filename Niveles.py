import netCDF4 as nc
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from mpl_toolkits import mplot3d
from wrf import getvar, interplevel
import wrf
import matplotlib as mpl
import utm
wrfin = nc.Dataset("D:/WRF/WRF260919/wrfout_d04_2018-03-01_00_00_00")

p = getvar(wrfin, "pressure")
ht = getvar(wrfin, "z", units="m")
Ws = getvar(wrfin, 'uvmet_wspd_wdir', units = 'm s-1')
Ws_10 = getvar(wrfin, 'uvmet10_wspd_wdir', units = 'm s-1')
Lat = getvar(wrfin, 'lat', meta = False)
Lon = getvar(wrfin, 'lon', meta = False)
Elevacion = getvar(wrfin, 'ter', units='m', meta = False)
coor = wrf.ll_to_xy(wrfin, -17.6290737, -65.2842807, timeidx=0, meta=False, as_int=True) # Lat Lon cerca de Qollpana
# U = getvar(wrfin, 'ua', )
print(ht.shape)
Speed10 = np.array(Ws_10[0,:,:])
print(Ws.shape)
# Interpolar cada 10 m
Levels = range(2700,2810,10)
print(Levels)
ht_500 = interplevel(Ws, ht, Levels)
print(ht_500.shape)
print('wind 1')
# print(ht_500[1,49,:,:])    # Wind Direction
print('wind 2')
# print(ht_500[0,0,:,:])    # Wind Speed
# Guardando los archivos como extension Geotiff 
print(Lon.shape)
Dim = Lon.shape[0] * Lon.shape[1]
np.array(Lat)
np.array(Lon)
# Create tuples of cordinates
# coor = [cord for cord in (Lon, Lat)]
print(utm.from_latlon(51.2, 7.5))
M = np.array([Lat.reshape(Dim), Lon.reshape(Dim)]).T
print(M.shape)
# Tupla=tuple(M)
# for i in range(0, )
print(M[0][0])
print(M[0][1])
print(M[0])

coord = utm.from_latlon(M[0][0],M[0][1])
UTM = np.array(coord)
a = utm.from_latlon(M[0][0],M[0][1])
lat2 = a[0]
lon2 = a[1]
b=  [lat2, lon2]
print(b[0])
print(b[1])
print(M)
print(M.shape)
latutm = np.zeros(M.shape[0])
lonutm = np.zeros(M.shape[0])
for i in range(0, M.shape[0]):
    var = M[i]
    # print(var)
    aa = utm.from_latlon(var[0], var[1])
    latutm[i] = aa[0]
    lonutm[i] = aa[1]
Lat1 = latutm.reshape(Lat.shape)
Lon1 = lonutm.reshape(Lon.shape)

print(Elevacion)
ElevacionW = Elevacion +10
print(ElevacionW)
print(ElevacionW.shape)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Lon, Lat, ElevacionW, rstride=1, cstride=1, cmap='viridis')
ax.set_xlabel('longitud')
ax.set_ylabel('latitude')
ax.set_zlabel('Elevacion [m]')
plt.show()
plt.close()
# Graficando los planos de velocidades de viento interpolados a 2750 2760 y 2740 msnm

# 2750 msnm
print(ht_500.shape)
print(np.array(ht_500[0,:,:,:]).shape)
z1 = np.array(ht_500[0,0,:,:])
where_are_NaNs = np.isnan(z1)
z1[where_are_NaNs] = 0
fig1 = plt.figure(figsize=(15,10))
plt.pcolormesh(Lon, Lat, Speed10)
# plt.plot(xx, yy, marker='.', color='white', linestyle='none')

z_min = Speed10.min()
z_max = Speed10.max()
cmap = mpl.cm.viridis
fig1.canvas.draw()
cbar_ax = fig1.add_axes([0.92, 0.2, 0.02, 0.6])
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap= cmap, ticks=np.arange(0,1.01,0.25), orientation='vertical')
cb1.set_ticklabels([str(int(z_min)),str(int(z_max/4)),str(int(z_max/3)), str(int(z_max/2)), str(int(z_max))])
cbar_ax.tick_params(labelsize=12)
cbar_ax.text(1.1, -0.05, 'm/s', size=10)
plt.show()

# Guardando los niveles de velocidad de viento interpolados en formato TXT
z1 = np.array(ht_500[0,:,:,:])
where_are_NaNs = np.isnan(z1)
z1[where_are_NaNs] = 0
# Write the array to disk
with open('speed.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(z1.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in z1:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')
# Guardando los niveles de direccion de viento interpolados en formato TXT
z1 = np.array(ht_500[1,:,:,:])
where_are_NaNs = np.isnan(z1)
z1[where_are_NaNs] = 0
# Write the array to disk
with open('direction.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(z1.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in z1:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')
# Guardando los coordenadas de latitud en formato TXT
z1 = np.array(Lat)
where_are_NaNs = np.isnan(z1)
z1[where_are_NaNs] = 0
# Write the array to disk
with open('latitud.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(z1.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in z1:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')
# Guardando los coordenadas de longitud en formato TXT
z1 = np.array(Lon)
where_are_NaNs = np.isnan(z1)
z1[where_are_NaNs] = 0
# Write the array to disk
with open('longitud.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(z1.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in z1:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')

Speed10 = np.array(Ws_10[0,:,:])

# Guardando los coordenadas de W10ms en formato TXT
z1 = np.array(Speed10)
where_are_NaNs = np.isnan(Speed10)
z1[where_are_NaNs] = 0
# Write the array to disk
with open('Speed10.txt', 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    outfile.write('# Array shape: {0}\n'.format(z1.shape))

    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for data_slice in z1:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, data_slice, fmt='%-7.2f')

        # Writing out a break to indicate different slices...
        outfile.write('# New slice\n')
