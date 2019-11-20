from scipy.spatial import distance
import numpy as np
# Entrada de datos
nx = 90 #10
ny = 90 #10
nv = nx*ny
grid_lat = np.arange(0,nv)
grid_lon = np.arange(0,nv)
xx, yy = grid_lon.reshape(nx,ny), grid_lat.reshape(nx,ny)
print(xx)
np.random.seed(42)
values = np.random.normal(20,5,nv)
M = np.array([xx.reshape(nv), yy.reshape(nv), values]).T


# Generando los puntos a Interpolar con Kriging Orinario
nxi = 180 #10
nyi = 180 #10
nvi = nxi*nyi
grid_lati = np.linspace(0, nyi-1, nxi)
grid_loni = np.linspace(0, nxi-1, nyi)
xxi, yyi = np.meshgrid(grid_loni, grid_lati)
Mi = np.array([xxi.reshape(nvi), yyi.reshape(nvi)]).T
print(Mi.shape)
D = distance.cdist(Mi,M[:,:2],  'euclidean')
print(D.shape)
print(Mi[:,:])
print(M[:,:2])
print(D)
