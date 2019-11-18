import pandas as pd
import matplotlib as mpl
from numba import cuda, vectorize, jit
import numpy as np
from numpy.linalg import inv
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time
from sys import getsizeof
import time

@jit(nopython= True)
def MatrizKrig(nvi,M,Mi, constantes,valuesK):
    for puntos in range(0,nvi):
        vector = np.array([np.ones(M.shape[0]), np.ones(M.shape[0])]).T
        DPE = np.power(M[:,:2] - vector*Mi[puntos], 2) # Distancia Punto a Estimar
        print(M[0,:2])
        print(Mi[0])
        print(DPE)
        DPEF = np.sqrt(DPE[:,0] + DPE[:,1])
        print(DPEF)
        print(len(DPEF))
        # Calculando el vector de semivarianza de los datos estimados
        VSVA = VSVAR(DPEF,constantes)
        b = np.ones(VSVA.shape[0]+1)
        b[:-1] = VSVA
        print(VSVA)
        print(b)
        # Calculando el vector de pesos para la interpolacion de kriging
        Weight =np.matmul(inversa, b)
        print(Weight)
        # Calculando el punto a estimar
        Estimado = np.dot(M[:,2], Weight[:-1])
        print(Estimado)
        valuesK[puntos] = Estimado

@jit(nopython= True)
def VSVAR(D, constantes):
    D1d = D
    SVM = [constantes[2] + constantes[1]*(1-np.exp(-h/constantes[0])) for  h in D1d]

    # SVM = [(constantes[2] + constantes[1]*(3.0*h/(2.0*constantes[0]) - ((h/constantes[0])**3.0)/2.0)) for  h in D1d]
    SVM = np.array(SVM, dtype=np.float32)
    # SVM = SVM.reshape(D.shape[0], D.shape[0])
    return SVM

@jit(nopython= True)
def MSVAR(D, constantes):
    D1d = D.reshape(D.shape[0]*D.shape[0]).astype(np.float32)
    SVM = [constantes[2] + constantes[1]*(1-np.exp(-h/constantes[0])) for  h in D1d]
    # SVM = [(constantes[2] + constantes[1]*(3.0*h/(2.0*constantes[0]) - ((h/constantes[0])**3.0)/2.0)) for  h in D1d]
    SVM = np.array(SVM, dtype=np.float32)
    SVM = SVM.reshape(D.shape[0], D.shape[0])
    return SVM
@jit(nopython = True)
def Generate(nv, M):
    a = np.zeros(shape=(nv,nv))
    for i in range(0, nv):
        for j in range(0, nv):
            a[i][j] = np.power(M[i,2] - M[j,2], 2)
    return a
@jit(nopython = True)
def MDist(Matriz):
    D1 = squareform(pdist(Matriz[:,:2]))
    return D1
@np.vectorize
def spherical(h, c0, c, a):
    """
    Input
    h: distance
    c0: sill
    c: nugget
    a: range
    Output
    Theoretical semivariogram at distance h
    """
    if h<=a:
        return c0 + c*(3.0*h/(2.0*a) - ((h/a)**3.0)/2.0)
    else:
        return c0 + c
@np.vectorize
def gaussian(h, c0, c, a):
    """
    Same as spherical
    """
    return c0 + c*(1-np.exp(-h*h/((a)**2)))
@np.vectorize
def exponential(h, c0, c, a):
    """
    Same as spherical
    """
    return c0 + c*(1-np.exp(-h/a))
@np.vectorize
def linear(h, c0, c, a):
    """
    Same as spherical
    """
    if h<=a:
        return c0 + c*(h/a)
    else:
        return c0 + c
def fitsemivariogram(z, s, model, numranges=200):
    """
    Fits a theoretical semivariance model.
    Input
    z: data, NumPy 2D array, each row has (X, Y, Value)
    s: empirical semivariances
    model: one of the semivariance models: spherical,
    Gaussian, exponential, and linear
    Output
    A lambda function that serves as a fitted model of
    semivariogram. This function will require one parameter
    (distance).
    """
    c = np.var(z[:,2]) # c, sill
    if s[0][0] is not 0.0: # c0, nugget
        c0 = 0.0
    else:
        c0 = s[0][1]
    minrange, maxrange = s[0][1], s[0][-1]
    ranges = np.linspace(minrange, maxrange, numranges)
    errs = [np.mean((s[1] - model(s[0], c0, c, r))**2)
            for r in ranges]
    a = ranges[errs.index(min(errs))] # optimal range
    print(a, h, c0, c)
    return lambda h: model(h, c0, c, a)
def fitsemivariogram2(z, s, model, numranges=200):
    """
    Fits a theoretical semivariance model.
    Input
    z: data, NumPy 2D array, each row has (X, Y, Value)
    s: empirical semivariances
    model: one of the semivariance models: spherical,
    Gaussian, exponential, and linear
    Output
    A lambda function that serves as a fitted model of
    semivariogram. This function will require one parameter
    (distance).
    """
    c = np.var(z[:,2]) # c, sill
    if s[0][0] is not 0.0: # c0, nugget
        c0 = 0.0
    else:
        c0 = s[0][1]
    minrange, maxrange = s[0][1], s[0][-1]
    ranges = np.linspace(minrange, maxrange, numranges)
    errs = [np.mean((s[1] - model(s[0], c0, c, r))**2)
            for r in ranges]
    a = ranges[errs.index(min(errs))] # optimal range
    return np.array([a,c,c0])
# Entrada de datos
nx = 10 #10
ny = 10 #10
nv = nx*ny
grid_lat = np.linspace(12, 13, nx)
grid_lon = np.linspace(-66, -65, ny)
xx, yy = np.meshgrid(grid_lon, grid_lat)
np.random.seed(42)
values = np.random.normal(20,5,nv)
M = np.array([xx.reshape(nv), yy.reshape(nv), values]).T
print(M.shape)
# Calculando las ditancias
# D = squareform(pdist(M[:,:2]))
D = distance.cdist(M[:,:2], M[:,:2], 'euclidean')

# Calculando la matriz de diferencias cuadradas
a = np.zeros(shape=(nv,nv))
a = Generate(nv, M)
print(a)
# Filtrando la matrix inferior
UM = np.triu_indices(nv)
a[UM] = 0
print(a)
# Binning Data
D_max = D.max()
D_min = D.min()
print(D_min, D_max)
Nbins = 5
DeltaH = D_max/(Nbins-1)
print(DeltaH, D_max)
h = np.linspace(0, D_max,Nbins)
h1 = h - DeltaH/2
h1_mask = h1 >0 
h1 = abs(h1 *h1_mask)
h2 = h + DeltaH/2
# calculando los 11 valores repartidos
Vario = np.zeros(len(h1))
for k in range(1, len(h1)):
    Mascara = (D > h1[k]) & (D<= h2[k])
    Bin1 = Mascara*a
    N = np.count_nonzero(Bin1)
    Vario[k] = np.sum(Bin1)/(N*2)
print(Vario)
print(h)
valuesv = np.array([h , Vario])
print(valuesv)
svs = fitsemivariogram(M, valuesv, exponential)
p1, = plt.plot(valuesv[0], valuesv[1], 'o')
p2, = plt.plot(valuesv[0], svs(valuesv[0]), color='grey', lw=2)
plt.show()
# # Calculando el modelo del Variograma
# #   Modelo Esferico
# # Calculando los coeficientes de a = range,
# # C0 : nugget variance, C0+C1 : sill
constantes = fitsemivariogram2(M, valuesv, exponential)
print( 'constantes', " ", constantes)
# Ordinary Kriging
# Generando la matriz Inversa de semivarianza de Datos Observados
# spherical()
print(D.shape)
SVM = MSVAR(D, constantes)
# D1d = D.reshape(D.shape[0]*D.shape[0]).astype(np.float32)
# SVM = [spherical(data,constantes[2], constantes[1],constantes[0] ) for data in D1d]
# SVM = np.array(SVM, dtype=np.float32)
# print(SVM.reshape(D.shape[0],D.shape[0]))
print(np.array(SVM))
print(D)
# Añadiendo una columna y fila de unos y una fila de zeros
SVMz = np.ones((SVM.shape[0]+1, SVM.shape[0]+1))
SVMz[:-1,:-1] = SVM
print(SVM.shape)
print(SVMz.shape)
print(SVMz)
# Introduciendo un zero en ell ultimo elemento de Fila y Columna
SVMz[-1,-1] =0
print(SVMz)
print(SVM)

# Calculando la Matriz Inversa del semivariograma de los datos Observados
inversa = inv(SVMz)
print(inversa)
# Generando los puntos a Interpolar con Kriging Orinario
nxi = 100 #10
nyi = 100 #10
nvi = nxi*nyi
grid_lati = np.linspace(12, 13, nxi)
grid_loni = np.linspace(-66, -65, nyi)
xxi, yyi = np.meshgrid(grid_loni, grid_lati)
Mi = np.array([xxi.reshape(nvi), yyi.reshape(nvi)]).T
# Calculando distancias con el punto a estimar
print(Mi.shape)
print(M.shape)
# vector = np.ones(M.shape[0])
vector = np.array([np.ones(M.shape[0]), np.ones(M.shape[0])]).T
print(vector)
# Repetir 180*180 veeces
start = time.time()
valuesK = np.zeros(nvi)



for puntos in range(0,nvi):
    vector = np.array([np.ones(M.shape[0]), np.ones(M.shape[0])]).T
    DPE = np.power(M[:,:2] - vector*Mi[puntos], 2) # Distancia Punto a Estimar
    print(M[0,:2])
    print(Mi[0])
    print(DPE)
    DPEF = np.sqrt(DPE[:,0] + DPE[:,1])
    print(DPEF)
    print(len(DPEF))
    # Calculando el vector de semivarianza de los datos estimados
    VSVA = VSVAR(DPEF,constantes)
    b = np.ones(VSVA.shape[0]+1)
    b[:-1] = VSVA
    print(VSVA)
    print(b)
    # Calculando el vector de pesos para la interpolacion de kriging
    Weight =np.matmul(inversa, b)
    print(Weight)
    # Calculando el punto a estimar
    Estimado = np.dot(M[:,2], Weight[:-1])
    print(Estimado)
    valuesK[puntos] = Estimado
    # print(M[puntos,2])
    # print(M[puntos+1,2])

end = time.time()
print('tiempo de ejecucion ', end - start, ' s' )
fig1 = plt.figure(figsize=(15,10))
plt.subplot(121)
z_min = valuesK.min()
z_max = valuesK.max()

plt.pcolormesh(xxi, yyi, valuesK.reshape(nxi,nyi))
# plt.plot(xx, yy, marker='.', color='white', linestyle='none')

plt.subplot(122)
# z_min = z.min()
# z_max = z.max()
np.random.seed(42)
values = np.random.normal(20,5,nv)
cmap = mpl.cm.cool
plt.pcolormesh(xx, yy, values.reshape(nx,ny))
plt.title('Mapa de Viento a resolucion de 1km por pixel')
# plt.plot(Xorig, Yorig, marker='.', color='green', linestyle='none')
fig1.canvas.draw()
cbar_ax = fig1.add_axes([0.92, 0.2, 0.02, 0.6])
cb1 = mpl.colorbar.ColorbarBase(cbar_ax, ticks=np.arange(0,1.01,0.25), orientation='vertical')
cb1.set_ticklabels([str(int(z_min)),str(int(z_max/4)),str(int(z_max/3)), str(int(z_max/2)), str(int(z_max))])
cbar_ax.tick_params(labelsize=12)
cbar_ax.text(1.1, -0.05, 'm/s', size=10)
plt.show()
