from scipy.optimize import minimize
import mat4py as mt
from modelo_max import max_nkll
from modelo_diff import diff_nkll

## Cargar los datos de algún participante
mat = mt.loadmat('data/EXP1/S1_log.mat') ## loadmat los carga como un diccionario.

## Máximos
# def max_func(x):
#     return max_nkll(x, mat, n_sim=1000)

# params = [4.13896701, 0.03808483, 0.55879685, 0.77827881, 0.99271046] # 783

# result = minimize(max_func, params, method='Nelder-Mead', bounds=[(-1, 3), (-2, 3), (0, 1), (0, 1), (0, 1)])
# print(result)


## Diferencias
def diff_func(x):
    return diff_nkll(x, mat, n_sim=1000)

params = [4.03963042, 0.03203068, 0.20263638, 0.68078228, 0.99974641] # 776

result = minimize(diff_func, params, method='Nelder-Mead', bounds=[(-1, 3), (-2, 3), (0, 1), (0, 1), (0, 1)])
print(result)

## AIC = 2 * n_param - 2 * log-likelihood

AIC_max = 2 * len(params) - 2 * (- 783) # 1576
AIC_diff = 2 * len(params) - 2 * (- 776) # 1562

## En nuestro modelo casero, un participante consigue una mejor puntuación con el modelo de diferencias.
## AIC_max - AIC_diff = 14

## Los investigadores solo reportaron la diferencia sumada entre el AIC de todos los participantes, no los resultados individuales.
## Por lo tanto, es difícil saber que tan lejos o cerca estamos de sus resultados, solo sabemos que reprodujimos parte de 
## ellos en forma cualitativa.