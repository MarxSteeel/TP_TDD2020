from scipy.stats import gamma
from scipy.stats import multivariate_normal
import numpy as np

def diff_nkll(x, data, n_sim=10000):
    '''Función que toma una serie de parámetros y los ajusta siguiendo un modelo que estima la confianza
       como la diferencia entre las dos distribuciones posteriores más grandes. Devuelve el negativo del 
       log-likelihood de la función, para poder ser optimizado'''
    ## Cargar datos
    trl = data['trl']
    trl_x = np.array(trl['target_x']) ## 336 x 1, coordenada x del S.
    trl_y = np.array(trl['target_y']) ## 336 x 1, coordenada y del S.
    trl_color = np.array(trl['estD']).T[0]
    trl_config = np.array(trl['config']).T[0]
    trl_conf = np.array(trl['estC']).T[0]

    n_trials = trl_x.shape[0]
    n_configs = 4
    n_cat = 3

    ## Parámetros
    sigma_x = 10**x[0] * np.identity(2)
    alpha = 10**x[1]
    b1 = x[2]
    b2 = x[3]
    b3 = x[4]

    ## Medias de las distribuciones de los estímulos.
    mux_sti  = np.array([[-96,0,96], [-128,0,128], [-96,-59,96], [-96,59,96], [-64,0,64], [-51,0,51], [-64,-64,64], [-64,64,64]])
    muy_sti  = np.array([[0,0,0], [0,0,0], [0,0,0], [0,0,0], [37,-74,37], [30,-59,30], [37,0,37], [37,0,37]])

    ## Desviación del estímulo
    sigma_s = 64 * np.identity(2)
    ## Desviación del likelihood
    sigma_l = np.sqrt(sigma_s**2 + sigma_x**2)

    ## Todas las estimaciones por MAP de cada una de las 10000 repeticiones por trial
    ## Como el modelo utiliza una simulación para estimar las probabilidades de éxito en cada trial, tenemos que
    ## usar una matriz en 3 dimensiones:
    maps = np.zeros((n_trials, 2, n_sim)) # 336 x 2 x 10000, coordenadas del estímulo en cada trial para 10000 simulaciones.
    i = 0
    for sx, sy in zip(trl_x, trl_y):
        mu = [sx[0], sy[0]]
        xs = multivariate_normal.rvs(mean=mu, cov=sigma_l, size=n_sim)
        ## El prior es uniforme, por lo que la estimación por MAP en este caso es
        ## identica a la estimación por MLE:
        mu_l = xs
        maps[i] = mu_l.T
        i += 1

    ## Probabilidades para cada distribución, para 10000 repeticiones por trial.
    probs = np.zeros((n_trials, n_cat, n_sim)) # 336 x 3 x 10000, probabilidad de cada categoría en cada trial para 10000 simulaciones.
    for i in range(n_configs):
        xs_sti = mux_sti[i]
        ys_sti = muy_sti[i]
        mu1 = [xs_sti[0], ys_sti[0]]
        mu2 = [xs_sti[1], ys_sti[1]]
        mu3 = [xs_sti[2], ys_sti[2]]
        for j in range(n_sim):
            map_config = maps[trl_config==i+1, :, j]
            ## La suma por 1e-22 es para evitar problemas numéricos en los casos en que Python
            ## redondea la probabilidad a 0.
            p1 = multivariate_normal.pdf(map_config, mean=mu1, cov=sigma_s) + 1e-22
            p2 = multivariate_normal.pdf(map_config, mean=mu2, cov=sigma_s) + 1e-22
            p3 = multivariate_normal.pdf(map_config, mean=mu3, cov=sigma_s) + 1e-22
            p_sum = p1 + p2 + p3

            ## Agregar ruido de Dirichlet
            p1 = gamma.rvs((p1 / p_sum) * alpha, scale=1)
            p2 = gamma.rvs((p2 / p_sum) * alpha, scale=1)
            p3 = gamma.rvs((p3 / p_sum) * alpha, scale=1)
            p_sum = p1 + p2 + p3
            p1 /= p_sum
            p2 /= p_sum
            p3 /= p_sum

            probs[trl_config==i+1, 0, j] = p1
            probs[trl_config==i+1, 1, j] = p2
            probs[trl_config==i+1, 2, j] = p3  
    
    ## Colores para cada trial
    labels = np.argmax(probs, axis=1) + 1

    ## Diferencia entre probs más altas para las 10000 repeticiones de cada trial
    maximos = np.zeros((n_trials, 2, n_sim)) # 336 x 2 x 10000, se remueve la categoría con menor probabilidad.
    for i in range(n_trials):
        for j in range(n_sim):
            maximos[i, :, j] = np.delete(probs[i, :, j], probs[i, :, j].argmin())

    maximos = abs(maximos[:, 0] - maximos[:, 1])
    
    maximos[maximos >= b3] = 4
    maximos[maximos <= b1] = 1
    maximos[(maximos > b1) & (maximos < b2)] = 2
    maximos[(maximos > b2) & (maximos < b3)] = 3

    ## Para imprimir los valores reportados y los calculados por nosotros (confianza):
    # for i in range(n_trials):
    #     print(trl_conf[i], maximos[i])

    ## Maximum Likelihood
    # Primero se necesita el promedio de respuestas correctas por trial

    correctas = np.zeros((n_trials, n_sim))
    for i in range(n_trials):
        for j in range(n_sim):
            correctas[i, j] = (labels[i, j] == trl_color[i]) & (maximos[i, j] == trl_conf[i])

    ## Probabilidad de éxito por trial
    correctas = np.array(np.sum(correctas, axis=1) / n_sim) + 1e-22 
    # De nuevo, la suma es para evitar problemas numéricos (logaritmo de 0).

    ## Negativo del logaritmo de la probabilidad de éxito por trial.
    nllk = - sum(np.log(correctas))

    ## Para visualizar los valores durante la optimización
    print('Parámetros:', x)
    print('- log-likelihood:', nllk)
    
    return(nllk)
