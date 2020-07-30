# TP_TDD2020
Trabajamos sobre el paper:
Li, H.H. and Ma, W.J. (2020). 
Confidence reports in decision-making with multiple alternatives violate the Bayesian confidence hypothesis. 
Nature Communications. DOI: 10.1038/s41467-020-15581-6
https://www.nature.com/articles/s41467-020-15581-6

## Objetivo
El código presentado es una versión simplificada de los modelos de Máximos y Diferencias utilizados en el paper mencionado más arriba, escrito integramente
en Python. Si bien podría mejorarse en su implementación, esperamos que sea suficiente para comprender las ideas fundamentales de los modelos planteados
y para comparar sus predicciones.

## Uso
modelo_diff.py y modelo_max.py presentan los modelos (Diferencias y Máximos respectivamente). Son los más interesantes de analizar para comprender la investigación,
y pueden usarse para contrastar las respuestas de los participantes con las predicciones del modelo.

Por su parte, optimizer.py permite entrenar los modelos utilizando Maximum Likelihood, y descubrir cual ajusta mejor a cada participante.
Para cargar los datos, se puede modificar la primer fila del archivo (mat = ...), la cual accede a las respuestas y las convierte a un formato legible en Python.
Manipulando los comentarios, se puede elegir entrenar tanto el modelo de Máximos como el de Diferencias.


Los datos utilizados para entrenar los modelos no son nuestros, fueron obtenidos por Hsin-Hung Li y Wei Ji Ma para su investigación. Accedimos a los mismos
a través del repositorio https://github.com/hsinhungli/confidence-multiple-alternatives.
