import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Generar datos de ejemplo para la serie de tiempo
np.random.seed(0)
tiempo = np.arange(1, 101)
nivel = 100 + 2 * tiempo  # Nivel creciente con el tiempo
tendencia = 2 * tiempo  # Tendencia lineal
estacionalidad = 10 * np.sin(2 * np.pi * tiempo / 12)  # Estacionalidad mensual
ruido = np.random.normal(0, 5, size=100)  # Ruido aleatorio

# Combinar componentes para formar la serie de tiempo
serie_tiempo = nivel + tendencia + estacionalidad + ruido

# Descomponer la serie de tiempo en sus componentes
descomposicion = seasonal_decompose(serie_tiempo, model='additive', period=12)

# Graficar la serie de tiempo y sus componentes
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(tiempo, serie_tiempo, label='Serie de tiempo')
plt.legend()
plt.title('Ventas de servicios segmento B2B')

plt.subplot(4, 1, 2)
plt.plot(tiempo, descomposicion.trend, label='Tendencia')
plt.legend()
plt.title('Tendencia')

plt.subplot(4, 1, 3)
plt.plot(tiempo, descomposicion.seasonal, label='Estacionalidad')
plt.legend()
plt.title('Estacionalidad')

plt.subplot(4, 1, 4)
plt.plot(tiempo, descomposicion.resid, label='Ruido')
plt.legend()
plt.title('Ruido')

plt.tight_layout()
plt.show()
