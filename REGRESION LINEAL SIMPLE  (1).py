#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Ejemplo regresión lineal simple
"""Supóngase que un analista de deportes quiere saber si existe una relación entre el número de veces que batean los jugadores 
de un equipo de béisbol y el número de runs que consigue. 
En caso de existir y de establecer un modelo, podría predecir el resultado del partido"""

#Librerías: Las librerías utilizadas en este ejemplo para una Regresión Lineal Simple son:

"""# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

(Matplotlib es una biblioteca para la generación de gráficos a partir de datos contenidos en listas o arrays en el lenguaje 
de programación Python y su extensión matemática NumPy.)
(Seaborn es una biblioteca de visualización de datos para Python que se ejecuta sobre la popular biblioteca de visualización 
de datos Matplotlib, aunque proporciona una interfaz sencilla y gráficos de mejor aspecto estético.)


# Preprocesado y modelado
# ==============================================================================

###from scipy.stats import pearsonr

(Coeficiente de correlación de Pearson y valor p para probar la no correlación.
El coeficiente de correlación de Pearson mide la relación lineal entre dos conjuntos de datos. 
El cálculo del valor p se basa en el supuesto de que cada conjunto de datos se distribuye normalmente)

(El valor p indica aproximadamente la probabilidad de que un sistema no correlacionado produzca conjuntos de datos 
que tengan una correlación de Pearson al menos tan extrema como la calculada a partir de estos conjuntos de datos.

Parámetros
x (N,) array_like
Matriz de entrada.

y (N,) array_like
Matriz de entrada.

Devoluciones
r flotar
Coeficiente de correlación de Pearson.

flotación del valor p
Valor p de dos colas.)

###from sklearn.linear_model import LinearRegression
(Codigo: Regresion lineal)

###from sklearn.model_selection import train_test_split
(La función sklearn. model_selection. train_test_split nos permite dividir un dataset en dos bloques, 
típicamente bloques destinados al entrenamiento y validación del modelo (llamemos
a estos bloques "bloque de entrenamiento " y "bloque de pruebas" para mantener la coherencia con el nombre de la función))

###from sklearn.metrics import r2_score
(Medida para el R2)
###from sklearn.metrics import mean_squared_error
(Error estandar)
###import statsmodels.api as sm
(Descrpción modelo)
###import statsmodels.formula.api as smf
(Estadisticas para la regresión del modelo)

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')
(Sirve para ignorar advertencias de control de python)

"""


# In[5]:


# Tratamiento de datos
import pandas as pd
import numpy as np


# In[6]:


# Gráficos
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# In[7]:


# Preprocesado y modelado
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[8]:


# Configuración matplotlib
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')


# In[9]:


# Configuración warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


"""
En Python se usan los paréntesis ( y ) para crear tuplas, definir y llamar funciones e indicar 
el orden de evaluación de expresiones. Los corchetes [ y ], para crear listas y acceder a elementos de colecciones. 
Las llaves { y }, para crear diccionarios y formatear cadenas de texto.
#Un diccionario es un tipo de datos que sirve para asociar pares de objetos. 
Un diccionario puede ser visto como una colección de llaves, cada una de las cuales tiene asociada un valor.
Un Diccionario es una estructura de datos y un tipo de dato en Python con características especiales que nos permite 
almacenar cualquier tipo de valor como enteros, cadenas, listas e incluso otras funciones.

Python es un lenguaje orientado a objetos

Cuando creas una variable y le asignas un valor entero, ese valor es un objeto; una función es un objeto; las listas, tuplas, diccionarios, conjuntos, 
… son objetos; una cadena de caracteres es un objeto. Y así podría seguir indefinidamente.
Todo en Python es un objeto, y casi todo tiene atributos y métodos. Los diferentes lenguajes de programación definen "objeto" de maneras diferentes. 
En algunos significa que todos los objetos deben tener atributos y métodos; en otros esto significa que todos los objetos pueden tener subclases.

"""


# In[10]:


# Datos
# ===============================================================================
equipos = ["Texas","Boston","Detroit","Kansas","St.","New_S.","New_Y.",
           "Milwaukee","Colorado","Houston","Baltimore","Los_An.","Chicago",
           "Cincinnati","Los_P.","Philadelphia","Chicago","Cleveland","Arizona",
           "Toronto","Minnesota","Florida","Pittsburgh","Oakland","Tampa",
           "Atlanta","Washington","San.F","San.I","Seattle"]
bateos = [5659,  5710, 5563, 5672, 5532, 5600, 5518, 5447, 5544, 5598,
          5585, 5436, 5549, 5612, 5513, 5579, 5502, 5509, 5421, 5559,
          5487, 5508, 5421, 5452, 5436, 5528, 5441, 5486, 5417, 5421]

runs = [855, 875, 787, 730, 762, 718, 867, 721, 735, 615, 708, 644, 654, 735,
        667, 713, 654, 704, 731, 743, 619, 625, 610, 645, 707, 641, 624, 570,
        593, 556]

datos = pd.DataFrame({'equipos': equipos, 'bateos': bateos, 'runs': runs})
datos.head(3)


# In[ ]:


"""¿Qué significa Figsize en Python?
Matplotlib permite especificar la relación de aspecto, DPI (Dots per inches= puntos por pulgada) y el tamaño de la figura cuando 
se crea el objeto Figura, utilizando los argumentos de palabras clave
figsize y dpi.figsize es una tupla del ancho y alto de la figura en pulgadas, 
y dpi son los puntos por pulgada (píxel por pulgada)."""


# In[19]:


#Representación gráfica:
#El primer paso antes de generar un modelo de regresión simple es representar los datos para poder 
#intuir si existe una relación y cuantificar dicha relación mediante un coeficiente de correlación.

# Gráfico
# ==============================================================================
#Dimensiones fig,ax: eje (a= ancho, b=largo) como quiero que se vea el grafico
fig, ax = plt.subplots(figsize=(6, 3.84))
#Se definen los datos x=predictora y=dependiente c= hace referencia al color "b: blue- g: green - r: red - c: cyan -m: magenta y: yellow -k: black -w: white" 
#kind sirve para definir el tipo de grafico en este caso de dispersi+on(scatter)
datos.plot(
    x    = 'bateos',
    y    = 'runs',
    c    = 'purple',
    kind = "scatter",
    ax   = ax
)
ax.set_title('Distribución de bateos y runs');


# In[20]:


# Correlación lineal entre las dos variables
# ==============================================================================
#(para correr la correlación de pearsonr se llama el directorio datos en donde se encuentran guardadas las listas de bateos y runs)
corr_test = pearsonr(x = datos['bateos'], y =  datos['runs'])

#corr_test(0) muestra coefeciente de correlación y corr_test(1) muestra p-value
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])


#El gráfico y el test de correlación muestran una relación lineal, de intensidad considerable (r = 0.61) 
#y significativa (p-value = 0.000339). 

#Tiene sentido intentar generar un modelo de regresión lineal con el objetivo de predecir el número de runs en función del número de bateos del equipo.


# In[ ]:


#Ajuste del modelo
#Se ajusta un modelo empleando como variable respuesta runs y como predictor bateos. 
#Como en todo estudio predictivo, no solo es importante ajustar el modelo, sino también cuantificar su capacidad para predecir nuevas observaciones. 
#Para poder hacer esta evaluación, se dividen los datos en dos grupos, uno de entrenamiento y otro de test.


# In[21]:


#Scikit-learn
# División de los datos en train(entrenamiento) y test
# ==============================================================================
X = datos[['bateos']]
y = datos['runs']

#También con esta función es posible utilizar el valor -1,1
#para dejar que sea Numpy quien calcule el tamaño adecuado para la dimensión correspondiente.
#Simplemente significa que es una dimensión desconocida y queremos que Numpy se dé cuenta. 
#Y numpy calculará esto mirando la 'longitud de la matriz y las dimensiones restantes' 
#y asegurándose de que cumpla con los criterios mencionados anteriormente.


X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
#dimensión de la matriz X y Y la calcula numpy con base en el set de datos partiendo de -1 a 1

#Train_size como 0,8, nuestro objetivo es poner el 80% de los datos en nuestro conjunto de entrenamiento, 
#y el resto de los datos en el conjunto de prueba puede valer entre 0 y 1.

#RandomStates un generador de números pseudoaleatorios, lo que significa 
#que no puede generar números verdaderamente aleatorios, sino sólo números que "parecen" aleatorios.
#Para hacer esto, necesita darle alguna "semilla" inicial que pueda usar para generar los números.
#El argumento al que te refieres es la semilla 1234; Preferiblemente, debería ser único para cada llamada de función, 
#ya que si se llama con la misma semilla dos veces, generará exactamente la misma secuencia de números.

#shuffle = Falso. Ya sea para mezclar los datos antes de dividirlos en lotes. Tenga en cuenta que las muestras dentro de cada división no se mezclarán.
#random_state int, instancia de RandomState o None, predeterminado = None
#Cuando shufflees Verdadero, random_stateafecta el orden de los índices, que controla la aleatoriedad de cada pliegue. De lo contrario, este parámetro no tiene ningún efecto. Pase un int para una salida reproducible a través de múltiples llamadas a funciones. 
#shufflebool, default=True.Si mezclar o no los datos antes de dividirlos. f shuffle=False  estratificar debe ser None.


# In[ ]:


"""Parámetros
* secuencia de matrices de indexables con la misma longitud / forma [0]
Las entradas permitidas son listas, matrices numpy, matrices scipy-sparse o marcos de datos pandas.

test_size float o int, predeterminado = Ninguno
Si es flotante, debe estar entre 0.0 y 1.0 y representar la proporción del conjunto de datos para incluir en la división de prueba. Si es int, representa el número absoluto de muestras de prueba. Si es Ninguno, el valor se establece como complemento del tamaño del tren. Si train_sizetambién es Ninguno, se establecerá en 0,25.

train_size float o int, predeterminado = Ninguno
Si es flotante, debe estar entre 0.0 y 1.0 y representar la proporción del conjunto de datos para incluir en la división del tren. 
Si es int, representa el número absoluto de muestras de trenes. 
Si es Ninguno, el valor se establece automáticamente como complemento del tamaño de la prueba.

random_state int, instancia de RandomState o None, predeterminado = None
Controla la mezcla aplicada a los datos antes de aplicar la división. Pase un int para una salida reproducible a través de múltiples llamadas a funciones. Consulte el glosario .

shuffle bool, predeterminado = True
Si mezclar o no los datos antes de dividirlos. Si shuffle = False, estratificar debe ser Ninguno.

estratificar como una matriz, predeterminado = Ninguno
Si no es Ninguno, los datos se dividen de forma estratificada, utilizando esto como etiquetas de clase. Leer más en la Guía del usuario .

Devoluciones
lista de división , longitud = 2 * len (matrices)
Lista que contiene la división de entradas de prueba de tren.

Nuevo en la versión 0.16: si la entrada es escasa, la salida será a scipy.sparse.csr_matrix. De lo contrario, el tipo de salida es el mismo que el tipo de entrada."""


# In[22]:


# Creación del modelo
# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train.reshape(-1, 1), y = y_train)


# In[23]:


# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))


# In[ ]:


# https://www.cienciadedatos.net/documentos/py10-regresion-lineal-python.html

