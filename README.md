# Proyecto de Limpieza de Datos, Análisis Exploratorio y Predicción de Precios de Steam

## Descripción

Este proyecto tiene como objetivo realizar un análisis integral de un conjunto de datos de Steam, una plataforma de distribución digital de videojuegos. Comenzaremos con la limpieza y preparación de los datos para asegurarnos de que estén listos para su análisis. Luego, realizaremos un Análisis Exploratorio (EDA) para obtener una comprensión profunda de los patrones y características presentes en los datos. Finalmente, desarrollaremos un modelo de Machine Learning para predecir los precios de los videojuegos de Steam.

## Dataset

El conjunto de datos utilizado en este proyecto contiene información relevante sobre los videojuegos disponibles en la plataforma. Incluye atributos como nombre del juego, género, fecha de lanzamiento, puntuaciones de usuarios, precios, entre otros.

## Objetivos

- Realizar una limpieza de los datos para eliminar datos faltantes, valores duplicados y asegurar la calidad de los datos para el análisis posterior.
- Identificar patrones, tendencias y relaciones entre las diferentes características de los videojuegos utilizando técnicas de Análisis Exploratorio de Datos (EDA).
- Desarrollar un modelo de Machine Learning para predecir los precios de los videojuegos de Steam basado en su genero.
- Evaluar el rendimiento del modelo y compararlo con métricas para determinar su precisión.

## Tecnologías utilizadas

- Python: Se utilizará Python como lenguaje principal para la implementación de la limpieza de datos, EDA y el desarrollo del modelo de Machine Learning.
- Bibliotecas de Python: Pandas, NumPy, Matplotlib y Scikit-learn serán algunas de las bibliotecas clave utilizadas en el proyecto.

## Limpieza
En este caso decido eliminar valores duplicados como se ve en la tabla
| Title                                            | Género              | Fecha de lanzamiento | Precio  |
|--------------------------------------------------|---------------------|---------------------|---------|
| Batman: Arkham City - Game of the Year Edition   | Action,Adventure    | 2012-09-07          | $19.99  |
| The Dream Machine: Chapter 4                     | Adventure           | 2013-08-05          | $4.99   |
| The Dream Machine: Chapter 4                     | Adventure           | 2013-08-05          | $4.99   |
| Batman: Arkham City - Game of the Year Edition   | Action,Adventure    | 2012-09-07          | $19.99  |

## EDA

Durante el Análisis Exploratorio de Datos (EDA), realicé una investigación sobre las diferentes características de los videojuegos presentes en el conjunto de datos de Steam. Mi objetivo principal era identificar las columnas más relevantes y significativas para el desarrollo del modelo de Machine Learning para predecir los precios de los videojuegos.
**Visualización de Distribuciones:** Utilicé un gráfico para comprender cómo se distribuyen los precios de los videojuegos en función de sus géneros

![grafico](https://i.imgur.com/W9oQlOC.png)

### Características Seleccionadas para el Modelo

Basándonos en los resultados de nuestro análisis exploratorio, seleccioné las siguientes columnas como características clave para mi modelo de Machine Learning:

| Genres                                         | Price | early_access |
|-----------------------------------------------|-------|--------------|
| ['Action', 'Casual', 'Indie', 'Simulation', 'S... | 4.99  | True         |
| ['Free to Play', 'Indie', 'RPG', 'Strategy']       | 0.00  | False        |
| ['Casual', 'Free to Play', 'Indie', 'Simulatio... | 0.99  | False        |
| ['Action', 'Adventure', 'Simulation']             | 3.99  | False        |

Luego, lo transformé a dummies para entrenar el modelo de predicción.

| Indie | Action | Adventure | Price | early_access |
|-------|--------|-----------|-------|--------------|
| 1     | 0      | 1         | 4.99  | True         |
| 0     | 1      | 0         | 0.00  | False        |
| 0     | 0      | 0         | 0.99  | False        |
| 1     | 1      | 1         | 3.99  | False        |

Utilicé estas columnas para la construcción del modelo de ML.

## Modelo de Predicción Polinomial y Resultados

Después de seleccionar las características clave y transformar los datos a dummies, procedi a construir un modelo de predicción utilizando el algoritmo de Regresión Polinomial. Este modelo nos permite capturar relaciones no lineales entre las características y el precio de los videojuegos.

### Entrenamiento del Modelo

Utilicé el conjunto de datos transformado para entrenar el modelo de Regresión Polinomial. El objetivo del modelo era predecir el precio de los videojuegos en función del género.

### Resultados del Modelo

Una vez que entrenamos el modelo, realice predicciones sobre el conjunto de datos de prueba. Evaluamos el rendimiento del modelo utilizando el error cuadrático medio (RMSE) para medir la precisión de las predicciones.

El modelo de predicción polinomial devolvió los siguientes resultados:

- Precio predicho (y_pred): [5.63, 8.30, 10.12, 13.28]
- Precio real (y_test): [4.99, 6.80, 6.98, 8.09]
- Error Cuadrático Medio (RMSE): 13.99

### Gráfico Comparativo

A continuación, se muestra un gráfico comparativo entre los precios reales (y_test) y los precios predichos por el modelo (y_pred). Este gráfico nos permite visualizar cómo se comparan las predicciones con los valores reales y cómo de cerca está nuestro modelo de ajustarse a los datos.

![grafico2](https://i.imgur.com/tfBLFNC.png)

El gráfico muestra que las predicciones del modelo están relativamente cerca de los valores reales, lo que indica que el modelo ha sido capaz de capturar patrones y relaciones importantes en los datos.

## Instrucciones para Reproducir el Proyecto

1. Clona este repositorio en tu máquina local.

```bash
git clone https://github.com/Nehuen505/PI-ML.git
```

2. Instala las dependencias requeridas

```bash
pip install -r requirements.txt
```

3. Explora los notebooks en el orden deseado para seguir el flujo del proyecto.
