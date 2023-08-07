from fastapi import FastAPI
import ast
import pandas as pd

import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

app = FastAPI()

df = pd.read_csv('steam_games_limpio.csv', encoding='utf-8')

@app.on_event("startup")
async def startup_event():
    # Convertir la columna "release_date" al tipo datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    # Eliminamos una columna
    df.drop('Unnamed: 0', axis=1,inplace=True)
    pass

@app.get('/genero')
def genero(Año: str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]

    # Eliminar los valores nulos en la columna "genres"
    df_filtered = df_filtered.dropna(subset=['genres'])

    # Convertir las cadenas de géneros en listas reales
    df_filtered['genres'] = df_filtered['genres'].apply(ast.literal_eval)

    # Unir todas las listas de géneros en una única lista
    all_genres = [genre for sublist in df_filtered['genres'] for genre in sublist]

    # Contar la frecuencia de cada género
    genre_counts = pd.Series(all_genres).value_counts()

    # Obtener los 5 géneros más vendidos en orden correspondiente
    top_5_genres = genre_counts.head(5).index.tolist()

    # Convertir la lista de géneros en una cadena con saltos de línea
    top_5_genres_str = ', '.join(top_5_genres)

    return top_5_genres_str

@app.get('/juegos')
def juegos(Año:str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]

    # Eliminar los valores nulos en la columna "app_name"
    df_filtered = df_filtered.dropna(subset=['app_name'])

    return df_filtered['app_name']

@app.get('/specs')
def specs(Año:str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]

    # Eliminar los valores nulos en la columna "specs"
    df_filtered = df_filtered.dropna(subset=['specs'])

    # Convertir las cadenas de specs en listas reales
    df_filtered['specs'] = df_filtered['specs'].apply(ast.literal_eval)

    # Unir todas las listas de specs en una única lista
    all_specs = [specs for sublist in df_filtered['specs'] for specs in sublist]

    # Contar la frecuencia de cada género
    specs_counts = pd.Series(all_specs).value_counts()

    # Obtener los 5 géneros más vendidos en orden correspondiente
    top_5_specs = specs_counts.head(5).index.tolist()

    # Convertir la lista de géneros en una cadena con saltos de línea
    top_5_specs_str = ', '.join(top_5_specs)
    
    return top_5_specs_str

@app.get('/earlyaccess')
def earlyacces(Año:str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    # Eliminar los valores nulos en la columna "early_access"
    df_filtered = df_filtered.dropna(subset=['early_access'])
    
    # Contar la cantidad de juegos con early access
    cantidad_early_access = df_filtered['early_access'].sum()
    
    # Convertir el resultado a un tipo de dato nativo de Python (int)
    cantidad_early_access = int(cantidad_early_access)
    
    return cantidad_early_access

@app.get('/sentiment')
def sentiment(Año: str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    # Eliminar los valores nulos en la columna "sentiment"
    df_filtered = df_filtered.dropna(subset=['sentiment'])
    
    # Eliminar las filas que contienen la categoría de sentimiento que tiene "user reviews"
    df_filtered = df_filtered[~df_filtered['sentiment'].str.contains('user reviews')]
    
    # Contar la cantidad de registros para cada categoría de sentimiento
    sentiment_counts = df_filtered['sentiment'].value_counts()
    
    # Convertir la serie de conteos en un diccionario
    sentiment_dict = sentiment_counts.to_dict()
    
    return sentiment_dict

@app.get('/metascore')
def metascore(Año: str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    # Eliminar los valores nulos en la columna "metascore"
    df_filtered = df_filtered.dropna(subset=['metascore'])
    
    # Ordenar el DataFrame por la columna "metascore" de forma descendente para obtener los mejores puntajes primero
    df_sorted = df_filtered.sort_values(by='metascore', ascending=False)
    
    # Tomar los primeros 5 juegos con mayor metascore
    top_5_games = df_sorted.head(5)
    
    # Obtener la lista de nombres de los juegos y sus puntajes de metascore como una lista de tuplas
    juegos_y_metascore = [(nombre, puntaje) for nombre, puntaje in zip(top_5_games['title'], top_5_games['metascore'])]
    
    return juegos_y_metascore

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Cargar los datos y el modelo desde el archivo pkl
with open('modelo.pkl', 'rb') as file:
    data = pickle.load(file)

modelo_regresion = data['modelo']
X_test_poly = data['X_test_poly']
y_test = data['y_test']
y_pred = data['y_pred']
poly = data['poly']
X = data['X']

print(type(X))
print(type(poly))

@app.get('/prediccion')
def predecir_precio_y_rmse(generos: str, early_access: int):
    # Crear un DataFrame con las características de los géneros a predecir
    generos_a_predecir_df = pd.DataFrame({genero: [1 if genero in generos.split(',') else 0] for genero in X.columns})

    # Agregar la columna 'early_access' al DataFrame con el valor proporcionado por el usuario
    generos_a_predecir_df['early_access'] = early_access

    # Transformar las características de los géneros a predecir utilizando el mismo transformador polinomial
    generos_a_predecir_poly = poly.transform(generos_a_predecir_df)

    # Realizar la predicción utilizando el modelo cargado
    precio_predicho = modelo_regresion.predict(generos_a_predecir_poly)[0]

    # Calcular el Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)

    # Calcular el Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    return {"precio_predicho": round(precio_predicho, 2), "rmse": rmse}

print(predecir_precio_y_rmse('Action', True))