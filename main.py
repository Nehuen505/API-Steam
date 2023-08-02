from fastapi import FastAPI
import ast
import pandas as pd


app = FastAPI()

df = pd.read_csv('steam_games_limpio.csv', encoding='utf-8')

@app.on_event("startup")
async def startup_event():
    # Convertir la columna "release_date" al tipo datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    # Eliminamos una columna
    df.drop('Unnamed: 0', axis=1,inplace=True)
    pass

@app.get('/')
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
    top_5_genres_str = '\n'.join(top_5_genres)

    return top_5_genres_str

@app.get('/')
def juegos(Año:str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]

    # Eliminar los valores nulos en la columna "app_name"
    df_filtered = df_filtered.dropna(subset=['app_name'])

    return df_filtered['app_name']

@app.get('/')
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
    top_5_specs_str = '\n'.join(top_5_specs)
    
    return top_5_specs_str

@app.get('/')
def earlyacces(Año:str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    # Contar la cantidad de juegos con early access
    cantidad_early_access = df_filtered['early_access'].sum()
    
    return cantidad_early_access

@app.get('/')
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

@app.get('/')
def metascore(Año: str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    # Eliminar los valores nulos en la columna "metascore"
    df_filtered = df_filtered['metascore'].dropna()
    
    # Ordenar el DataFrame por la columna "metascore" de forma descendente para obtener los mejores puntajes primero
    df_sorted = df_filtered.sort_values(ascending=False)
    
    # Tomar los primeros 5 juegos con mayor metascore
    top_5_games = df_sorted.head(5)
    
    return top_5_games