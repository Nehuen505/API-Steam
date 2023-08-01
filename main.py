from fastapi import FastAPI
import ast
import pandas as pd


app = FastAPI()

rows = []
with open('steam_games.json') as f:
    for line in f.readlines():
        rows.append(ast.literal_eval(line))

df = pd.DataFrame(rows)

@app.on_event("startup")
async def startup_event():
    # Convertir la columna "release_date" al tipo datetime
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    pass

@app.get('/')
def genero(Año:str):
    # Filtrar solo los registros correspondientes al año ingresado
    df_filtered = df[df['release_date'].dt.year == int(Año)]

    # Eliminar los valores nulos en la columna "genres"
    df_filtered = df_filtered.dropna(subset=['genres'])

    # Unir todas las listas de géneros en una única lista
    all_genres = [genre for sublist in df_filtered['genres'] for genre in sublist]

    # Contar la frecuencia de cada género
    genre_counts = pd.Series(all_genres).value_counts()

    # Obtener los 5 géneros más vendidos en orden correspondiente
    top_5_genres = genre_counts.head(5).index.tolist()

    # Agregar un salto de línea después de cada género en la lista
    top_5_genres_str = '\n'.join(top_5_genres)

    return top_5_genres_str