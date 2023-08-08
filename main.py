from fastapi import FastAPI
import ast
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

app = FastAPI()

df = pd.read_csv('steam_games_limpio.csv', encoding='utf-8')

@app.on_event("startup")
async def startup_event():
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df.drop('Unnamed: 0', axis=1,inplace=True)
    
    pass

@app.get('/genero')
def genero(Año: str):
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    df_filtered = df_filtered.dropna(subset=['genres'])
    
    df_filtered['genres'] = df_filtered['genres'].apply(ast.literal_eval)
    
    all_genres = [genre for sublist in df_filtered['genres'] for genre in sublist]
    
    genre_counts = pd.Series(all_genres).value_counts()
    
    top_genres_dict = genre_counts.head(5).to_dict()
    
    return top_genres_dict

@app.get('/juegos')
def juegos(Año:str):
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    df_filtered = df_filtered.dropna(subset=['app_name'])
    
    return df_filtered['app_name']

@app.get('/specs')
def specs(Año:str):
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    df_filtered = df_filtered.dropna(subset=['specs'])
    
    df_filtered['specs'] = df_filtered['specs'].apply(ast.literal_eval)
    
    all_specs = [specs for sublist in df_filtered['specs'] for specs in sublist]
    
    specs_counts = pd.Series(all_specs).value_counts()
    
    top_specs_dict = specs_counts.head(5).to_dict()
    
    return top_specs_dict

@app.get('/earlyaccess')
def earlyacces(Año:str):
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    df_filtered = df_filtered.dropna(subset=['early_access'])
    
    cantidad_early_access = df_filtered['early_access'].sum()
    
    cantidad_early_access = int(cantidad_early_access)
    
    return cantidad_early_access

@app.get('/sentiment')
def sentiment(Año: str):
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    df_filtered = df_filtered.dropna(subset=['sentiment'])
    
    df_filtered = df_filtered[~df_filtered['sentiment'].str.contains('user reviews')]
    
    sentiment_counts = df_filtered['sentiment'].value_counts()
    
    sentiment_dict = sentiment_counts.to_dict()
    
    return sentiment_dict

@app.get('/metascore')
def metascore(Año: str):
    df_filtered = df[df['release_date'].dt.year == int(Año)]
    
    df_filtered = df_filtered.dropna(subset=['metascore'])
    
    df_sorted = df_filtered.sort_values(by='metascore', ascending=False)
    
    top_5_games = df_sorted.head(5)
    
    juegos_y_metascore = dict(zip(top_5_games['title'], top_5_games['metascore']))
    
    return juegos_y_metascore

# -----ML-----

# Cargar los datos y el modelo desde el archivo pkl
with open('modelo_ml.pkl', 'rb') as file:
    data = pickle.load(file)

modelo_regresion = data['modelo']
X_test_poly = data['X_test_poly']
y_test = data['y_test']
y_pred = data['y_pred']
poly = data['poly']
X = data['X']
generos_permitidos = ['Accounting', 'Action', 'Adventure', 'Animation & Modeling',
       'Audio Production', 'Casual', 'Design & Illustration',
       'Education', 'Indie', 'Massively Multiplayer', 'Photo Editing', 'RPG',
       'Racing', 'Simulation', 'Software Training', 'Sports', 'Strategy',
       'Utilities', 'Video Production', 'Web Publishing']

@app.get('/prediccion')
def predecir_precio_y_rmse(generos: str, early_access: bool):
    generos_ingresados = generos.split(',')
    
    generos_no_permitidos = [genero for genero in generos_ingresados if genero not in generos_permitidos]
    
    if generos_no_permitidos:
        return {"message": f"Los siguientes géneros no están permitidos: {', '.join(generos_no_permitidos)}. Debes elegir entre los siguientes géneros: {', '.join(generos_permitidos)}."}
    else:
        generos_a_predecir_df = pd.DataFrame({genero: [1 if genero in generos_ingresados else 0] for genero in X.columns})
        generos_a_predecir_df['early_access'] = early_access
        generos_a_predecir_poly = poly.transform(generos_a_predecir_df)
        precio_predicho = modelo_regresion.predict(generos_a_predecir_poly)[0]
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return {"Precio predicho": round(precio_predicho, 2), "RMSE": round(rmse, 2)}