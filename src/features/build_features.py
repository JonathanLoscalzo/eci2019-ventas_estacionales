import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def impute_values(df):
    """
    está funcion se encaga de imputar los valores del dataframe que recibe (todas las columnas por igual)
    """
    #estrategia de imputación, con valores medios.
    imp = SimpleImputer(strategy='mean')

    #imputamos
    df_impute = pd.DataFrame(imp.fit_transform(df))
    df_impute.columns = df.columns
    return df_impute

def assign_tipo_pos(df_pos, df_ventas_pos_freq):
    df_pos.loc[lambda df: 
               (df.ventas >= df_ventas_pos_freq.loc["POCAS","ini"]) 
               & (df.ventas <= df_ventas_pos_freq.loc["POCAS","fin"]), "tipo_pos_ventas"] = "POCAS"
    
    df_pos.loc[lambda df: 
               (df.ventas >= df_ventas_pos_freq.loc["MEDIO","ini"]) 
               & (df.ventas <= df_ventas_pos_freq.loc["MEDIO","fin"]), "tipo_pos_ventas"] = "MEDIO"
    
    df_pos.loc[lambda df: 
               (df.ventas >= df_ventas_pos_freq.loc["MUCHAS","ini"]) 
               & (df.ventas <= df_ventas_pos_freq.loc["MUCHAS","fin"]), "tipo_pos_ventas"] = "MUCHAS"
    
    df_pos.loc[lambda df: 
               (df.ventas >= df_ventas_pos_freq.loc["SIN VENTAS","ini"]) 
               & (df.ventas <= df_ventas_pos_freq.loc["SIN VENTAS","fin"]), "tipo_pos_ventas"] = "SIN VENTAS"

def add_date_features(df, col_name="fecha"):
    """
    Esta función transforma una columna que se recibe como parámetro, 
    en varias columnas derivadas. 
    """
    df['date'] = df[col_name]
#     df['dayofweek'] = df['date'].dt.dayofweek
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
#     df['dayofyear'] = df['date'].dt.dayofyear
#     df['dayofmonth'] = df['date'].dt.day
#     df['weekofyear'] = df['date'].dt.weekofyear
#     df['is_weekend'] = np.where(df['date'].dt.dayofweek.isin([5,6]), 1, 0)
    df = df.drop([col_name,'date'], axis=1, errors='ignore')
    
    return df

def group_canal(df_ventas):
    df_ventas.canal.loc[df_ventas.canal != "ALMACEN"] = "OTROS"


def assign_unidades_anteriores(ventas_totales, year, month):
    return (
        ventas_totales
        .set_index('id_pos')
        [lambda df: (df.year == year) & (df.month==month)]
        .unidades
        .reindex(ventas_totales.id_pos.unique())
        .fillna(0) # o -1?
    )
