# =========================
# STREAMLIT APP COMPLETO
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

from PIL import Image
# ── MODELADO | Setup ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, accuracy_score)
from utils.load_data import *

# st.set_page_config(layout="wide")
def show():

    df = load_data()
    def map_mutcd_to_severity(mutcd_category):
        if mutcd_category == 'Menor':
            return 1
        elif mutcd_category == 'Intermedio':
            return 2
        elif mutcd_category == 'Mayor':
            return 3
        return None
    # Función reutilizable de clasificación de riesgo por cuartiles
    def clasificar_ocurrencia(serie):
        p25 = serie.quantile(0.25)
        p50 = serie.quantile(0.50)
        p75 = serie.quantile(0.75)
        def nivel(val):
            if val <= p25:   return 'Bajo'
            elif val <= p50: return 'Moderado'
            elif val <= p75: return 'Alto'
            else:            return 'Crítico'
        return serie.apply(nivel)

    # Función para clasificar según tiempo real (no por severidad)
    def clasificar_mutcd_por_tiempo(duracion):
        """Clasificación MUTCD basada en duración real del evento"""
        if pd.isna(duracion):
            return 'Menor'  # Asignar por defecto si no hay duración
        elif duracion < 30:
            return 'Menor'
        elif duracion <= 120:  # 2 horas = 120 minutos
            return 'Intermedio'
        else:
            return 'Mayor'
    st.markdown("""
    ### C. RESULTADOS DE MODELOS PREDICTIVOS
    ####  3.1 Comparación de Modelos + Feature Importance
    """)  
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier   

    # Extraer variables temporales
    df['Start_Time'] = pd.to_datetime(
        df['Start_Time'], errors='coerce'
    )
    df['Hour']      = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month']     = df['Start_Time'].dt.month_name()
    # Calculamos la duración de cada accidente desde Start_Time y End_Time
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60        

    # Consideramos válidas duraciones entre 1 minuto y 24 horas
    duration_clean = df[
        (df['Duration_min'] >= 1) &
        (df['Duration_min'] <= 1440)
    ].copy() # Explicitly create a copy to avoid SettingWithCopyWarning

    # Crear MUTCD_Category basada en duración real, no en severidad
    df['MUTCD_Category'] = df['Duration_min'].apply(clasificar_mutcd_por_tiempo)

    # Crear MUTCD_Category usando .loc para evitar SettingWithCopyWarning
    duration_clean.loc[:, 'MUTCD_Category'] = duration_clean['Duration_min'].apply(clasificar_mutcd_por_tiempo)    
    
    df['MUTCD_Severity'] = (
        df['MUTCD_Category'].apply(map_mutcd_to_severity)
    )      

    # Excluir MUTCD_Category del encoding porque la usaremos como Target
    scaled_categorical_columns = df.select_dtypes(
        include=['object', 'bool']
    ).columns.to_list()

    # Remover MUTCD_Category para no encodearla
    scaled_categorical_columns = [
        col for col in scaled_categorical_columns
        if col != 'MUTCD_Category'
    ]

    df_label_encoded = df.copy()
    le = LabelEncoder()
    for col in scaled_categorical_columns:
        df_label_encoded[col] = le.fit_transform(df_label_encoded[col].astype(str))

    # 1. Dataset con encoding
    df = df_label_encoded.copy()

    # 2. Agregar Hour si no existe
    if 'Hour' not in df.columns:
        df['Hour'] = pd.to_datetime(
            df['Start_Time'], errors='coerce'
        ).dt.hour

    # 3. Seleccionar columnas relevantes
    cols = [
        'MUTCD_Severity',
        # Geográficas
        'Start_Lat', 'Start_Lng', 'Street', 'City', 'County',
        # Infraestructura vial
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
        'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
        'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
        # Climáticas
        'Wind_Chill(F)', 'Temperature(F)', 'Humidity(%)',
        'Visibility(mi)', 'Wind_Speed(mph)',
        'Weather_Condition', 'Precipitation(in)',
        # Temporal
        'Hour'
    ]
    df = df[cols].dropna()

    # 4. Codificar Weather_Condition
    df = pd.get_dummies(df, columns=['Weather_Condition'], drop_first=True)

    # 5. Variable objetivo — 3 categorías
    df['Target'] = pd.cut(
        df['MUTCD_Severity'],
        bins=[0, 1, 2, 4],
        labels=['Low', 'Moderate', 'High']
    )
    df = df.drop(columns=['MUTCD_Severity'])
    df = df.dropna(subset=['Target'])

    print('Distribución de la variable objetivo:')
    print(df['Target'].value_counts())

    # 6. X e y sin split aún
    X = df.drop('Target', axis=1)
    y = df['Target']   

    # Undersampling — igualar todas las clases al mínimo
    print('Distribución antes del balanceo:')
    print(y.value_counts())

    min_class_count = y.value_counts().min()

    # 1. Une X e y en un solo DataFrame para poder muestrear juntos
    df_balanced = pd.concat([X, y], axis=1)

    # 2.  Agrupa por clase (Low/Moderate/High) y toma exactamente
    #     min_class_count muestras de CADA grupo → undersampling
    df_balanced = (
        df_balanced
        .groupby('Target', group_keys=False, observed=True)
        .apply(lambda x: x.sample(n=min_class_count, random_state=42))
        .reset_index(drop=True)
        # 3. Mezcla aleatoriamente las filas para que no queden
        #    agrupadas por clase al entrenar
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    X = df_balanced.drop(columns=['Target'])
    y = df_balanced['Target']

    print(f'\nDistribución después del balanceo:')
    print(y.value_counts())
    print(f'\nTotal registros balanceados: {len(df_balanced):,}')

    # Split 80/20 DESPUÉS del balanceo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print('=' * 55)
    print('           RANDOM FOREST CLASSIFIER')
    print('=' * 55)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Validación cruzada
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy')
    print(f'\n Validación Cruzada (5-Fold):')
    print(f'   Scores:  {cv_scores_rf.round(4)}')
    print(f'   Media:   {cv_scores_rf.mean():.4f}')
    print(f'   Std:     {cv_scores_rf.std():.4f}')

    # Entrenamiento y evaluación
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf  = rf.predict_proba(X_test)

    print('=' * 55)
    print('           EXTRA TREES CLASSIFIER')
    print('=' * 55)

    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # Validación cruzada
    cv_scores_et = cross_val_score(et, X_train, y_train, cv=kf, scoring='accuracy')
    print(f'\n Validación Cruzada (5-Fold):')
    print(f'   Scores:  {cv_scores_et.round(4)}')
    print(f'   Media:   {cv_scores_et.mean():.4f}')
    print(f'   Std:     {cv_scores_et.std():.4f}')

    # Entrenamiento y evaluación
    et.fit(X_train, y_train)
    y_pred_et = et.predict(X_test)
    y_prob_et  = et.predict_proba(X_test)

    ### 7.5 XGBoost Classifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    print('=' * 55)
    print('           XGBOOST CLASSIFIER')
    print('=' * 55)

    # XGBoost necesita el target como entero, no string
    le_target = LabelEncoder()
    y_train_xgb = le_target.fit_transform(y_train)
    y_test_xgb  = le_target.transform(y_test)

    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',   # rápido para datasets grandes
        eval_metric='mlogloss'
    )

    # Validación cruzada
    cv_scores_xgb = cross_val_score(xgb, X_train, y_train_xgb, cv=kf, scoring='accuracy')
    print(f'\n Validación Cruzada (5-Fold):')
    print(f'   Scores:  {cv_scores_xgb.round(4)}')
    print(f'   Media:   {cv_scores_xgb.mean():.4f}')
    print(f'   Std:     {cv_scores_xgb.std():.4f}')

    # Entrenamiento y evaluación
    xgb.fit(X_train, y_train_xgb)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)

    print('=' * 55)
    print('         COMPARACIÓN DE MODELOS')
    print('=' * 55)

    results_df = pd.DataFrame({
        'Modelo':        ['Random Forest', 'Extra Trees', 'XGBoost'],
        'CV Media':      [cv_scores_rf.mean(), cv_scores_et.mean(), cv_scores_xgb.mean()],
        'CV Std':        [cv_scores_rf.std(),  cv_scores_et.std(),  cv_scores_xgb.std()],
        'Test Accuracy': [accuracy_score(y_test, y_pred_rf),
                        accuracy_score(y_test, y_pred_et),
                        accuracy_score(y_test_xgb, y_pred_xgb)],
        'ROC AUC':       [
            roc_auc_score(y_test, y_prob_rf,  multi_class='ovr', average='weighted'),
            roc_auc_score(y_test, y_prob_et,  multi_class='ovr', average='weighted'),
            roc_auc_score(y_test_xgb, y_prob_xgb, multi_class='ovr', average='weighted')
        ]
    }).round(4)

    # display(results_df)
    st.write(results_df)

    st.markdown("""
    - Random Forest presenta el mejor desempeño global, con mayor accuracy (0.60) y 
    mejor capacidad de discriminación (ROC AUC = 0.80), por lo que se selecciona como modelo final.
    ---
    """)  
    # Mejor modelo por ROC AUC
    best_idx   = results_df['ROC AUC'].idxmax()
    best_name  = results_df.loc[best_idx, 'Modelo']
    best_model = rf if best_idx == 0 else et
    print(f'\n Mejor modelo: {best_name}')

    # Feature Importance
    feat_importance = (pd.DataFrame({
        'Feature':    X.columns,
        'Importance': best_model.feature_importances_
    })
    .sort_values('Importance', ascending=False)
    .head(15))

    fig10 = px.bar(feat_importance, x='Importance', y='Feature',
                orientation='h',
                color='Importance', color_continuous_scale='PuBuGn',
                template='seaborn',
                title=f'Top 15 Variables más Importantes — {best_name}')
    fig10.update_layout(height=600, width=1000, font=dict(size=13))

    st.plotly_chart(fig10, use_container_width=True)
 
    st.markdown("""
    - Las variables más influyentes son principalmente geográficas: Street, Start_Lat y Start_Lng
    - Factores temporales y ambientales como Hour, Humidity, Temperature y Wind Speed también tienen impacto relevante.
    - Variables como precipitación, visibilidad o señales de tráfico tienen menor importancia en la predicción
    ---
    """)   
