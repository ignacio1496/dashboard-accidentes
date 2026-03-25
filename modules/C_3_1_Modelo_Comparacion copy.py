# =========================
# STREAMLIT APP COMPLETO
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import joblib

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

from PIL import Image

# ── MODELADO ──────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils.load_data import *

# =========================
# MODELOS (RUTAS)
# =========================
MODEL_PATHS = {
    "rf": "assets/modelo_rf.pkl",
    "et": "assets/modelo_et.pkl",
    "xgb": "assets/modelo_xgb.pkl"
}

def save_model(model, path, metrics=None):
    os.makedirs("assets", exist_ok=True)
    joblib.dump({
        "model": model,
        "metrics": metrics
    }, path)   

def load_model(path):
    if os.path.exists(path):
        data = joblib.load(path)
        return data.get("model"), data.get("metrics")
    return None, None


def show():

    df = load_data()

    # =========================
    # UI CONTROL
    # =========================

    DEV_MODE = True  # cámbialo a True cuando quieras usar el botón

    if DEV_MODE:
        st.subheader("Control de Modelos")

        col1, col2 = st.columns(2)

        with col1:
            recalcular = st.button("Recalcular modelos")

        with col2:
            modelos_existentes = all(os.path.exists(p) for p in MODEL_PATHS.values())
            if modelos_existentes:
                st.success("Modelos guardados disponibles")
            else:
                st.warning("No hay modelos guardados")
    else:
        recalcular = False
    st.markdown("""
    ### C. RESULTADOS DE MODELOS PREDICTIVOS
    ####  3.1 Comparación de Modelos + Feature Importance
    """)  
    # =========================
    # PREPROCESAMIENTO 
    # =========================
    def map_mutcd_to_severity(mutcd_category):
        return {"Menor": 1, "Intermedio": 2, "Mayor": 3}.get(mutcd_category, None)

    def clasificar_mutcd_por_tiempo(duracion):
        if pd.isna(duracion):
            return 'Menor'
        elif duracion < 30:
            return 'Menor'
        elif duracion <= 120:
            return 'Intermedio'
        else:
            return 'Mayor'

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    df['MUTCD_Category'] = df['Duration_min'].apply(clasificar_mutcd_por_tiempo)
    df['MUTCD_Severity'] = df['MUTCD_Category'].apply(map_mutcd_to_severity)

    df['Hour'] = df['Start_Time'].dt.hour

    # Encoding
    scaled_categorical_columns = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    scaled_categorical_columns = [col for col in scaled_categorical_columns if col != 'MUTCD_Category']

    le = LabelEncoder()
    for col in scaled_categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))

    cols = [
        'MUTCD_Severity','Start_Lat','Start_Lng','Street','City','County',
        'Amenity','Bump','Crossing','Give_Way','Junction','No_Exit',
        'Railway','Roundabout','Station','Stop','Traffic_Calming',
        'Traffic_Signal','Turning_Loop','Wind_Chill(F)','Temperature(F)',
        'Humidity(%)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Hour'
    ]

    df = df[cols].dropna()

    df['Target'] = pd.cut(
        df['MUTCD_Severity'],
        bins=[0, 1, 2, 4],
        labels=['Low', 'Moderate', 'High']
    )

    df = df.drop(columns=['MUTCD_Severity']).dropna()

    X = df.drop('Target', axis=1)
    y = df['Target']

    # Balanceo
    min_class = y.value_counts().min()
    df_balanced = pd.concat([X, y], axis=1).groupby('Target', group_keys=False).apply(
        lambda x: x.sample(min_class, random_state=42)
    ).reset_index(drop=True)

    X = df_balanced.drop(columns=['Target'])
    y = df_balanced['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost encoding
    le_target = LabelEncoder()
    y_train_xgb = le_target.fit_transform(y_train)
    y_test_xgb = le_target.transform(y_test)

    # =========================
    # MODELOS
    # =========================
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='mlogloss'
    )

    # =========================
    # ENTRENAR O CARGAR
    # =========================
    if recalcular:
        with st.spinner("Entrenando modelos..."):

            rf.fit(X_train, y_train)
            et.fit(X_train, y_train)
            xgb.fit(X_train, y_train_xgb)
            save_model(rf, MODEL_PATHS["rf"], {
                "cv_mean": cv_scores_rf.mean(),
                "cv_std": cv_scores_rf.std()
            })

            save_model(et, MODEL_PATHS["et"], {
                "cv_mean": cv_scores_et.mean(),
                "cv_std": cv_scores_et.std()
            })

            save_model(xgb, MODEL_PATHS["xgb"], {
                "cv_mean": cv_scores_xgb.mean(),
                "cv_std": cv_scores_xgb.std()
            })

            # CV solo aquí (optimización)
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy')
            cv_scores_et = cross_val_score(et, X_train, y_train, cv=kf, scoring='accuracy')
            cv_scores_xgb = cross_val_score(xgb, X_train, y_train_xgb, cv=kf, scoring='accuracy')

            st.success("Modelos entrenados y guardados")

    else:
        rf, rf_metrics = load_model(MODEL_PATHS["rf"])
        et, et_metrics = load_model(MODEL_PATHS["et"])
        xgb, xgb_metrics = load_model(MODEL_PATHS["xgb"])

        if rf is None or et is None or xgb is None:
            st.warning("No existen modelos guardados. Presiona 'Recalcular modelos'")
            st.stop()
        else:
            st.info("Usando modelos guardados")

    # =========================
    # PREDICCIONES
    # =========================
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)

    y_pred_et = et.predict(X_test)
    y_prob_et = et.predict_proba(X_test)

    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)

    # =========================
    # RESULTADOS
    # =========================
    results_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'Extra Trees', 'XGBoost'],

        'CV Media': [
            rf_metrics["cv_mean"] if rf_metrics else None,
            et_metrics["cv_mean"] if et_metrics else None,
            xgb_metrics["cv_mean"] if xgb_metrics else None
        ],

        'CV Std': [
            rf_metrics["cv_std"] if rf_metrics else None,
            et_metrics["cv_std"] if et_metrics else None,
            xgb_metrics["cv_std"] if xgb_metrics else None
        ],

        'Test Accuracy': [
            accuracy_score(y_test, y_pred_rf),
            accuracy_score(y_test, y_pred_et),
            accuracy_score(y_test_xgb, y_pred_xgb)
        ],

        'ROC AUC': [
            roc_auc_score(y_test, y_prob_rf, multi_class='ovr', average='weighted'),
            roc_auc_score(y_test, y_prob_et, multi_class='ovr', average='weighted'),
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

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    # best_model = rf
    # Mejor modelo por ROC AUC
    best_idx   = results_df['ROC AUC'].idxmax()
    best_name  = results_df.loc[best_idx, 'Modelo']
    best_model = rf if best_idx == 0 else et
    print(f'\n Mejor modelo: {best_name}')

    feat_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)


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
