import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils.load_data import load_data


def show():
    st.title(" Análisis Exploratorio de Datos (EDA)")

    # =============================
    # 1. CONTEXTO (Storytelling)
    # =============================
    st.markdown("""
    ##  Contexto del Dataset

    El dataset utilizado corresponde a accidentes de tráfico en el estado de Florida,
    seleccionado por su similitud con Panamá en términos de:

    - Clima subtropical
    - Infraestructura urbana/rural mixta
    - Alta densidad vehicular

    ###  Enfoque analítico
    - Predicción de severidad del accidente
    """)

    # =============================
    # 2. CARGA DE DATOS
    # =============================
    df = load_data()

    st.subheader(" Vista general del dataset")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)

    with col1:
        st.metric(" Total registros", f"{df.shape[0]:,}")

    with col2:
        st.metric(" Total variables", df.shape[1])

    # =============================
    # 3. TIPOS DE DATOS
    # =============================
    st.subheader(" Tipos de variables")

    dtypes_df = pd.DataFrame({
        "Variable": df.columns,
        "Tipo": df.dtypes.astype(str)
    })

    st.dataframe(dtypes_df)

    # =============================
    # 4. VARIABLES NUMÉRICAS
    # =============================
    st.subheader(" Variables numéricas")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)

    st.write(f"Total variables numéricas: {numeric_df.shape[1]}")

    # =============================
    # 5. DISTRIBUCIÓN DE SEVERIDAD
    # =============================
    if "Severity" in df.columns:
        st.subheader(" Distribución de la Severidad")

        fig = px.histogram(
            df,
            x="Severity",
            title="Distribución de Severidad"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Insight
        La distribución de severidad permite identificar si el dataset está balanceado
        o presenta sesgos hacia ciertos niveles de gravedad.
        """)

    # =============================
    # 6. VARIABLES CLAVE (EJEMPLO)
    # =============================
    posibles_vars = ["Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)"]

    vars_existentes = [v for v in posibles_vars if v in df.columns]

    if len(vars_existentes) > 0:
        st.subheader(" Variables ambientales")

        var_selected = st.selectbox("Selecciona variable:", vars_existentes)

        fig2 = px.box(
            df,
            x="Severity" if "Severity" in df.columns else None,
            y=var_selected,
            title=f"{var_selected} vs Severidad"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        ### Interpretación
        Se analiza cómo las condiciones ambientales influyen en la severidad
        de los accidentes.
        """)

    # =============================
    # 7. CORRELACIÓN
    # =============================
    if numeric_df.shape[1] > 1:
        st.subheader(" Correlación entre variables")

        corr = numeric_df.corr()

        fig3, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr, annot=True, cmap="PuBuGn", fmt='.2f', ax=ax)

        st.pyplot(fig3)

        st.markdown("""
        ## Insight
        La matriz de correlación permite identificar relaciones lineales
        entre variables numéricas.
        """)

    # =============================
    # 8. CONCLUSIÓN DEL EDA
    # =============================
    st.markdown("""
    ---
    ##  Conclusión del Análisis Exploratorio

    - Se identificaron variables clave relacionadas con la severidad
    - Existen patrones en variables ambientales
    - Se detectan posibles correlaciones útiles para modelado

     Este análisis sienta las bases para la construcción del modelo predictivo.
    """)
