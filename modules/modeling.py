import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import plotly.express as px

from utils.load_data import load_data


def show():
    st.title(" Modelo Predictivo de Severidad")

    # =============================
    # 1. CONTEXTO
    # =============================
    st.markdown("""
    ##  Objetivo del Modelo

    Construir un modelo de aprendizaje automático capaz de predecir la
    **severidad de un accidente** basado en variables ambientales y operativas.

    ##  Algoritmo utilizado
    - Random Forest Classifier

    Este modelo fue seleccionado por:
    - Capacidad de manejar datos no lineales
    - Robustez ante ruido
    - Interpretabilidad (feature importance)
    """)

    # =============================
    # 2. CARGA DE DATOS
    # =============================
    df = load_data()

    # =============================
    # 3. SELECCIÓN DE VARIABLES
    # =============================
    st.subheader(" Selección de variables")

    target = "Severity"

    posibles_features = [
        "Temperature(F)",
        "Humidity(%)",
        "Visibility(mi)",
        "Wind_Speed(mph)",
        "Pressure(in)"
    ]

    features = [col for col in posibles_features if col in df.columns]

    if target not in df.columns:
        st.error(" No se encontró la variable objetivo 'Severity'")
        return

    df_model = df[features + [target]].dropna()

    st.write("Variables utilizadas:", features)

    # =============================
    # 4. SPLIT
    # =============================
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================
    # 5. PIPELINE
    # =============================
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # =============================
    # 6. ENTRENAMIENTO
    # =============================
    with st.spinner("Entrenando modelo..."):
        pipeline.fit(X_train, y_train)

    # =============================
    # 7. EVALUACIÓN
    # =============================
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.subheader(" Evaluación del modelo")

    st.metric("Accuracy", f"{acc:.2f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        title="Matriz de Confusión"
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report
    st.text(" Classification Report")
    st.text(classification_report(y_test, y_pred))

    # =============================
    # 8. IMPORTANCIA DE VARIABLES
    # =============================
    st.subheader(" Importancia de variables")

    model = pipeline.named_steps["model"]

    importances = model.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig_imp = px.bar(
        feat_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Importancia de Variables"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    ### Insight
    Las variables con mayor importancia son aquellas que más influyen en la
    predicción de la severidad del accidente.
    """)

    # =============================
    # 9. PREDICCIÓN EN VIVO
    # =============================
    st.subheader(" Predicción en tiempo real")

    st.markdown("Ingrese valores para simular un accidente:")

    input_data = {}

    cols = st.columns(len(features))

    for i, col in enumerate(features):
        with cols[i]:
            input_data[col] = st.number_input(
                col,
                value=float(df[col].mean())
            )

    input_df = pd.DataFrame([input_data])

    if st.button("Predecir severidad"):
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df).max()

        st.success(f" Severidad predicha: {prediction}")
        st.info(f" Confianza del modelo: {proba:.2f}")

    # =============================
    # 10. CONCLUSIÓN
    # =============================
    st.markdown("""
    ---
    ## Conclusión del Modelo

    - El modelo logra predecir la severidad con un nivel aceptable de precisión.
    - Variables ambientales tienen impacto significativo.
    - Puede ser utilizado como base para sistemas de alerta temprana.

    ## Aplicación real
    - Sistemas de monitoreo vial
    - Planeación urbana
    - Prevención de accidentes
    """)