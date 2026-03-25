import streamlit as st
import numpy as np
from PIL import Image
def show():
    st.title(" Predicción de Severidad en Siniestros Viales")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ##  Introducción

        Los siniestros viales representan un problema crítico de salud pública y movilidad urbana.
        Este proyecto tiene como objetivo analizar y modelar la **severidad de los accidentes**
        mediante técnicas de **aprendizaje automático**.

        - Uso del dataset US Accidents (2016–2023) de Kaggle 
        por su amplitud y nivel de detalle
        - Incluye variables clave: ubicación, fecha y hora, clima, tipo de vía y severidad
        - La metodología servirá como base para adaptación al contexto panameño, utilizando datos de la INEC para alimentar modelos similares.
        """)
    with col2:
        img = Image.open('assets/Imagen5.png')
        st.image(img)

    st.markdown("""
    ---
    """)