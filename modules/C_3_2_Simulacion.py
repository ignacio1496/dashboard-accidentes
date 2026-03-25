import streamlit as st
import numpy as np
from PIL import Image

def show():
    st.markdown("""
    ####  3.2 SIMULACIÓN DATA CONTEXTO PANAMÁ
    """) 

    st.image(np.array(Image.open(r'assets/Imagen1.png')))

    st.markdown("""
    - La curva muestra un crecimiento sostenido de los accidentes hasta 2019, seguido de una caída abrupta en 2020 (pandemia y restricciones de movilidad). 
    A partir de 2021, la tendencia vuelve a subir, aunque sin alcanzar los picos previos. 
    Esto refleja cómo factores externos como la crisis sanitaria pueden alterar drásticamente la siniestralidad, 
    pero también cómo el problema persiste estructuralmente.
    ---
    """)   

    st.image(np.array(Image.open(r'assets/Imagen2.png')))


    st.markdown("""
    - Panamá concentra la mayoría de los accidentes (281 mil), seguido por Panamá Oeste y Chiriquí. 
    Provincias como Darién y Ngäbe Buglé registran cifras mucho menores, lo que refleja tanto diferencias poblacionales 
    como de infraestructura vial.
    ---
    """)

    st.image(np.array(Image.open(r'assets/Imagen3.png')))

    st.markdown("""
    - La lluvia y las tormentas aparecen como condiciones críticas, con altos niveles de accidentes graves. Aunque los días despejados también acumulan cifras importantes, 
    la severidad tiende a intensificarse bajo condiciones climáticas adversas.
    ---
    """)  

    st.image(np.array(Image.open(r'assets/Imagen4.png')))

    st.markdown("""
    - Panamá lidera con una tasa de 1,955 accidentes por cada 10,000 habitantes, seguido por Panamá Oeste y Colón. 
    En contraste, Ngäbe Buglé apenas registra 28. Esta métrica ajustada por población revela dónde el riesgo relativo es mayor, 
    más allá de los números absolutos.
    ---
    """)           