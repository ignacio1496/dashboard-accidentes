import streamlit as st

def show():
    st.title(" Insights Clave")

    st.markdown("""
    ##  Hallazgos principales

    ### 1. Velocidad
    A mayor velocidad, mayor severidad del accidente.

    ### 2. Condiciones climáticas
    Los accidentes en lluvia presentan mayor riesgo.

    ### 3. Hora del día
    Mayor severidad en horarios nocturnos.

    ---
    ##  Interpretación estratégica

    - Implementar controles de velocidad
    - Mejorar iluminación vial
    - Campañas de concientización

    """)