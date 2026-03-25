import streamlit as st

# Importar módulos
from modules import (intro, eda, insights, conclusions, Panorama, modeling, presentacion
        , A_1_Ocurrencia, A_1_2_Duracion,A_1_3_MU,A_1_4_Temporal, A_1_5_Clima, A_1_6_MapaCalor
        , B_2_1_Severidad, B_2_2_Severidad_Prom, B_2_3_Severidad_Ocurr
        , C_3_1_Modelo_Comparacion, C_3_2_Simulacion
        )
st.set_page_config(
    page_title="Proyecto Final - Analítica de Accidentes",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navegación
st.sidebar.title(" Navegación")
section = st.sidebar.radio(
    "Ir a:",
    [
        "Presentación",
        "Introducción",
        "Panorama General",
        # "A 1 Ocurrencia",
        # "A 2 Duración",
        # "A 3 MUTCD",
        # "A 4 Analisis Temporal",
        # "A 5 Clima",
        # "A 6 Mapa de Calor",
        # "B 1 Severidad",
        # "B 2 Severidad Promedio",
        # "B 3 Severidad vs Ocurrencia",
        # "C 1 Comparación de Modelos",
        # "C 2 Simulación",
        "Análisis Exploratorio",
        "Insights",
        "Modelos",
        "Conclusiones"
    ]
)

# Routing tipo storytelling
if section == "Presentación":
    presentacion.show()

elif section == "Introducción":
    intro.show()
elif section == "A 1 Ocurrencia":
    A_1_Ocurrencia.show()
elif section == "A 2 Duración":
    A_1_2_Duracion.show()
elif section == "A 3 MUTCD":
    A_1_3_MU.show()
elif section == "A 4 Analisis Temporal":
    A_1_4_Temporal.show()
elif section == "A 5 Clima":
    A_1_5_Clima.show()
elif section == "A 6 Mapa de Calor":
    A_1_6_MapaCalor.show()
elif section == "B 1 Severidad":
    B_2_1_Severidad.show()
elif section == "B 2 Severidad Promedio":
    B_2_2_Severidad_Prom.show()
elif section == "B 3 Severidad vs Ocurrencia":
    B_2_3_Severidad_Ocurr.show()
elif section == "C 1 Comparación de Modelos":
    C_3_1_Modelo_Comparacion.show()

elif section == "C 2 Simulación":
    C_3_2_Simulacion.show()

elif section == "Panorama General":
    Panorama.show()

elif section == "Análisis Exploratorio":
    eda.show()

elif section == "Insights":
    insights.show()

elif section == "Modelos":
    modeling.show()

elif section == "Conclusiones":
    conclusions.show()