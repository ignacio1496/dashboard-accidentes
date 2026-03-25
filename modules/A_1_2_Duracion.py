# =========================
# STREAMLIT APP COMPLETO
# =========================
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.load_data import *
# st.set_page_config(layout="wide")
def show():

    df = load_data()
    # =========================
    # 4.2 
    # =========================
    st.markdown("""
    ####   1.2 Duración real por nivel de Severidad
    """) 
    # ── Duración real por nivel de Severidad ──────────────────────────────────────

    # Calculamos la duración de cada accidente desde Start_Time y End_Time
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    df['Duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    # Filtrar duraciones negativas o extremas (outliers de registro)
    # Consideramos válidas duraciones entre 1 minuto y 24 horas
    duration_clean = df[
        (df['Duration_min'] >= 1) &
        (df['Duration_min'] <= 1440)
    ].copy() # Explicitly create a copy to avoid SettingWithCopyWarning

    print("=== DURACIÓN REAL DEL IMPACTO EN TRÁFICO POR NIVEL DE SEVERIDAD ===")
    print("(en minutos)\n")
    # Estadísticas reales de duración por nivel de severidad
    duration_stats = (duration_clean
        .groupby('Severity')['Duration_min']
        .agg(
            Registros='count',
            Minimo='min',
            Percentil_25=lambda x: x.quantile(0.25),
            Mediana='median',
            Promedio='mean',
            Percentil_75=lambda x: x.quantile(0.75),
            Maximo='max'
        )
        .round(1)
        .reset_index()
    )
    # display(duration_stats)

    # Construir etiquetas dinámicas con rangos reales
    def formato_tiempo(minutos):
        minutos = round(minutos)  # ← redondear primero
        h = int(minutos // 60)
        m = int(minutos % 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m"

    print("\n=== RANGOS REALES POR NIVEL ===")
    for _, row in duration_stats.iterrows():
        sev  = int(row['Severity'])
        p25  = formato_tiempo(row['Percentil_25'])
        p75  = formato_tiempo(row['Percentil_75'])
        med  = formato_tiempo(row['Mediana'])
        print(f"Severidad {sev}: rango típico {p25} – {p75}  |  mediana: {med}")

    # Visualización: promedio por severidad con barra
    fig2 = px.bar(
        duration_stats,
        x='Severity', y='Promedio',
        color='Severity',
        color_discrete_sequence=px.colors.sequential.YlOrRd,
        text=duration_stats['Promedio'].apply(formato_tiempo),
        template='seaborn',
        labels={'Severity': 'Nivel de Severidad', 'Promedio': 'Promedio de Duración (min)'},
        title='Promedio de Duracion del Impacto en Trafico por Nivel de Severidad - Florida'
    )
    fig2.update_layout(height=450, width=700, font=dict(size=13), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    - La severidad del accidente se asocia a variables de impacto: delay, distance y duration
    - Se detectaron inconsistencias en duration, que da indicio de duraciones estimadas por el sistema
    - Se excluye duration como variable predictora directa
    ---
    """) 

