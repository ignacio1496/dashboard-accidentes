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


    # =========================
    # 1. CARGA
    # =========================
    # @st.cache_data
    # def load_data():
    #     df = pd.read_csv("data/US_Accidents_FL.csv")
    #     return df

    df = load_data()

    # =========================
    # 3. LIMPIEZA
    # =========================
    missing = df.isnull().sum()
    # st.subheader("Valores Nulos")
    # st.dataframe(missing[missing > 0])

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # st.write("Nulos restantes:", df.isnull().sum().sum())

    # =========================
    # 3.2 ÚNICOS
    # =========================
    unique_df = pd.DataFrame({
        'Columna': df.columns,
        'Unicos': [df[col].nunique() for col in df.columns]
    })

    # st.dataframe(unique_df)

    # =========================
    # 3.5 OUTLIERS IQR
    # =========================
    # st.subheader("Eliminación de Outliers (IQR)")

    clean_df = df.copy()

    for col in clean_df.select_dtypes(include=['float64','int64']).columns:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1

        clean_df = clean_df[
            (clean_df[col] >= Q1 - 1.5 * IQR) &
            (clean_df[col] <= Q3 + 1.5 * IQR)
        ]

    # st.write(f"Antes: {len(df):,}")
    # st.write(f"Después: {len(clean_df):,}")

    # =========================
    # 4.1 DISTRIBUCIÓN
    # =========================
    st.title("A. Análisis de Ocurrencia")

    st.markdown("""
    #### 1.1 Distribución de variables categóricas
    """)    
    # Distribución de variables categóricas (Top 10 por columna)
    dist_cat_columns = clean_df[[
        'Street', 'City', 'County',
        # 'State',  # comentado: dataset filtrado a Florida únicamente
        'Wind_Direction', 'Weather_Condition',
        'Sunrise_Sunset', 'Civil_Twilight',
        'Nautical_Twilight', 'Astronomical_Twilight'
    ]]

    all_frecuencias = {}
    for column in dist_cat_columns:
        top_10_freq = clean_df[column].str.strip().value_counts()[:10]
        all_frecuencias[column] = pd.DataFrame({
            'Columna': column,
            'Valor': top_10_freq.index,
            'Frecuencia': top_10_freq.values
        })

    frecuencias_df = pd.concat(all_frecuencias.values(), ignore_index=True)
    columnas = frecuencias_df['Columna'].unique()
    n_cols = 3
    n_rows = math.ceil(len(columnas) / n_cols)
    colores = px.colors.qualitative.Pastel

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=list(columnas),
        vertical_spacing=0.18,
        horizontal_spacing=0.08
    )

    for i, columna in enumerate(columnas):
        row = i // n_cols + 1
        col = i %  n_cols + 1

        df_col = frecuencias_df[frecuencias_df['Columna'] == columna].copy()
        ratio = df_col['Frecuencia'] / df_col['Frecuencia'].sum() * 100

        # Top 10 para columnas con muchos valores
        top_idx = ratio.nlargest(10).index
        df_col = df_col.loc[top_idx]
        ratio = ratio.loc[top_idx]

        # Color por label
        color_map = {val: colores[j % len(colores)] for j, val in enumerate(df_col['Valor'].astype(str))}
        marker_colors = df_col['Valor'].astype(str).map(color_map).tolist()

        fig.add_trace(
            go.Bar(
                x=df_col['Valor'].astype(str),
                y=ratio.round(1),
                text=ratio.round(1).astype(str) + '%',
                textposition='outside',
                marker_color=marker_colors,
                cliponaxis=True,
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_yaxes(
            title_text='Ratio (%)',
            range=[0, ratio.max() * 1.30],
            row=row, col=col
        )

    fig.update_xaxes(tickangle=90)

    fig.update_layout(
        height=420 * n_rows,
        width=1100,
        template='plotly_white',
        font=dict(size=11),
        title_text='Distribución de Variables',
        title_x=0.5
    )    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    - Mayor concentración de accidentes en ciudades como Miami y Orlando, y en el condado Miami-Dade        
    - Vías principales concentran más incidentes → relación con alto flujo vehicular
    - Predominan condiciones climáticas favorables (Fair, Mostly Cloudy, Partly Cloudy)
    - Más del 75 % de los accidentes ocurren de día → asociado a mayor volumen de tráfico
    ---
    """)

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

    # =========================
    # MUTCD
    # =========================
  
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

    # Crear MUTCD_Category basada en duración real, no en severidad
    df['MUTCD_Category'] = df['Duration_min'].apply(clasificar_mutcd_por_tiempo)
    resumen_mutcd = (df
        .groupby('MUTCD_Category')['Severity']
        .count()
        .reset_index(name='Registros')
    ) 

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

    color_ocurrencia  = {
        'Bajo':     '#a8d5a2',
        'Moderado': '#f6d860',
        'Alto':     '#f4a261',
        'Crítico':  '#e63946'
    } 

    st.markdown("""
    ####   1.3 Calcula la mediana de duración por categoría MUTCD
    - Nivel de Severidad ajustado
    """) 
    # Crear MUTCD_Category usando .loc para evitar SettingWithCopyWarning
    duration_clean.loc[:, 'MUTCD_Category'] = duration_clean['Duration_min'].apply(clasificar_mutcd_por_tiempo)

    # Calcula la mediana de duración por categoría MUTCD
    duration_stats_mutcd = (
        duration_clean
        .groupby('MUTCD_Category')['Duration_min']
        .median()
        .reset_index()
        .rename(columns={'Duration_min': 'Mediana'})
    )

    # Visualización
    fig3 = px.bar(
        duration_stats_mutcd,
        x='MUTCD_Category', y='Mediana',
        color='MUTCD_Category',
        category_orders={'MUTCD_Category': ['Menor', 'Intermedio', 'Mayor']},  # <-- esto
        color_discrete_map={
            'Menor':      '#a8d5a2',
            'Intermedio': '#f6d860',
            'Mayor':      '#e63946'
        },
        text=duration_stats_mutcd['Mediana'].apply(formato_tiempo),
        template='plotly_white',
        labels={
            'MUTCD_Category': 'Categoría MUTCD',
            'Mediana': 'Mediana de Duración (min)'
        },
        title='Mediana de Duración por Categoría MUTCD — Florida'
    )
    fig3.update_layout(height=450, width=700, font=dict(size=13), showlegend=False)    

    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    - Se creó la variable categórica Duration_Category basada en criterios de la Federal Highway Administration (FHWA)
    - Clasificación: menor (< 30 min), intermedio (30 min – 2 h) y mayor (> 2 h)
    - Mejora la coherencia en la medición del impacto en el flujo vehicular
    ---
    """)  

    st.markdown("""
    ####   1.4 Análisis Temporal
    """) 
    # Extraer variables temporales
    df['Start_Time'] = pd.to_datetime(
        df['Start_Time'], errors='coerce'
    )
    df['Hour']      = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month']     = df['Start_Time'].dt.month_name()

    day_order   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    month_order = ['January','February','March','April','May','June',
                'July','August','September','October','November','December']

    # Prepara los datos
    hourly  = df.groupby('Hour').size().reset_index(name='Accidentes')

    daily   = (df.groupby('DayOfWeek').size()
            .reindex(day_order).reset_index(name='Accidentes'))

    monthly = (df.groupby('Month').size()
            .reindex(month_order).reset_index(name='Accidentes'))

    # Config de cada subplot
    plots = [
        {'df': hourly,   'x': 'Hour',      'title': 'Por Hora del Día'},
        {'df': daily,    'x': 'DayOfWeek', 'title': 'Por Día de la Semana'},
        {'df': monthly,  'x': 'Month',     'title': 'Por Mes'},
    ]

    fig4 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[p['title'] for p in plots],
        horizontal_spacing=0.08
    )

    for i, p in enumerate(plots):
        col = i + 1
        accidentes = p['df']['Accidentes']

        # Escala de color YlOrRd mapeada a los valores
        min_val, max_val = accidentes.min(), accidentes.max()
        normed = (accidentes - min_val) / (max_val - min_val)
        import plotly.colors as pc
        colors = [pc.sample_colorscale('YlOrRd', v)[0] for v in normed]

        fig4.add_trace(
            go.Bar(
                x=p['df'][p['x']].astype(str),
                y=accidentes,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=col
        )
        fig4.update_yaxes(title_text='Accidentes', row=1, col=col)
        fig4.update_xaxes(title_text=p['x'], tickangle=45, row=1, col=col)
        fig4.update_xaxes(title_text=p['x'], tickangle=90, row=1, col=col)

    fig4.update_layout(
        height=450,
        width=1200,
        template='plotly_white',
        font=dict(size=12),
        title_text='Distribución Temporal de Accidentes — Florida',
        title_x=0.5
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    - Mayor frecuencia a las 8:00 hrs, los viernes y en diciembre
    - Asociado a horas pico, fin de semana laboral y mayor actividad vehicular de fin de año
    ---
    """)

    st.markdown("""
    ####   1.5 Análisis Climático vs Ocurrencia
    """)    
    top_weather = (df['Weather_Condition']
                .value_counts().head(10).reset_index())
    top_weather.columns = ['Weather_Condition', 'Accidentes']

    fig5 = px.bar(top_weather, x='Accidentes', y='Weather_Condition',
                orientation='h',
                color='Accidentes', color_continuous_scale='PuBuGn',
                text='Accidentes', template='seaborn',
                title='Top 10 Condiciones Climáticas con Mayor Ocurrencia de Accidentes')
    fig5.update_layout(height=500, width=1000, font=dict(size=13))
    st.plotly_chart(fig5, use_container_width=True)
    
    st.markdown("""
    - Mayor ocurrencia de accidentes en clima “Fair”, seguido de “Mostly Cloudy” y “Partly Cloudy”
    - Sugiere mayor exposición al riesgo en climas estables
    ---
    """) 

    st.markdown("""
    ####  Mapa de calor geográfico de la sección: 
    #####  1.6 Índice de Riesgo por Frecuencia Geográfica
    """) 
    def map_mutcd_to_severity(mutcd_category):
        if mutcd_category == 'Menor':
            return 1
        elif mutcd_category == 'Intermedio':
            return 2
        elif mutcd_category == 'Mayor':
            return 3
        return None

    df['MUTCD_Severity'] = (
        df['MUTCD_Category'].apply(map_mutcd_to_severity)
    )    
    # Riesgo por coordenadas (~1 km²)
    df['lat_round'] = df['Start_Lat'].round(2)
    df['lng_round'] = df['Start_Lng'].round(2)

    coord_risk = (
        df
        .groupby(['lat_round','lng_round'])
        .agg(ocurrencias=('MUTCD_Severity','count'), severidad_promedio_mutcd=('MUTCD_Severity','mean'))
        .reset_index()
        .sort_values('ocurrencias', ascending=False)
    )
    coord_risk['nivel_ocurrencia'] = clasificar_ocurrencia(coord_risk['ocurrencias'])

    # Mapa de calor geográfico
    top_coords = coord_risk[coord_risk['ocurrencias'] >= coord_risk['ocurrencias'].quantile(0.90)]

    fig6 = px.density_mapbox(
        top_coords, lat='lat_round', lon='lng_round', z='ocurrencias',
        radius=8, center={'lat': 27.5, 'lon': -81.5}, zoom=5,
        mapbox_style='carto-positron', color_continuous_scale='YlOrRd',
        title='Mapa de Calor — Zonas de Mayor Ocurrencia por Frecuencia (Florida)'
    )
    fig6.update_layout(height=600, width=1100, font=dict(size=13))
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
    Los accidentes no están distribuidos de forma uniforme, sino altamente concentrados en zonas específicas, lo que permite identificar puntos críticos de intervención.

    - Los niveles se clasifican según frecuencia: bajo (1), moderado (2–4), alto (5–13) y crítico (>14 accidentes)
    - Predomina el nivel moderado y bajo, aunque existe una cantidad significativa de casos críticos.
    - Se identifican puntos geográficos críticos con alta concentración de accidentes (hasta 3,800 eventos)
    - A nivel administrativo, Miami-Dade y Orange presentan la mayor cantidad de incidentes.

    ---
    """)

    st.markdown("""
    ###  B. Subsección: Severidad promedio de accidentes en Florida según variables
    ####  2.1 Análisis de Severidad 
    """)  
    # ── Datos ─────────────────────────────────────────────────────────────────────
    severity_hour = (df
                    .groupby('Hour')['MUTCD_Severity']
                    .mean().reset_index(name='Severidad_Promedio'))

    severity_weather = (df
                        .groupby('Weather_Condition')
                        .agg(
                            Severidad_Promedio=('MUTCD_Severity','mean'),
                            Total_Accidentes=('MUTCD_Severity','count')
                        )
                        .reset_index()
                        .query('Total_Accidentes >= 100')
                        .sort_values('Severidad_Promedio', ascending=False)
                        .head(15))

    severity_county = (df
                    .groupby('County')
                    .agg(
                        Severidad_Promedio=('MUTCD_Severity','mean'),
                        Total_Accidentes=('MUTCD_Severity','count')
                    )
                    .reset_index()
                    .query('Total_Accidentes >= 50')
                    .sort_values('Severidad_Promedio', ascending=False)
                    .head(20))

    # ── Subplots: 1 fila, 3 columnas ─────────────────────────────────────────────
    fig7 = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Severidad por Hora del Día',
            'Top 15 Condiciones Climáticas',
            'Top 20 Condados'
        ),
        horizontal_spacing=0.12
    )

    # ── Gráfica 1: Line chart por hora ───────────────────────────────────────────
    fig7.add_trace(
        go.Scatter(
            x=severity_hour['Hour'],
            y=severity_hour['Severidad_Promedio'],
            mode='lines+markers',
            line=dict(color='#e63946', width=2),
            marker=dict(size=6),
            name='Por Hora'
        ),
        row=1, col=1
    )

    # ── Gráfica 2: Bar horizontal por clima ──────────────────────────────────────
    fig7.add_trace(
        go.Bar(
            x=severity_weather['Severidad_Promedio'],
            y=severity_weather['Weather_Condition'],
            orientation='h',
            text=severity_weather['Severidad_Promedio'].round(2),
            textposition='outside',
            marker=dict(
                color=severity_weather['Severidad_Promedio'],
                colorscale='YlOrRd',
                showscale=False
            ),
            name='Por Clima'
        ),
        row=1, col=2
    )

    # ── Gráfica 3: Bar vertical por condado ──────────────────────────────────────
    fig7.add_trace(
        go.Bar(
            x=severity_county['County'],
            y=severity_county['Severidad_Promedio'],
            text=severity_county['Severidad_Promedio'].round(2),
            textposition='outside',
            marker=dict(
                color=severity_county['Severidad_Promedio'],
                colorscale='YlOrRd',
                showscale=False
            ),
            name='Por Condado'
        ),
        row=1, col=3
    )

    # ── Ejes ──────────────────────────────────────────────────────────────────────
    fig7.update_xaxes(title_text='Hora',            tickangle=90,
                    row=1, col=1)
    fig7.update_xaxes(title_text='Severidad Prom.', tickangle=90,
                    title_standoff=15, automargin=True, row=1, col=2)
    fig7.update_xaxes(title_text='Condado',         tickangle=90,
                    row=1, col=3)

    fig7.update_yaxes(title_text='Severidad Promedio', row=1, col=1)
    fig7.update_yaxes(title_text='',                   row=1, col=2)
    fig7.update_yaxes(title_text='Severidad Promedio', row=1, col=3)

    # ── Layout ────────────────────────────────────────────────────────────────────
    fig7.update_layout(
        height=500,
        width=1800,
        template='seaborn',
        font=dict(size=11),
        title_text='Análisis de Severidad — Florida',
        title_x=0.5,
        showlegend=False
    )

    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("""
    -	Variable temporal: Mayor severidad en la madrugada, mediodía y medianoche. 
    -	Factores: fatiga, dismunicion de atención, alcohol, mayor cantidad de autos.
    -	Condiciones Climaticas: Mayor severidad en escenarios con viento & nubosidad/lluvia ligera
    -	Factores: baja visibilidad, menor adherencia, menor estabilidad del auto.
    -	Variable geografica: Variabilidad moderada entre condados. Se destaca Miami- Dade, Orange y Sarasota.
    -	La severidad no es aleatoria: depende del tiempo, el clima y la ubicación
    ---
    """)

    st.markdown("""
    ####   2.2 Análisis de Severidad
    ##### Top 3 condados con mayor severidad promedio distribuidas por hora 
    """)  
    # Top 3 condados con más ocurrencias
    top3_condados = (df
                    .groupby('County')
                    .agg(
                        Severidad_Promedio=('MUTCD_Severity', 'mean'),
                        Total_Accidentes=('MUTCD_Severity', 'count')
                    )
                    .query('Total_Accidentes >= 50')
                    .sort_values('Severidad_Promedio', ascending=False)
                    .head(3)
                    .index
                    .tolist())
    # Nuevo dataset: severidad promedio por hora y condado
    severity_hour_county = (df[df['County'].isin(top3_condados)]
                            .groupby(['County', 'Hour'])['MUTCD_Severity']
                            .mean()
                            .reset_index(name='Severidad_Promedio'))


    # Paleta de colores, uno por condado
    colores = ['#e63946', '#457b9d', '#2a9d8f']

    fig8 = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{c} County' for c in top3_condados],
        horizontal_spacing=0.08
    )

    for i, condado in enumerate(top3_condados):
        df_condado = severity_hour_county[severity_hour_county['County'] == condado]

        fig8.add_trace(
            go.Scatter(
                x=df_condado['Hour'],
                y=df_condado['Severidad_Promedio'],
                mode='lines+markers',
                line=dict(color=colores[i], width=2),
                marker=dict(size=6),
                name=condado
            ),
            row=1, col=i + 1
        )

        fig8.update_xaxes(title_text='Hora del Día', tickmode='linear', dtick=1,
                        tickangle=0, row=1, col=i + 1)
        fig8.update_yaxes(title_text='Severidad Promedio', row=1, col=i + 1)

    fig8.update_layout(
        height=400, width=1800,
        template='plotly_white',
        font=dict(size=12),
        title_text='Severidad Promedio por Hora — Top 3 Condados',
        title_x=0.5
    )

    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("""
    - Patrones consistentes entre condados
    - Picos en madrugada, mediodía y noche
    - Variaciones explicadas por entorno urbano y comportamiento
    ---
    """)              

    st.markdown("""
    ### Análisis de relación ocurrencia vs severidad por condado 
    ####  2.3 Análisis de Severidad  
    ##### Resumen integrado: Ocurrencia + Severidad por condado 
    """)  
    # Resumen integrado: Ocurrencia + Severidad por condado
    resumen_county = (df
                    .groupby('County')
                    .agg(
                        Ocurrencias=('MUTCD_Severity','count'),
                        Severidad_Promedio=('MUTCD_Severity','mean')
                    )
                    .reset_index())

    resumen_county['Riesgo_Ocurrencia'] = clasificar_ocurrencia(resumen_county['Ocurrencias'])
    resumen_county['Nivel_Severidad']   = pd.cut(
        resumen_county['Severidad_Promedio'],
        bins=[0, 1.5, 2.0, 2.5, 4],
        labels=['Baja','Moderada','Alta','Crítica']
    )

    fig9 = px.scatter(resumen_county,
                    x='Ocurrencias', y='Severidad_Promedio',
                    color='Riesgo_Ocurrencia', size='Ocurrencias',
                    hover_name='County',
                    color_discrete_map=color_ocurrencia,
                    template='seaborn',
                    title='Relación Ocurrencia vs Severidad por Condado — Florida<br>'
                        '<sup>Cada punto es un condado. Tamaño = número de accidentes</sup>')
    fig9.update_layout(height=550, width=1000, font=dict(size=13))

    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("""
    - No se observa una relación lineal fuerte entre ocurrencia y severidad
    - Condados con más accidentes tienden a severidad moderada–alta
    - Destaca Miami-Dade County con alto volumen y alta severidad (riesgo crítico)
    - La mayoría presenta menor volumen y severidad moderada/baja
    - El riesgo se concentra en regiones específicas
    ---
    """)   

    st.markdown("""
    ### C. RESULTADOS DE MODELOS PREDICTIVOS
    ####  3.1 Comparación de Modelos + Feature Importance
    """)  
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier    

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

