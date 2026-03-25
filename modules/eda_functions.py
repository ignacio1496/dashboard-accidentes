import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# CARGA Y PREPROCESAMIENTO
# =========================
def load_data(path):
    df = pd.read_csv(path)

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time']   = pd.to_datetime(df['End_Time'], errors='coerce')

    df['Duration_min'] = (
        df['End_Time'] - df['Start_Time']
    ).dt.total_seconds() / 60

    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()

    return df


# =========================
# 4.1 DISTRIBUCIÓN CATEGÓRICA
# =========================
def plot_categorical_distribution(df, column, top_n=10):
    data = df[column].value_counts().head(top_n).reset_index()
    data.columns = [column, 'Frecuencia']

    fig = px.bar(
        data,
        x=column,
        y='Frecuencia',
        title=f'Top {top_n} - {column}'
    )
    return fig


# =========================
# 4.2 DURACIÓN VS SEVERIDAD
# =========================
def plot_duration_vs_severity(df):
    df_clean = df[(df['Duration_min'] >= 1) & (df['Duration_min'] <= 1440)]

    fig = px.box(
        df_clean,
        x='Severity',
        y='Duration_min',
        color='Severity',
        title='Duración vs Severidad'
    )
    return fig


# =========================
# MUTCD
# =========================
def add_mutcd_category(df):
    def clasificar(d):
        if d < 30:
            return "Menor"
        elif d <= 120:
            return "Intermedio"
        else:
            return "Mayor"

    df = df.copy()
    df['MUTCD_Category'] = df['Duration_min'].apply(clasificar)
    return df


def plot_mutcd_distribution(df):
    data = df['MUTCD_Category'].value_counts().reset_index()
    data.columns = ['Categoria', 'Accidentes']

    fig = px.bar(
        data,
        x='Categoria',
        y='Accidentes',
        title='Distribución MUTCD'
    )
    return fig


# =========================
# 5.1 ANÁLISIS TEMPORAL
# =========================
def plot_temporal_analysis(df):
    hourly = df.groupby('Hour').size().reset_index(name='Accidentes')

    fig = px.bar(
        hourly,
        x='Hour',
        y='Accidentes',
        title='Accidentes por Hora'
    )
    return fig


# =========================
# 5.2 CLIMA
# =========================
def plot_weather(df):
    data = df['Weather_Condition'].value_counts().head(10).reset_index()
    data.columns = ['Clima', 'Accidentes']

    fig = px.bar(
        data,
        x='Accidentes',
        y='Clima',
        orientation='h',
        title='Top Condiciones Climáticas'
    )
    return fig


# =========================
# 5.3 MAPA DE CALOR
# =========================
def plot_heatmap(df):
    df = df.copy()
    df['lat'] = df['Start_Lat'].round(2)
    df['lon'] = df['Start_Lng'].round(2)

    coords = df.groupby(['lat', 'lon']).size().reset_index(name='Accidentes')

    fig = px.density_mapbox(
        coords,
        lat='lat',
        lon='lon',
        z='Accidentes',
        radius=8,
        zoom=5,
        mapbox_style="carto-positron",
        title='Mapa de Calor'
    )
    return fig


# =========================
# 6.2 SEVERIDAD
# =========================
def plot_severity_by_hour(df):
    data = df.groupby('Hour')['Severity'].mean().reset_index()

    fig = px.line(
        data,
        x='Hour',
        y='Severity',
        title='Severidad Promedio por Hora'
    )
    return fig