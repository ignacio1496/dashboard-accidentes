import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_categorical_distributions_streamlit(df):

    variables = [
        "Street", "City", "County",
        "Wind_Direction", "Weather_Condition", "Sunrise_Sunset",
        "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        if var in df.columns:

            data = df[var].value_counts(normalize=True).head(10) * 100

            sns.barplot(
                x=data.index,                
                y=data.values,
                # x=data.values,
                # y=data.index,
                ax=axes[i]
            )

            axes[i].set_title(var)
            axes[i].set_ylabel("Ratio (%)")
            # axes[i].set_xlabel("Ratio (%)")

        else:
            axes[i].set_visible(False)

    plt.suptitle("Distribución de Variables", fontsize=16)
    plt.tight_layout()

    return fig #plt.gcf()
    # st.pyplot(fig)

def plot_duration_by_severity(df):

    # =============================
    # 1. CONVERTIR TIEMPO
    # =============================
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors='coerce')
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors='coerce')

    # Eliminar filas inválidas
    df = df.dropna(subset=["Start_Time", "End_Time"])

    df["Duration_min"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60

    # =============================
    # 2. AGRUPAR
    # =============================
    duration_avg = df.groupby("Severity")["Duration_min"].mean().reset_index()

    # =============================
    # 3. CREAR LABELS (h m)
    # =============================
    def format_time(mins):
        h = int(mins // 60)
        m = int(mins % 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m"

    duration_avg["label"] = duration_avg["Duration_min"].apply(format_time)

    # =============================
    # 4. GRAFICA
    # =============================
    plt.figure(figsize=(10, 6))

    cmap = sns.color_palette("rocket", as_cmap=True)

    bars = sns.barplot(
        data=duration_avg,
        x="Severity",
        y="Duration_min",
        palette="rocket"
    )

    # Labels sobre barras
    for i, row in duration_avg.iterrows():
        bars.text(
            i,
            row["Duration_min"] + 5,
            row["label"],
            ha='center',
            fontsize=11,
            fontweight='bold'
        )

    plt.title("Promedio de Duración del Impacto en Tráfico por Nivel de Severidad - Florida")
    plt.xlabel("Nivel de Severidad")
    plt.ylabel("Promedio de Duración (min)")

    plt.tight_layout()

    return plt.gcf()