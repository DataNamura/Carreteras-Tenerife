import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Análisis de Incidentes en Carreteras", layout="wide")

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/guaja/proyectos/incidencias/incidencias_modificacion_1.csv')
    df['incidencia_fecha_inicio'] = pd.to_datetime(df['incidencia_fecha_inicio'])
    df['año'] = df['incidencia_fecha_inicio'].dt.year
    df['hora'] = df['incidencia_fecha_inicio'].dt.hour
    df['mes'] = df['incidencia_fecha_inicio'].dt.month
    df['dia_semana'] = df['incidencia_fecha_inicio'].dt.dayofweek  # Lunes=0, Domingo=6
    return df

df = load_data()

# Título de la aplicación
st.title("Análisis Interactivo de Incidentes en Carreteras")

# Filtros en la barra lateral
st.sidebar.header("Filtros")

# Filtro por rango de años
min_year = int(df['año'].min())
max_year = int(df['año'].max())
años = st.sidebar.slider("Selecciona el Rango de Años", min_year, max_year, (min_year, max_year))

# Filtro por nombre de carretera (permitiendo múltiples selecciones)
carreteras_disponibles = df['carretera_nombre'].unique().tolist()
carreteras_seleccionadas = st.sidebar.multiselect("Selecciona Carreteras", carreteras_disponibles, default=carreteras_disponibles)

# Filtro por hora del día
hora = st.sidebar.slider("Selecciona la Hora del Día", 0, 23, (0, 23))

# Filtrar datos según las selecciones
df_filtrado = df[(df['año'] >= años[0]) & (df['año'] <= años[1]) &
                 (df['carretera_nombre'].isin(carreteras_seleccionadas)) &
                 (df['hora'] >= hora[0]) & (df['hora'] <= hora[1])]

# Mostrar el dataframe filtrado
st.write(f"Datos Filtrados: {df_filtrado.shape[0]} registros")
st.dataframe(df_filtrado)

# Gráfico 1: Número de Accidentes por Tramo
st.header(f"Número de Accidentes en las Carreteras Seleccionadas")
accidentes_por_tramo = df_filtrado['tramo_nombre'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=accidentes_por_tramo.values, y=accidentes_por_tramo.index, palette="Blues_r")
plt.title(f"Top 10 Tramos de Carretera con Más Accidentes")
st.pyplot(plt)

# Gráfico 2: Distribución de Accidentes por Hora del Día
st.header("Distribución de Accidentes por Hora del Día")
accidentes_por_hora = df_filtrado['hora'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x=accidentes_por_hora.index, y=accidentes_por_hora.values, marker='o', color="skyblue")
plt.title("Accidentes por Hora del Día")
plt.xlabel("Hora del Día")
plt.ylabel("Número de Accidentes")
plt.grid(True)
st.pyplot(plt)

# Gráfico 3: Frecuencia de Diferentes Tipos de Incidentes
st.header("Frecuencia de Diferentes Tipos de Incidentes")
incidentes_por_tipo = df_filtrado['incidencia_tipo'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=incidentes_por_tipo.values, y=incidentes_por_tipo.index, palette="Blues_r")
plt.title("Frecuencia de Tipos de Incidentes")
plt.xlabel("Número de Incidentes")
plt.ylabel("Tipo de Incidente")
st.pyplot(plt)

# Gráfico 4: Correlación entre Mes y Número de Accidentes
st.header("Correlación entre Mes y Número de Accidentes")

# Preparar los datos para la correlación
correlacion_df = df_filtrado[df_filtrado['es_accidente'] == 'Accidente'].groupby('mes').size().reset_index(name='accidentes')

plt.figure(figsize=(10, 6))
sns.regplot(x='mes', y='accidentes', data=correlacion_df, scatter_kws={'s':50}, line_kws={'color':'blue'})
plt.title('Correlación entre Mes y Número de Accidentes')
plt.xlabel('Mes')
plt.ylabel('Número de Accidentes')

# Define meses_nombres with month abbreviations
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

plt.xticks(ticks=correlacion_df['mes'], labels=meses_nombres, rotation=45)
plt.grid(True)
st.pyplot(plt)

# Gráfico 5: Mapa de Calor de Accidentes por Día de la Semana y Hora
st.header("Mapa de Calor: Accidentes por Día de la Semana y Hora del Día")
heatmap_data = df_filtrado[df_filtrado['es_accidente'] == 'Accidente'].groupby(['dia_semana', 'hora']).size().unstack()
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="Blues_r", annot=True, fmt="d")
plt.title("Accidentes por Día de la Semana y Hora del Día")
plt.xlabel("Hora del Día")
plt.ylabel("Día de la Semana")
plt.xticks(rotation=45)
plt.yticks(ticks=[0,1,2,3,4,5,6], labels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
st.pyplot(plt)

# Gráfico 6: Barras Apiladas por Tipo de Incidente y Año
st.header("Distribución de Tipos de Incidentes por Año")
stacked_data = df_filtrado.groupby(['año', 'incidencia_tipo']).size().unstack()
stacked_data.plot(kind='bar', stacked=True, figsize=(12, 6), colormap="Blues_r")
plt.title("Distribución de Tipos de Incidentes por Año")
plt.xlabel("Año")
plt.ylabel("Número de Incidentes")
st.pyplot(plt)


# Gráfico 8: Gráfico de Líneas Comparativo de Accidentes por Carretera
st.header("Comparación de Accidentes por Carretera a lo Largo del Tiempo")
accidentes_por_carretera = df_filtrado[df_filtrado['es_accidente'] == 'Accidente'].groupby(['año', 'carretera_nombre']).size().unstack()
accidentes_por_carretera.plot(kind='line', figsize=(12, 6), colormap="Blues_r")
plt.title("Comparación de Accidentes por Carretera a lo Largo del Tiempo")
plt.xlabel("Año")
plt.ylabel("Número de Accidentes")
st.pyplot(plt)

# Gráfico 9: Accidentes por Año
st.header("Número de Accidentes por Año")
accidentes_por_año = df_filtrado[df_filtrado['es_accidente'] == 'Accidente']['año'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.barplot(x=accidentes_por_año.index, y=accidentes_por_año.values)
plt.title("Número de Accidentes por Año")
plt.xlabel("Año")
plt.ylabel("Número de Accidentes")
plt.grid(True)
st.pyplot(plt)
