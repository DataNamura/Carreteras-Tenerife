import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown  # Asegurarse de importar gdown para la descarga de archivos desde Google Drive
import datetime

# Configuración de la página
st.set_page_config(page_title="🚗 Análisis de Accidentes en Carreteras", layout="wide")

# Instrucciones y contexto
st.markdown("""
# 🚗 Análisis y Predicción de Accidentes en Carreteras de Tenerife

Esta aplicación interactiva proporciona un análisis detallado y predicciones sobre los accidentes de tráfico ocurridos en las carreteras de Tenerife entre los años **2010 y 2024**. 
Es importante destacar que:
- **Datos de 2010**: Hay una menor cantidad de datos disponibles, lo que podría influir en la precisión del análisis para ese año.
- **Datos de 2024**: Los datos están disponibles solo hasta **julio de 2024**.
""")

# Función para cargar datos desde Google Drive
@st.cache_data
def load_data():
    file_id = '1ZNtIooyj_3dAQ8dVpbnNOSFYHr0vDZdG'  # El ID del archivo en Google Drive
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'incidencias_modificacion_1.csv'
    gdown.download(url, output, quiet=False)
    df = pd.read_csv(output)
    df['incidencia_fecha_inicio'] = pd.to_datetime(df['incidencia_fecha_inicio'])
    df['año'] = df['incidencia_fecha_inicio'].dt.year
    df['hora'] = df['incidencia_fecha_inicio'].dt.hour
    df['mes'] = df['incidencia_fecha_inicio'].dt.month
    df['dia_semana'] = df['incidencia_fecha_inicio'].dt.dayofweek
    return df

df = load_data()

# Filtros en la barra lateral
st.sidebar.header("🔎 Filtros")

# Subtítulo y descripción para el filtro de años
st.sidebar.subheader("📅 Rango de Años")
st.sidebar.markdown("Selecciona el rango de años que deseas analizar.")
min_year = int(df['año'].min())
max_year = int(df['año'].max())
años = st.sidebar.slider("Selecciona el Rango de Años", min_year, max_year, (min_year, max_year))

# Separador para organizar los filtros
st.sidebar.markdown("---")

# Subtítulo y filtro para seleccionar carreteras
st.sidebar.subheader("🛣️ Selección de Carreteras")
st.sidebar.markdown("Elige una o varias carreteras para filtrar los datos.")
carreteras_disponibles = df['carretera_nombre'].unique().tolist()
carreteras_disponibles.append("Seleccionar Todas")
carreteras_seleccionadas = st.sidebar.multiselect(
    "Carreteras",
    carreteras_disponibles,
    default=["Seleccionar Todas"]
)

# Separador
st.sidebar.markdown("---")

# Subtítulo y descripción para el filtro de horas
st.sidebar.subheader("⏰ Hora del Día")
st.sidebar.markdown("Filtra los datos según la hora del día en la que ocurrieron los incidentes.")
hora = st.sidebar.slider("Selecciona la Hora del Día", 0, 23, (0, 23))

# Separador
st.sidebar.markdown("---")

# Filtrar el dataframe por años, carreteras y hora seleccionada
if "Seleccionar Todas" in carreteras_seleccionadas:
    carreteras_seleccionadas = carreteras_disponibles[:-1]  # Excluye la opción "Seleccionar Todas"

df_filtrado = df[(df['año'] >= años[0]) & (df['año'] <= años[1]) & 
                 (df['carretera_nombre'].isin(carreteras_seleccionadas)) & 
                 (df['hora'] >= hora[0]) & (df['hora'] <= hora[1])]

# Filtrar solo los accidentes (excluir incidentes si es necesario)
df_filtrado_accidentes = df_filtrado[df_filtrado['nombre_accidentes'] != 'incidente']

# Mapa de carreteras seleccionadas
st.title(f'Mapa de Carreteras Seleccionadas')

# Cargar puntos kilométricos desde un archivo GeoJSON (si lo tienes)
gdf_puntos = gpd.read_file('puntos-kilometricos (1).geojson')

# Filtrar los puntos kilométricos por las carreteras seleccionadas
gdf_puntos_filtrados = gdf_puntos[gdf_puntos['via_nombre'].isin(carreteras_seleccionadas)]

# Crear el mapa
if not gdf_puntos_filtrados.empty:
    map_center = [gdf_puntos_filtrados['pk_latitud'].mean(), gdf_puntos_filtrados['pk_longitud'].mean()]
    mapa = folium.Map(location=map_center, zoom_start=10)

    # Añadir las líneas conectando los puntos de cada carretera
    for via, group in gdf_puntos_filtrados.groupby('via_nombre'):
        puntos = list(zip(group['pk_latitud'], group['pk_longitud']))
        folium.PolyLine(puntos, color='blue', weight=2.5, opacity=0.7).add_to(mapa)

    # Mostrar el mapa interactivo
    folium_static(mapa)
else:
    st.warning("No se encontraron puntos kilométricos para las carreteras seleccionadas.")

# Configuración global del estilo del gráfico
sns.set(style="darkgrid")  # Configura el estilo de fondo oscuro
plt.style.use("dark_background")  # Alternativa para fondo oscuro

# Configuración global del estilo del gráfico usando solo seaborn
sns.set(style="darkgrid")  # Configura el estilo de fondo oscuro para seaborn
# Elimina plt.style.use("dark_background") para evitar conflictos

# Configuración global del estilo del gráfico usando solo matplotlib
plt.style.use("dark_background")  # Alternativa para fondo oscuro
# Elimina sns.set(style="darkgrid") para evitar conflictos

# Gráfico 1: Número de Accidentes por Tramo
st.header("🔝 Número de Accidentes en las Carreteras Seleccionadas")

# Contar los accidentes por tramo en el DataFrame filtrado
accidentes_por_tramo = df_filtrado_accidentes['tramo_nombre'].value_counts().head(10)

# Verificar si hay datos, y si no hay, mostrar un gráfico vacío con una advertencia
if accidentes_por_tramo.empty:
    st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
    # Crear un gráfico vacío para que no quede en negro
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, 'No hay suficientes datos', horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.title("🚧 Top 10 Tramos de Carretera con Más Accidentes")
    plt.tight_layout()
    st.pyplot(plt)
else:
    # Si hay datos suficientes, mostrar el gráfico normal
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accidentes_por_tramo.values, y=accidentes_por_tramo.index, palette="Blues_r")
    plt.title("🚧 Top 10 Tramos de Carretera con Más Accidentes")
    plt.xlabel("Número de Accidentes")
    plt.ylabel("Tramo")
    plt.tight_layout()
    plt.savefig("grafico_accidentes_por_tramo.png")
    st.pyplot(plt)
    with open("grafico_accidentes_por_tramo.png", "rb") as file:
        st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_accidentes_por_tramo.png", mime="image/png")



# Gráfico 2: Distribución de Accidentes por Hora del Día
st.header("⏱️ Distribución de Accidentes por Hora del Día")

# Contar el número de accidentes por hora y asegurarse de que todas las horas del día estén presentes
accidentes_por_hora = df_filtrado_accidentes['hora'].value_counts().sort_index()

# Crear un DataFrame con todas las horas del día (0 a 23)
todas_las_horas = pd.DataFrame({'hora': range(24)})

# Convertir accidentes_por_hora a DataFrame
accidentes_por_hora_df = accidentes_por_hora.reset_index()
accidentes_por_hora_df.columns = ['hora', 'numero_accidentes']

# Unir con el DataFrame de todas las horas
accidentes_por_hora_df = todas_las_horas.merge(accidentes_por_hora_df, how='left', on='hora')

# Rellenar valores nulos con 0 para las horas sin accidentes
accidentes_por_hora_df['numero_accidentes'] = accidentes_por_hora_df['numero_accidentes'].fillna(0).astype(int)

# Graficar
plt.figure(figsize=(10, 6))
sns.lineplot(x=accidentes_por_hora_df['hora'], y=accidentes_por_hora_df['numero_accidentes'], marker='o', color="skyblue")
plt.title("⏰ Accidentes por Hora del Día")
plt.xlabel("Hora del Día")
plt.ylabel("Número de Accidentes")
plt.xticks(range(24))  # Asegurarse de que todas las horas estén en el eje x
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_accidentes_por_hora.png")
st.pyplot(plt)
with open("grafico_accidentes_por_hora.png", "rb") as file:
    st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_accidentes_por_hora.png", mime="image/png")

# Gráfico 3: Correlación entre Día de la Semana y Número de Accidentes
st.header("📅 Correlación entre Día de la Semana y Número de Accidentes")
correlacion_df = df_filtrado_accidentes.groupby('dia_semana').size().reset_index(name='accidentes')

if correlacion_df.empty:
    st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.regplot(x='dia_semana', y='accidentes', data=correlacion_df, scatter_kws={'s':50}, line_kws={'color':'blue'})
    plt.title('📊 Correlación entre Día de la Semana y Número de Accidentes')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Número de Accidentes')
    
    dias_semana_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    plt.xticks(ticks=correlacion_df['dia_semana'], labels=dias_semana_nombres, rotation=45)
    
    plt.tight_layout()
    plt.savefig("grafico_correlacion_dia_semana_accidentes.png")
    st.pyplot(plt)
    
    with open("grafico_correlacion_dia_semana_accidentes.png", "rb") as file:
        st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_correlacion_dia_semana_accidentes.png", mime="image/png")

# Gráfico 4: Correlación entre Mes y Número de Accidentes
st.header("📆 Correlación entre Mes y Número de Accidentes")
correlacion_df = df_filtrado_accidentes.groupby('mes').size().reset_index(name='accidentes')
if correlacion_df.empty:
    st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.regplot(x='mes', y='accidentes', data=correlacion_df, scatter_kws={'s':50}, line_kws={'color':'blue'})
    plt.title('📈 Correlación entre Mes y Número de Accidentes')
    plt.xlabel('Mes')
    plt.ylabel('Número de Accidentes')
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    plt.xticks(ticks=correlacion_df['mes'], labels=meses_nombres, rotation=45)
    plt.tight_layout()
    plt.savefig("grafico_correlacion_mes_accidentes.png")
    st.pyplot(plt)
    with open("grafico_correlacion_mes_accidentes.png", "rb") as file:
        st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_correlacion_mes_accidentes.png", mime="image/png")

# Gráfico 5: Mapa de Calor de Accidentes por Día de la Semana y Hora
st.header("🔥 Mapa de Calor: Accidentes por Día de la Semana y Hora del Día")
heatmap_data = df_filtrado_accidentes.groupby(['dia_semana', 'hora']).size().unstack()
if heatmap_data.empty or heatmap_data.shape[1] == 0:
    st.write("⚠️ No hay suficientes datos para mostrar el mapa de calor.")
else:
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="Blues_r", annot=True, fmt=".0f")
    plt.title("🌡️ Accidentes por Día de la Semana y Hora del Día")
    plt.xlabel("Hora del Día")
    plt.ylabel("Día de la Semana")
    plt.yticks(ticks=[0,1,2,3,4,5,6], labels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
    plt.tight_layout()
    plt.savefig("grafico_mapa_calor_accidentes.png")
    st.pyplot(plt)
    with open("grafico_mapa_calor_accidentes.png", "rb") as file:
        st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_mapa_calor_accidentes.png", mime="image/png")

# Gráfico 7: Gráfico de Líneas Comparativo de Accidentes por Carretera
st.header("📉 Comparación de Accidentes por Carretera a lo Largo del Tiempo")

# Filtro para seleccionar múltiples carreteras para el gráfico comparativo
carreteras_disponibles = df['carretera_nombre'].unique().tolist()
carreteras_seleccionadas_default = ['TF-1']  # Carretera por defecto

carreteras_seleccionadas = st.multiselect(
    "🚧 Selecciona una o más Carreteras para Comparar",
    carreteras_disponibles,
    default=carreteras_seleccionadas_default
)

# Filtrar el dataframe para las carreteras seleccionadas
df_filtrado_comparativo = df_filtrado_accidentes[df_filtrado_accidentes['carretera_nombre'].isin(carreteras_seleccionadas)]

# Agrupación y visualización
accidentes_por_carretera = df_filtrado_comparativo.groupby(['año', 'carretera_nombre']).size().unstack()

if accidentes_por_carretera.empty or accidentes_por_carretera.shape[1] == 0:
    st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(14, 8))  # Aumenta el tamaño de la figura
    ax = accidentes_por_carretera.plot(kind='line', colormap="Blues_r", linewidth=2, marker='o', ax=plt.gca())
    
    # Ajustar la leyenda a la derecha del gráfico
    plt.legend(title='Carreteras', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.title("📈 Comparación de Accidentes a lo Largo del Tiempo")
    plt.xlabel("Año")
    plt.ylabel("Número de Accidentes")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajustar el espacio para la leyenda
    
    plt.savefig("grafico_comparativo_accidentes_carretera.png")
    st.pyplot(plt)
    with open("grafico_comparativo_accidentes_carretera.png", "rb") as file:
        st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_comparativo_accidentes_carretera.png", mime="image/png")

# Gráfico 8: Accidentes por Año
st.header("📅 Número de Accidentes por Año")
accidentes_por_año = df_filtrado[df_filtrado['es_accidente'] == 'Accidente']['año'].value_counts().sort_index()
if accidentes_por_año.empty:
    st.write("⚠️ No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accidentes_por_año.index, y=accidentes_por_año.values, palette="Blues_r")
    plt.title("📅 Número de Accidentes por Año")
    plt.xlabel("Año")
    plt.ylabel("Número de Accidentes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_accidentes_por_ano.png")  # Guarda el archivo primero
    st.pyplot(plt)
    with open("grafico_accidentes_por_ano.png", "rb") as file:
        st.download_button(label="💾 Descargar gráfico", data=file, file_name="grafico_accidentes_por_ano.png", mime="image/png")
st.header("🔍 Insights Clave del Análisis Basado en los Filtros Aplicados")

# 1. Insight sobre el número de accidentes por tramo
top_tramo = df_filtrado_accidentes['tramo_nombre'].value_counts().idxmax()
top_accidentes = df_filtrado_accidentes['tramo_nombre'].value_counts().max()

st.markdown(f"""
### 1. 🛣️ **Tramo con Mayor Número de Accidentes**
El tramo con más accidentes en el rango seleccionado es **{top_tramo}** con un total de **{top_accidentes} accidentes**.
""")

# 2. Insight sobre la hora con más accidentes
hora_pico = df_filtrado_accidentes['hora'].value_counts().idxmax()
num_accidentes_hora = df_filtrado_accidentes['hora'].value_counts().max()

st.markdown(f"""
### 2. ⏰ **Hora del Día con Más Accidentes**
La hora del día con más accidentes es a las **{hora_pico}:00 horas** con **{num_accidentes_hora} accidentes**.
""")

# 3. Insight sobre el día de la semana con más accidentes
dia_semana_pico = df_filtrado_accidentes['dia_semana'].value_counts().idxmax()
dias_semana_nombres = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
dia_semana_nombre = dias_semana_nombres[dia_semana_pico]
num_accidentes_dia = df_filtrado_accidentes['dia_semana'].value_counts().max()

st.markdown(f"""
### 3. 📅 **Día de la Semana con Mayor Riesgo**
El día de la semana con más accidentes es el **{dia_semana_nombre}** con **{num_accidentes_dia} accidentes**.
""")

# 4. Insight sobre el mes con más accidentes
mes_pico = df_filtrado_accidentes['mes'].value_counts().idxmax()
meses_nombres = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
mes_nombre = meses_nombres[mes_pico - 1]
num_accidentes_mes = df_filtrado_accidentes['mes'].value_counts().max()

st.markdown(f"""
### 4. 📆 **Mes con Mayor Número de Accidentes**
El mes con más accidentes es **{mes_nombre}** con **{num_accidentes_mes} accidentes**.
""")


# 5. Insight sobre el día de la semana y hora (patrón)
if not df_filtrado_accidentes.empty:
    heatmap_data = df_filtrado_accidentes.groupby(['dia_semana', 'hora']).size().unstack()
    dia_hora_pico = heatmap_data.stack().idxmax()
    dia_pico_nombre = dias_semana_nombres[dia_hora_pico[0]]
    hora_pico_nombre = dia_hora_pico[1]

    st.markdown(f"""
    ### 5. 🔥 **Patrón de Accidentes por Día de la Semana y Hora del Día**
    El momento más crítico en el rango seleccionado es el **{dia_pico_nombre}** a las **{hora_pico_nombre}:00 horas**.
    """)

# 6. Insight sobre el número de accidentes a lo largo del tiempo
if not df_filtrado_accidentes.empty:
    accidentes_por_año = df_filtrado_accidentes['año'].value_counts().sort_index()
    año_pico = accidentes_por_año.idxmax()
    num_accidentes_año = accidentes_por_año.max()

    st.markdown(f"""
    ### 6. 📈 **Aumento de Accidentes a lo Largo del Tiempo**
    El año con más accidentes en el rango seleccionado es **{año_pico}** con **{num_accidentes_año} accidentes**.
    """)

# 7. Insight sobre las carreteras específicas seleccionadas
if len(carreteras_seleccionadas) > 1:
    carretera_pico = df_filtrado_accidentes['carretera_nombre'].value_counts().idxmax()
    num_accidentes_carretera = df_filtrado_accidentes['carretera_nombre'].value_counts().max()

    st.markdown(f"""
    ### 7. 🛣️ **Carretera con Mayor Número de Accidentes**
    De las carreteras seleccionadas, la **{carretera_pico}** es la más peligrosa con **{num_accidentes_carretera} accidentes**.
    """)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st
import numpy as np

# Preprocesamiento de datos para el modelo
def preprocess_data(df):
    df_model = df.copy()
    df_model['hora'] = pd.to_datetime(df_model['incidencia_fecha_inicio']).dt.hour
    df_model['mes'] = pd.to_datetime(df_model['incidencia_fecha_inicio']).dt.month
    df_model['dia_semana'] = pd.to_datetime(df_model['incidencia_fecha_inicio']).dt.dayofweek
    df_model['es_accidente'] = df_model['es_accidente'].apply(lambda x: 1 if x == 'Accidente' else 0)

    # Codificar variables categóricas
    le_carretera = LabelEncoder()
    le_tramo = LabelEncoder()
    df_model['carretera_nombre'] = le_carretera.fit_transform(df_model['carretera_nombre'])
    df_model['tramo_nombre'] = le_tramo.fit_transform(df_model['tramo_nombre'])

    return df_model, le_carretera, le_tramo

df_model, le_carretera, le_tramo = preprocess_data(df)

# Seleccionar características y etiqueta
X = df_model[['carretera_nombre', 'tramo_nombre', 'hora', 'mes', 'dia_semana']]
y = df_model['es_accidente']

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo de clasificación
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba y mostrar la precisión
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"🎯 Precisión del modelo: {accuracy:.2f}")

# Predecir probabilidad de accidente según los filtros seleccionados por el usuario
st.header("🔮 Predicción de Accidente")

# Listas para mostrar los nombres en lugar de números
meses_nombres = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
horas_dia = list(range(0, 24))

# Filtros para la predicción
carretera_seleccionada = st.selectbox("🛣️ Selecciona una Carretera", carreteras_disponibles)
tramos_disponibles = ['Todos los Tramos'] + df[df['carretera_nombre'] == carretera_seleccionada]['tramo_nombre'].unique().tolist()
tramo_seleccionado = st.selectbox("📍 Selecciona un Tramo", tramos_disponibles)

# Multiselect para seleccionar múltiples meses
meses_seleccionados = st.multiselect("📅 Selecciona los Meses", meses_nombres, default=meses_nombres)

# Multiselect para seleccionar múltiples días de la semana
dias_seleccionados = st.multiselect("🗓️ Selecciona los Días de la Semana", dias_semana, default=dias_semana)

# Multiselect para seleccionar múltiples horas del día
horas_seleccionadas = st.multiselect("⏰ Selecciona las Horas del Día", horas_dia, default=horas_dia)

# Convertir los nombres de meses y días seleccionados a sus correspondientes números para el modelo
meses_numeros = [meses_nombres.index(mes) + 1 for mes in meses_seleccionados]
dias_numeros = [dias_semana.index(dia) for dia in dias_seleccionados]

# Verificar si se seleccionó "Todos los Tramos"
if tramo_seleccionado == 'Todos los Tramos':
    tramos_numeros = df[df['carretera_nombre'] == carretera_seleccionada]['tramo_nombre'].unique()
    tramos_numeros = le_tramo.transform(tramos_numeros)  # Codificar los tramos para el modelo
else:
    tramos_numeros = [le_tramo.transform([tramo_seleccionado])[0]]  # Codificar el tramo seleccionado

# Asegurar que todas las listas tengan la misma longitud
# Calcular la cantidad de combinaciones
n_combinations = len(tramos_numeros) * len(meses_numeros) * len(dias_numeros) * len(horas_seleccionadas)

# Expandir las listas para que coincidan en longitud usando numpy.repeat
input_data = pd.DataFrame({
    'carretera_nombre': np.repeat(le_carretera.transform([carretera_seleccionada]), n_combinations),
    'tramo_nombre': np.repeat(tramos_numeros, len(meses_numeros) * len(dias_numeros) * len(horas_seleccionadas)),
    'hora': np.tile(horas_seleccionadas, n_combinations // len(horas_seleccionadas)),
    'mes': np.tile(meses_numeros, n_combinations // len(meses_numeros)),
    'dia_semana': np.tile(dias_numeros, n_combinations // len(dias_numeros))
})

# Hacer la predicción
pred_probs = model.predict_proba(input_data)[:, 1]  # Probabilidad de accidente

# Calcular la probabilidad promedio de accidente
avg_pred_prob = pred_probs.mean()

# Mostrar resultados en la app
st.write(f"🔮 **Probabilidad promedio de que ocurra un accidente**: {avg_pred_prob:.2%}")

# Explicación de los cálculos
st.markdown("""
**🔍 Explicación de los Cálculos**:
Este valor es la media de las probabilidades de accidente calculadas para cada combinación de los filtros seleccionados (meses, días, horas, tramos). Refleja el riesgo general, pero no implica que siempre ocurra un accidente.
""")
# Explicación de la precisión del modelo
st.markdown("""
**🎯 Precisión del Modelo**:
La precisión muestra qué tan bien el modelo está funcionando en términos de clasificar correctamente los accidentes. Un valor de **0.78** indica que el 78% de las veces, el modelo predijo correctamente si hubo o no un accidente, proporcionando una evaluación general de su desempeño en el conjunto de prueba.
""")

#--------------------------- BAROMETRO hemos excluido la variante mes para que sea mas preciso la prediccion 
#solo teniendo en cuenta las horas del dia y el dia
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import plotly.graph_objects as go

# Preprocesamiento de datos para el modelo sin usar el mes
def preprocess_data_no_month(df):
    df_model = df.copy()
    df_model['incidencia_fecha_inicio'] = pd.to_datetime(df_model['incidencia_fecha_inicio'])
    df_model['hora'] = df_model['incidencia_fecha_inicio'].dt.hour
    df_model['dia_semana'] = df_model['incidencia_fecha_inicio'].dt.dayofweek
    df_model['es_accidente'] = df_model['es_accidente'].apply(lambda x: 1 if x == 'Accidente' else 0)

    # Codificar variables categóricas
    le_carretera = LabelEncoder()
    df_model['carretera_nombre'] = le_carretera.fit_transform(df_model['carretera_nombre'])
    return df_model, le_carretera

# Preprocesar los datos sin el mes
df_model_no_month, le_carretera_no_month = preprocess_data_no_month(df)

# Seleccionar características (sin el mes) y etiqueta
X_no_month = df_model_no_month[['carretera_nombre', 'hora', 'dia_semana']]
y_no_month = df_model_no_month['es_accidente']

# Dividir datos en entrenamiento y prueba
X_train_no_month, X_test_no_month, y_train_no_month, y_test_no_month = train_test_split(X_no_month, y_no_month, test_size=0.3, random_state=42)

# Entrenar un modelo de clasificación
model_no_month = RandomForestClassifier(n_estimators=100, random_state=42)
model_no_month.fit(X_train_no_month, y_train_no_month)

# Obtener la hora y el día de la semana actuales
hora_actual = datetime.now().hour
dia_semana_actual = datetime.now().weekday()

# Traducir días al español
dias_semana_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# Mostrar título y el filtro de carreteras en el cuerpo principal
st.header("🔮 Predicción de Accidente Actual")

# Seleccionar la carretera
carretera_seleccionada = st.selectbox("🛣️ Selecciona una Carretera", df['carretera_nombre'].unique(), key='carretera_select')

# Convertir la carretera seleccionada a su valor codificado
carretera_encoded = le_carretera_no_month.transform([carretera_seleccionada])[0]

# Crear un dataframe con las características actuales (sin mes) para predecir
input_data_no_month = pd.DataFrame({
    'carretera_nombre': [carretera_encoded],
    'hora': [hora_actual],
    'dia_semana': [dia_semana_actual]
})

# Predecir la probabilidad de accidente
prob_accidente_no_month = model_no_month.predict_proba(input_data_no_month)[:, 1][0]

# Mostrar la predicción
st.write(f"🔮 **Probabilidad de Accidente Actual**: {prob_accidente_no_month:.2%}")

# Crear un gráfico de tipo Gauge para mostrar la probabilidad de accidente
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prob_accidente_no_month * 100,
    title={'text': "Probabilidad de Accidente Actual"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "black"},
        'steps': [
            {'range': [0, 33], 'color': "green"},
            {'range': [33, 66], 'color': "orange"},
            {'range': [66, 100], 'color': "red"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': prob_accidente_no_month * 100
        }
    }
))

# Mostrar la hora y fecha actual en Streamlit
st.write(f"**Hora Actual**: {datetime.now().strftime('%H:%M:%S')}")
st.write(f"**Día Actual**: {dias_semana_es[dia_semana_actual]}")

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

