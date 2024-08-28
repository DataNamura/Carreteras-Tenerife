import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gdown

# Configuración de la página
st.set_page_config(page_title="Análisis de Incidentes en Carreteras", layout="wide")

# Cargar los datos desde Google Drive
@st.cache_data
def load_data():
    file_id = '1ZNtIooyj_3dAQ8dVpbnNOSFYHr0vDZdG'
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

# Título de la aplicación
st.title("Análisis Interactivo de Incidentes en Carreteras")

# Filtros en la barra lateral
st.sidebar.header("Filtros")

min_year = int(df['año'].min())
max_year = int(df['año'].max())
años = st.sidebar.slider("Selecciona el Rango de Años", min_year, max_year, (min_year, max_year))

carreteras_disponibles = df['carretera_nombre'].unique().tolist()
carreteras_seleccionadas = st.sidebar.multiselect("Selecciona Carreteras", carreteras_disponibles, default=carreteras_disponibles)

hora = st.sidebar.slider("Selecciona la Hora del Día", 0, 23, (0, 23))

# Filtrar el dataframe
df_filtrado = df[(df['año'] >= años[0]) & (df['año'] <= años[1]) &
                 (df['carretera_nombre'].isin(carreteras_seleccionadas)) &
                 (df['hora'] >= hora[0]) & (df['hora'] <= hora[1])]

# Excluir los registros donde 'nombre_accidentes' es 'incidente'
df_filtrado_accidentes = df_filtrado[df_filtrado['nombre_accidentes'] != 'incidente']

# Mostrar el dataframe filtrado
st.write(f"Datos Filtrados: {df_filtrado_accidentes.shape[0]} registros")
st.dataframe(df_filtrado_accidentes)

# Gráfico 1: Número de Accidentes por Tramo
st.header("Número de Accidentes en las Carreteras Seleccionadas")
accidentes_por_tramo = df_filtrado_accidentes['tramo_nombre'].value_counts().head(10)
if accidentes_por_tramo.empty:
    st.write("No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accidentes_por_tramo.values, y=accidentes_por_tramo.index, palette="Blues_r")
    plt.title("Top 10 Tramos de Carretera con Más Accidentes")
    plt.xlabel("Número de Accidentes")
    plt.ylabel("Tramo")
    plt.tight_layout()
    plt.savefig("grafico_accidentes_por_tramo.png")
    st.pyplot(plt)
    with open("grafico_accidentes_por_tramo.png", "rb") as file:
        st.download_button(label="Descargar gráfico", data=file, file_name="grafico_accidentes_por_tramo.png", mime="image/png")

# Gráfico 2: Distribución de Accidentes por Hora del Día
st.header("Distribución de Accidentes por Hora del Día")

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
plt.title("Accidentes por Hora del Día")
plt.xlabel("Hora del Día")
plt.ylabel("Número de Accidentes")
plt.xticks(range(24))  # Asegurarse de que todas las horas estén en el eje x
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_accidentes_por_hora.png")
st.pyplot(plt)
with open("grafico_accidentes_por_hora.png", "rb") as file:
    st.download_button(label="Descargar gráfico", data=file, file_name="grafico_accidentes_por_hora.png", mime="image/png")


# Gráfico 3: Correlación entre Día de la Semana y Número de Accidentes
st.header("Correlación entre Día de la Semana y Número de Accidentes")
correlacion_df = df_filtrado_accidentes.groupby('dia_semana').size().reset_index(name='accidentes')

if correlacion_df.empty:
    st.write("No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.regplot(x='dia_semana', y='accidentes', data=correlacion_df, scatter_kws={'s':50}, line_kws={'color':'blue'})
    plt.title('Correlación entre Día de la Semana y Número de Accidentes')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Número de Accidentes')
    
    dias_semana_nombres = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    plt.xticks(ticks=correlacion_df['dia_semana'], labels=dias_semana_nombres, rotation=45)
    
    plt.tight_layout()
    plt.savefig("grafico_correlacion_dia_semana_accidentes.png")
    st.pyplot(plt)
    
    with open("grafico_correlacion_dia_semana_accidentes.png", "rb") as file:
        st.download_button(label="Descargar gráfico", data=file, file_name="grafico_correlacion_dia_semana_accidentes.png", mime="image/png")
# Gráfico 4: Correlación entre Mes y Número de Accidentes
st.header("Correlación entre Mes y Número de Accidentes")
correlacion_df = df_filtrado_accidentes.groupby('mes').size().reset_index(name='accidentes')
if correlacion_df.empty:
    st.write("No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.regplot(x='mes', y='accidentes', data=correlacion_df, scatter_kws={'s':50}, line_kws={'color':'blue'})
    plt.title('Correlación entre Mes y Número de Accidentes')
    plt.xlabel('Mes')
    plt.ylabel('Número de Accidentes')
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    plt.xticks(ticks=correlacion_df['mes'], labels=meses_nombres, rotation=45)
    plt.tight_layout()
    plt.savefig("grafico_correlacion_mes_accidentes.png")
    st.pyplot(plt)
    with open("grafico_correlacion_mes_accidentes.png", "rb") as file:
        st.download_button(label="Descargar gráfico", data=file, file_name="grafico_correlacion_mes_accidentes.png", mime="image/png")

# Gráfico 5: Mapa de Calor de Accidentes por Día de la Semana y Hora
st.header("Mapa de Calor: Accidentes por Día de la Semana y Hora del Día")
heatmap_data = df_filtrado_accidentes.groupby(['dia_semana', 'hora']).size().unstack()
if heatmap_data.empty or heatmap_data.shape[1] == 0:
    st.write("No hay suficientes datos para mostrar el mapa de calor.")
else:
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="Blues_r", annot=True, fmt=".0f")
    plt.title("Accidentes por Día de la Semana y Hora del Día")
    plt.xlabel("Hora del Día")
    plt.ylabel("Día de la Semana")
    plt.yticks(ticks=[0,1,2,3,4,5,6], labels=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
    plt.tight_layout()
    plt.savefig("grafico_mapa_calor_accidentes.png")
    st.pyplot(plt)
    with open("grafico_mapa_calor_accidentes.png", "rb") as file:
        st.download_button(label="Descargar gráfico", data=file, file_name="grafico_mapa_calor_accidentes.png", mime="image/png")

# Gráfico 7: Gráfico de Líneas Comparativo de Accidentes por Carretera
st.header("Comparación de Accidentes por Carretera a lo Largo del Tiempo")
accidentes_por_carretera = df_filtrado_accidentes.groupby(['año', 'carretera_nombre']).size().unstack()
if accidentes_por_carretera.empty or accidentes_por_carretera.shape[1] == 0:
    st.write("No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(12, 6))
    accidentes_por_carretera.plot(kind='line', colormap="Blues_r")
    plt.title("Comparación de Accidentes por Carretera a lo Largo del Tiempo")
    plt.xlabel("Año")
    plt.ylabel("Número de Accidentes")
    plt.tight_layout()
    plt.savefig("grafico_comparativo_accidentes_carretera.png")
    st.pyplot(plt)
    with open("grafico_comparativo_accidentes_carretera.png", "rb") as file:
        st.download_button(label="Descargar gráfico", data=file, file_name="grafico_comparativo_accidentes_carretera.png", mime="image/png")


# Gráfico 8: Accidentes por Año
st.header("Número de Accidentes por Año")
accidentes_por_año = df_filtrado[df_filtrado['es_accidente'] == 'Accidente']['año'].value_counts().sort_index()
if accidentes_por_año.empty:
    st.write("No hay suficientes datos para mostrar el gráfico.")
else:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accidentes_por_año.index, y=accidentes_por_año.values, palette="Blues_r")
    plt.title("Número de Accidentes por Año")
    plt.xlabel("Año")
    plt.ylabel("Número de Accidentes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_accidentes_por_ano.png")  # Guarda el archivo primero
    st.pyplot(plt)
    with open("grafico_accidentes_por_ano.png", "rb") as file:
        st.download_button(label="Descargar gráfico", data=file, file_name="grafico_accidentes_por_ano.png", mime="image/png")

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
st.write(f"Precisión del modelo: {accuracy:.2f}")

# Predecir probabilidad de accidente según los filtros seleccionados por el usuario
st.header("Predicción de Accidente")

# Listas para mostrar los nombres en lugar de números
meses_nombres = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
horas_dia = list(range(0, 24))

# Filtros para la predicción
carretera_seleccionada = st.selectbox("Selecciona una Carretera", carreteras_disponibles)
tramos_disponibles = ['Todos los Tramos'] + df[df['carretera_nombre'] == carretera_seleccionada]['tramo_nombre'].unique().tolist()
tramo_seleccionado = st.selectbox("Selecciona un Tramo", tramos_disponibles)

# Multiselect para seleccionar múltiples meses
meses_seleccionados = st.multiselect("Selecciona los Meses", meses_nombres, default=meses_nombres)

# Multiselect para seleccionar múltiples días de la semana
dias_seleccionados = st.multiselect("Selecciona los Días de la Semana", dias_semana, default=dias_semana)

# Multiselect para seleccionar múltiples horas del día
horas_seleccionadas = st.multiselect("Selecciona las Horas del Día", horas_dia, default=horas_dia)

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
st.write(f"Probabilidad promedio de que ocurra un accidente: {avg_pred_prob:.2%}")

# Explicación de los cálculos
st.markdown("""
**Probabilidad promedio de que ocurra un accidente**: 
Este valor es la media de las probabilidades de accidente calculadas para cada combinación de los filtros seleccionados (meses, días, horas, tramos). Refleja el riesgo general, pero no implica que siempre ocurra un accidente.
""")

