import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ftfy

# 1. Configuración de la interfaz web
st.set_page_config(page_title="FIFA 19 - Buscador de Similitudes", layout="centered")
st.title("FIFA 19 – Buscador de similitudes entre jugadores")

# 2. Carga y limpieza adaptativa de la base de datos
@st.cache_data
def cargar_datos_fifa19():
    try:
        # Intentamos leer con codificación utf-8-sig para capturar Ćaleta-Car automáticamente
        df = pd.read_csv("data.csv", encoding="utf-8-sig")
    except (UnicodeDecodeError, TypeError):
        try:
            # Si falla la codificación estándar, usamos latin-1
            df = pd.read_csv("data.csv", encoding="latin-1")
        except FileNotFoundError:
            st.error("⚠️ No se encontró el archivo 'data.csv'. Asegúrate de que esté en la misma carpeta.")
            st.stop()
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'data.csv'. Asegúrate de que esté en la misma carpeta.")
        st.stop()
        
    # Rellenamos valores vacíos con texto plano para proteger la librería ftfy de errores float
    df["Name"] = df["Name"].fillna("").astype(str).apply(ftfy.fix_text)
    if "Club" in df.columns:
        df["Club"] = df["Club"].fillna("").astype(str).apply(ftfy.fix_text)
        
    df = df.dropna(subset=['Name', 'Age', 'Overall', 'ShortPassing', 'Dribbling'])
    return df

# Inicialización de la variable global de datos
df = cargar_datos_fifa19()

# 3. Formulario de selección y filtros
with st.container():
    jugadores_disponibles = sorted(df["Name"].unique())
    jugador_seleccionado = st.selectbox("Seleccionar jugador:", jugadores_disponibles)
    
    edad_maxima = st.slider("Edad máxima:", min_value=15, max_value=45, value=25, step=1)
    calificacion_maxima = st.slider("Calificación máxima general:", min_value=47, max_value=94, value=70, step=1)
    
    n_resultados = st.number_input("Los N mejores partidos:", min_value=1, max_value=30, value=5)
    buscar = st.button("Encuentra jugadores similares")

# 4. Procesamiento e índice de similitud matemática
if buscar:
    # Extraer el ID único del jugador seleccionado para evitar colisiones por nombres duplicados
    fila_objetivo = df[df["Name"] == jugador_seleccionado].iloc[0]
    id_objetivo = fila_objetivo["ID"] if "ID" in df.columns else jugador_seleccionado
    
    # Aplicar filtros de restricciones del usuario
    df_filtrado = df[
        (df["Age"] <= edad_maxima) & 
        (df["Overall"] <= calificacion_maxima) & 
        (df["Name"] != jugador_seleccionado)
    ]
    
    if df_filtrado.empty:
        st.warning("No hay jugadores en tu base de datos que cumplan simultáneamente con los filtros de Edad y General establecidos.")
    else:
        # Atributos nativos del archivo de FIFA 19 para estructurar los vectores
        columnas_metricas = [
            'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 
            'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 
            'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 
            'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots'
        ]
        
        columnas_metricas = [col for col in columnas_metricas if col in df.columns]
        columna_indice = "ID" if "ID" in df.columns else df.index
        
        # Escalar las métricas de 0 a 1 para normalizar el peso matemático
        scaler = MinMaxScaler()
        df_metricas_norm = scaler.fit_transform(df[columnas_metricas])
        df_norm_completo = pd.DataFrame(df_metricas_norm, columns=columnas_metricas, index=df[columna_indice])
        
        # Aislar vectores técnicos
        vector_objetivo = df_norm_completo.loc[[id_objetivo]]
        vectores_filtrados = df_norm_completo.loc[df_filtrado[columna_indice]]
        
        # Calcular similitud de coseno
        similitudes = cosine_similarity(vectores_filtrados, vector_objetivo)
        
        df_filtrado = df_filtrado.copy()
        df_filtrado["Similitud (%)"] = (similitudes.flatten() * 100).round(2)
        
        # Ordenar de mayor a menor y recortar según la cantidad N solicitada
        resultados = df_filtrado.sort_values(by="Similitud (%)", ascending=False).head(int(n_resultados))
        
        # Renderizar resultados en pantalla
        st.subheader(f"Jugadores más similares a {jugador_seleccionado}:")
        
        columnas_visibles = ["Name", "Age", "Overall"]
        if "Club" in df.columns:
            columnas_visibles.append("Club")
        if "Position" in df.columns:
            columnas_visibles.append("Position")
        columnas_visibles.append("Similitud (%)")
        
        st.dataframe(resultados[columnas_visibles], use_container_width=True)
