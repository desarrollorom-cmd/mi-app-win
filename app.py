import streamlit as st
import joblib
import numpy as np

modelo = joblib.load("modelo_ventas_win.pkl")

st.title("📈 Predicción de Ventas – WIN Perú")

st.write("Simula las ventas esperadas de un asesor según su contexto")

pdv_trafico = st.number_input("Tráfico promedio del PDV", min_value=0)
dias_laborados = st.number_input("Días laborados", min_value=1, max_value=31)
experiencia_meses = st.number_input("Experiencia del asesor (meses)", min_value=0)
visitas_supervisor = st.number_input("Visitas del supervisor", min_value=0)
promociones = st.selectbox("¿Promociones activas?", [0, 1])

if st.button("🔮 Predecir ventas"):
    X = np.array([[pdv_trafico, dias_laborados, experiencia_meses,
                   visitas_supervisor, promociones]])
    
    prediccion = modelo.predict(X)[0]
    
    st.success(f"Ventas estimadas: {round(prediccion)} líneas")
