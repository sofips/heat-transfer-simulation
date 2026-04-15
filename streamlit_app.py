import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Barra térmica", layout="wide")

st.title("🌡️ Simulación de barra con conducción + convección")

st.markdown("""
Modelo basado en estado estacionario:

d²θ/dx² = (hP / kA) θ

con θ = T - T0
""")

# -------------------------
# SIDEBAR (controles)
# -------------------------
st.sidebar.header("Parámetros")

L = st.sidebar.slider("Longitud L", 0.5, 5.0, 1.0)
h = st.sidebar.slider("Convección h", 0.1, 50.0, 10.0)
k = st.sidebar.slider("Conductividad k", 0.1, 50.0, 10.0)

TA = st.sidebar.slider("Temperatura TA", 0.0, 200.0, 100.0)
TB = st.sidebar.slider("Temperatura TB", 0.0, 200.0, 40.0)
T0 = st.sidebar.slider("Temperatura ambiente T0", 0.0, 200.0, 20.0)

P = 1.0
A = 1.0

# -------------------------
# MODELO
# -------------------------
x = np.linspace(0, L, 300)

alpha = np.sqrt(h * P / (k * A))

# solución general
C1 = (TB - T0 - (TA - T0)*np.exp(-alpha*L)) / (np.exp(alpha*L) - np.exp(-alpha*L))
C2 = (TA - T0) - C1

T = T0 + C1*np.exp(alpha*x) + C2*np.exp(-alpha*x)

# -------------------------
# MÉTRICAS
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Parámetro clave")

st.sidebar.latex(r"\alpha = \sqrt{\frac{hP}{kA}}")
st.sidebar.write(f"α = {alpha:.3f}")

# -------------------------
# PLOTS
# -------------------------
col1, col2 = st.columns(2)

# ---- gráfico 1: perfil ----
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(x, T)
    ax1.set_xlabel("x")
    ax1.set_ylabel("T(x)")
    ax1.set_title("Perfil de temperatura")
    st.pyplot(fig1)

# ---- gráfico 2: barra ----
with col2:
    bar = np.tile(T, (30,1))

    fig2, ax2 = plt.subplots()
    im = ax2.imshow(bar, aspect='auto', extent=[0, L, 0, 1])
    ax2.set_title("Visualización de la barra")
    ax2.set_xlabel("x")
    ax2.set_yticks([])

    plt.colorbar(im, ax=ax2)
    st.pyplot(fig2)

# -------------------------
# INTERPRETACIÓN
# -------------------------
st.markdown("## 🧠 Interpretación física")

st.markdown("""
- **α grande** → domina la convección → la barra tiende rápido a T0  
- **α chico** → domina la conducción → perfil casi lineal  
- **TA y TB** fijan las condiciones de borde  
- **T0 actúa como atractor térmico**
""")