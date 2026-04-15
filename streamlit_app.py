import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Barra térmica", layout="wide")

HEAT_PALETTE = {
    "bg_gradient_start": "#fff6e9",
    "bg_gradient_end": "#ffe0b2",
    "sidebar_start": "#ffd8a8",
    "sidebar_end": "#ffb370",
    "figure_bg": "#fff1df",
    "axes_bg": "#fff7ee",
    "text": "#5b250f",
    "line": "#d94801",
    "grid": "#f5a26f",
    "ta": "#b22222",
    "tb": "#ff8c00",
}

st.markdown(
    f"""
    <style>
        .stApp {{
            background: linear-gradient(120deg, {HEAT_PALETTE['bg_gradient_start']}, {HEAT_PALETTE['bg_gradient_end']});
            color: {HEAT_PALETTE['text']};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {HEAT_PALETTE['sidebar_start']}, {HEAT_PALETTE['sidebar_end']});
        }}
        h1, h2, h3, p, li, label {{
            color: {HEAT_PALETTE['text']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌡️ Visualización de barra con conducción + convección")

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
delta_T = max(np.ptp(T), 1.0)

# ---- gráfico 1: perfil ----
with col1:
    fig1, ax1 = plt.subplots(facecolor=HEAT_PALETTE["figure_bg"])
    ax1.set_facecolor(HEAT_PALETTE["axes_bg"])
    ax1.plot(x, T, color=HEAT_PALETTE["line"], linewidth=3)
    ax1.scatter([0, L], [TA, TB], color=[HEAT_PALETTE["ta"], HEAT_PALETTE["tb"]], s=90, zorder=5)
    ax1.annotate(
        f"TA = {TA:.1f}°C",
        xy=(0, TA),
        xytext=(0.08 * L, TA + 0.08 * delta_T),
        color=HEAT_PALETTE["ta"],
        arrowprops=dict(arrowstyle="->", color=HEAT_PALETTE["ta"], lw=1.5),
    )
    ax1.annotate(
        f"TB = {TB:.1f}°C",
        xy=(L, TB),
        xytext=(0.68 * L, TB + 0.08 * delta_T),
        color=HEAT_PALETTE["tb"],
        arrowprops=dict(arrowstyle="->", color=HEAT_PALETTE["tb"], lw=1.5),
    )
    ax1.grid(alpha=0.35, color=HEAT_PALETTE["grid"])
    ax1.tick_params(colors=HEAT_PALETTE["text"])
    for spine in ax1.spines.values():
        spine.set_color(HEAT_PALETTE["text"])
    ax1.set_xlabel("x")
    ax1.set_ylabel("T(x)")
    ax1.set_title("Perfil de temperatura", color=HEAT_PALETTE["text"])
    ax1.xaxis.label.set_color(HEAT_PALETTE["text"])
    ax1.yaxis.label.set_color(HEAT_PALETTE["text"])
    st.pyplot(fig1)

# ---- gráfico 2: barra ----
with col2:
    bar = np.tile(T, (30,1))

    fig2, ax2 = plt.subplots(facecolor=HEAT_PALETTE["figure_bg"])
    ax2.set_facecolor(HEAT_PALETTE["axes_bg"])
    im = ax2.imshow(bar, aspect='auto', extent=[0, L, 0, 1], cmap="inferno")
    ax2.axvline(0, color=HEAT_PALETTE["ta"], linestyle="--", linewidth=2)
    ax2.axvline(L, color=HEAT_PALETTE["tb"], linestyle="--", linewidth=2)
    ax2.text(0.02 * L, 1.03, f"TA = {TA:.1f}°C", color=HEAT_PALETTE["ta"], fontsize=10, clip_on=False)
    ax2.text(0.70 * L, 1.03, f"TB = {TB:.1f}°C", color=HEAT_PALETTE["tb"], fontsize=10, clip_on=False)
    ax2.set_title("Visualización de la barra", color=HEAT_PALETTE["text"])
    ax2.set_xlabel("x")
    ax2.set_yticks([])
    ax2.tick_params(colors=HEAT_PALETTE["text"])
    for spine in ax2.spines.values():
        spine.set_color(HEAT_PALETTE["text"])
    ax2.xaxis.label.set_color(HEAT_PALETTE["text"])

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label("Temperatura", color=HEAT_PALETTE["text"])
    cbar.ax.tick_params(color=HEAT_PALETTE["text"])
    plt.setp(cbar.ax.get_yticklabels(), color=HEAT_PALETTE["text"])
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