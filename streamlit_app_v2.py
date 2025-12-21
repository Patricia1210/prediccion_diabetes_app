# streamlit_app_v2.py
# Dashboard de predicción de diabetes con modo dual (Clínico/Investigación)
# Versión actualizada con class_weight, umbral 0.30, IMC automático

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
import math
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CONFIGURACIÓN Y ESTILOS
# ============================================================

st.set_page_config(
    page_title="Sistema de Predicción de Diabetes | ENSANUT 2023",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-blue: #1e3a5f;
        --secondary-blue: #2d5986;
        --accent-gold: #b8860b;
        --light-gray: #f8f9fa;
        --medium-gray: #e9ecef;
        --dark-gray: #495057;
        --success-green: #2d6a4f;
        --warning-amber: #d97706;
        --danger-red: #991b1b;
    }
    
    .main {
        background-color: #f4f6f8;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Navegación mejorada */
    .nav-section {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid var(--accent-gold);
    }
    
    .nav-item {
        padding: 0.5rem 0;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
    }
    
    .nav-icon {
        font-size: 1.2rem;
        margin-right: 0.8rem;
    }
    
    .header-academico {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        padding: 2.5rem 3rem;
        border-radius: 0;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 4px solid var(--accent-gold);
    }
    
    .header-title {
        font-family: 'Libre Baskerville', serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .header-authors {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        font-size: 0.95rem;
    }
    
    .author-name {
        font-weight: 600;
        margin-right: 1.5rem;
    }
    
    .paper-section {
        background: white;
        padding: 2.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border-left: 4px solid var(--accent-gold);
    }
    
    .section-title {
        font-family: 'Libre Baskerville', serif;
        color: var(--primary-blue);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--medium-gray);
    }
    
    .subsection-title {
        font-family: 'Libre Baskerville', serif;
        color: var(--secondary-blue);
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        padding: 1.8rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 10px rgba(30, 58, 95, 0.25);
        border-top: 3px solid var(--accent-gold);
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.8rem 0;
        font-family: 'Libre Baskerville', serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 300;
    }
    
    .alert-box {
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1.5rem 0;
        border-left: 5px solid;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .alert-low {
        background-color: #d1fae5;
        border-color: var(--success-green);
        color: #065f46;
    }
    
    .alert-moderate {
        background-color: #fef3c7;
        border-color: var(--warning-amber);
        color: #92400e;
    }
    
    .alert-high {
        background-color: #fee2e2;
        border-color: var(--danger-red);
        color: #7f1d1d;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.05rem;
        box-shadow: 0 4px 12px rgba(30, 58, 95, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 3px solid var(--accent-gold);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(30, 58, 95, 0.4);
    }
    
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 4px;
        border: 2px solid var(--medium-gray);
        padding: 0.6rem;
        font-size: 0.95rem;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--secondary-blue);
    }
    
    label {
        font-weight: 600 !important;
        color: var(--dark-gray) !important;
        font-size: 0.92rem !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: white;
        border-bottom: 2px solid var(--medium-gray);
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: var(--dark-gray);
        background-color: transparent;
        border-bottom: 3px solid transparent;
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        border-bottom: 3px solid var(--accent-gold);
        color: var(--primary-blue);
    }
    
    .dataframe {
        border-radius: 6px;
        overflow: hidden;
        font-size: 0.9rem;
    }
    
    thead tr th {
        background-color: var(--primary-blue) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.8rem !important;
    }
    
    tbody tr:nth-child(even) {
        background-color: var(--light-gray);
    }
    
    .info-box {
        background: var(--light-gray);
        padding: 1.5rem;
        border-radius: 6px;
        border-left: 4px solid var(--secondary-blue);
        margin: 1rem 0;
    }
    
    .academic-divider {
        height: 2px;
        background: linear-gradient(90deg, var(--accent-gold) 0%, var(--medium-gray) 100%);
        margin: 2.5rem 0;
        border: none;
    }
    
    .footer-academic {
        background: var(--primary-blue);
        color: white;
        padding: 2rem;
        margin-top: 3rem;
        text-align: center;
        border-top: 4px solid var(--accent-gold);
    }
    
    .section-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        vertical-align: middle;
    }
    
    .mode-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .mode-clinical {
        background: rgba(45, 106, 79, 0.2);
        border: 1px solid var(--success-green);
        color: var(--success-green);
    }
    
    .mode-research {
        background: rgba(217, 119, 6, 0.2);
        border: 1px solid var(--warning-amber);
        color: var(--warning-amber);
    }
    
    h3 {
        color: var(--primary-blue);
        font-family: 'Libre Baskerville', serif;
        font-weight: 600;
    }
    
    /* IMC calculated display */
    .imc-display {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #0284c7;
        margin: 1rem 0;
    }
    
    .imc-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0369a1;
        font-family: 'Libre Baskerville', serif;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURACIÓN DE MODELOS Y UMBRALES
# ============================================================

BASE = Path(__file__).parent
MODELOS_DIR = BASE / "modelos_entrenados_ext"

# Modelos actualizados con class_weight
RUTAS_MODELOS = {
    "Regresión Logística": MODELOS_DIR / "class_weight_logistic_regression_best.joblib",
    "Gradient Boosting": MODELOS_DIR / "class_weight_gradient_boosting_best.joblib",
    "Random Forest": MODELOS_DIR / "class_weight_random_forest_best.joblib",
    "Ensamble Voting": MODELOS_DIR / "ensemble_voting_classweight.joblib"

}

# Umbral óptimo basado en análisis
OPTIMAL_THRESHOLD = 0.30

# Features (sin Region)
FEATURES = [
    "Edad", "Sexo", "Peso", "Talla", "Cintura", "IMC",
    "Sistolica", "Diastolica", "Depresion", "Prediabetico", "Infarto",
    "Padre_diabetico", "Madre_diabetica", "Colesterol", "Trigliceridos",
    "Fumar", "Alcohol",
]

# Métricas de los modelos (para modo investigación)
METRICAS_MODELOS = {
    "Regresión Logística": {
        "auc": 0.778,
        "recall": 0.926,
        "precision": 0.215,
        "f1": 0.349,
        "descripcion": "Máxima detección de casos (92.6%). Ideal para screening inicial."
    },
    "Gradient Boosting": {
        "auc": 0.797,
        "recall": 0.296,
        "precision": 0.369,
        "f1": 0.329,
        "descripcion": "Balance entre precisión y detección. Mejor AUC-ROC."
    },
    "Random Forest": {
        "auc": 0.783,
        "recall": 0.247,
        "precision": 0.308,
        "f1": 0.274,
        "descripcion": "Modelo conservador con menos falsos positivos."
    },
    "Ensamble Voting": {
        "auc": 0.787,
        "recall": 0.679,
        "precision": 0.281,
        "f1": 0.397,
        "descripcion": "Combinación de modelos para mayor robustez."
    }
}

# ============================================================
# CARGA DE MODELOS
# ============================================================

@st.cache_resource
def cargar_modelos():
    modelos = {}
    for nombre, ruta in RUTAS_MODELOS.items():
        try:
            modelos[nombre] = load(ruta)
        except FileNotFoundError:
            st.warning(f"⚠️ No se encontró el modelo: {nombre}")
    return modelos

MODELOS = cargar_modelos()

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_imc(peso_kg, talla_cm):
    """Calcula el IMC a partir de peso y talla."""
    talla_m = talla_cm / 100
    return peso_kg / (talla_m ** 2)

def clasificar_imc(imc):
    """Clasifica el IMC según estándares OMS."""
    if imc < 18.5:
        return "Bajo peso", "#3b82f6"
    elif imc < 25:
        return "Normal", "#10b981"
    elif imc < 30:
        return "Sobrepeso", "#f59e0b"
    elif imc < 35:
        return "Obesidad I", "#ef4444"
    elif imc < 40:
        return "Obesidad II", "#dc2626"
    else:
        return "Obesidad III", "#991b1b"

def map_si_no(valor_str: str) -> int:
    """Mapea Sí/No a 1/0."""
    return 1 if valor_str == "Sí" else 0

def clasificar_riesgo(prob):
    """Clasifica el riesgo basado en probabilidad (ajustado a umbral 0.30)."""
    if prob < 0.30:
        return "Bajo", "✓ Riesgo bajo detectado. Se recomienda mantener hábitos saludables y realizar controles periódicos preventivos.", "alert-low"
    elif prob < 0.60:
        return "Moderado", "⚠ Riesgo moderado identificado. Es recomendable una evaluación médica detallada y seguimiento especializado.", "alert-moderate"
    else:
        return "Alto", "✕ Riesgo alto detectado. Se recomienda valoración clínica inmediata y estudios complementarios especializados.", "alert-high"

def obtener_coeficientes_logistic(modelo_logistic):
    """Obtiene coeficientes del modelo de regresión logística."""
    try:
        clf = modelo_logistic.named_steps["clf"]
    except Exception:
        clf = modelo_logistic

    coefs = clf.coef_[0]
    df = pd.DataFrame({
        "Variable": FEATURES,
        "Coeficiente": coefs,
        "Odds Ratio": np.exp(coefs)
    })
    df["Impacto"] = df["Coeficiente"].apply(
        lambda c: "↑ Incrementa" if c > 0 else "↓ Disminuye"
    )
    df["abs_coef"] = df["Coeficiente"].abs()
    df = df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])
    return df

def crear_grafico_comparacion_modelos():
    """Crea gráfico comparativo de modelos."""
    df_metrics = pd.DataFrame([
        {"Modelo": k, "Métrica": "AUC-ROC", "Valor": v["auc"]}
        for k, v in METRICAS_MODELOS.items()
    ] + [
        {"Modelo": k, "Métrica": "Recall", "Valor": v["recall"]}
        for k, v in METRICAS_MODELOS.items()
    ] + [
        {"Modelo": k, "Métrica": "Precision", "Valor": v["precision"]}
        for k, v in METRICAS_MODELOS.items()
    ])
    
    fig = px.bar(
        df_metrics,
        x="Modelo",
        y="Valor",
        color="Métrica",
        barmode="group",
        title="Comparación de Métricas por Modelo",
        color_discrete_map={
            "AUC-ROC": "#1e3a5f",
            "Recall": "#2d6a4f",
            "Precision": "#d97706"
        }
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="",
        yaxis_title="Valor de Métrica",
        font={'family': "Source Sans Pro"},
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    return fig

# ============================================================
# SIDEBAR CON MODO DUAL
# ============================================================

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
            <h2 style="margin: 0; font-family: 'Libre Baskerville', serif; font-size: 1.3rem;">
                Sistema de Predicción
            </h2>
            <p style="margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.9;">
                ENSANUT 2023
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MODO DE OPERACIÓN
    st.markdown("""
        <div style="margin: 1rem 0;">
            <h3 style="color: white; font-size: 1rem; margin-bottom: 1rem; font-family: 'Libre Baskerville', serif;">
                🔧 MODO DE OPERACIÓN
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    modo = st.radio(
        "",
        ["🏥 Modo Clínico", "🔬 Modo Investigación"],
        index=0,
        help="**Clínico:** Interfaz simplificada con modelo óptimo pre-seleccionado.\n\n**Investigación:** Acceso completo a todos los modelos y métricas técnicas."
    )
    
    if "Clínico" in modo:
        st.markdown("""
            <div class="nav-section">
                <p style="margin: 0; font-size: 0.85rem; line-height: 1.6;">
                    <strong>Modo optimizado</strong> para profesionales de la salud.
                    Utiliza el modelo con mayor sensibilidad (92.6% de detección).
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="nav-section">
                <p style="margin: 0; font-size: 0.85rem; line-height: 1.6;">
                    <strong>Modo académico</strong> con acceso a comparaciones
                    de modelos y métricas de evaluación detalladas.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # NAVEGACIÓN
    st.markdown("""
        <div style="margin: 1.5rem 0;">
            <h3 style="color: white; font-size: 1rem; margin-bottom: 1rem; font-family: 'Libre Baskerville', serif;">
                📋 SECCIONES
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="nav-section">
            <div class="nav-item">
                <span class="nav-icon">🔬</span>
                <span>Análisis Predictivo</span>
            </div>
            <div class="nav-item">
                <span class="nav-icon">📊</span>
                <span>Análisis de Variables</span>
            </div>
            <div class="nav-item">
                <span class="nav-icon">📖</span>
                <span>Metodología del Sistema</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # DESARROLLADORES
    st.markdown("""
        <div style="margin: 1.5rem 0;">
            <h3 style="color: white; font-size: 1rem; margin-bottom: 1rem; font-family: 'Libre Baskerville', serif;">
                👥 DESARROLLADORES
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="nav-section">
            <p style="margin: 0 0 0.8rem 0; font-weight: 600; font-size: 0.95rem;">
                Patricia Herrejón Calderón
            </p>
            <p style="margin: 0; font-weight: 600; font-size: 0.95rem;">
                Luis Corona Alcantar
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ACERCA DE
    st.markdown("""
        <div style="margin: 1.5rem 0;">
            <h3 style="color: white; font-size: 1rem; margin-bottom: 1rem; font-family: 'Libre Baskerville', serif;">
                ℹ️ ACERCA DE
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="nav-section" style="font-size: 0.85rem; line-height: 1.6;">
            <p style="margin: 0;">
                Sistema desarrollado con base en datos de la Encuesta Nacional 
                de Salud y Nutrición 2023 de México, utilizando técnicas avanzadas 
                de machine learning.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style="text-align: center; padding-top: 1rem; font-size: 0.75rem; opacity: 0.8;">
            <p style="margin: 0;">Versión 1.0.0</p>
            <p style="margin: 0.5rem 0 0 0;">© 2025</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

# Header académico con badge de modo
badge_class = "mode-clinical" if "Clínico" in modo else "mode-research"
badge_text = "MODO CLÍNICO" if "Clínico" in modo else "MODO INVESTIGACIÓN"

st.markdown(f"""
    <div class="header-academico">
        <h1 class="header-title">
            Sistema de Predicción de Riesgo de Diabetes Mellitus
            <span class="mode-badge {badge_class}">{badge_text}</span>
        </h1>
        <p class="header-subtitle">
            Modelo predictivo basado en análisis multivariado de datos ENSANUT 2023
        </p>
        <div class="header-authors">
            <div style="display: inline-block; margin-right: 2rem;">
                <span class="author-name">Patricia Herrejón Calderón</span><br>
                <span style="font-size: 0.85rem; opacity: 0.85;">📧 nutriherrejon@gmail.com</span>
            </div>
            <div style="display: inline-block;">
                <span class="author-name">Luis Corona Alcantar</span><br>
                <span style="font-size: 0.85rem; opacity: 0.85;">📧 lca1643@gmail.com</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Tabs principales
tab_pred, tab_factores, tab_info = st.tabs([
    "🔬 Análisis Predictivo", 
    "📊 Análisis de Variables",
    "📖 Metodología"
])

# ============================================================
# TAB DE PREDICCIÓN
# ============================================================
with tab_pred:
    st.markdown('<div class="paper-section">', unsafe_allow_html=True)
    
    # SELECCIÓN DE MODELO SEGÚN MODO
    if "Clínico" in modo:
        # Modo clínico: modelo fijo (el mejor)
        nombre_modelo = "Regresión Logística"
        st.markdown('<div class="section-title"><span class="section-icon">🤖</span>Modelo Predictivo Optimizado</div>', unsafe_allow_html=True)
        st.info("🎯 **Modelo seleccionado:** Regresión Logística con umbral óptimo 0.30 | **Sensibilidad:** 92.6% | **AUC-ROC:** 0.778")
        
    else:
        # Modo investigación: selector completo
        st.markdown('<div class="section-title"><span class="section-icon">🤖</span>Selección de Modelo Predictivo</div>', unsafe_allow_html=True)
        
        col_m1, col_m2 = st.columns([3, 1])
        
        with col_m1:
            nombre_modelo = st.selectbox(
                "Algoritmo de Machine Learning",
                list(MODELOS.keys()),
                index=0,
                help="Seleccione el modelo de predicción. Cada algoritmo presenta características estadísticas distintas."
            )
        
        with col_m2:
            metrics = METRICAS_MODELOS[nombre_modelo]
            st.markdown(f'''
                <div class="info-box" style="margin-top: 1.8rem;">
                    <strong>AUC:</strong> {metrics["auc"]:.3f}<br>
                    <strong>Recall:</strong> {metrics["recall"]:.3f}
                </div>
            ''', unsafe_allow_html=True)
        
        # Mostrar descripción del modelo
        st.info(f"📝 **{nombre_modelo}:** {metrics['descripcion']}")
    
    st.markdown('<hr class="academic-divider">', unsafe_allow_html=True)
    
    # Sección de datos antropométricos
    st.markdown('<div class="subsection-title">I. Parámetros Antropométricos y Demográficos</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.number_input("Edad (años)", min_value=18, max_value=110, value=40, step=1)
        sexo_str = st.selectbox("Sexo biológico", ["Femenino", "Masculino"])
        peso = st.number_input("Peso corporal (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5)
    
    with col2:
        talla = st.number_input("Talla (cm)", min_value=120.0, max_value=220.0, value=165.0, step=0.5)
        cintura = st.number_input("Perímetro de cintura (cm)", min_value=50.0, max_value=200.0, value=90.0, step=0.5)
    
    with col3:
        # Calcular IMC automáticamente
        imc_calculado = calcular_imc(peso, talla)
        clasificacion_imc, color_imc = clasificar_imc(imc_calculado)
        
        st.markdown(f"""
            <div class="imc-display">
                <div style="font-size: 0.9rem; margin-bottom: 0.5rem; font-weight: 600; color: #0369a1;">
                    📊 Índice de Masa Corporal (Calculado)
                </div>
                <div class="imc-value">{imc_calculado:.2f} kg/m²</div>
                <div style="margin-top: 0.8rem; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid {color_imc};">
                    <strong style="color: {color_imc};">Clasificación:</strong> {clasificacion_imc}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Datos de presión arterial
    col4, col5 = st.columns(2)
    with col4:
        sistolica = st.number_input("Presión sistólica (mmHg)", min_value=80.0, max_value=260.0, value=120.0, step=1.0)
    with col5:
        diastolica = st.number_input("Presión diastólica (mmHg)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
    
    st.markdown('<hr class="academic-divider">', unsafe_allow_html=True)
    
    # Sección de antecedentes clínicos
    st.markdown('<div class="subsection-title">II. Historia Clínica y Antecedentes Familiares</div>', unsafe_allow_html=True)
    
    col6, col7, col8 = st.columns(3)
    
    with col6:
        depresion_str = st.selectbox("Diagnóstico de depresión", ["No", "Sí"])
        prediabetico_str = st.selectbox("Estado prediabético", ["No", "Sí"])
        infarto_str = st.selectbox("Antecedente de infarto", ["No", "Sí"])
    
    with col7:
        padre_str = st.selectbox("Diabetes en padre", ["No", "Sí"])
        madre_str = st.selectbox("Diabetes en madre", ["No", "Sí"])
        colesterol_str = st.selectbox("Hipercolesterolemia", ["No", "Sí"])
    
    with col8:
        trigliceridos_str = st.selectbox("Hipertrigliceridemia", ["No", "Sí"])
        fumar_str = st.selectbox("Tabaquismo", ["No", "Sí"])
        alcohol_str = st.selectbox("Consumo de alcohol", ["No", "Sí"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de cálculo
    st.markdown('<br>', unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular = st.button("🔍 EJECUTAR ANÁLISIS PREDICTIVO", use_container_width=True)
    
    if calcular:
        # Mapear variables
        sexo = 1 if sexo_str == "Masculino" else 0
        depresion = map_si_no(depresion_str)
        prediabetico = map_si_no(prediabetico_str)
        infarto = map_si_no(infarto_str)
        padre_diabetico = map_si_no(padre_str)
        madre_diabetica = map_si_no(madre_str)
        colesterol = map_si_no(colesterol_str)
        trigliceridos = map_si_no(trigliceridos_str)
        fumar = map_si_no(fumar_str)
        alcohol = map_si_no(alcohol_str)
        
        # Crear diccionario de datos (sin Region, con IMC calculado)
        datos_dict = {
            "Edad": edad, 
            "Sexo": sexo, 
            "Peso": peso,
            "Talla": talla, 
            "Cintura": cintura, 
            "IMC": imc_calculado,
            "Sistolica": sistolica, 
            "Diastolica": diastolica,
            "Depresion": depresion, 
            "Prediabetico": prediabetico,
            "Infarto": infarto, 
            "Padre_diabetico": padre_diabetico,
            "Madre_diabetica": madre_diabetica, 
            "Colesterol": colesterol,
            "Trigliceridos": trigliceridos, 
            "Fumar": fumar, 
            "Alcohol": alcohol,
        }
        
        X_nuevo = pd.DataFrame([[datos_dict[feat] for feat in FEATURES]], columns=FEATURES)
        modelo = MODELOS[nombre_modelo]
        
        # Predicción con umbral óptimo 0.30
        if hasattr(modelo, "predict_proba"):
            prob = float(modelo.predict_proba(X_nuevo)[0, 1])
        else:
            prob = float(modelo.decision_function(X_nuevo))
            prob = 1 / (1 + math.exp(-prob))
        
        # Aplicar umbral óptimo
        pred = 1 if prob >= OPTIMAL_THRESHOLD else 0
        
        riesgo_label, mensaje, clase_css = clasificar_riesgo(prob)
        
        # Resultados
        st.markdown('<br><div class="paper-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Resultados del Análisis Predictivo</div>', unsafe_allow_html=True)
        
        # Métricas principales en 3 columnas (SIN GRÁFICO)
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Probabilidad Estimada</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Nivel de Riesgo</div>
                    <div class="metric-value" style="font-size: 2.2rem;">{riesgo_label.upper()}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_r3:
            clase_pred = "Positivo" if pred == 1 else "Negativo"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Clasificación Binaria</div>
                    <div class="metric-value" style="font-size: 2rem;">{clase_pred}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Mensaje de recomendación
        st.markdown(f'<div class="alert-box {clase_css}"><strong>Interpretación clínica:</strong> {mensaje}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # En modo investigación, mostrar métricas del modelo
        if "Investigación" in modo:
            st.markdown('<hr class="academic-divider">', unsafe_allow_html=True)
            st.markdown('<div class="subsection-title">Métricas del Modelo Seleccionado</div>', unsafe_allow_html=True)
            
            metrics = METRICAS_MODELOS[nombre_modelo]
            
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            
            with col_met1:
                st.metric("AUC-ROC", f"{metrics['auc']:.3f}")
            with col_met2:
                st.metric("Recall (Sensibilidad)", f"{metrics['recall']:.3f}")
            with col_met3:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col_met4:
                st.metric("F1-Score", f"{metrics['f1']:.3f}")

# ============================================================
# TAB DE FACTORES DE RIESGO
# ============================================================
with tab_factores:
    st.markdown('<div class="paper-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Análisis de Variables Predictoras</div>', unsafe_allow_html=True)
    
    if "Regresión Logística" in MODELOS:
        st.markdown("Interpretación de coeficientes del modelo de **Regresión Logística**")
        
        modelo_logistic = MODELOS["Regresión Logística"]
        df_coefs = obtener_coeficientes_logistic(modelo_logistic)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **Marco teórico de interpretación:**
        - **Odds Ratio (OR) > 1**: La variable presenta asociación positiva con incremento del riesgo de diabetes
        - **Odds Ratio (OR) < 1**: La variable presenta asociación negativa con disminución del riesgo
        - **Coeficiente β positivo**: Incremento en la log-odds de la probabilidad
        - **Coeficiente β negativo**: Decremento en la log-odds de la probabilidad
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabla de coeficientes
        st.dataframe(
            df_coefs.style.format({
                "Coeficiente": "{:.4f}",
                "Odds Ratio": "{:.4f}",
            }).background_gradient(subset=['Coeficiente'], cmap='RdYlGn_r'),
            use_container_width=True,
            height=400
        )
        
        # Top factores
        st.markdown('<div class="subsection-title">Variables de Mayor Relevancia Estadística</div>', unsafe_allow_html=True)
        
        top_n = 5
        top_vars = df_coefs.head(top_n)
        
        for idx, row in top_vars.iterrows():
            emoji = "↑" if row['Coeficiente'] > 0 else "↓"
            color = "#991b1b" if row['Coeficiente'] > 0 else "#2d6a4f"
            
            st.markdown(f"""
            <div class="info-box" style="border-left-color: {color};">
                <strong>{emoji} {row['Variable']}</strong><br>
                <span style="font-size: 0.9rem;">
                Coeficiente β: <code>{row['Coeficiente']:.4f}</code> | 
                Odds Ratio: <code>{row['Odds Ratio']:.4f}</code> | 
                {row['Impacto']} el riesgo de diabetes
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ El modelo de Regresión Logística no está disponible.")
    
    # Gráfico comparativo en modo investigación
    if "Investigación" in modo:
        st.markdown('<hr class="academic-divider">', unsafe_allow_html=True)
        st.markdown('<div class="subsection-title">Comparación de Modelos</div>', unsafe_allow_html=True)
        
        fig_comp = crear_grafico_comparacion_modelos()
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Tabla comparativa
        st.markdown("**Tabla Comparativa de Métricas**")
        df_comparison = pd.DataFrame([
            {
                "Modelo": k,
                "AUC-ROC": v["auc"],
                "Recall": v["recall"],
                "Precision": v["precision"],
                "F1-Score": v["f1"]
            }
            for k, v in METRICAS_MODELOS.items()
        ])
        st.dataframe(
            df_comparison.style.format({
                "AUC-ROC": "{:.3f}",
                "Recall": "{:.3f}",
                "Precision": "{:.3f}",
                "F1-Score": "{:.3f}"
            }).background_gradient(subset=['Recall'], cmap='Greens'),
            use_container_width=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB DE INFORMACIÓN
# ============================================================
with tab_info:
    st.markdown('<div class="paper-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📖 Metodología y Fundamentos del Sistema</div>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 1. Marco Contextual
        
        Este sistema predictivo ha sido desarrollado utilizando datos de la 
        **Encuesta Nacional de Salud y Nutrición (ENSANUT) 2023** de México.
        
        **Características del estudio:**
        - Base de datos: ENSANUT 2023
        - Población objetivo: Adultos ≥18 años
        - Variables predictoras: 17 indicadores validados
        - Estrategia de balanceo: Class Weight
        - Umbral óptimo: 0.30 (maximiza detección)
        - Validación: Cross-validation estratificada
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 2. Arquitectura de Modelos
        
        **A. Regresión Logística** ⭐ (Recomendado)
        - Sensibilidad: 92.6% (detecta casi todos los casos)
        - AUC-ROC: 0.778
        - Modelo interpretable con coeficientes clínicos
        - Ideal para screening inicial
        
        **B. Gradient Boosting**
        - AUC-ROC más alto: 0.797
        - Balance moderado (Recall: 29.6%)
        - Captura interacciones complejas
        
        **C. Random Forest**
        - Ensemble robusto de árboles
        - Recall: 24.7% (más conservador)
        - Menos falsos positivos
        
        **D. Ensamble Voting**
        - Combinación de los 3 modelos
        - Recall: 67.9%
        - Mayor robustez ante nuevos datos
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_info2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 3. Variables Predictoras (17)
        
        **Dominio Antropométrico:**
        - Edad, sexo, peso, talla
        - Índice de Masa Corporal (IMC) - calculado automáticamente
        - Perímetro de cintura
        
        **Dominio Clínico:**
        - Presión arterial (sistólica/diastólica)
        - Estado prediabético
        - Antecedente de infarto
        - Diagnóstico de depresión
        
        **Dominio Bioquímico y Familiar:**
        - Perfil lipídico (colesterol, triglicéridos)
        - Historia familiar de diabetes (padre/madre)
        - Factores de estilo de vida (tabaco, alcohol)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 4. Optimización del Sistema
        
        **Umbral de Decisión:**
        - Umbral estándar: 0.50
        - Umbral optimizado: **{OPTIMAL_THRESHOLD}**
        - Criterio: Maximizar Recall (sensibilidad)
        
        **Justificación:**
        En diagnóstico de diabetes, es preferible un falso positivo
        (que genera una prueba adicional) que un falso negativo
        (caso no detectado con posibles complicaciones graves).
        
        **Sistema de dos etapas sugerido:**
        1. Screening inicial con este sistema (alta sensibilidad)
        2. Confirmación mediante estudios de laboratorio
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        ### 5. Consideraciones Éticas
        
        **Limitaciones del sistema:**
        - Herramienta de **apoyo diagnóstico**, no diagnóstico definitivo
        - Requiere interpretación por profesional médico calificado
        - Debe complementarse con estudios de laboratorio
        - No sustituye la evaluación clínica integral
        
        **Uso responsable:**
        - Los resultados son estimaciones probabilísticas
        - Variabilidad interpersonal significativa
        - Contexto poblacional: México (ENSANUT 2023)
        - Validación externa requerida para otras poblaciones
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<hr class="academic-divider">', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 6. Referencias y Desarrollo
    
    **Equipo de desarrollo:**
    - Patricia Herrejón Calderón - Desarrollo e implementación de modelos
    - Luis Corona Alcantar - Arquitectura de datos y validación estadística
    
    **Contexto académico:**
    - Tesis de Maestría en Ciencia de Datos
    - Comparación exhaustiva de algoritmos de machine learning
    - Optimización de hiperparámetros y umbrales de decisión
    
    **Base de datos:**
    - Instituto Nacional de Salud Pública (INSP). Encuesta Nacional de Salud y Nutrición 2023 (ENSANUT 2023). México, 2023.
    
    **Tecnologías implementadas:**
    - Python 3.x, Scikit-learn, Pandas, NumPy
    - Imbalanced-learn (para class weighting)
    - Streamlit para interfaz web
    - Plotly para visualizaciones interactivas
    - Joblib para persistencia de modelos
    
    **Contacto y retroalimentación:**
    Para consultas técnicas, sugerencias o colaboraciones académicas, 
    por favor contacte a los autores.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer académico
st.markdown("""
    <div class="footer-academic">
        <p style="margin: 0; font-size: 1.1rem; font-weight: 600;">
            🔬 Sistema de Predicción de Diabetes Mellitus
        </p>
        <p style="margin: 0.5rem 0; font-size: 0.95rem;">
            Basado en ENSANUT 2023 | Desarrollado con fines académicos y de investigación
        </p>
        <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.3); margin: 1.5rem 0;">
        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">
            <strong>Autores:</strong> Patricia Herrejón Calderón & Luis Corona Alcantar
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.8;">
            © 2025 | Versión 1.0.0 | Todos los derechos reservados
        </p>
    </div>
""", unsafe_allow_html=True)