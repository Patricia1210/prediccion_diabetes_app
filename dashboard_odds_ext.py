# dashboard_odds_ext.py
# Dashboard clínico de odds ratios (modelo extendido ENSANUT)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------- CONFIGURACIÓN BÁSICA ----------------
BASE = Path(__file__).parent
RUTA_TABLA = BASE / "PASO11_tabla_interpretativa_ext.csv"

st.set_page_config(
    page_title="Modelo de riesgo de diabetes - Odds Ratios",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- ESTILOS PERSONALIZADOS ----------------
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    h3 {
        color: #34495e;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- CARGA DE DATOS ----------------
@st.cache_data
def cargar_tabla():
    df = pd.read_csv(RUTA_TABLA)
    columnas_necesarias = [
        "variable",
        "coeficiente",
        "odds_ratio",
        "ci_inf_95",
        "ci_sup_95",
        "p_value",
        "clasificacion_efecto",
        "interpretacion_clinica_auto",
    ]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]
    if faltantes:
        st.error(f"⚠ La tabla no tiene estas columnas necesarias: {faltantes}")
    return df

df = cargar_tabla()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Logo", width=150)
    st.title("🧬 Panel de Control")
    st.markdown("---")
    
    st.markdown("### 📊 Información del Modelo")
    st.info(f"""
    **Total de variables:** {len(df)}  
    **Fuente de datos:** ENSANUT 2023 + Pima  
    **Tipo de modelo:** Regresión Logística
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Navegación Rápida")
    seccion = st.radio(
        "Ir a sección:",
        ["📈 Resumen Ejecutivo", "🌲 Análisis de Odds Ratios", "📋 Tabla Detallada", "💊 Recomendaciones"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Acerca de")
    st.caption("""
    Dashboard desarrollado para el análisis de factores de riesgo 
    de diabetes tipo 2 en población mexicana.
    """)

# ---------------- HEADER ----------------
st.markdown("""
    <h1 style='text-align: center;'>
        🩺 Modelo Predictivo de Diabetes Tipo 2
    </h1>
    <p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>
        Análisis de Odds Ratios basado en ENSANUT 2023
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- MÉTRICAS PRINCIPALES ----------------
st.markdown("## 📊 Métricas Clave del Modelo")

col1, col2, col3 = st.columns(3)

# Calcular métricas
n_riesgo = len(df[df["odds_ratio"] > 1])
n_protector = len(df[df["odds_ratio"] < 1])
max_or = df["odds_ratio"].max()

with col1:
    st.metric(
        label="Variables Analizadas",
        value=f"{len(df)}",
        delta="Total en el modelo"
    )

with col2:
    st.metric(
        label="Factores de Riesgo",
        value=f"{n_riesgo}",
        delta="OR > 1",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Factores Protectores",
        value=f"{n_protector}",
        delta="OR < 1",
        delta_color="normal"
    )

st.markdown("---")

# ---------------- TABS PRINCIPALES ----------------
tab1, tab2, tab3 = st.tabs(["📈 Visualizaciones", "📋 Tabla Interactiva", "💊 Recomendaciones Clínicas"])

# ==================== TAB 1: VISUALIZACIONES ====================
with tab1:
    col_viz1, col_viz2 = st.columns([1.5, 1])
    
    with col_viz1:
        st.markdown("### 🌲 Forest Plot - Odds Ratios con Intervalos de Confianza")
        
        # Forest plot con Plotly
        df_plot = df.sort_values("odds_ratio", ascending=True)
        
        # Crear colores según el efecto
        colors = ['#e74c3c' if or_val > 1 else '#27ae60' for or_val in df_plot["odds_ratio"]]
        
        fig = go.Figure()
        
        # Agregar intervalos de confianza
        for idx, row in df_plot.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['ci_inf_95'], row['ci_sup_95']],
                y=[row['variable'], row['variable']],
                mode='lines',
                line=dict(color='lightgray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Agregar puntos de OR
        fig.add_trace(go.Scatter(
            x=df_plot['odds_ratio'],
            y=df_plot['variable'],
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"OR: {or_val:.2f}<br>IC 95%: [{ci_low:.2f}, {ci_high:.2f}]<br>p-valor: {pval:.4f}"
                  for or_val, ci_low, ci_high, pval in 
                  zip(df_plot['odds_ratio'], df_plot['ci_inf_95'], df_plot['ci_sup_95'], df_plot['p_value'])],
            hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>',
            showlegend=False
        ))
        
        # Línea de referencia en OR = 1
        fig.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            xaxis_title="Odds Ratio",
            yaxis_title="",
            height=600,
            hovermode='closest',
            plot_bgcolor='white',
            xaxis=dict(
                type='log',
                gridcolor='lightgray',
                showline=True,
                linecolor='black'
            ),
            yaxis=dict(
                showgrid=False,
                showline=True,
                linecolor='black'
            ),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 Los puntos rojos indican aumento de riesgo (OR > 1), los verdes indican efecto protector (OR < 1)")
    
    with col_viz2:
        st.markdown("### 📊 Distribución de Efectos")
        
        # Gráfico de pie con clasificación de efectos
        clasificacion_counts = df['clasificacion_efecto'].value_counts()
        
        fig_pie = px.pie(
            values=clasificacion_counts.values,
            names=clasificacion_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
        )
        
        fig_pie.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Significancia estadística
        st.markdown("### 📉 Significancia Estadística")
        
        df['significancia'] = df['p_value'].apply(
            lambda x: 'p < 0.001' if x < 0.001 else ('p < 0.05' if x < 0.05 else 'No significativo')
        )
        
        sig_counts = df['significancia'].value_counts()
        
        fig_bar = px.bar(
            x=sig_counts.index,
            y=sig_counts.values,
            color=sig_counts.index,
            color_discrete_map={
                'p < 0.001': '#27ae60',
                'p < 0.05': '#f39c12',
                'No significativo': '#95a5a6'
            },
            labels={'x': 'Nivel de Significancia', 'y': 'Número de Variables'}
        )
        
        fig_bar.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Top 5 factores
    st.markdown("---")
    col_top1, col_top2 = st.columns(2)
    
    with col_top1:
        st.markdown("### 🔴 Top 5 - Mayor Riesgo")
        top_riesgo = df[df["odds_ratio"] > 1].nlargest(5, "odds_ratio")
        
        for idx, row in top_riesgo.iterrows():
            with st.container():
                st.markdown(f"""
                <div style='background-color: #ffebee; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #e74c3c;'>
                    <h4 style='margin: 0; color: #c0392b;'>{row['variable']}</h4>
                    <p style='margin: 5px 0; font-size: 1.1rem;'><b>OR:</b> {row['odds_ratio']:.2f} 
                    (IC 95%: {row['ci_inf_95']:.2f} - {row['ci_sup_95']:.2f})</p>
                    <p style='margin: 5px 0; color: #7f8c8d;'>{row['interpretacion_clinica_auto']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col_top2:
        st.markdown("### 🟢 Top 5 - Mayor Protección")
        top_protector = df[df["odds_ratio"] < 1].nsmallest(5, "odds_ratio")
        
        for idx, row in top_protector.iterrows():
            with st.container():
                st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #27ae60;'>
                    <h4 style='margin: 0; color: #1e8449;'>{row['variable']}</h4>
                    <p style='margin: 5px 0; font-size: 1.1rem;'><b>OR:</b> {row['odds_ratio']:.2f} 
                    (IC 95%: {row['ci_inf_95']:.2f} - {row['ci_sup_95']:.2f})</p>
                    <p style='margin: 5px 0; color: #7f8c8d;'>{row['interpretacion_clinica_auto']}</p>
                </div>
                """, unsafe_allow_html=True)

# ==================== TAB 2: TABLA INTERACTIVA ====================
with tab2:
    st.markdown("### 🔍 Explorador de Variables")
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        opciones_clasif = ["Todos"] + sorted(df["clasificacion_efecto"].unique().tolist())
        clasif_sel = st.selectbox("📊 Clasificación de efecto:", opciones_clasif)
    
    with col_filter2:
        tipo_efecto = st.selectbox(
            "🎯 Tipo de efecto:",
            ["Todos", "Aumento de riesgo (OR > 1)", "Efecto protector (OR < 1)"]
        )
    
    with col_filter3:
        nivel_sig = st.selectbox(
            "📈 Significancia estadística:",
            ["Todos", "Altamente significativo (p < 0.001)", "Significativo (p < 0.05)", "No significativo"]
        )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    if clasif_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["clasificacion_efecto"] == clasif_sel]
    
    if tipo_efecto == "Aumento de riesgo (OR > 1)":
        df_filtrado = df_filtrado[df_filtrado["odds_ratio"] > 1]
    elif tipo_efecto == "Efecto protector (OR < 1)":
        df_filtrado = df_filtrado[df_filtrado["odds_ratio"] < 1]
    
    if nivel_sig == "Altamente significativo (p < 0.001)":
        df_filtrado = df_filtrado[df_filtrado["p_value"] < 0.001]
    elif nivel_sig == "Significativo (p < 0.05)":
        df_filtrado = df_filtrado[df_filtrado["p_value"] < 0.05]
    elif nivel_sig == "No significativo":
        df_filtrado = df_filtrado[df_filtrado["p_value"] >= 0.05]
    
    # Mostrar resumen de filtros
    st.info(f"📋 Mostrando {len(df_filtrado)} de {len(df)} variables")
    
    # Preparar tabla para mostrar
    df_mostrar = df_filtrado.copy()
    for col in ["coeficiente", "odds_ratio", "ci_inf_95", "ci_sup_95", "p_value"]:
        if col in df_mostrar.columns:
            df_mostrar[col] = df_mostrar[col].astype(float).round(4)
    
    df_mostrar = df_mostrar.rename(columns={
        "variable": "Variable",
        "coeficiente": "Coeficiente",
        "odds_ratio": "Odds Ratio",
        "ci_inf_95": "IC 95% Inf",
        "ci_sup_95": "IC 95% Sup",
        "p_value": "p-valor",
        "clasificacion_efecto": "Clasificación",
        "interpretacion_clinica_auto": "Interpretación Clínica",
    })
    
    # Mostrar tabla estilizada
    st.dataframe(
        df_mostrar,
        use_container_width=True,
        height=500,
        hide_index=True
    )
    
    # Botón de descarga
    csv = df_mostrar.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar tabla filtrada como CSV",
        data=csv,
        file_name="modelo_diabetes_filtrado.csv",
        mime="text/csv",
    )

# ==================== TAB 3: RECOMENDACIONES ====================
with tab3:
    st.markdown("### 💊 Recomendaciones Clínicas Basadas en Evidencia")
    
    st.success("""
    **🎯 Objetivo:** Proporcionar guías de intervención preventiva basadas en los factores de riesgo 
    identificados en el modelo predictivo.
    """)
    
    # Sección de factores de riesgo
    st.markdown("---")
    st.markdown("## 🔴 Factores de Alto Riesgo - Intervenciones Prioritarias")
    
    top_riesgo = df[df["odds_ratio"] > 1].nlargest(5, "odds_ratio")
    
    for idx, (_, row) in enumerate(top_riesgo.iterrows(), 1):
        with st.expander(f"**{idx}. {row['variable']}** - OR: {row['odds_ratio']:.2f}", expanded=(idx <= 2)):
            col_rec1, col_rec2 = st.columns([1, 2])
            
            with col_rec1:
                st.metric("Odds Ratio", f"{row['odds_ratio']:.2f}")
                st.metric("p-valor", f"{row['p_value']:.4f}")
                st.metric("IC 95%", f"[{row['ci_inf_95']:.2f}, {row['ci_sup_95']:.2f}]")
            
            with col_rec2:
                st.markdown("**📋 Interpretación:**")
                st.write(row['interpretacion_clinica_auto'])
                
                st.markdown("**💡 Recomendaciones sugeridas:**")
                st.markdown("""
                - Tamizaje y monitoreo frecuente
                - Educación específica del paciente
                - Intervención temprana y seguimiento
                - Derivación a especialista si aplica
                """)
    
    # Sección de factores protectores
    st.markdown("---")
    st.markdown("## 🟢 Factores Protectores - Promoción de la Salud")
    
    top_protector = df[df["odds_ratio"] < 1].nsmallest(5, "odds_ratio")
    
    for idx, (_, row) in enumerate(top_protector.iterrows(), 1):
        with st.expander(f"**{idx}. {row['variable']}** - OR: {row['odds_ratio']:.2f}", expanded=(idx <= 2)):
            col_rec1, col_rec2 = st.columns([1, 2])
            
            with col_rec1:
                st.metric("Odds Ratio", f"{row['odds_ratio']:.2f}")
                st.metric("p-valor", f"{row['p_value']:.4f}")
                st.metric("IC 95%", f"[{row['ci_inf_95']:.2f}, {row['ci_sup_95']:.2f}]")
            
            with col_rec2:
                st.markdown("**📋 Interpretación:**")
                st.write(row['interpretacion_clinica_auto'])
                
                st.markdown("**💡 Estrategias de promoción:**")
                st.markdown("""
                - Fomentar y reforzar comportamientos saludables
                - Programas educativos comunitarios
                - Políticas de salud pública enfocadas
                - Intervenciones poblacionales
                """)
    
    # Resumen ejecutivo
    st.markdown("---")
    st.markdown("## 📝 Resumen Ejecutivo para Toma de Decisiones")
    
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.warning("""
        ### ⚠️ Áreas de Intervención Crítica
        
        Basado en los odds ratios más altos, se recomienda:
        
        1. **Programas de detección temprana** enfocados en población con factores de alto riesgo
        2. **Intervenciones multicomponente** que aborden simultáneamente varios factores modificables
        3. **Seguimiento longitudinal** de pacientes en grupos de riesgo elevado
        """)
    
    with col_sum2:
        st.success("""
        ### ✅ Oportunidades de Prevención
        
        Los factores protectores identificados sugieren:
        
        1. **Promoción activa** de comportamientos saludables comprobados
        2. **Políticas públicas** que faciliten el acceso a factores protectores
        3. **Educación poblacional** sobre modificación de riesgo
        """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p><b>Dashboard de Análisis Predictivo de Diabetes</b></p>
        <p>Desarrollado con datos de ENSANUT 2023 | Modelo de Regresión Logística</p>
        <p style='font-size: 0.9rem;'>💡 Para más información sobre interpretación de odds ratios y significancia estadística, 
        consulte la documentación técnica del proyecto.</p>
    </div>
""", unsafe_allow_html=True)