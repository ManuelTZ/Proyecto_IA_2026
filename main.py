# =============================================================================
# main.py
# Simulador de Pandemia Global con IA — Interfaz Streamlit
# Ejecutar: streamlit run main.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import os

# ── Dependencias opcionales con fallback ───────────────────────────────────
try:
    import pydeck as pdk
    HAS_PYDECK = True
except ImportError:
    HAS_PYDECK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from simulation_engine import PandemiaSimulator, NODES, CONNECTIVITY
from ai_logic import AIMonitor
from report_generator import ReportGenerator

# ─────────────────────────────────────────────────────────── PAGE CONFIG
st.set_page_config(
    page_title="🦠 Simulador de Pandemia Global",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────── CSS
st.markdown("""
<style>
/* Fondo oscuro global */
[data-testid="stAppViewContainer"] { background: #0a0e17; }
[data-testid="stSidebar"]          { background: #111827; }
[data-testid="stHeader"]           { background: transparent; }

/* Tarjetas de métricas */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2035, #1e2640);
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid #2a3550;
}
/* Badges de prioridad */
.badge-critica { color:#ff4444; font-weight:700; font-size:0.85em; }
.badge-alta    { color:#ff8800; font-weight:700; font-size:0.85em; }
.badge-media   { color:#ffcc00; font-weight:700; font-size:0.85em; }
.badge-baja    { color:#8888ff; font-weight:700; font-size:0.85em; }

/* Scrollable AI panel */
.ai-scroll { max-height: 460px; overflow-y: auto; padding-right: 6px; }

/* Subtle divider */
hr { border-color: #1e2640 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────── HELPERS
PRIORITY_ICON = {"CRÍTICA": "🔴", "ALTA": "🟠", "MEDIA": "🟡", "BAJA": "🔵"}
POLICY_ICON   = {
    "CERRAR_FRONTERAS":     "🚫✈️",
    "CUARENTENA_ESTRICTA":  "🏠🔒",
    "VACUNACIÓN_MASIVA":    "💉",
    "MONITOREO_INTENSIVO":  "🔬",
}

def fmt(n: float) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return str(int(n))


# ─────────────────────────────────────────────────────────── MAPA PYDECK
def build_pydeck(sim: PandemiaSimulator):
    state_df = sim.get_state_df()
    arc_df   = sim.get_arc_data()

    # ScatterplotLayer: círculos proporcionales a la población
    scatter_rows = []
    for _, r in state_df.iterrows():
        radius = max(150_000, min(1_800_000, np.sqrt(r["population"]) * 420))
        scatter_rows.append({
            "lat": r["lat"], "lon": r["lon"],
            "radius": radius,
            "color": r["color"],
            "node": r["node"],
            "infected": fmt(r["infected"]),
            "infected_pct": f"{r['infected_pct']*100:.2f}%",
            "deaths": fmt(r["deaths"]),
            "icu": "⚠️ COLAPSADA" if r["icu_overflow"] else f"{r['icu_load']*100:.0f}%",
        })
    scatter_df = pd.DataFrame(scatter_rows)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        scatter_df,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
        opacity=0.82,
        stroked=True,
        get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=1,
    )

    # ArcLayer: rutas de transporte
    arc_rows = []
    for _, r in arc_df.iterrows():
        arc_rows.append({
            "src": [r["source_lon"], r["source_lat"]],
            "tgt": [r["target_lon"], r["target_lat"]],
            "cs":  r["color_source"],
            "ct":  r["color_target"],
        })
    arc_layer = pdk.Layer(
        "ArcLayer",
        arc_rows,
        get_source_position="src",
        get_target_position="tgt",
        get_source_color="cs",
        get_target_color="ct",
        width_min_pixels=1,
        width_scale=2,
        pickable=False,
        auto_highlight=False,
    )

    view = pdk.ViewState(latitude=18, longitude=5, zoom=1.35, pitch=28)

    tooltip = {
        "html": (
            "<div style='background:#1a2035;padding:8px 12px;border-radius:8px;"
            "color:white;font-size:12px;border:1px solid #2a3550'>"
            "<b style='font-size:14px'>{node}</b><br/>"
            "🤒 Infectados: <b>{infected}</b> ({infected_pct})<br/>"
            "💀 Fallecidos: <b>{deaths}</b><br/>"
            "🏥 UCI: <b>{icu}</b></div>"
        ),
        "style": {"z-index": "9999"},
    }

    return pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/dark-v11",
    )


# ─────────────────────────────────────────────────────────── CHARTS
def plot_global_curve(sim: PandemiaSimulator) -> io.BytesIO:
    totals = sim.get_global_totals_series()
    if len(totals["day"]) < 2:
        return None

    BG = "#0e1317"
    fig, ax = plt.subplots(figsize=(10, 3.4), facecolor=BG)
    ax.set_facecolor("#131b27")

    mil = FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
    ax.yaxis.set_major_formatter(mil)
    ax.grid(True, color="#1e2a3a", linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2a3a")

    days = totals["day"]
    ax.fill_between(days, totals["infected"],  alpha=0.28, color="#ff4040")
    ax.plot(days, totals["infected"],  color="#ff4040", lw=2,   label="Infectados")
    ax.fill_between(days, totals["recovered"], alpha=0.20, color="#40d870")
    ax.plot(days, totals["recovered"], color="#40d870", lw=1.5, label="Recuperados", ls="--")
    ax.plot(days, totals["deaths"],    color="#aaaaaa", lw=1.5, label="Fallecidos",  ls=":")

    ax.set_xlabel("Día", color="#c0cce0", fontsize=10)
    ax.tick_params(colors="#c0cce0")
    ax.legend(facecolor="#1a2035", labelcolor="#c0cce0", fontsize=9,
              loc="upper left", framealpha=0.85)
    ax.set_title(f"Curva Global — Día {sim.day}", color="#c0cce0", fontsize=11)

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_node_sparklines(sim: PandemiaSimulator) -> io.BytesIO:
    history  = sim.get_history_df()
    state_df = sim.get_state_df().sort_values("infected_pct", ascending=False)
    nodes    = state_df["node"].tolist()
    n_nodes  = len(nodes)
    cols     = 4
    rows     = (n_nodes + cols - 1) // cols

    BG = "#0e1317"
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 2.4), facecolor=BG)
    axes = axes.flatten()
    fig.suptitle(f"Evolución por Región — Día {sim.day}",
                 color="#c0cce0", fontsize=12, fontweight="bold")

    for k, nd in enumerate(nodes):
        ax = axes[k]
        ax.set_facecolor("#131b27")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a3a")
        ax.tick_params(colors="#c0cce0", labelsize=7)

        col = f"{nd}_I"
        if col in history.columns and len(history) > 1:
            series = history[col].values
            days   = np.arange(len(series))
            ax.fill_between(days, series, color="#ff4040", alpha=0.45)
            ax.plot(days, series, color="#ff5555", lw=1.2)

        pct = state_df[state_df["node"] == nd]["infected_pct"].values[0] * 100
        icu_ov = state_df[state_df["node"] == nd]["icu_overflow"].values[0]
        title_str = f"{'⚠️ ' if icu_ov else ''}{nd}"
        title_color = "#ff4040" if icu_ov else "#c0cce0"
        ax.set_title(f"{title_str}\n{pct:.2f}%", color=title_color, fontsize=8.5,
                     fontweight="bold" if icu_ov else "normal")
        ax.set_xticks([])
        ymax = max(ax.get_ylim()[1], 1)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))

    # Ocultar ejes sobrantes
    for k in range(n_nodes, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout(pad=0.5, h_pad=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────── SIDEBAR
def render_sidebar():
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Covid-19_without_background.png/240px-Covid-19_without_background.png",
            width=60,
        )
        st.title("⚙️ Control de Simulación")

        with st.expander("🦠 Parámetros del Patógeno", expanded=True):
            beta  = st.slider("β  Transmisibilidad", 0.05, 1.0,  0.30, 0.01,
                              help="Prob. de contagio por contacto/día")
            gamma = st.slider("γ  Tasa de Recuperación", 0.01, 0.30, 0.05, 0.005,
                              help="Fracción de infectados que se recuperan/día")
            delta = st.slider("δ  Letalidad", 0.001, 0.10, 0.010, 0.001,
                              help="Fracción de infectados que fallecen")
            R0    = beta / gamma
            col1, col2 = st.columns(2)
            col1.metric("R₀", f"{R0:.2f}")
            col2.metric("Inmunidad rebaño", f"{(1-1/max(1,R0))*100:.0f}%")

        with st.expander("🌍 Escenario", expanded=True):
            origin = st.selectbox("País de origen", list(NODES.keys()), index=0)
            seed   = st.number_input("Infectados iniciales", 10, 500_000, 1_000, 500)

        with st.expander("🚦 Intervenciones"):
            closed_borders = st.multiselect(
                "Cerrar fronteras", list(NODES.keys()),
                help="Reduce movilidad internacional en 96%"
            )
            quarantine_nodes = st.multiselect(
                "Cuarentena (↓β 55%)", list(NODES.keys()),
                help="Reduce transmisión local"
            )
            vaccination_nodes = st.multiselect(
                "Vacunación masiva (30% cobertura)", list(NODES.keys()),
                help="Vacuna al 30% de la población susceptible"
            )

        with st.expander("⏱️ Velocidad"):
            speed    = st.slider("Días por frame", 1, 15, 4)
            max_days = st.slider("Duración máxima (días)", 30, 730, 200)
            delay    = st.slider("Pausa entre frames (ms)", 50, 500, 100)

        st.divider()
        col_a, col_b = st.columns(2)
        start = col_a.button("▶ Iniciar",  type="primary",    use_container_width=True)
        stop  = col_b.button("⏹ Detener", use_container_width=True)
        reset = st.button("🔄 Reiniciar", use_container_width=True)

    return {
        "beta": beta, "gamma": gamma, "delta": delta,
        "origin": origin, "seed": seed,
        "closed_borders": closed_borders,
        "quarantine_nodes": quarantine_nodes,
        "vaccination_nodes": vaccination_nodes,
        "speed": speed, "max_days": max_days, "delay": delay,
        "start": start, "stop": stop, "reset": reset,
    }


# ─────────────────────────────────────────────────────────── PANEL AI
def render_ai_panel(sim: PandemiaSimulator):
    ai   = AIMonitor(sim)
    recs = ai.analyze_global_state()
    st.session_state["last_recs"] = recs

    st.markdown("### 🤖 Panel de IA")

    # Predicción de pico global
    peak_msg = ai.predict_global_peak()
    st.info(peak_msg, icon="📈")

    # Scores de riesgo
    scores = ai.get_risk_scores()
    top3   = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    st.markdown("**🎯 Top riesgo:**")
    for nd, sc in top3:
        bar_w = int(sc)
        color = "#ff4040" if sc > 70 else "#ff8800" if sc > 40 else "#40c870"
        st.markdown(
            f"<div style='margin:3px 0'>"
            f"<span style='color:#c0cce0;font-size:12px'>{nd}</span><br>"
            f"<div style='background:#1e2640;border-radius:4px;height:8px'>"
            f"<div style='background:{color};width:{bar_w}%;height:8px;border-radius:4px'></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Recomendaciones
    if recs:
        st.markdown(f"**📋 Recomendaciones ({len(recs)})**")
        st.markdown("<div class='ai-scroll'>", unsafe_allow_html=True)
        for rec in recs[:7]:
            icon_p = PRIORITY_ICON.get(rec["priority"], "⚪")
            icon_t = POLICY_ICON.get(rec["type"], "📌")
            with st.expander(f"{icon_p} {icon_t} {rec['node']} — {rec['type']}"):
                st.caption(f"**Prioridad:** {rec['priority']}  |  {rec['metric']}")
                st.write(rec["explanation"])
                c1, c2 = st.columns(2)
                if rec["type"] == "CERRAR_FRONTERAS":
                    if c1.button("🚫 Cerrar", key=f"close_{rec['node']}_{rec['type']}"):
                        sim.close_borders(rec["node"])
                        st.toast(f"Fronteras cerradas: {rec['node']}", icon="✅")
                elif rec["type"] == "CUARENTENA_ESTRICTA":
                    if c1.button("🔒 Cuarentena", key=f"quar_{rec['node']}_{rec['type']}"):
                        sim.apply_quarantine(rec["node"], 0.65)
                        st.toast(f"Cuarentena aplicada: {rec['node']}", icon="✅")
                elif rec["type"] == "VACUNACIÓN_MASIVA":
                    if c1.button("💉 Vacunar", key=f"vax_{rec['node']}_{rec['type']}"):
                        sim.apply_vaccination(rec["node"], 0.30)
                        st.toast(f"Vacunación aplicada: {rec['node']}", icon="✅")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Sin alertas críticas activas", icon="✅")


# ─────────────────────────────────────────────────────────── UCI PANEL
def render_icu_panel(sim: PandemiaSimulator):
    state_df = sim.get_state_df().sort_values("icu_load", ascending=False)
    ai = AIMonitor(sim)

    st.markdown("### 🏥 Capacidad UCI")
    for _, r in state_df.iterrows():
        load  = r["icu_load"]
        color = "#ff2020" if load > 1 else "#ff8000" if load > 0.75 else "#40c870"
        icon  = "🔴" if load > 1 else "🟠" if load > 0.75 else "🟢"
        days_c = ai.predict_hospital_collapse(r["node"])

        caption_txt = f"UCI {min(load,1.5)*100:.0f}%"
        if days_c is not None and days_c == 0:
            caption_txt = "⚠️ COLAPSADA"
        elif days_c is not None and days_c < 30:
            caption_txt += f" | Colapso ~{days_c}d"

        st.markdown(
            f"<div style='margin:4px 0'>"
            f"<span style='color:#c0cce0;font-size:11px'>{icon} {r['node']}</span>"
            f"<span style='float:right;color:#888;font-size:10px'>{caption_txt}</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, float(load)))


# ─────────────────────────────────────────────────────────── REPORT
def render_report_section(sim: PandemiaSimulator):
    st.markdown("### 📄 Informe Técnico")
    st.caption("Genera un reporte académico LaTeX con figuras, tablas y análisis de IA.")

    if st.button("📥 Generar Informe LaTeX", use_container_width=True):
        with st.spinner("Generando figuras y compilando LaTeX..."):
            try:
                ai  = AIMonitor(sim)
                gen = ReportGenerator(sim, ai)
                tex_path = gen.save_tex("pandemic_report.tex")

                col1, col2 = st.columns(2)
                with open(tex_path, "r", encoding="utf-8") as f:
                    tex_content = f.read()
                col1.download_button(
                    "⬇️ Descargar .tex",
                    data=tex_content.encode("utf-8"),
                    file_name="pandemic_report.tex",
                    mime="text/plain",
                    use_container_width=True,
                )

                pdf_path = gen.compile_pdf(tex_path)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        col2.download_button(
                            "⬇️ Descargar PDF",
                            data=f.read(),
                            file_name="pandemic_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    col2.info("PDF: instala pdflatex para compilación directa.")

                st.success("✅ Informe generado correctamente.")
            except Exception as e:
                st.error(f"Error al generar informe: {e}")


# ─────────────────────────────────────────────────────────── MAIN APP
def main():
    params = render_sidebar()

    # ── Gestión de estado ─────────────────────────────────────────────────
    need_new_sim = (
        "sim" not in st.session_state
        or params["reset"]
        or params["start"]
    )
    if need_new_sim:
        st.session_state.sim      = PandemiaSimulator(
            beta            = params["beta"],
            gamma           = params["gamma"],
            fatality_rate   = params["delta"],
            origin          = params["origin"],
            seed_infections = params["seed"],
        )
        st.session_state.running  = params["start"]
        st.session_state.last_recs = []

    if params["stop"]:
        st.session_state.running = False

    sim: PandemiaSimulator = st.session_state.sim

    # ── Aplicar intervenciones manuales (sidebar) ─────────────────────────
    for nd in list(NODES.keys()):
        if nd in params["closed_borders"]:
            sim.close_borders(nd)
        elif nd not in params["quarantine_nodes"]:
            sim.open_borders(nd)

        if nd in params["quarantine_nodes"]:
            sim.apply_quarantine(nd, 0.55)
        else:
            sim.lift_quarantine(nd)

    for nd in params["vaccination_nodes"]:
        if sim.vaccination_progress.get(nd, 0.0) < 0.01:
            sim.apply_vaccination(nd, 0.30)

    # ── Avanzar simulación (N pasos) ─────────────────────────────────────
    if st.session_state.running and sim.day < params["max_days"]:
        for _ in range(params["speed"]):
            if sim.day >= params["max_days"]:
                break
            sim.step()

    if sim.day >= params["max_days"] or sim.is_pandemic_over():
        if st.session_state.get("running", False):
            st.session_state.running = False
            st.balloons()

    # ─────────────────────────────────────────────────────── LAYOUT PRINCIPAL
    st.markdown(
        "<h1 style='text-align:center;color:#c0cce0;margin-bottom:0'>🦠 Simulador de Pandemia Global</h1>"
        "<p style='text-align:center;color:#6a7a9a;margin-top:2px'>Modelo SIR · Metapoblaciones · IA Predictiva</p>",
        unsafe_allow_html=True,
    )

    # ── Métricas globales ─────────────────────────────────────────────────
    mc = st.columns(6)
    mc[0].metric("📅 Día",          sim.day)
    mc[1].metric("🤒 Infectados",   fmt(sim.total_infected))
    mc[2].metric("💚 Recuperados",  fmt(sim.total_recovered))
    mc[3].metric("💀 Fallecidos",   fmt(sim.total_deaths))
    mc[4].metric("R₀",              f"{sim.R0:.2f}")
    mc[5].metric("Nodos activos",
                 f"{sum(1 for nd in sim.nodes if sim.I[sim.idx[nd]] > 100)}/{len(sim.nodes)}")

    st.divider()

    # ── Mapa + paneles IA ─────────────────────────────────────────────────
    col_map, col_right = st.columns([3, 1])

    with col_map:
        map_ph = st.empty()
        curve_ph = st.empty()
        sparks_ph = st.empty()

    with col_right:
        ai_ph  = st.empty()
        icu_ph = st.empty()

    # ── Sección inferior ─────────────────────────────────────────────────
    col_bottom_l, col_bottom_r = st.columns([2, 1])
    with col_bottom_l:
        st.subheader("📊 Estado Detallado por Región")
        state_df_display = sim.get_state_df()[[
            "node", "infected", "recovered", "deaths", "infected_pct", "icu_load", "border_closed"
        ]].copy()
        state_df_display["infected_pct"] = (state_df_display["infected_pct"] * 100).round(3).astype(str) + "%"
        state_df_display["icu_load"]     = (state_df_display["icu_load"] * 100).round(1).astype(str) + "%"
        state_df_display.columns = ["Región", "Infectados", "Recuperados", "Fallecidos",
                                     "% Pob.", "Carga UCI", "Frontera cerrada"]
        state_df_display["Infectados"]  = state_df_display["Infectados"].apply(lambda x: f"{int(float(x)):,}")
        state_df_display["Recuperados"] = state_df_display["Recuperados"].apply(lambda x: f"{int(float(x)):,}")
        state_df_display["Fallecidos"]  = state_df_display["Fallecidos"].apply(lambda x: f"{int(float(x)):,}")
        st.dataframe(state_df_display.set_index("Región"), use_container_width=True, height=380)

    with col_bottom_r:
        render_report_section(sim)

    # ── Renderizado de paneles ─────────────────────────────────────────────
    # Mapa
    with map_ph.container():
        if HAS_PYDECK:
            try:
                deck = build_pydeck(sim)
                st.pydeck_chart(deck, use_container_width=True, height=440)
            except Exception as e:
                state_df_map = sim.get_state_df()
                st.map(state_df_map.rename(columns={"lat": "latitude", "lon": "longitude"}),
                       zoom=1)
                st.caption(f"pydeck: {e}")
        else:
            state_df_map = sim.get_state_df()
            st.map(state_df_map.rename(columns={"lat": "latitude", "lon": "longitude"}),
                   zoom=1)
            st.info("Instala `pydeck` para el mapa interactivo completo.")

    # Curva global
    with curve_ph.container():
        buf_curve = plot_global_curve(sim)
        if buf_curve:
            st.image(buf_curve, use_column_width=True)

    # Sparklines por nodo
    with sparks_ph.container():
        buf_sparks = plot_node_sparklines(sim)
        if buf_sparks:
            st.image(buf_sparks, use_column_width=True)

    # Panel IA
    with ai_ph.container():
        render_ai_panel(sim)

    # Panel UCI
    with icu_ph.container():
        render_icu_panel(sim)

    # ── Loop de simulación ────────────────────────────────────────────────
    if st.session_state.running and sim.day < params["max_days"]:
        time.sleep(params["delay"] / 1000)
        st.rerun()


if __name__ == "__main__":
    main()
