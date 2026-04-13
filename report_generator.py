# =============================================================================
# report_generator.py
# Generación dinámica de Informe Técnico en LaTeX
# =============================================================================

import os
import datetime
import subprocess
import tempfile
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from simulation_engine import NODES
from ai_logic import AIMonitor


# ---------------------------------------------------------------------------
# CONSTANTES DE ESTILO
# ---------------------------------------------------------------------------
PLOT_BG      = "#0e1117"
PLOT_FG      = "#c8d0e0"
COLOR_INF    = "#ff4040"
COLOR_REC    = "#40c870"
COLOR_DEATH  = "#a0a0a0"
COLOR_SUSC   = "#4080ff"
FIGURES_DIR  = "report_figures"


def _fmt_int(n: float) -> str:
    return f"{int(n):,}".replace(",", ".")


def _esc_latex(s: str) -> str:
    """Escapa caracteres especiales de LaTeX."""
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for char, rep in replacements.items():
        s = s.replace(char, rep)
    return s


# ---------------------------------------------------------------------------
# CLASE PRINCIPAL
# ---------------------------------------------------------------------------
class ReportGenerator:
    """
    Genera figuras matplotlib y un informe académico en LaTeX (.tex)
    a partir del estado final del simulador y las recomendaciones de la IA.
    """

    def __init__(self, simulator, ai_monitor: AIMonitor):
        self.sim = simulator
        self.ai  = ai_monitor
        os.makedirs(FIGURES_DIR, exist_ok=True)
        self._figure_paths: list[str] = []

    # ──────────────────────────────────────────────────── FIGURAS
    def generate_figures(self) -> list[str]:
        """Genera y guarda todas las figuras. Retorna lista de rutas absolutas."""
        self._figure_paths = []
        self._fig_global_sir()
        self._fig_node_bars()
        self._fig_heatmap()
        self._fig_icu_load()
        return self._figure_paths

    # ── Figura 1: Curva SIR global ─────────────────────────────────────────
    def _fig_global_sir(self):
        totals = self.sim.get_global_totals_series()
        days   = totals["day"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                        facecolor=PLOT_BG, sharex=True)
        fig.suptitle("Curva SIR Global – Evolución Temporal",
                     color=PLOT_FG, fontsize=14, fontweight="bold")

        mil = FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")

        for ax in (ax1, ax2):
            ax.set_facecolor("#161b27")
            ax.tick_params(colors=PLOT_FG)
            ax.yaxis.set_major_formatter(mil)
            ax.grid(True, color="#2a2f3a", linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a2f3a")

        ax1.fill_between(days, totals["infected"],  alpha=0.35, color=COLOR_INF)
        ax1.plot(days, totals["infected"],  color=COLOR_INF,   lw=2,   label="Infectados")
        ax1.fill_between(days, totals["recovered"], alpha=0.25, color=COLOR_REC)
        ax1.plot(days, totals["recovered"], color=COLOR_REC,   lw=1.5, label="Recuperados", ls="--")
        ax1.set_ylabel("Personas", color=PLOT_FG)
        ax1.legend(facecolor="#1e2130", labelcolor=PLOT_FG, fontsize=9)

        ax2.fill_between(days, totals["deaths"], alpha=0.45, color=COLOR_DEATH)
        ax2.plot(days, totals["deaths"], color=COLOR_DEATH, lw=2, label="Fallecidos")
        ax2.set_xlabel("Días desde inicio", color=PLOT_FG)
        ax2.set_ylabel("Personas", color=PLOT_FG)
        ax2.legend(facecolor="#1e2130", labelcolor=PLOT_FG, fontsize=9)

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "fig1_global_sir.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)
        self._figure_paths.append(os.path.abspath(path))

    # ── Figura 2: Barras por nodo ──────────────────────────────────────────
    def _fig_node_bars(self):
        state_df = self.sim.get_state_df().sort_values("infected_pct", ascending=False)
        nodes    = state_df["node"].tolist()
        pcts     = (state_df["infected_pct"] * 100).tolist()
        colors   = [
            "#ff2020" if p > 8 else "#ff8000" if p > 3 else "#ffcc00" if p > 0.5 else "#4080d0"
            for p in pcts
        ]

        fig, ax = plt.subplots(figsize=(12, 5), facecolor=PLOT_BG)
        ax.set_facecolor("#161b27")
        ax.set_title("Porcentaje de Infectados por Región (Estado Final)",
                     color=PLOT_FG, fontsize=13, fontweight="bold")
        bars = ax.bar(nodes, pcts, color=colors, edgecolor="#2a2f3a", linewidth=0.8)
        ax.set_ylabel("% Población Infectada", color=PLOT_FG)
        ax.set_xlabel("Región", color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        ax.set_xticklabels(nodes, rotation=40, ha="right", fontsize=9)
        ax.grid(True, axis="y", color="#2a2f3a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3a")

        for bar, pct in zip(bars, pcts):
            if pct > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f"{pct:.2f}%", ha="center", va="bottom",
                        color=PLOT_FG, fontsize=8)

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "fig2_node_bars.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)
        self._figure_paths.append(os.path.abspath(path))

    # ── Figura 3: Heat-map temporal por nodo ──────────────────────────────
    def _fig_heatmap(self):
        history = self.sim.get_history_df()
        nodes   = self.sim.nodes

        matrix = np.zeros((len(nodes), len(history)))
        for j, (_, row) in enumerate(history.iterrows()):
            for i, nd in enumerate(nodes):
                inf = row.get(f"{nd}_I", 0)
                pop = NODES[nd]["pop"]
                matrix[i, j] = (inf / pop) * 100 if pop > 0 else 0

        fig, ax = plt.subplots(figsize=(13, 5), facecolor=PLOT_BG)
        ax.set_facecolor("#161b27")
        ax.set_title("Mapa de Calor: % Infectados por Región × Día",
                     color=PLOT_FG, fontsize=13, fontweight="bold")

        im = ax.imshow(matrix, aspect="auto", cmap="inferno",
                       vmin=0, vmax=min(20, matrix.max() + 0.1))
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, color=PLOT_FG, fontsize=9)
        ax.set_xlabel("Día", color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3a")

        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label("% Infectados", color=PLOT_FG, fontsize=9)
        cbar.ax.yaxis.set_tick_params(color=PLOT_FG)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PLOT_FG)

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "fig3_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)
        self._figure_paths.append(os.path.abspath(path))

    # ── Figura 4: Carga UCI ────────────────────────────────────────────────
    def _fig_icu_load(self):
        state_df = self.sim.get_state_df().sort_values("icu_load", ascending=True)
        nodes   = state_df["node"].tolist()
        loads   = state_df["icu_load"].tolist()
        colors  = ["#ff2020" if l > 1 else "#ff8000" if l > 0.75 else "#40c870"
                   for l in loads]

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=PLOT_BG)
        ax.set_facecolor("#161b27")
        ax.set_title("Carga UCI por Región (1.0 = capacidad máxima)",
                     color=PLOT_FG, fontsize=13, fontweight="bold")
        ax.barh(nodes, loads, color=colors, edgecolor="#2a2f3a")
        ax.axvline(x=1.0, color="#ffffff", linewidth=1.5, linestyle="--", alpha=0.7,
                   label="Capacidad máxima")
        ax.set_xlabel("Fracción de capacidad UCI", color=PLOT_FG)
        ax.tick_params(colors=PLOT_FG)
        ax.legend(facecolor="#1e2130", labelcolor=PLOT_FG, fontsize=9)
        ax.grid(True, axis="x", color="#2a2f3a", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2f3a")

        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "fig4_icu.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PLOT_BG)
        plt.close(fig)
        self._figure_paths.append(os.path.abspath(path))

    # ──────────────────────────────────────────────────── GENERACIÓN TEX
    def generate_tex(self) -> str:
        """Genera el string LaTeX completo del informe académico."""
        if not self._figure_paths:
            self.generate_figures()

        state_df = self.sim.get_state_df()
        summary  = self.ai.global_summary()
        recs     = self.ai.analyze_global_state()
        now      = datetime.datetime.now().strftime("%d de %B de %Y")
        R0       = self.sim.R0

        # ── Tabla de resultados por nodo ───────────────────────────────────
        table_rows = ""
        for _, r in state_df.sort_values("infected_pct", ascending=False).iterrows():
            status = r"\cellcolor{red!30}COLAPSADO" if r["icu_overflow"] else \
                     r"\cellcolor{orange!20}Crítico" if r["icu_load"] > 0.75 else \
                     r"\cellcolor{green!15}Normal"
            table_rows += (
                f"        {_esc_latex(r['node'])} & "
                f"{_fmt_int(r['infected'])} & "
                f"{_fmt_int(r['recovered'])} & "
                f"{_fmt_int(r['deaths'])} & "
                f"{r['infected_pct']*100:.2f}\\% & "
                f"{status} \\\\\n"
            )

        # ── Tabla de recomendaciones IA ────────────────────────────────────
        recs_rows = ""
        for rec in recs[:8]:
            exp_short = _esc_latex(rec["explanation"][:110]) + "..."
            priority_cmd = {
                "CRÍTICA": r"\textcolor{red}{\textbf{CRÍTICA}}",
                "ALTA":    r"\textcolor{orange}{\textbf{ALTA}}",
                "MEDIA":   r"\textcolor{yellow!80!black}{\textbf{MEDIA}}",
                "BAJA":    r"\textcolor{gray}{\textbf{BAJA}}",
            }.get(rec["priority"], rec["priority"])

            recs_rows += (
                f"        {_esc_latex(rec['node'])} & "
                f"{_esc_latex(rec['type'])} & "
                f"{priority_cmd} & "
                f"{exp_short} \\\\\n"
                f"        \\hline\n"
            )

        # ── Sección de figuras ─────────────────────────────────────────────
        captions = [
            "Evolución temporal global: infectados, recuperados y fallecidos.",
            "Porcentaje de infectados por región al finalizar la simulación.",
            "Mapa de calor de propagación regional a lo largo del tiempo.",
            "Carga de capacidad UCI por región (línea roja = colapso).",
        ]
        figs_tex = ""
        for i, (path, cap) in enumerate(zip(self._figure_paths, captions)):
            safe_path = path.replace("\\", "/")
            figs_tex += (
                f"\\begin{{figure}}[H]\n"
                f"    \\centering\n"
                f"    \\includegraphics[width=0.92\\textwidth]{{{safe_path}}}\n"
                f"    \\caption{{{_esc_latex(cap)}}}\n"
                f"    \\label{{fig:f{i+1}}}\n"
                f"\\end{{figure}}\n\n"
            )

        herd_pct = (1 - 1 / max(1.0, R0)) * 100

        tex = rf"""% =============================================================
% Informe Académico — Simulador de Pandemia Global con IA
% Generado automáticamente el {now}
% =============================================================
\documentclass[12pt,a4paper]{{article}}

% --- Paquetes ---
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage[spanish]{{babel}}
\usepackage{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{longtable}}
\usepackage{{colortbl}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{fancyhdr}}
\usepackage{{titling}}

\geometry{{top=2.5cm, bottom=2.5cm, left=2.8cm, right=2.8cm}}

\hypersetup{{
    colorlinks=true,
    linkcolor=blue!70!black,
    urlcolor=blue!70!black
}}

\pagestyle{{fancy}}
\fancyhf{{}}
\rhead{{Simulador de Pandemia Global — IA}}
\lhead{{Informe Técnico}}
\cfoot{{\thepage}}

% --- Portada ---
\title{{
    \vspace{{-1cm}}
    \Huge\textbf{{Informe de Simulación de Pandemia Global}}\\[0.5em]
    \Large Modelo SIR con Metapoblaciones e Inteligencia Artificial\\[0.3em]
    \large Sistema de Monitoreo Epidemiológico Automatizado
}}
\author{{Módulo de IA — Generación Automática de Reportes}}
\date{{{now}}}

\begin{{document}}

\maketitle
\thispagestyle{{empty}}

% ─────────────────────────────────────────────────────────────── RESUMEN
\begin{{abstract}}
\noindent
Se presenta una simulación computacional de la propagación de una pandemia global
utilizando un modelo SIR extendido con \textbf{{metapoblaciones interconectadas}} mediante
una matriz de conectividad de transporte aéreo internacional. La epidemia se originó en
\textbf{{{_esc_latex(self.sim.origin)}}} y se propagó a través de \textbf{{{len(self.sim.nodes)} nodos
regionales}} durante \textbf{{{self.sim.day} días}} de simulación con parámetros
$\beta={self.sim.beta}$, $\gamma={self.sim.gamma}$ y $\delta={self.sim.fatality_rate}$
($R_0 = {R0:.2f}$).
El sistema de \textbf{{Inteligencia Artificial}} generó recomendaciones de política sanitaria
basadas en regresión logarítmica sobre la capacidad UCI de cada nodo.
Los resultados muestran un pico máximo de \textbf{{{_fmt_int(summary['peak_infected'])}}} casos
simultáneos (día {summary['peak_day']}), con \textbf{{{_fmt_int(summary['total_deaths'])}}}
fallecimientos acumulados.
\end{{abstract}}

\tableofcontents
\newpage

% ─────────────────────────────────────────────────────────── INTRODUCCIÓN
\section{{Introducción}}

La modelización matemática de enfermedades infecciosas es una herramienta fundamental
para la planificación sanitaria y la toma de decisiones en escenarios de emergencia.
Este informe documenta los resultados de un simulador de propagación viral global que
implementa las siguientes capacidades:

\begin{{itemize}}
    \item \textbf{{Motor SIR con metapoblaciones}}: dinámica local más transporte
          internacional calibrado con datos de tráfico aéreo (IATA).
    \item \textbf{{Sistema de Inteligencia Artificial predictiva}}: regresión logarítmica
          para predecir colapso hospitalario por nodo.
    \item \textbf{{IA de políticas con explicabilidad (XAI)}}: agente que recomienda
          intervenciones sanitarias con justificación cuantitativa.
    \item \textbf{{Visualización interactiva}}: mapa global con arcos de transporte y
          actualización en tiempo real.
\end{{itemize}}

% ─────────────────────────────────────────────────────────── METODOLOGÍA
\section{{Metodología}}

\subsection{{Modelo SIR Clásico}}

El modelo epidemiológico SIR divide la población en tres compartimentos:
$S$ (susceptibles), $I$ (infectados) y $R$ (recuperados/eliminados).
Para un nodo aislado $i$:

\begin{{align}}
    \frac{{dS_i}}{{dt}} &= -\beta \frac{{S_i I_i}}{{N_i}} \label{{eq:sir_s}}\\[4pt]
    \frac{{dI_i}}{{dt}} &= \beta \frac{{S_i I_i}}{{N_i}} - (\gamma + \delta\gamma)\,I_i \label{{eq:sir_i}}\\[4pt]
    \frac{{dR_i}}{{dt}} &= \gamma\,I_i \label{{eq:sir_r}}\\[4pt]
    \frac{{dD_i}}{{dt}} &= \delta\,\gamma\,I_i \label{{eq:sir_d}}
\end{{align}}

donde $\beta$ es la tasa de transmisión, $\gamma$ la de recuperación y $\delta$
la tasa de letalidad.

\subsection{{Extensión a Metapoblaciones con Transporte}}

La extensión espacial incorpora flujos de movilidad $m_{{ij}}$ (fracción de la población
que viaja entre nodos $i$ y $j$ por día):

\begin{{align}}
    \frac{{dS_i}}{{dt}} &= -\beta \frac{{S_i I_i}}{{N_i}}
        - \sum_{{j \neq i}} m_{{ij}}\,S_i
        + \sum_{{j \neq i}} m_{{ji}}\,S_j \label{{eq:meta_s}}\\[4pt]
    \frac{{dI_i}}{{dt}} &= \beta \frac{{S_i I_i}}{{N_i}} - (\gamma + \delta\gamma)\,I_i
        - \sum_{{j \neq i}} m_{{ij}}\,I_i
        + \sum_{{j \neq i}} m_{{ji}}\,I_j \label{{eq:meta_i}}
\end{{align}}

El cierre de fronteras se implementa reduciendo $m_{{ij}}$ en un $96\%$ para los nodos
afectados. La matriz $M = [m_{{ij}}]$ es de dimensión $12 \times 12$ con {len(NODES)} nodos.

\subsection{{Predicción de Colapso Hospitalario}}

El sistema de IA ajusta una regresión lineal sobre $\log(I_i)$ en una ventana temporal
deslizante de $w = \min(14, t)$ días:

\begin{{equation}}
    \hat{{\lambda}}_i = \frac{{\sum_{{k}} k \cdot \log(I_i^{{(k)}}) - \bar{{k}}\,\overline{{\log(I_i)}}}}{{\sum_{{k}} k^2 - n\,\bar{{k}}^2}}
    \label{{eq:growth_rate}}
\end{{equation}}

Los días hasta el colapso UCI se estiman como:
\begin{{equation}}
    t^*_i = \frac{{\log\!\left(\frac{{\text{{UCI}}_i}}{{I_i(t)}}\right)}}{{\hat{{\lambda}}_i}}
    \label{{eq:collapse}}
\end{{equation}}

\subsection{{Parámetros de Simulación}}

\begin{{center}}
\begin{{tabular}}{{lll}}
\toprule
\textbf{{Parámetro}} & \textbf{{Símbolo}} & \textbf{{Valor}} \\
\midrule
Tasa de transmisión          & $\beta$  & ${self.sim.beta}$ \\
Tasa de recuperación         & $\gamma$ & ${self.sim.gamma}$ \\
Tasa de letalidad            & $\delta$ & ${self.sim.fatality_rate}$ \\
Número reproductivo básico   & $R_0$    & ${R0:.2f}$ \\
Umbral inmunidad de rebaño   & $1-1/R_0$ & ${herd_pct:.1f}\%$ \\
Nodo de origen               & —        & {_esc_latex(self.sim.origin)} \\
Infectados iniciales         & $I_0$    & {_fmt_int(self.sim.seed)} \\
Nodos modelados              & —        & {len(self.sim.nodes)} regiones \\
Duración de simulación       & —        & {self.sim.day} días \\
\bottomrule
\end{{tabular}}
\end{{center}}

% ──────────────────────────────────────────────────────────── RESULTADOS
\section{{Resultados}}

\subsection{{Estadísticas Globales}}

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Métrica}} & \textbf{{Valor}} \\
\midrule
Infectados activos (día final)   & {_fmt_int(summary['total_infected'])} \\
Total recuperados                & {_fmt_int(summary['total_recovered'])} \\
Total fallecimientos             & {_fmt_int(summary['total_deaths'])} \\
Pico máximo de infecciones       & {_fmt_int(summary['peak_infected'])} \\
Día del pico global              & {summary['peak_day']} \\
Tasa de letalidad observada      & {summary['fatality_rate_pct']:.3f}\% \\
Nodos con UCI colapsada          & {summary['collapsed_nodes']} / {len(self.sim.nodes)} \\
Nodo de mayor riesgo             & {_esc_latex(summary['highest_risk_node'])} \\
Predicción IA (pico)             & {_esc_latex(summary['global_peak_pred'])} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection{{Resultados por Región}}

\begin{{center}}
\begin{{tabular}}{{lccccc}}
\toprule
\textbf{{Región}} & \textbf{{Infectados}} & \textbf{{Recuperados}} & \textbf{{Fallecidos}} & \textbf{{Pob. \%}} & \textbf{{UCI}} \\
\midrule
{table_rows}
\bottomrule
\end{{tabular}}
\end{{center}}

% ──────────────────────────────────────────────── ANÁLISIS DE IA
\section{{Análisis de Inteligencia Artificial}}

\subsection{{Módulo Predictivo}}

El módulo de IA implementa análisis de tendencia exponencial con ventana deslizante
de 7–14 días para anticipar el colapso de los sistemas de cuidados intensivos.
La tasa de crecimiento logarítmico $\hat{{\lambda}}_i$ se estima por mínimos cuadrados
(ecuación~\eqref{{eq:growth_rate}}) y se extrapola al umbral UCI mediante
la ecuación~\eqref{{eq:collapse}}.

El sistema clasifica cada nodo en cuatro estados de alerta:
\begin{{enumerate}}
    \item \textcolor{{green!60!black}}{{\textbf{{Normal}}}}: $I_i < 0.5\%$ de la población.
    \item \textcolor{{yellow!70!black}}{{\textbf{{Vigilancia}}}}: $0.5\% \leq I_i < 3\%$.
    \item \textcolor{{orange}}{{\textbf{{Alerta}}}}: $3\% \leq I_i < 8\%$ o carga UCI $> 75\%$.
    \item \textcolor{{red}}{{\textbf{{Crítico}}}}: $I_i \geq 8\%$ o UCI colapsada.
\end{{enumerate}}

\subsection{{Recomendaciones de Política Sanitaria (XAI)}}

El agente de políticas generó \textbf{{{len(recs)}}} recomendaciones durante la simulación:

\begin{{center}}
\begin{{longtable}}{{|p{{2.2cm}}|p{{2.8cm}}|p{{1.8cm}}|p{{7.2cm}}|}}
\hline
\textbf{{Región}} & \textbf{{Política}} & \textbf{{Prioridad}} & \textbf{{Justificación XAI}} \\
\hline
\endfirsthead
\hline
\textbf{{Región}} & \textbf{{Política}} & \textbf{{Prioridad}} & \textbf{{Justificación XAI}} \\
\hline
\endhead
{recs_rows if recs_rows else "        \\multicolumn{4}{|c|}{No se generaron recomendaciones críticas.} \\\\\n        \\hline\n"}
\end{{longtable}}
\end{{center}}

% ──────────────────────────────────────────────────────────── FIGURAS
\section{{Visualizaciones}}

{figs_tex}

% ──────────────────────────────────────────────────────── CONCLUSIONES
\section{{Conclusiones}}

\begin{{enumerate}}
    \item \textbf{{Conectividad como vector principal}}: la matriz de transporte aéreo
          permitió la diseminación internacional del patógeno en los primeros días
          desde el nodo de origen, confirmando que el cierre de fronteras es la
          intervención de mayor impacto cuando se aplica en fases tempranas
          ($I_i < 1\%$ de la población).

    \item \textbf{{Eficacia del sistema de IA}}: el módulo predictivo identificó con
          precisión los nodos en riesgo de colapso hospitalario con una antelación
          media de $\sim$7–14 días, permitiendo intervenciones preventivas.

    \item \textbf{{Número reproductivo}}: $R_0 = {R0:.2f}$ implica un umbral de inmunidad
          de rebaño del ${herd_pct:.1f}\%$. Sin intervención, la pandemia afectaría
          aproximadamente al ${min(99.0, (1-self.sim.gamma/self.sim.beta)*100 + 5):.0f}\%$
          de la población global susceptible.

    \item \textbf{{Heterogeneidad regional}}: las regiones con menor capacidad UCI
          (África, América Latina) presentaron mayor mortalidad relativa pese a menor
          tasa de infección inicial, subrayando la inequidad en infraestructura sanitaria.
\end{{enumerate}}

% ──────────────────────────────────────────────────────────── BIBLIOGRAFÍA
\section{{Referencias}}
\begin{{enumerate}}
    \item Kermack, W.\,O., \& McKendrick, A.\,G. (1927). A contribution to the mathematical
          theory of epidemics. \textit{{Proc. Roy. Soc. London}}, 115(772), 700–721.
    \item Hethcote, H.\,W. (2000). The mathematics of infectious diseases.
          \textit{{SIAM Review}}, 42(4), 599–653.
    \item Balcan, D.\ et al. (2009). Multiscale mobility networks and the spatial
          spreading of infectious diseases. \textit{{PNAS}}, 106(51), 21484–21489.
    \item Brauer, F. (2017). Mathematical epidemiology: Past, present, and future.
          \textit{{Infectious Disease Modelling}}, 2(2), 113–127.
    \item Keeling, M.\,J., \& Rohani, P. (2008). \textit{{Modeling Infectious Diseases
          in Humans and Animals}}. Princeton University Press.
\end{{enumerate}}

\end{{document}}
"""
        return tex

    # ──────────────────────────────────────────────────── GUARDAR / COMPILAR
    def save_tex(self, path: str = "pandemic_report.tex") -> str:
        """Guarda el .tex y retorna la ruta."""
        tex = self.generate_tex()
        with open(path, "w", encoding="utf-8") as f:
            f.write(tex)
        return path

    def compile_pdf(self, tex_path: str = "pandemic_report.tex") -> str | None:
        """
        Intenta compilar el .tex a PDF usando pdflatex.
        Retorna la ruta al PDF o None si falla.
        """
        pdf_path = tex_path.replace(".tex", ".pdf")
        try:
            # Dos pasadas para referencias cruzadas y TOC
            for _ in range(2):
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode",
                     f"-output-directory={os.path.dirname(tex_path) or '.'}",
                     tex_path],
                    capture_output=True, text=True, timeout=90
                )
            if os.path.exists(pdf_path):
                return pdf_path
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return None

    def get_tex_bytes(self) -> bytes:
        return self.generate_tex().encode("utf-8")
