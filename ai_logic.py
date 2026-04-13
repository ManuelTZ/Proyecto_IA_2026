# =============================================================================
# ai_logic.py
# Módulo de Inteligencia Artificial: Predictivo, Diagnóstico y Explicable (XAI)
# =============================================================================

import numpy as np
import pandas as pd
from typing import Optional
from simulation_engine import NODES, CONNECTIVITY


# ---------------------------------------------------------------------------
# CLASE PRINCIPAL DEL MONITOR DE IA
# ---------------------------------------------------------------------------
class AIMonitor:
    """
    Sistema de IA para monitoreo y respuesta ante pandemia.

    Componentes:
    1. IA Predictiva: regresión logística / tendencia exponencial para
       predecir colapso hospitalario por nodo.
    2. IA de Diagnóstico: analiza el estado global y emite alertas.
    3. IA de Políticas (XAI): recomienda intervenciones con explicación.
    4. Scoring de riesgo: puntuación 0-100 por nodo.
    """

    POLICY_PRIORITY = {"CRÍTICA": 3, "ALTA": 2, "MEDIA": 1, "BAJA": 0}

    def __init__(self, simulator):
        self.sim = simulator

    # ────────────────────────────────────────────── PREDICCIÓN HOSPITALARIA
    def predict_hospital_collapse(self, node_name: str) -> Optional[int]:
        """
        Predice en cuántos días colapsará el sistema UCI de un nodo.

        Método:
          - Extrae la serie temporal de infectados (últimos 14 días).
          - Ajusta una regresión lineal sobre log(I) → tasa de crecimiento.
          - Extrapola hasta el umbral de capacidad UCI.
          - Retorna días restantes (0 = ya colapsado, None = sin riesgo).
        """
        idx = self.sim.idx.get(node_name)
        if idx is None:
            return None

        history = self.sim.get_history_df()
        col = f"{node_name}_I"
        if col not in history.columns or len(history) < 4:
            return None

        series = history[col].values
        icu_cap = NODES[node_name]["icu_capacity"] * NODES[node_name]["pop"]
        current = float(series[-1])

        # Ya colapsado
        if current >= icu_cap:
            return 0

        # Ventana adaptativa (7-14 días)
        window = min(14, len(series))
        recent = series[-window:]
        recent_pos = recent[recent > 0]

        if len(recent_pos) < 3:
            return None

        # Regresión lineal sobre log(I)
        x = np.arange(len(recent_pos), dtype=float)
        log_y = np.log(recent_pos + 1.0)

        try:
            coeffs = np.polyfit(x, log_y, 1)
            growth_rate = coeffs[0]   # pendiente en espacio logarítmico
        except (np.linalg.LinAlgError, ValueError):
            return None

        if growth_rate <= 1e-6:
            return None   # declinando o plano

        # Días hasta I = icu_cap
        # current · exp(growth_rate · t) = icu_cap
        # t = log(icu_cap / current) / growth_rate
        if current <= 0:
            return None

        days_to_collapse = np.log(icu_cap / (current + 1.0)) / growth_rate
        return max(0, int(np.ceil(days_to_collapse)))

    # ──────────────────────────────────────────── PREDICCIÓN DE PICO GLOBAL
    def predict_global_peak(self) -> str:
        """Estima días restantes hasta el pico global de infecciones."""
        history = self.sim.get_history_df()
        if len(history) < 7:
            return "Datos insuficientes (< 7 días)."

        totals = [
            sum(row.get(f"{nd}_I", 0) for nd in self.sim.nodes)
            for _, row in history.iterrows()
        ]
        recent = np.array(totals[-7:])
        diffs = np.diff(recent)
        avg_daily_growth = float(np.mean(diffs))

        if avg_daily_growth <= 0:
            peak_day = history["day"].iloc[int(np.argmax(totals))]
            return f"✅ Pico global alcanzado en el día {peak_day}. Tendencia decreciente."

        # Umbral de inmunidad de rebaño
        total_pop = float(sum(NODES[nd]["pop"] for nd in self.sim.nodes))
        herd_threshold = 1.0 - self.sim.gamma / self.sim.beta
        s_at_peak = total_pop * (1.0 - herd_threshold)
        current_s = float(self.sim.S.sum())

        if current_s <= s_at_peak:
            return "⚠️ Pico global inminente o ya alcanzado."

        # Extrapolación por log-regresión
        log_recent = np.log(recent + 1.0)
        x = np.arange(len(log_recent), dtype=float)
        try:
            slope = np.polyfit(x, log_recent, 1)[0]
        except Exception:
            slope = 0.0

        if slope <= 0:
            return "✅ Tendencia global decreciente."

        current_total = float(totals[-1])
        needed_ratio = max(1.0, (current_s - s_at_peak) / (current_total + 1.0))
        days_to_peak = int(np.log(needed_ratio) / (slope + 1e-9))
        days_to_peak = max(1, min(days_to_peak, 365))

        return f"📈 Pico global estimado en ~{days_to_peak} días (día {self.sim.day + days_to_peak})."

    # ────────────────────────────────────────── ANÁLISIS GLOBAL Y POLÍTICAS
    def analyze_global_state(self) -> list[dict]:
        """
        Analiza el estado pandémico global y genera recomendaciones de política
        con explicaciones XAI.

        Retorna lista de dicts:
            {type, node, priority, explanation, metric}
        """
        state_df = self.sim.get_state_df()
        recommendations = []

        for _, row in state_df.iterrows():
            node        = row["node"]
            inf_pct     = row["infected_pct"]
            icu_load    = row["icu_load"]
            icu_overflow = row["icu_overflow"]
            node_idx    = self.sim.idx.get(node)

            if node_idx is None:
                continue

            # ── 1. Alerta UCI crítica → Cuarentena Estricta ────────────────
            if icu_overflow:
                recs = self._build_recommendation(
                    rtype    = "CUARENTENA_ESTRICTA",
                    node     = node,
                    priority = "CRÍTICA",
                    metric   = f"Carga UCI: {icu_load*100:.0f}%",
                    explanation = (
                        f"COLAPSO HOSPITALARIO en {node}: la capacidad UCI ha sido "
                        f"superada ({icu_load*100:.0f}%). "
                        f"El modelo predice un incremento de mortalidad del "
                        f"{self.fatality_amplification(icu_load):.0f}% por saturación. "
                        f"Se requiere cuarentena estricta inmediata (reducción β ≥ 60%) "
                        f"para aplanar la curva y liberar camas críticas."
                    )
                )
                recommendations.append(recs)

            # ── 2. Alta conectividad + infectividad → Cerrar fronteras ──────
            connected_healthy = self._get_connected_healthy_nodes(node_idx, node, inf_pct)
            if inf_pct > 0.04 and len(connected_healthy) >= 2 and node not in self.sim.closed_borders:
                healthy_str = ", ".join([n for n, _, _ in connected_healthy[:4]])
                avg_rate = np.mean([r for _, r, _ in connected_healthy[:4]])
                jump_risk = self._compute_jump_risk(node_idx, connected_healthy)

                recs = self._build_recommendation(
                    rtype    = "CERRAR_FRONTERAS",
                    node     = node,
                    priority = "ALTA" if inf_pct < 0.10 else "CRÍTICA",
                    metric   = f"Inf: {inf_pct*100:.1f}% | Nodos en riesgo: {len(connected_healthy)}",
                    explanation = (
                        f"Se recomienda cerrar fronteras en {node} porque el modelo "
                        f"predictivo detectó un salto inminente de la cepa hacia "
                        f"{len(connected_healthy)} nodos sanos con alta conectividad: "
                        f"{healthy_str}. "
                        f"Tasa de movilidad media: {avg_rate*1e4:.1f}×10⁻⁴. "
                        f"Riesgo de salto calculado: {jump_risk:.1%} por día. "
                        f"El cierre reduciría la dispersión internacional en un 96%."
                    )
                )
                recommendations.append(recs)

            # ── 3. Crecimiento acelerado → Vacunación Masiva ────────────────
            if 0.005 < inf_pct < 0.06 and not icu_overflow:
                days_collapse = self.predict_hospital_collapse(node)
                if days_collapse is not None and 0 < days_collapse < 45:
                    vax_coverage = self.sim.vaccination_progress.get(node, 0.0)
                    if vax_coverage < 0.4:
                        recs = self._build_recommendation(
                            rtype    = "VACUNACIÓN_MASIVA",
                            node     = node,
                            priority = "ALTA" if days_collapse < 20 else "MEDIA",
                            metric   = f"Colapso estimado en {days_collapse} días",
                            explanation = (
                                f"El modelo de regresión logarítmica estima colapso UCI en "
                                f"{node} en {days_collapse} días. "
                                f"Cobertura vacunal actual: {vax_coverage*100:.0f}%. "
                                f"Se recomienda campaña de vacunación masiva (objetivo ≥ 40% pop.) "
                                f"para reducir la población susceptible y elevar el umbral de "
                                f"inmunidad de rebaño. Con R₀={self.sim.R0:.1f}, se requiere "
                                f"{(1-1/self.sim.R0)*100:.0f}% de inmunidad poblacional."
                            )
                        )
                        recommendations.append(recs)

            # ── 4. Nodo en incubación temprana → Monitoreo intensivo ────────
            if 0.0002 < inf_pct <= 0.005:
                origin_node = self.sim.origin
                if node != origin_node:
                    recs = self._build_recommendation(
                        rtype    = "MONITOREO_INTENSIVO",
                        node     = node,
                        priority = "BAJA",
                        metric   = f"Inf: {inf_pct*100:.3f}% (fase incipiente)",
                        explanation = (
                            f"Detección temprana de contagios en {node} "
                            f"({inf_pct*100:.3f}% de la población). "
                            f"Con β={self.sim.beta} y γ={self.sim.gamma}, "
                            f"R₀={self.sim.R0:.1f} implica duplicación cada "
                            f"{np.log(2)/max(0.001, self.sim.beta - self.sim.gamma):.1f} días. "
                            f"Se recomienda activar protocolos de rastreo de contactos "
                            f"y cuarentena selectiva para contener el brote."
                        )
                    )
                    recommendations.append(recs)

        # Ordenar por prioridad descendente
        recommendations.sort(
            key=lambda r: self.POLICY_PRIORITY.get(r["priority"], 0),
            reverse=True
        )
        return recommendations

    # ──────────────────────────────────────────────────── RISK SCORING
    def get_risk_scores(self) -> dict[str, float]:
        """Puntuación de riesgo 0-100 por nodo."""
        state_df = self.sim.get_state_df()
        scores = {}
        for _, row in state_df.iterrows():
            node      = row["node"]
            inf_pct   = row["infected_pct"]
            icu_load  = row["icu_load"]
            # Conectividad: suma de tasas de rutas activas
            ni = self.sim.idx.get(node, -1)
            connectivity_score = 0.0
            if ni >= 0:
                connectivity_score = float(self.sim.connectivity_matrix[ni].sum()) * 1e4

            risk = (
                inf_pct * 400
                + icu_load * 35
                + connectivity_score * 2
            )
            scores[node] = min(100.0, risk)
        return scores

    # ──────────────────────────────────────────────────── HELPERS INTERNOS
    def _get_connected_healthy_nodes(
        self, node_idx: int, node_name: str, inf_pct: float
    ) -> list[tuple[str, float, float]]:
        """Retorna lista de (nodo_sano, tasa_movilidad, infectados_pct) conectados a node."""
        result = []
        for j, other in enumerate(self.sim.nodes):
            if other == node_name:
                continue
            rate = float(self.sim.connectivity_matrix[node_idx][j])
            if rate == 0.0:
                continue
            other_pct = self.sim.I[j] / self.sim.N[j] if self.sim.N[j] > 0 else 0
            if other_pct < 0.002 and inf_pct > 0.01:
                result.append((other, rate, other_pct))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _compute_jump_risk(
        self, node_idx: int, healthy_nodes: list[tuple]
    ) -> float:
        """Probabilidad diaria de que al menos un viajero infeccioso llegue a nodos sanos."""
        if not healthy_nodes:
            return 0.0
        total_travelers = sum(
            rate * float(self.sim.I[node_idx]) for _, rate, _ in healthy_nodes
        )
        prob_at_least_one = 1.0 - np.exp(-total_travelers)
        return float(prob_at_least_one)

    @staticmethod
    def fatality_amplification(icu_load: float) -> float:
        """Aumento % en tasa de mortalidad por saturación UCI."""
        if icu_load <= 1.0:
            return 0.0
        return min(300.0, (icu_load - 1.0) * 180.0)

    @staticmethod
    def _build_recommendation(
        rtype: str, node: str, priority: str, explanation: str, metric: str
    ) -> dict:
        return {
            "type":        rtype,
            "node":        node,
            "priority":    priority,
            "explanation": explanation,
            "metric":      metric,
        }

    # ──────────────────────────────────────────────────── MÉTRICAS GLOBALES
    def global_summary(self) -> dict:
        """Resumen ejecutivo del estado global."""
        totals = self.sim.get_global_totals_series()
        infected_series = totals["infected"]
        deaths_series   = totals["deaths"]

        peak_infected = max(infected_series) if infected_series else 0
        peak_day      = int(np.argmax(infected_series)) if infected_series else 0
        fatality_rate = (deaths_series[-1] / (peak_infected + 1)) if peak_infected else 0

        active_nodes     = sum(1 for nd in self.sim.nodes if self.sim.I[self.sim.idx[nd]] > 100)
        collapsed_nodes  = sum(1 for nd in self.sim.nodes
                              if self.sim.I[self.sim.idx[nd]] >
                              NODES[nd]["icu_capacity"] * NODES[nd]["pop"])

        scores = self.get_risk_scores()
        highest_risk = max(scores, key=scores.get) if scores else "N/A"

        return {
            "current_day":       self.sim.day,
            "total_infected":    int(self.sim.total_infected),
            "total_deaths":      int(self.sim.total_deaths),
            "total_recovered":   int(self.sim.total_recovered),
            "peak_infected":     int(peak_infected),
            "peak_day":          peak_day,
            "fatality_rate_pct": fatality_rate * 100,
            "active_nodes":      active_nodes,
            "collapsed_nodes":   collapsed_nodes,
            "highest_risk_node": highest_risk,
            "R0":                self.sim.R0,
            "herd_immunity_pct": (1 - 1 / max(1.0, self.sim.R0)) * 100,
            "global_peak_pred":  self.predict_global_peak(),
        }
