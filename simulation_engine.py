# =============================================================================
# simulation_engine.py
# Motor de Simulación SIR con Metapoblaciones y Sistema de Transporte Global
# =============================================================================

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# DEFINICIÓN DE NODOS (Países/Regiones) con poblaciones reales (2023)
# ---------------------------------------------------------------------------
NODES = {
    "China":         {"pop": 1_412_600_000, "lat": 35.86,  "lon": 104.19,  "icu_capacity": 0.00042, "gdp_index": 0.72},
    "USA":           {"pop":   331_900_000, "lat": 37.09,  "lon": -95.71,  "icu_capacity": 0.00290, "gdp_index": 1.00},
    "India":         {"pop": 1_380_000_000, "lat": 20.59,  "lon":  78.96,  "icu_capacity": 0.00023, "gdp_index": 0.30},
    "Brazil":        {"pop":   215_300_000, "lat": -14.24, "lon": -51.93,  "icu_capacity": 0.00021, "gdp_index": 0.45},
    "Germany":       {"pop":    83_200_000, "lat": 51.16,  "lon":  10.45,  "icu_capacity": 0.00290, "gdp_index": 0.92},
    "Nigeria":       {"pop":   218_500_000, "lat":   9.08, "lon":   8.67,  "icu_capacity": 0.00005, "gdp_index": 0.18},
    "Russia":        {"pop":   144_700_000, "lat": 61.52,  "lon": 105.31,  "icu_capacity": 0.00082, "gdp_index": 0.55},
    "Japan":         {"pop":   125_700_000, "lat": 36.20,  "lon": 138.25,  "icu_capacity": 0.00130, "gdp_index": 0.88},
    "Australia":     {"pop":    25_900_000, "lat": -25.27, "lon": 133.77,  "icu_capacity": 0.00220, "gdp_index": 0.94},
    "Mexico":        {"pop":   130_200_000, "lat": 23.63,  "lon": -102.55, "icu_capacity": 0.00015, "gdp_index": 0.42},
    "South Africa":  {"pop":    60_100_000, "lat": -30.56, "lon":  22.94,  "icu_capacity": 0.00009, "gdp_index": 0.28},
    "UK":            {"pop":    67_200_000, "lat": 55.37,  "lon":  -3.43,  "icu_capacity": 0.00240, "gdp_index": 0.90},
}

# ---------------------------------------------------------------------------
# MATRIZ DE CONECTIVIDAD (rutas aéreas internacionales)
# Valor = fracción de la población que viaja entre nodos por día
# Basado en datos de tráfico aéreo IATA 2019
# ---------------------------------------------------------------------------
CONNECTIVITY = {
    # Asia - América
    ("China",        "USA"):           0.000280,
    ("China",        "Japan"):         0.000820,
    ("China",        "Australia"):     0.000190,
    ("China",        "India"):         0.000110,
    ("China",        "Russia"):        0.000200,
    ("Japan",        "USA"):           0.000310,
    ("Japan",        "Australia"):     0.000140,
    ("Japan",        "India"):         0.000080,
    # Europa
    ("Germany",      "UK"):            0.000520,
    ("Germany",      "Russia"):        0.000290,
    ("Germany",      "USA"):           0.000410,
    ("Germany",      "China"):         0.000180,
    ("Germany",      "Nigeria"):       0.000095,
    ("UK",           "USA"):           0.000650,
    ("UK",           "India"):         0.000220,
    ("UK",           "South Africa"):  0.000175,
    ("UK",           "Nigeria"):       0.000195,
    ("UK",           "Australia"):     0.000130,
    # Américas
    ("USA",          "Brazil"):        0.000290,
    ("USA",          "Mexico"):        0.001050,
    ("USA",          "Australia"):     0.000195,
    ("Brazil",       "Mexico"):        0.000185,
    ("Brazil",       "South Africa"):  0.000090,
    # África
    ("Nigeria",      "South Africa"):  0.000110,
    # Rutas adicionales
    ("India",        "USA"):           0.000310,
    ("Russia",       "China"):         0.000200,
    ("Mexico",       "Brazil"):        0.000155,
    ("Australia",    "South Africa"):  0.000055,
    ("Germany",      "Brazil"):        0.000120,
    ("India",        "Japan"):         0.000075,
    ("China",        "Germany"):       0.000200,
}


# ---------------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ---------------------------------------------------------------------------
def build_connectivity_matrix():
    """Construye la matriz N×N de conectividad entre nodos."""
    node_names = list(NODES.keys())
    n = len(node_names)
    idx_map = {name: i for i, name in enumerate(node_names)}
    matrix = np.zeros((n, n), dtype=float)

    for (a, b), rate in CONNECTIVITY.items():
        if a in idx_map and b in idx_map:
            ia, ib = idx_map[a], idx_map[b]
            matrix[ia][ib] = rate
            matrix[ib][ia] = rate   # bidireccional

    return matrix, node_names, idx_map


def infection_color(pct: float) -> list:
    """Devuelve [R, G, B, A] según el porcentaje de infectados."""
    if pct > 0.15:
        return [180, 0,   0,   240]
    elif pct > 0.08:
        return [230, 20,  20,  220]
    elif pct > 0.03:
        return [255, 80,  0,   200]
    elif pct > 0.01:
        return [255, 160, 0,   180]
    elif pct > 0.002:
        return [255, 220, 50,  160]
    elif pct > 0.0001:
        return [100, 210, 100, 140]
    else:
        return [30,  90,  220, 110]


# ---------------------------------------------------------------------------
# CLASE PRINCIPAL DEL SIMULADOR
# ---------------------------------------------------------------------------
class PandemiaSimulator:
    """
    Simulador SIR con metapoblaciones.

    Compartimentos por nodo i:
        S_i  — Susceptibles
        I_i  — Infectados
        R_i  — Recuperados
        D_i  — Fallecidos

    Ecuaciones diferenciales (discretizadas, Euler forward, Δt=1 día):

        ΔS_i = −β · S_i · I_i / N_i  −  Σ_j m_ij · S_i
        ΔI_i =  β · S_i · I_i / N_i  − (γ + δγ) · I_i  +  Σ_j m_ji · I_j  −  Σ_j m_ij · I_i
        ΔR_i =  γ · I_i
        ΔD_i =  δ · γ · I_i
    """

    def __init__(
        self,
        beta: float = 0.30,
        gamma: float = 0.05,
        fatality_rate: float = 0.010,
        origin: str = "China",
        seed_infections: int = 1_000,
    ):
        self.beta = beta
        self.gamma = gamma
        self.fatality_rate = fatality_rate
        self.origin = origin
        self.seed = seed_infections

        self.connectivity_matrix, self.nodes, self.idx = build_connectivity_matrix()
        self.n = len(self.nodes)

        # Poblaciones
        self.N = np.array([NODES[nd]["pop"] for nd in self.nodes], dtype=float)

        # Estado inicial
        self.S = self.N.copy()
        self.I = np.zeros(self.n, dtype=float)
        self.R = np.zeros(self.n, dtype=float)
        self.D = np.zeros(self.n, dtype=float)

        # Semilla de infección
        origin_idx = self.idx.get(origin, 0)
        seed_val = min(seed_infections, self.S[origin_idx])
        self.I[origin_idx] = seed_val
        self.S[origin_idx] -= seed_val

        # Control de políticas (por nodo)
        self.closed_borders: set = set()          # nodos con fronteras cerradas
        self.quarantine_factor: dict = {}          # reducción local de β (0-1)
        self.vaccination_progress: dict = {}       # fracción vacunada por nodo

        self.day = 0
        self.history: list[dict] = []
        self._record()   # día 0

    # ------------------------------------------------------------------ STEP
    def step(self):
        """Avanza la simulación un día (Δt = 1)."""
        S, I, R, D = self.S.copy(), self.I.copy(), self.R.copy(), self.D.copy()

        dS = np.zeros(self.n)
        dI = np.zeros(self.n)
        dR = np.zeros(self.n)
        dD = np.zeros(self.n)

        # ── Dinámica local SIR ─────────────────────────────────────────────
        for i in range(self.n):
            ni = self.N[i]
            if ni <= 0:
                continue

            # Factor de cuarentena local
            q = self.quarantine_factor.get(self.nodes[i], 1.0)
            beta_eff = self.beta * q

            # Tasa de vacunación (reduce susceptibles)
            vax = self.vaccination_progress.get(self.nodes[i], 0.0)
            effective_S = S[i] * (1.0 - vax)

            new_infections = beta_eff * effective_S * I[i] / ni
            new_recoveries = self.gamma * I[i]
            new_deaths     = self.fatality_rate * self.gamma * I[i]

            dS[i] -= new_infections
            dI[i] += new_infections - new_recoveries - new_deaths
            dR[i] += new_recoveries
            dD[i] += new_deaths

        # ── Transporte entre nodos ─────────────────────────────────────────
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue

                base_rate = self.connectivity_matrix[i][j]
                if base_rate == 0.0:
                    continue

                # Reducir movilidad si algún nodo tiene fronteras cerradas
                border_factor = 1.0
                if self.nodes[i] in self.closed_borders:
                    border_factor *= 0.04   # 96 % reducción emisor
                if self.nodes[j] in self.closed_borders:
                    border_factor *= 0.04   # 96 % reducción receptor

                rate = base_rate * border_factor

                # Viajeros infecciosos de i → j
                travelers_I = rate * I[i]
                travelers_I = min(travelers_I, I[i] * 0.5)

                # Viajeros susceptibles de i → j (efecto menor)
                travelers_S = rate * 0.15 * S[i]
                travelers_S = min(travelers_S, S[i] * 0.05)

                dI[i] -= travelers_I
                dI[j] += travelers_I
                dS[i] -= travelers_S
                dS[j] += travelers_S

        # ── Aplicar deltas ─────────────────────────────────────────────────
        self.S = np.maximum(self.S + dS, 0.0)
        self.I = np.maximum(self.I + dI, 0.0)
        self.R = np.maximum(self.R + dR, 0.0)
        self.D = np.maximum(self.D + dD, 0.0)

        # Normalizar (S+I+R no puede superar N)
        total = self.S + self.I + self.R
        mask = total > self.N
        if mask.any():
            factor = self.N[mask] / total[mask]
            self.S[mask] *= factor
            self.I[mask] *= factor
            self.R[mask] *= factor

        self.day += 1
        self._record()

    # ─────────────────────────────────────────────────────────── POLÍTICAS
    def close_borders(self, node: str):
        self.closed_borders.add(node)

    def open_borders(self, node: str):
        self.closed_borders.discard(node)

    def apply_quarantine(self, node: str, reduction: float = 0.5):
        """Reduce la β local en `reduction` fracción (0 = sin efecto, 1 = β=0)."""
        self.quarantine_factor[node] = 1.0 - min(max(reduction, 0.0), 1.0)

    def lift_quarantine(self, node: str):
        self.quarantine_factor.pop(node, None)

    def apply_vaccination(self, node: str, coverage: float = 0.3):
        """Marca `coverage` fracción de la población como vacunada (inmune)."""
        prev = self.vaccination_progress.get(node, 0.0)
        self.vaccination_progress[node] = min(1.0, prev + coverage)
        # Inmediatamente mover susceptibles a recuperados
        newly_immune = self.S[self.idx[node]] * coverage
        self.S[self.idx[node]] -= newly_immune
        self.R[self.idx[node]] += newly_immune
        self.S = np.maximum(self.S, 0.0)

    # ─────────────────────────────────────────────────────────── REGISTROS
    def _record(self):
        row = {"day": self.day}
        for i, nd in enumerate(self.nodes):
            row[f"{nd}_S"] = float(self.S[i])
            row[f"{nd}_I"] = float(self.I[i])
            row[f"{nd}_R"] = float(self.R[i])
            row[f"{nd}_D"] = float(self.D[i])
        self.history.append(row)

    # ──────────────────────────────────────────────────────────── GETTERS
    def get_state_df(self) -> pd.DataFrame:
        rows = []
        for i, nd in enumerate(self.nodes):
            pop = float(self.N[i])
            inf = float(self.I[i])
            inf_pct = inf / pop if pop > 0 else 0.0
            icu_cap = NODES[nd]["icu_capacity"] * pop
            rows.append({
                "node":          nd,
                "lat":           NODES[nd]["lat"],
                "lon":           NODES[nd]["lon"],
                "population":    pop,
                "susceptible":   float(self.S[i]),
                "infected":      inf,
                "recovered":     float(self.R[i]),
                "deaths":        float(self.D[i]),
                "infected_pct":  inf_pct,
                "icu_capacity":  icu_cap,
                "icu_load":      min(1.5, inf / (icu_cap + 1)),
                "icu_overflow":  inf > icu_cap,
                "border_closed": nd in self.closed_borders,
                "color":         infection_color(inf_pct),
            })
        return pd.DataFrame(rows)

    def get_arc_data(self) -> pd.DataFrame:
        arcs = []
        for (a, b), rate in CONNECTIVITY.items():
            if a not in self.idx or b not in self.idx:
                continue
            ia, ib = self.idx[a], self.idx[b]
            pct_a = self.I[ia] / self.N[ia] if self.N[ia] > 0 else 0
            pct_b = self.I[ib] / self.N[ib] if self.N[ib] > 0 else 0
            max_pct = max(pct_a, pct_b)

            # Color de la ruta según carga infecciosa
            if max_pct > 0.05:
                color_s = [255, 30,  30,  200]
                color_t = [255, 30,  30,  200]
            elif max_pct > 0.005:
                color_s = [255, 140, 0,   170]
                color_t = [255, 140, 0,   170]
            else:
                color_s = [40,  100, 255, 90]
                color_t = [40,  100, 255, 90]

            border_reduced = (a in self.closed_borders or b in self.closed_borders)
            if border_reduced:
                color_s = [100, 100, 100, 60]
                color_t = [100, 100, 100, 60]

            arcs.append({
                "source_lon":    NODES[a]["lon"],
                "source_lat":    NODES[a]["lat"],
                "target_lon":    NODES[b]["lon"],
                "target_lat":    NODES[b]["lat"],
                "color_source":  color_s,
                "color_target":  color_t,
                "source":        a,
                "target":        b,
                "border_closed": border_reduced,
                "base_rate":     rate,
            })
        return pd.DataFrame(arcs)

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    # ─────────────────────────────────────────────────────── ESTADÍSTICAS
    @property
    def total_infected(self) -> float:
        return float(self.I.sum())

    @property
    def total_deaths(self) -> float:
        return float(self.D.sum())

    @property
    def total_recovered(self) -> float:
        return float(self.R.sum())

    @property
    def R0(self) -> float:
        return self.beta / self.gamma

    def is_pandemic_over(self) -> bool:
        return self.I.sum() < 50 and self.day > 15

    def get_global_totals_series(self) -> dict:
        """Devuelve series temporales agregadas globales."""
        hist = self.get_history_df()
        infected  = [sum(r.get(f"{nd}_I", 0) for nd in self.nodes) for _, r in hist.iterrows()]
        recovered = [sum(r.get(f"{nd}_R", 0) for nd in self.nodes) for _, r in hist.iterrows()]
        deaths    = [sum(r.get(f"{nd}_D", 0) for nd in self.nodes) for _, r in hist.iterrows()]
        suscept   = [sum(r.get(f"{nd}_S", 0) for nd in self.nodes) for _, r in hist.iterrows()]
        return {
            "day":       hist["day"].tolist(),
            "infected":  infected,
            "recovered": recovered,
            "deaths":    deaths,
            "susceptible": suscept,
        }
