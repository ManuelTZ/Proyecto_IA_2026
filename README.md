# 🦠 Simulador de Pandemia Global con IA

Simulador interactivo tipo "Plague Inc." con modelo SIR de metapoblaciones,
visualización pydeck y módulo de Inteligencia Artificial explicable.

---

## 📁 Estructura de Archivos

```
├── main.py                 # App principal Streamlit (dashboard completo)
├── simulation_engine.py    # Motor SIR + metapoblaciones + transporte global
├── ai_logic.py             # IA predictiva, diagnóstico y XAI
├── report_generator.py     # Generador de informe académico LaTeX
├── requirements.txt        # Dependencias Python
└── README.md               # Este archivo
```

---

## 🚀 Instalación y Ejecución

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Para compilar PDFs directamente
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install mactex

# Windows: Instalar MiKTeX desde miktex.org
```

### 4. Lanzar la aplicación
```bash
streamlit run main.py
```

---

## ⚙️ Parámetros Principales

| Parámetro | Símbolo | Rango | Descripción |
|-----------|---------|-------|-------------|
| Transmisibilidad | β | 0.05 – 1.0 | Prob. de contagio por contacto/día |
| Recuperación | γ | 0.01 – 0.30 | Fracción que se recupera/día |
| Letalidad | δ | 0.001 – 0.10 | Fracción que fallece |
| R₀ | β/γ | — | Calculado automáticamente |

---

## 🗺️ Nodos Modelados (12 regiones)

China · USA · India · Brazil · Germany · Nigeria ·
Russia · Japan · Australia · Mexico · South Africa · UK

Cada nodo tiene: población real, coordenadas GPS, capacidad UCI calibrada.

---

## 🤖 Módulo de IA

- **Predictiva**: Regresión log-lineal sobre ventana deslizante (7-14 días)  
  → Estima días hasta colapso UCI por nodo.

- **Diagnóstico**: Analiza infección, conectividad y carga hospitalaria  
  → Emite 4 tipos de alerta: MONITOREO / VACUNACIÓN / FRONTERAS / CUARENTENA.

- **XAI**: Cada recomendación incluye justificación cuantitativa:  
  > *"Se recomienda cerrar fronteras en China porque el modelo predictivo detectó un  
  > salto inminente de la cepa hacia 3 nodos sanos con alta conectividad: USA, Japan,  
  > Germany. Riesgo de salto calculado: 8.3% por día."*

---

## 📄 Informe LaTeX

El módulo `ReportGenerator` produce un `.tex` estructurado con:

1. Portada y resumen ejecutivo
2. Metodología (ecuaciones SIR con metapoblaciones)
3. Tablas de resultados por región
4. Análisis de IA y recomendaciones XAI
5. 4 figuras matplotlib (curva SIR, barras por nodo, heatmap temporal, carga UCI)
6. Conclusiones y bibliografía científica

---

## 🎮 Controles Interactivos

- **Cerrar fronteras**: reduce movilidad aérea en 96%
- **Cuarentena estricta**: reduce β local en 55%  
- **Vacunación masiva**: inmuniza al 30% de la población susceptible
- **Velocidad**: 1-15 días por frame (ajustable)
- **Aplicar recomendaciones IA**: botón en cada alerta del panel derecho

---

## 📐 Arquitectura del Motor

```
PandemiaSimulator
├── step()              → Avanza 1 día (Euler forward)
│   ├── Dinámica local SIR (por nodo)
│   └── Transporte internacional (matriz M 12×12)
├── close_borders()     → m_ij *= 0.04
├── apply_quarantine()  → β_local *= (1-reduction)
└── apply_vaccination() → S → R para fracción vacunada

AIMonitor
├── predict_hospital_collapse()   → Regresión log-lineal
├── predict_global_peak()         → Extrapolación R₀
├── analyze_global_state()        → 4 tipos de política XAI
└── get_risk_scores()             → Score 0-100 por nodo
```
# Proyecto_IA_2026
