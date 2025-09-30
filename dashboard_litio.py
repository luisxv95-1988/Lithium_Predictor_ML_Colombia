import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pathlib
import numpy as np
import plotly.figure_factory as ff
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Configuración inicial
# ==============================
st.set_page_config(page_title="Dashboard Geoquímico - Litio", layout="wide", initial_sidebar_state="expanded", page_icon="🔋")

# ==============================
# Cargar datos
# ==============================
df = pd.read_csv("df_filtrado.csv", sep=";", decimal=",", encoding="latin1")

# ==============================
# Encabezado con el mapa HTML
# ==============================
st.title("Dashboard Geoquímico - Predicción de Litio")

st.markdown("### Mapa de concentración de Litio - Colombia")
html_path = pathlib.Path("mapa_dual_con_heatmap_y_escala.html")
html_str = html_path.read_text(encoding="latin1")
st.components.v1.html(html_str, height=600, scrolling=True)

# ==============================
# Sidebar con filtros
# ==============================
st.sidebar.header("Opciones de Visualización")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if "Li_ppm" in numeric_cols:
    numeric_cols.remove("Li_ppm")

features = ["Cs_ppm", "U_ppm",'k/Rb_ppm',"Zn_ppm","As_ppm","Th_ppm","W_ppm"
            ,"Sc_ppm","Y_ppm","Tb_ppm","Ga_ppm","Dy_ppm","In_ppm","Fe_ppm","Ho_ppm","Co_ppm","Er_ppm"]
    
# Selección para scatter
var_scatter = st.sidebar.selectbox("Variable para gráficos de dispersión y BoxPlot(vs Li_ppm):", features )
kde = st.sidebar.checkbox("Incluir KDE en el Histograma", value=True)


# ==============================
# Gráficos
# ==============================

# Gráfico de dispersión con línea de tendencia
st.subheader(f"Dispersión de Li_ppm vs {var_scatter}")

fig_scatter = px.scatter(df, x=var_scatter, y="Li_ppm",
                         labels={var_scatter: var_scatter, "Li_ppm": "Concentración de Li (ppm)"})

# Calcular la regresión lineal con numpy (evita usar statsmodels)
x = df[var_scatter].dropna()
y = df.loc[x.index, "Li_ppm"].dropna()

if len(x) > 1 and len(y) > 1:
    m, b = np.polyfit(x, y, 1)  # pendiente y intercepto
    fig_scatter.add_scatter(
        x=x,
        y=m * x + b,
        mode="lines",
        name="Línea de tendencia",
        line=dict(color="red")
    )
st.plotly_chart(fig_scatter, width='stretch')

# Histograma con KDE usando Plotly para el Litio
st.subheader(f"Histograma de Litio con KDE")
data = df["Li_ppm"].dropna()
fig_hist = ff.create_distplot([data], ["Li_ppm"], colors=['#636EFA'], show_hist=True, show_rug=False, bin_size=(data.max()-data.min())/40)
# Personalizar ejes y título
fig_hist.update_layout(
xaxis_title="Concentración de Li (ppm)",
yaxis_title="Densidad de Probabilidad"
)
st.plotly_chart(fig_hist, width='stretch')

if kde!= True:
    st.info("El histograma incluye una estimación de densidad de kernel (KDE) para mostrar la distribución de los datos.")
    # Histograma
    st.subheader(f"Histograma de {var_scatter}")
    fig_hist = px.histogram(df, x=var_scatter, nbins=40,
                            labels={var_scatter: var_scatter})
    st.plotly_chart(fig_hist, width='stretch')
else:
    # Histograma con KDE usando Plotly
    st.subheader(f"Histograma de {var_scatter} con KDE")
    data = df[var_scatter].dropna()
    fig_hist = ff.create_distplot([data], [var_scatter], show_hist=True, show_rug=False, bin_size=(data.max()-data.min())/40)
    # Personalizar ejes y título
    fig_hist.update_layout(
    xaxis_title=var_scatter,
    yaxis_title="Densidad de Probabilidad"
    )
    st.plotly_chart(fig_hist, width='stretch')

# Boxplot Litio
st.subheader(f"Gráfico de Bigotes para Litio (Li_ppm)")
fig_box = px.box(df, y="Li_ppm", points="outliers",
                 labels={"Li_ppm": "Concentración de Li (ppm)"},
                 color_discrete_sequence=['#636EFA'])
st.plotly_chart(fig_box, width='stretch')

# Boxplot
st.subheader(f"Gráfico de Bigotes para {var_scatter}")
fig_box = px.box(df, y=var_scatter, points="outliers",
                 labels={var_scatter: var_scatter})
st.plotly_chart(fig_box, width='stretch')


# ==============================
# Heatmap de correlaciones con Li_ppm
# ==============================
st.subheader("Heatmap de correlaciones (Li_ppm vs materiales seleccionados)")

# Filtrar solo columnas que existen en el DataFrame
valid_features = [col for col in features if col in df.columns]

# Calcular correlaciones solo entre Li_ppm y las features
corr = df[valid_features + ["Li_ppm"]].corr(numeric_only=True)["Li_ppm"].drop("Li_ppm").sort_values(ascending=False)

# Agregar grafico de burbujas para visualizar correlaciones donde el color represente la magnitud y el tamaño la fuerza de la correlación

fig_bubble = go.Figure(data=go.Scatter(
    x=corr.index,
    y=corr.values,
    mode='markers',
    marker=dict(
        size=np.abs(corr.values) * 100,  # Tamaño proporcional a la fuerza de la correlación
        color=corr.values,  # Color proporcional a la magnitud de la correlación
        colorscale='RdBu',
        showscale=True
    )
))

fig_bubble.update_layout(
    title="Correlación de Li_ppm con materiales seleccionados (Bubble Chart)",
    xaxis_title="Variables",
    yaxis_title="Correlación"
)

st.plotly_chart(fig_bubble, width='stretch')

fig_corr = go.Figure(data=go.Heatmap(
    z=[corr.values],
    x=corr.index,
    y=["Li_ppm"],
    colorscale="RdBu",
    zmid=0,
    text=np.round(corr.values, 3),
    texttemplate="%{text}"
))
fig_corr.update_layout(
    title="Correlación de Li_ppm con materiales seleccionados",
    xaxis_title="Variables",
    yaxis_title="",
    height=300
)

st.plotly_chart(fig_corr, width='stretch')



# Model Comparison
# ==============================
st.subheader("Comparación de Modelos de Machine Learning para predecir Li_ppm") 

X = df[features].copy()
y = df["Li_ppm"].copy()

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos a comparar
modelos = {
    "Ridge": Pipeline([("scaler", StandardScaler()), ("modelo", Ridge(alpha=0.3))]),
    "Lasso": Pipeline([("scaler", StandardScaler()), ("modelo", Lasso())]),
    "MLP": Pipeline([("scaler", StandardScaler()), ("modelo", MLPRegressor(max_iter=5000, random_state=42,solver='lbfgs',alpha=10,activation='relu'))]),
    "RandomForest":  RandomForestRegressor(random_state=42, n_estimators=1800,max_features=0.57),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

resultados = []

# Evaluar cada modelo
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    rpd = np.std(y_test) / rmse if rmse > 0 else np.nan
    accuracy = 1 - (rmse / np.mean(y_test))

    resultados.append({
        "Modelo": nombre,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "RPD": rpd,
        "Overall_Accuracy": accuracy
    })

# Resultados en DataFrame ordenado
df_resultados = pd.DataFrame(resultados).sort_values(by="R2", ascending=False).reset_index(drop=True)
st.dataframe(df_resultados.style.format({"R2": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}", "RPD": "{:.3f}", "Overall_Accuracy": "{:.3f}"}), use_container_width=True)

# Normalizar las métricas excepto R2 y Overall Accuracy (para que el color sea comparable)
df_heatmap = df_resultados.set_index("Modelo")

# Crear la figura
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", fmt=".3f", cbar=True, ax=ax)

ax.set_title("Comparación de Modelos - Métricas de Evaluación")
ax.set_ylabel("Modelo")
ax.set_xlabel("Métricas")

# Mostrar en Streamlit
st.pyplot(fig)