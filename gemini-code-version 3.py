import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página con icono de árbol de navidad
st.set_page_config(
    page_title="Método de Newton-Raphson (Buscador de Raíces)", page_icon="🎄"
)

# Inyectar estilos CSS específicos para colorear cada slider de la interfaz
st.markdown(
    """
    <style>
    /* Slider 1: Paso diferencial (Verde #556b2f) */
    div[data-testid="stSlider"]:nth-of-type(1) div[role="slider"] {
        background-color: #556b2f !important;
        border-color: #556b2f !important;
    }
    div[data-testid="stSlider"]:nth-of-type(1) div.st-bp {
        background-color: #556b2f !important;
    }

    /* Slider 2: Tolerancia / Exponente (Rojo #8b0000) */
    div[data-testid="stSlider"]:nth-of-type(2) div[role="slider"] {
        background-color: #8b0000 !important;
        border-color: #8b0000 !important;
    }
    div[data-testid="stSlider"]:nth-of-type(2) div.st-bp {
        background-color: #8b0000 !important;
    }

    /* Slider 3: Límite de iteraciones (Gris) */
    div[data-testid="stSlider"]:nth-of-type(3) div[role="slider"] {
        background-color: grey !important;
        border-color: grey !important;
    }
    div[data-testid="stSlider"]:nth-of-type(3) div.st-bp {
        background-color: grey !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Método Newton Raphson - Buscador de raíces")
st.markdown(
    "Elegí una función, ajustá los parámetros y presioná **Ejecutar NR** para"
    " descubrir si se puede utilizar este método para encontrar raíces o no."
)

# --- CONTROLES EN LA PÁGINA PRINCIPAL ---
st.subheader("1. Selección de Función")

funciones = {
    "f(x) = e^x - 10": {
        "func": lambda x: np.exp(x) - 10,
        "desc": "Función exponencial con una única raíz real.",
        "rango": (-2.0, 4.0),
    },
    "f(x) = ln(x)": {
        "func": lambda x: np.log(x) if x > 0 else np.nan,
        "desc": (
            "Función logarítmica (no definida para valores menores o iguales a"
            " cero)."
        ),
        "rango": (0.1, 5.0),
    },
    "f(x) = x^2 - 3": {
        "func": lambda x: x**2 - 3,
        "desc": "Polinomio cuadrático con dos raíces simétricas.",
        "rango": (-4.0, 4.0),
    },
    "f(x) = sin(x + 1)": {
        "func": lambda x: np.sin(x + 1),
        "desc": "Función trigonométrica con infinitas raíces periódicas.",
        "rango": (-6.0, 6.0),
    },
    "f(x) = x^3 - 4x": {
        "func": lambda x: x**3 - 4 * x,
        "desc": "Polinomio cúbico con tres raíces reales.",
        "rango": (-3.0, 3.0),
    },
    "f(x) = x^2 + 1": {
        "func": lambda x: x**2 + 1,
        "desc": "Polinomio estrictamente positivo sin raíces reales.",
        "rango": (-4.0, 4.0),
    },
    "f(x) = e^x": {
        "func": lambda x: np.exp(x),
        "desc": (
            "Función exponencial estricta con una asíntota horizontal en y = 0."
        ),
        "rango": (-10.0, 2.0),
    },
    "f(x) = arctan(x)": {
        "func": lambda x: np.arctan(x),
        "desc": (
            "Función arcotangente con raíz en el origen y comportamiento"
            " complejo para semillas alejadas."
        ),
        "rango": (-5.0, 5.0),
    },
}

opcion = st.selectbox("Elegí una función a evaluar:", list(funciones.keys()))
f_seleccionada = funciones[opcion]["func"]

st.markdown("---")
st.subheader("2. Parámetros del Algoritmo")

col_s1, col_s2 = st.columns(2)

with col_s1:
  x0_inicial = st.number_input(
      "Valor semilla (x₀) [Ingresar valor entero]:",
      value=1,
      step=1,
      format="%d",
  )

  # Slider 1: Verde #556b2f
  d = st.slider(
      "Paso diferencial (d):",
      min_value=1e-5,
      max_value=0.5,
      value=1e-3,
      step=1e-5,
      format="%.5f",
  )

with col_s2:
  # Slider 2: Rojo #8b0000
  e_exp = st.slider(
      "Exponente de tolerancia (Error e = 10^x):",
      min_value=-15,
      max_value=-3,
      value=-12,
      step=1,
  )
  e = 10.0**e_exp
  st.write(f"Tolerancia resultante (e): `{e:.1e}`")

  # Slider 3: Gris
  n = st.slider(
      "Límite de iteraciones (n):",
      min_value=50,
      max_value=2000,
      value=1000,
      step=50,
  )

st.markdown("---")


# --- ALGORITMO DE NEWTON-RAPHSON ---
def siguiente(x0, x1, f0, f1):
  if f0 == f1:
    return None
  return -f0 * ((x0 - x1) / (f0 - f1)) + x0


if st.button("Ejecutar NR", type="primary"):
  x0 = float(x0_inicial)
  x1 = x0 + d

  f0 = f_seleccionada(x0)
  f1 = f_seleccionada(x1)

  i = 0
  puntos_iteracion = [(x0, f0)]
  error_dominio = False
  division_cero = False

  if np.isnan(f0) or np.isnan(f1) or np.isinf(f0) or np.isinf(f1):
    error_dominio = True
  else:
    while abs(f0) >= e and i < n:
      i += 1
      x2 = siguiente(x0, x1, f0, f1)

      if x2 is None or np.isnan(x2) or np.isinf(x2):
        division_cero = True
        break

      f2 = f_seleccionada(x2)
      if np.isnan(f2) or np.isinf(f2):
        error_dominio = True
        break

      x0, f0 = x1, f1
      x1, f1 = x2, f2
      puntos_iteracion.append((x0, f0))

  # --- RESULTADOS Y DIAGNÓSTICO ---
  st.subheader("📊 Resultados y Comportamiento")

  if error_dominio or division_cero:
    st.error(
        "🪨 Recórcholis! No todo método es infalible, no hemos encontrado"
        " raíces (Error de dominio o división por cero)."
    )
  elif abs(f0) <= e:
    st.success(
        f"🎄 Eureka! Encontramos una solución en **$x = {round(x0, 6)}$**"
    )
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Iteraciones realizadas", i)
    col_res2.metric("Error |f(x)|", f"{abs(f0):.2e}")
    st.info(f"💡 **Diagnóstico:** {funciones[opcion]['desc']}")
  else:
    st.warning(
        "🪨 Recórcholis! No todo método es infalible, no hemos encontrado"
        " raíces."
    )
    st.write(
        f"Se alcanzó el límite de **{n} iteraciones**. La mejor aproximación"
        f" encontrada fue **$x = {round(x0, 4)}$** con un error de **$"
        f" {abs(f0):.4e} $**."
    )
    st.info(
        f"💡 **Diagnóstico sobre esta función:** {funciones[opcion]['desc']}"
    )

  # --- GRÁFICO PRINCIPAL ---
  st.subheader("📉 Visualización Gráfica")
  r_min, r_max = funciones[opcion]["rango"]

  if not error_dominio and not division_cero and abs(x0) < 50:
    r_min = min(r_min, x0 - 2, x0_inicial - 2)
    r_max = max(r_max, x0 + 2, x0_inicial + 2)

  x_vals = np.linspace(r_min, r_max, 500)
  y_vals = []
  for val in x_vals:
    val_f = f_seleccionada(val)
    y_vals.append(val_f if not np.isnan(val_f) else 0)

  fig, ax = plt.subplots(figsize=(8, 4))

  # Ejes cartesianos en gris
  ax.axhline(0, color="grey", linewidth=1.2, linestyle="-")
  ax.axvline(0, color="grey", linewidth=1.2, linestyle="-")

  # Función f(x) con color verde #556b2f
  ax.plot(x_vals, y_vals, label="f(x)", color="#556b2f", linewidth=2.5)

  # Valor semilla con color lightgrey, forma de rombo (diamond 'D') y reno (🦌)
  try:
    val_ini = f_seleccionada(x0_inicial)
    if not np.isnan(val_ini):
      ax.scatter(
          [x0_inicial],
          [val_ini],
          color="lightgrey",
          s=140,
          zorder=5,
          marker="D",
          edgecolors="black",
          label=f"Semilla x₀ (🦌) = {x0_inicial}",
      )
  except:
    pass

  # Raíz encontrada con color #8b0000 (regalo 🎁) o valor más cercano en darkgrey (carbón 🪨)
  if not error_dominio and not division_cero and abs(f0) <= e:
    ax.scatter(
        [x0],
        [f0],
        color="#8b0000",
        s=140,
        zorder=6,
        marker="o",
        edgecolors="black",
        label=f"Raíz hallada (🎁) ≈ {round(x0, 4)}",
    )
  elif not error_dominio and not division_cero:
    ax.scatter(
        [x0],
        [f0],
        color="darkgrey",
        s=140,
        zorder=6,
        marker="X",
        edgecolors="black",
        label=f"Valor más cercano (🪨) ≈ {round(x0, 4)}",
    )

  ax.set_xlabel("x")
  ax.set_ylabel("f(x)")
  ax.grid(True, alpha=0.3)
  ax.legend()
  st.pyplot(fig)

  # --- EXPANSOR DE LÍNEAS DE ITERACIÓN ---
  if not error_dominio and not division_cero:
    with st.expander("🎁 Ver líneas de las iteraciones"):
      st.markdown(
          "Este gráfico muestra los saltos y aproximaciones sucesivas del"
          " algoritmo paso a paso."
      )

      fig2, ax2 = plt.subplots(figsize=(8, 4))
      ax2.axhline(0, color="grey", linewidth=1.2, linestyle="-")
      ax2.axvline(0, color="grey", linewidth=1.2, linestyle="-")

      ax2.plot(
          x_vals,
          y_vals,
          label="f(x)",
          color="#556b2f",
          linewidth=2,
          alpha=0.7,
      )

      xs_it = [pt[0] for pt in puntos_iteracion]
      ys_it = [pt[1] for pt in puntos_iteracion]

      ax2.plot(
          xs_it,
          ys_it,
          color="#eead0e",
          linestyle="-.",
          marker="o",
          linewidth=1.5,
          markersize=5,
          label="Saltos NR (🦌)",
      )

      if len(xs_it) > 0:
        ax2.scatter(
            [xs_it[-1]],
            [ys_it[-1]],
            color="darkgrey" if abs(f0) > e else "#8b0000",
            s=100,
            zorder=6,
            edgecolors="black",
            label="Último punto",
        )

      ax2.set_xlabel("x")
      ax2.set_ylabel("f(x)")
      ax2.grid(True, alpha=0.3)
      ax2.legend()
      st.pyplot(fig2)
