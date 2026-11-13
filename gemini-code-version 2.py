import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Método de Newton-Raphson (Navideño)", page_icon="🎄"
)

st.title("🎄 Buscador Navideño de Raíces: Método de Newton-Raphson (NR)")
st.markdown("""
¡Ayudá a Papá Noel a encontrar las raíces de las funciones! 
Elegí una función, ajustá los parámetros con las barras deslizadoras en la página principal y presioná **Ejecutar NR** para ver si encontramos la solución o si nos quedamos con carbón.
""")

# --- CONTROLES EN LA PÁGINA PRINCIPAL ---
st.subheader("1. Selección de Función")

# Diccionario con 8 funciones (sin pistas en los nombres)
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
st.subheader("2. Parámetros del Algoritmo (Sliders)")

# Sliders principales distribuidos en columnas en la página principal
col_s1, col_s2 = st.columns(2)

with col_s1:
  x0_inicial = st.slider(
      "Valor semilla (x₀):",
      min_value=-10.0,
      max_value=10.0,
      value=1.5,
      step=0.1,
      format="%.2f",
  )

  d = st.slider(
      "Paso diferencial (d):",
      min_value=1e-12,
      max_value=1e-4,
      value=1e-10,
      format="%.1e",
  )

with col_s2:
  e_exp = st.slider(
      "Exponente de tolerancia (Error e = 10^x):",
      min_value=-15,
      max_value=-3,
      value=-12,
      step=1,
  )
  e = 10.0**e_exp
  st.write(f"Tolerancia resultante (e): `{e:.1e}`")

  n = st.slider(
      "Límite de iteraciones (n):", min_value=10, max_value=2000, value=1000
  )

st.markdown("---")


# --- ALGORITMO DE NEWTON-RAPHSON ---
def siguiente(x0, x1, f0, f1):
  if f0 == f1:
    return None
  return -f0 * ((x0 - x1) / (f0 - f1)) + x0


if st.button("🚀 Ejecutar NR Navideño", type="primary"):
  x0 = float(x0_inicial)
  x1 = x0 + d

  f0 = f_seleccionada(x0)
  f1 = f_seleccionada(x1)

  i = 0
  # Guardamos el historial completo de puntos (x, f(x)) para graficar las líneas de salto
  puntos_iteracion = [(x0, f0)]
  error_dominio = False
  division_cero = False

  # Validación inicial de dominio
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

  # --- RESULTADOS Y DIAGNÓSTICO POST-EJECUCIÓN ---
  st.subheader("📊 Resultados y Comportamiento")

  if error_dominio:
    st.error(
        "🪨 **¡Carbón!** Error de dominio: El algoritmo cayó fuera del dominio"
        " de la función o divergió a infinito."
    )
  elif division_cero:
    st.error(
        "🪨 **¡Carbón!** Pendiente nula ($f(x_0) = f(x_1)$), división por"
        " cero."
    )
  elif abs(f0) <= e:
    st.success(
        f"🎄 **¡Eureka! ¡Feliz Navidad!** La raíz está en **$x = {round(x0, 6)}$**"
    )
    col_res1, col_res2 = st.columns(2)
    col_res1.metric("Iteraciones realizadas", i)
    col_res2.metric("Error |f(x)|", f"{abs(f0):.2e}")
    st.info(f"💡 **Diagnóstico:** {funciones[opcion]['desc']}")
  else:
    st.warning(
        "🪨 **¡Carbón :( No hay regalos esta vez!** No se pudo encontrar la"
        " raíz."
    )
    st.write(
        f"Se alcanzó el límite de **{n} iteraciones**. La última aproximación"
        f" fue **$x = {round(x0, 4)}$** con un error de **$ {abs(f0):.4e} $**."
    )
    st.info(
        f"💡 **Diagnóstico sobre esta función:** {funciones[opcion]['desc']}"
        " (Nota: Si no tiene raíces reales o la semilla está lejos, NR puede"
        " fallar)."
    )

  # --- GRÁFICO VISUAL PRINCIPAL (PALETA NAVIDEÑA) ---
  st.subheader("📉 Visualización Gráfica (Estilo Navideño)")
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
  ax.set_facecolor("#f9f9f9")
  fig.patch.set_facecolor("#ffffff")

  ax.axhline(0, color="#95a5a6", linewidth=1.2, linestyle="--")  # Eje X plateado
  ax.axvline(0, color="#95a5a6", linewidth=0.8, linestyle="--")  # Eje Y plateado

  # Línea de la función en Verde Oscuro Navideño
  ax.plot(
      x_vals,
      y_vals,
      label="f(x)",
      color="#1b4332",
      linewidth=2.5,
  )

  # Marcar semilla inicial en Amarillo/Oro estelar
  try:
    val_ini = f_seleccionada(x0_inicial)
    if not np.isnan(val_ini):
      ax.scatter(
          [x0_inicial],
          [val_ini],
          color="#f1c40f",
          s=90,
          zorder=5,
          edgecolors="black",
          label=f"Semilla x₀ = {x0_inicial}",
      )
  except:
    pass

  # Marcar raíz encontrada si convergió en Rojo Navideño brillante con estrella
  if not error_dominio and not division_cero and abs(f0) <= e:
    ax.scatter(
        [x0],
        [f0],
        color="#c0392b",
        s=120,
        zorder=6,
        marker="*",
        label=f"Raíz hallada ≈ {round(x0, 4)}",
    )

  ax.set_xlabel("x", color="#1b4332", fontweight="bold")
  ax.set_ylabel("f(x)", color="#1b4332", fontweight="bold")
  ax.grid(True, color="#bdc3c7", linestyle=":", alpha=0.6)
  ax.legend(facecolor="#ffffff", edgecolor="#95a5a6")
  st.pyplot(fig)

  # --- BOTÓN SECRETO / REGALO CON LÍNEAS DE ITERACIÓN ---
  if not error_dominio and not division_cero:
    with st.expander("🎁 Abrir regalo: Ver líneas de las iteraciones"):
      st.markdown(
          "Este gráfico muestra los saltos y aproximaciones sucesivas del"
          " algoritmo paso a paso."
      )

      fig2, ax2 = plt.subplots(figsize=(8, 4))
      ax2.set_facecolor("#f9f9f9")
      fig2.patch.set_facecolor("#ffffff")

      ax2.axhline(0, color="#95a5a6", linewidth=1.2, linestyle="--")
      ax2.axvline(0, color="#95a5a6", linewidth=0.8, linestyle="--")

      # Curva principal
      ax2.plot(
          x_vals, y_vals, label="f(x)", color="#1b4332", linewidth=2, alpha=0.7
      )

      # Extraer coordenadas x de las iteraciones realizadas
      xs_it = [pt[0] for pt in puntos_iteracion]
      ys_it = [pt[1] for pt in puntos_iteracion]

      # Dibujar puntos y líneas conectoras de los saltos
      ax2.plot(
          xs_it,
          ys_it,
          color="#c0392b",
          linestyle="-.",
          marker="o",
          linewidth=1.5,
          markersize=5,
          label="Saltos NR",
      )

      # Resaltar la última iteración
      if len(xs_it) > 0:
        ax2.scatter(
            [xs_it[-1]],
            [ys_it[-1]],
            color="#f1c40f",
            s=100,
            zorder=6,
            edgecolors="black",
            label="Último punto",
        )

      ax2.set_xlabel("x", color="#1b4332", fontweight="bold")
      ax2.set_ylabel("f(x)", color="#1b4332", fontweight="bold")
      ax2.grid(True, color="#bdc3c7", linestyle=":", alpha=0.6)
      ax2.legend(facecolor="#ffffff", edgecolor="#95a5a6")
      st.pyplot(fig2)