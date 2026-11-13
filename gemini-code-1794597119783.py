import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Método de Newton-Raphson (NR)", page_icon="📈")

st.title("📈 Buscador de Raíces: Método de Newton-Raphson (NR)")
st.markdown("""
Esta aplicación implementa el algoritmo **NR** para encontrar raíces de funciones $f(x) = 0$. 
Elegí una función del menú y probá distintos valores semilla ($x_0$) para observar cuándo converge, 
cuándo encuentra solo una de varias raíces y cuándo falla.
""")

# --- BARRA LATERAL: PARÁMETROS DEL USUARIO ---
st.sidebar.header("⚙️ Configuración")

# Diccionario con 7 funciones (cumple con el rango de 5 a 10 y los casos pedidos)
funciones = {
    "1. f(x) = e^x - 10  (Funciona bien - 1 raíz)": {
        "func": lambda x: np.exp(x) - 10,
        "desc": "Función monótona creciente con una única raíz real en x ≈ 2.3026.",
        "rango": (-2.0, 4.0),
    },
    "2. f(x) = ln(x)  (Funciona bien si x0 > 0 - 1 raíz)": {
        "func": lambda x: np.log(x) if x > 0 else np.nan,
        "desc": (
            "Tiene una única raíz en x = 1. ¡Ojo! No está definida para x ≤ 0."
        ),
        "rango": (0.1, 5.0),
    },
    "3. f(x) = x^2 - 3  (Múltiples raíces: ±√3)": {
        "func": lambda x: x**2 - 3,
        "desc": (
            "Tiene dos raíces (x ≈ -1.732 y x ≈ 1.732). NR solo encuentra una"
            " dependiendo de la semilla."
        ),
        "rango": (-4.0, 4.0),
    },
    "4. f(x) = sin(x + 1)  (Infinitas raíces)": {
        "func": lambda x: np.sin(x + 1),
        "desc": (
            "Función periódica con infinitas raíces. NR es incompleto porque"
            " solo devuelve la primera que alcanza."
        ),
        "rango": (-6.0, 6.0),
    },
    "5. f(x) = x^3 - 4x  (Tres raíces: -2, 0, 2)": {
        "func": lambda x: x**3 - 4 * x,
        "desc": (
            "Polinomio cúbico con tres raíces reales. Dependiendo de x0, puede"
            " saltar a una raíz lejana."
        ),
        "rango": (-3.0, 3.0),
    },
    "6. f(x) = x^2 + 1  (Sin raíces reales)": {
        "func": lambda x: x**2 + 1,
        "desc": (
            "Nunca cruza el eje X (f(x) ≥ 1). NR oscilará o divergerá sin"
            " encontrar solución."
        ),
        "rango": (-4.0, 4.0),
    },
    "7. f(x) = e^x  (Sin raíces reales - Asíntota en y=0)": {
        "func": lambda x: np.exp(x),
        "desc": (
            "Siempre es positiva. Si x0 es muy negativo, f(x) < error y NR"
            " puede 'delirar' una raíz falsa."
        ),
        "rango": (-10.0, 2.0),
    },
    "8. f(x) = arctan(x)  (Caso difícil para NR)": {
        "func": lambda x: np.arctan(x),
        "desc": (
            "Tiene raíz en x = 0, pero si |x0| > 1.39, la recta tangente"
            " dispara las iteraciones hacia el infinito."
        ),
        "rango": (-5.0, 5.0),
    },
}

opcion = st.sidebar.selectbox(
    "Elegí una función:", list(funciones.keys())
)
f_seleccionada = funciones[opcion]["func"]

st.info(f"**Sobre esta función:** {funciones[opcion]['desc']}")

# Parámetros numéricos
x0_inicial = st.sidebar.number_input(
    "Valor semilla (x₀):", value=1.5, step=0.5, format="%.4f"
)

with st.sidebar.expander("Parámetros avanzados del algoritmo"):
  e = st.number_input("Tolerancia / Error (e):", value=1e-12, format="%.1e")
  d = st.number_input("Paso diferencial (d):", value=1e-10, format="%.1e")
  n = st.slider(
      "Límite de iteraciones (n):", min_value=10, max_value=2000, value=1000
  )


# --- ALGORITMO DE NEWTON-RAPHSON (Basado en tu código original) ---
def siguiente(x0, x1, f0, f1):
  if f0 == f1:
    return None  # Evita división por cero si la pendiente es horizontal
  return -f0 * ((x0 - x1) / (f0 - f1)) + x0


if st.button("🚀 Ejecutar NR", type="primary"):
  x0 = float(x0_inicial)
  x1 = x0 + d

  f0 = f_seleccionada(x0)
  f1 = f_seleccionada(x1)

  i = 0
  historial_x = [x0]
  error_dominio = False
  division_cero = False

  # Validación inicial de dominio (ej: logaritmo de número negativo)
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
      historial_x.append(x0)

  # --- RESULTADOS ---
  st.subheader("📊 Resultados")

  if error_dominio:
    st.error(
        "💥 **Error de dominio:** El algoritmo cayó fuera del dominio de la"
        " función (por ejemplo, $x \le 0$ en $\ln(x)$) o divergió a infinito."
    )
  elif division_cero:
    st.error(
        "⚠️ **Pendiente nula:** $f(x_0) = f(x_1)$, división por cero. NR no"
        " puede continuar."
    )
  elif abs(f0) <= e:
    st.success(f"🎉 **¡Eureka!** La raíz está en **$x = {round(x0, 6)}$**")
    col1, col2 = st.columns(2)
    col1.metric("Iteraciones realizadas", i)
    col2.metric("Error |f(x)|", f"{abs(f0):.2e}")
  else:
    st.warning("😕 **Mildis :( No se pudo encontrar la raíz.**")
    st.write(
        f"Se alcanzó el límite de **{n} iteraciones**. La última aproximación"
        f" fue **$x = {round(x0, 4)}$** con un error de **$ {abs(f0):.4e} $**."
    )

  # --- GRÁFICO VISUAL ---
  st.subheader("📉 Visualización de la función")
  r_min, r_max = funciones[opcion]["rango"]

  # Ajustamos el rango del gráfico si la raíz encontrada está dentro de un margen razonable
  if not error_dominio and not division_cero and abs(x0) < 50:
    r_min = min(r_min, x0 - 2, x0_inicial - 2)
    r_max = max(r_max, x0 + 2, x0_inicial + 2)

  x_vals = np.linspace(r_min, r_max, 500)
  y_vals = [f_seleccionada(val) for val in x_vals]

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.axhline(0, color="black", linewidth=1, linestyle="--")  # Eje X
  ax.axvline(0, color="gray", linewidth=0.5)  # Eje Y
  ax.plot(x_vals, y_vals, label="f(x)", color="#1f77b4", linewidth=2)

  # Marcar semilla inicial
  if not np.isnan(f_seleccionada(x0_inicial)):
    ax.scatter(
        [x0_inicial],
        [f_seleccionada(x0_inicial)],
        color="orange",
        zorder=5,
        label=f"Semilla x₀ = {x0_inicial}",
    )

  # Marcar raíz encontrada si convergió
  if not error_dominio and not division_cero and abs(f0) <= e:
    ax.scatter(
        [x0],
        [f0],
        color="green",
        s=100,
        zorder=6,
        label=f"Raíz hallada ≈ {round(x0, 4)}",
    )

  ax.set_xlabel("x")
  ax.set_ylabel("f(x)")
  ax.grid(True, alpha=0.3)
  ax.legend()
  st.pyplot(fig)