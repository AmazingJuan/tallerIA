import streamlit as st
from utils import (
    cargar_modelo_ocr,
    cargar_claves,
    procesar_imagen,
    analizar_con_groq,
    analizar_con_huggingface,
    hash_file,
)

# ============================================================
#      CONFIGURACIÓN DE VARIABLES PERSISTENTES (session_state)
# ============================================================

if "texto_extraido" not in st.session_state:
    st.session_state.texto_extraido = ""

if "imagen_hash" not in st.session_state:
    st.session_state.imagen_hash = None

if "archivo" not in st.session_state:
    st.session_state.archivo = None

if "proveedor" not in st.session_state:
    st.session_state.proveedor = "GROQ"

if "tarea" not in st.session_state:
    st.session_state.tarea = "Resumir en 3 puntos clave"

if "modelo" not in st.session_state:
    st.session_state.modelo = "llama-3.1-8b-instant"

if "temperatura" not in st.session_state:
    st.session_state.temperatura = 0.7

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 500

# ============================================================
#                 SIDEBAR: MENÚ DE SECCIONES
# ============================================================

st.sidebar.title("Taller Final IA")
st.sidebar.write("**Autor:** Juan Pablo Avendaño")

# Menú de navegación
menu = st.sidebar.radio(
    "Navegación:",
    [
        "Aplicación principal",
        "Reflexión",
    ],
    index=0,
)

# ============================================================
#                 SECCIÓN PRINCIPAL
# ============================================================

if menu == "Aplicación principal":
    groq_key, hf_key = cargar_claves()

    with st.spinner("Cargando modelo OCR..."):
        lector = cargar_modelo_ocr()

    st.title("Taller IA: OCR + LLM")
    st.header("Sección: OCR")
    st.write(
        "Sube una imagen para que su contenido sea reconocido mediante OCR y luego analizado por un modelo LLM."
    )

    archivo = st.file_uploader(
        "Ponga acá la imagen que desea analizar", type=["jpg", "jpeg", "png"]
    )

    # Si el usuario sube una nueva imagen, actualizar
    if archivo is not None:
        archivo_hash = hash_file(archivo)
        archivo.seek(0)
        st.session_state.archivo = archivo

        if archivo_hash != st.session_state.imagen_hash:
            with st.spinner("Procesando nueva imagen..."):
                archivo.seek(0)
                texto = procesar_imagen(lector, archivo)
                st.session_state.texto_extraido = texto
                st.session_state.imagen_hash = archivo_hash
            st.success("Lectura completada correctamente.")
        else:
            st.info("Lectura recuperada desde memoria (imagen no ha cambiado).")

    # Mostrar imagen si ya hay una cargada
    if st.session_state.archivo is not None:
        st.image(st.session_state.archivo, caption="Imagen cargada", width="stretch")

        # ============================================================
        #               SECCIÓN LLM (GROQ / HUGGING FACE)
        # ============================================================

        st.header("Sección: LLM (GROQ / Hugging Face)")
        st.session_state.proveedor = st.radio(
            "Seleccione el proveedor de API:",
            ["GROQ", "Hugging Face"],
            index=["GROQ", "Hugging Face"].index(st.session_state.proveedor),
        )

        st.session_state.tarea = st.selectbox(
            "Seleccione la tarea a realizar",
            [
                "Resumir en 3 puntos clave",
                "Identificar las entidades principales",
                "Traducir al inglés",
            ],
            index=[
                "Resumir en 3 puntos clave",
                "Identificar las entidades principales",
                "Traducir al inglés",
            ].index(st.session_state.tarea),
        )

        st.session_state.temperatura = st.slider(
            "Creatividad (temperature)", 0.0, 1.0, st.session_state.temperatura, 0.1
        )

        st.session_state.max_tokens = st.slider(
            "Máx. tokens (longitud de respuesta)",
            50,
            2000,
            st.session_state.max_tokens,
            50,
        )

        if st.session_state.proveedor == "GROQ":
            modelos_groq = [
                "llama-3.1-8b-instant",
                "openai/gpt-oss-20b",
                "openai/gpt-oss-120b",
                "meta-llama/llama-4-scout-17b-16e-instruct",
            ]
            st.session_state.modelo = st.selectbox(
                "Seleccione el modelo de GROQ",
                modelos_groq,
                index=(
                    modelos_groq.index(st.session_state.modelo)
                    if st.session_state.modelo in modelos_groq
                    else 0
                ),
            )
        else:
            modelos_hf = [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
            ]
            st.session_state.modelo = st.selectbox(
                "Seleccione el modelo de Hugging Face",
                modelos_hf,
                index=(
                    modelos_hf.index(st.session_state.modelo)
                    if st.session_state.modelo in modelos_hf
                    else 0
                ),
            )

        if st.button("Analizar Texto"):
            texto = st.session_state.texto_extraido.strip()
            if not texto:
                st.warning("Primero suba una imagen y realice la lectura OCR.")
            else:
                st.subheader("Resultado del análisis:")
                if st.session_state.proveedor == "GROQ":
                    with st.spinner("Analizando con GROQ..."):
                        instrucciones = {
                            "Resumir en 3 puntos clave": "Resume el siguiente texto en 3 puntos clave claros y concisos:",
                            "Identificar las entidades principales": "Identifica las entidades principales (personas, lugares, organizaciones, fechas):",
                            "Traducir al inglés": "Traduce el siguiente texto al inglés de manera natural y precisa:",
                        }
                        instruccion = instrucciones.get(
                            st.session_state.tarea, "Analiza el siguiente texto:"
                        )
                        respuesta = analizar_con_groq(
                            groq_key,
                            st.session_state.modelo,
                            instruccion,
                            texto,
                            st.session_state.temperatura,
                            st.session_state.max_tokens,
                        )
                else:
                    with st.spinner("Analizando con Hugging Face..."):
                        respuesta = analizar_con_huggingface(
                            hf_key,
                            st.session_state.modelo,
                            st.session_state.tarea,
                            texto,
                            st.session_state.temperatura,
                            st.session_state.max_tokens,
                        )

                st.markdown(respuesta or "No se generó ninguna respuesta.")

# ============================================================
#                 SECCIÓN DE REFLEXIÓN
# ============================================================

elif menu == "Reflexión":
    st.title("Reflexión sobre el Taller")

    st.write(
        "En esta sección se comparte una reflexión sobre lo aprendido durante el desarrollo del taller."
    )

    pregunta = st.selectbox(
        "Selecciona una pregunta para ver la respuesta:",
        [
            "¿Qué diferencias de velocidad notaron entre GROQ y Hugging Face?",
            "¿Cómo afecta el cambio de temperature a las respuestas del LLM?",
            "¿Qué tan importante fue la calidad del texto extraído por el OCR para la calidad del análisis del LLM?",
            "¿Qué otros modelos o tareas se podrían integrar en esta aplicación?",
        ],
    )

    if pregunta == "¿Qué diferencias de velocidad notaron entre GROQ y Hugging Face?":
        st.subheader("Diferencias de velocidad entre GROQ y Hugging Face")
        st.markdown(
            """
Basado en las pruebas realizadas y consultas que se hicieron, se puede decir que GROQ es más rapido que Hugging Face. Esto tiene sentido ya que su arquitectura está basada en **LPU (Language Processing Units)**, 
chips diseñados específicamente para acelerar modelos de lenguaje.

Por el contrario, Hugging Face ofrece una gran flexibilidad y variedad de modelos, pero esa diversidad implica 
latencias mayores, especialmente si los modelos se ejecutan en servidores compartidos o necesitan descargarse dinámicamente.
"""
        )

    elif pregunta == "¿Cómo afecta el cambio de temperature a las respuestas del LLM?":
        st.subheader("Efecto del parámetro *temperature*")
        st.markdown(
            """
El parámetro temperature tiene que ver con la creatividad que expone el modelo en sus respuestas.  
Al usar valores bajos (por ejemplo, 0.2 o 0.3), las respuestas eran muy estructuradas, y repetitivas.  
El modelo soltaba respuestas directas y concisas.

En cambio, con valores altos (0.8 o más), el modelo se volvía más creativo e impredecible. Esto era útil para tareas más abiertas o con tono libre, 
como generar ideas principales.
                    
Se pudo ver que si los valores de la temperatura bajan demasiado, el modelo no era capaz de generar una respuesta.
"""
        )

    elif (
        pregunta
        == "¿Qué tan importante fue la calidad del texto extraído por el OCR para la calidad del análisis del LLM?"
    ):
        st.subheader("Calidad del texto OCR y su impacto en el análisis")
        st.markdown(
            """
El OCR es donde comienza todo el análisis de los modelos, y cuando su salida tenía errores el modelo de lenguaje daba respuestas basadas en esos errores.

Por eso es necesario tener un OCR bien hecho para poder minimizar los errores que se cometen a la hora de generar respuestas.
"""
        )

    elif (
        pregunta
        == "¿Qué otros modelos o tareas se podrían integrar en esta aplicación?"
    ):
        st.subheader("Posibles extensiones y mejoras futuras")
        st.markdown(
            """
Se pueden integrar diversos modulos que harían el ejercicio mucho mas interesante:

- **Análisis de sentimientos:** determinar si el texto refleja emociones positivas, negativas o neutras.  
- **Clasificación de texto:** agrupar documentos por tema o contexto (por ejemplo: financiero, médico, educativo).  
- **Generación de preguntas y respuestas:** crear quizzes automáticos o asistentes de estudio.  
- **Reescritura y limpieza del texto OCR:** usar LLMs para corregir errores antes del análisis.  
- **Resúmenes inteligentes:** generar versiones breves orientadas a objetivos específicos (ejecutivo, académico, técnico).

Realmente cuando hablamos de posibilidades con un LLM se pueden hacer muchisimas cosas.
"""
        )
