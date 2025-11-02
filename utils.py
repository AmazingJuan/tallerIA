import streamlit as st
import easyocr
from groq import Groq
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image
import hashlib
import numpy as np

# ============================================================
#                     CARGA DE CLAVES
# ============================================================

@st.cache_resource
def cargar_claves():
    """Carga las claves API desde .env (solo una vez por sesión)."""
    load_dotenv()
    return os.getenv("GROQ_API_KEY"), os.getenv("HUGGINGFACE_API_KEY")


# ============================================================
#                    MODELOS Y CLIENTES
# ============================================================

@st.cache_resource
def cargar_modelo_ocr():
    """Carga el modelo OCR de EasyOCR (solo una vez)."""
    return easyocr.Reader(["es", "en"])


@st.cache_resource
def crear_cliente_groq(api_key):
    """Crea un cliente de GROQ."""
    return Groq(api_key=api_key)


@st.cache_resource
def crear_cliente_huggingface(modelo, api_key):
    """Crea un cliente de Hugging Face para un modelo específico."""
    return InferenceClient(model=modelo, token=api_key)


# ============================================================
#                  PROCESAMIENTO DE IMÁGENES
# ============================================================

@st.cache_data(show_spinner=False)
def procesar_imagen(_lector, archivo):
    """Realiza OCR sobre la imagen cargada usando EasyOCR y devuelve el texto."""
    if archivo is None:
        return ""

    imagen_bytes = archivo.read()
    imagen = Image.open(BytesIO(imagen_bytes)).convert("RGB")
    imagen_np = np.array(imagen)
    resultado = _lector.readtext(imagen_np, detail=0)
    texto = " ".join(resultado)
    return texto.strip()


# ============================================================
#                    ANÁLISIS CON GROQ
# ============================================================

@st.cache_data(show_spinner=False)
def analizar_con_groq(api_key, modelo, instruccion, texto, temperatura, max_tokens):
    """Analiza el texto usando GROQ con el modelo y parámetros especificados."""
    client = crear_cliente_groq(api_key)
    try:
        completion = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en procesamiento de texto."},
                {"role": "user", "content": f"{instruccion}\n\n{texto}"}
            ],
            temperature=temperatura,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al analizar con GROQ: {e}"


# ============================================================
#              ANÁLISIS CON HUGGING FACE (INSTRUCT)
# ============================================================

@st.cache_data(show_spinner=False)
def analizar_con_huggingface(api_key, modelo_hf, tarea, texto, temperatura, max_tokens):
    """
    Usa un modelo instructivo de Hugging Face (como LLaMA o Qwen).
    Intenta chat.completions, cae en text_generation si no es soportado.
    """
    client = crear_cliente_huggingface(modelo_hf, api_key)
    system_prompt = "Eres un asistente experto en procesamiento de texto."
    instrucciones = {
        "Resumir en 3 puntos clave": "Resume el siguiente texto en tres puntos clave claros y concisos.",
        "Identificar las entidades principales": "Identifica las entidades principales (personas, lugares, organizaciones, fechas).",
        "Traducir al inglés": "Traduce el siguiente texto al inglés de manera natural y precisa.",
    }

    user_content = instrucciones.get(tarea, "Analiza el siguiente texto.") + f"\n\n{texto.strip()}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        # Intento 1: API chat.completions (si el modelo lo soporta)
        resp = client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperatura,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        # Intento 2: fallback a text_generation
        try:
            prompt = f"{system_prompt}\n\nUsuario:\n{user_content}\n\nAsistente:"
            out = client.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperatura,
                top_p=0.9,  # Valor fijo
                do_sample=True,
                return_full_text=False,
            )
            return (out or "").strip()
        except Exception as e:
            return f"Error al analizar con Hugging Face: {e}"


# ============================================================
#                 HASH DE ARCHIVOS
# ============================================================

def hash_file(file):
    """Devuelve un hash único del archivo (para detectar cambios)."""
    file.seek(0)
    data = file.read()
    return hashlib.md5(data).hexdigest()
