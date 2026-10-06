"""
Idioma de la interfaz (los textos de la ventana, no los de la conversación).

Los textos se escriben en español en el código, dentro de T("...") (o N_("...")
en las tablas que se definen al importar y se traducen al mostrarlas). Cada
idioma tiene su tabla español -> traducción en interfaz\<código>.json; lo que
falta en ella se ve en español.

- en.json va revisado a mano con el programa.
- Los demás se generan con el propio traductor del programa (Opus-MT) a partir
  del inglés, que todos los idiomas del catálogo tienen, y se pueden corregir
  editando su .json (lo corregido se respeta al volver a generarlo).
"""
import ast
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE_DIR, "interfaz")
# Archivos con textos de la interfaz (de donde se sacan para generar un idioma).
SOURCES = ("conversacion.py", "motores.py", "voz.py", "traductor.py", "transcriptor.py")
# Lo que no se traduce nunca (nombres de programas, dispositivos, modelos...).
KEEP = ("CABLE Input", "CABLE Output", "VB-Cable", "Hugging Face", "WhatsApp Web", "Whisper",
        "Teams", "Zoom", "Piper", "Opus-MT", "Silero VAD", "faster-whisper", "whisper.cpp",
        "whisper-server", "CTranslate2", "Vulkan", "OpenAI", "GPU", "CPU", "NVIDIA", "CUDA",
        "Google", "Windows", "motor_gpu")

_table = {}
current = "es"


def N_(text):
    """Marca un texto para traducirlo al mostrarlo (con T)."""
    return text


def T(text, **data):
    """El texto en el idioma de la interfaz; data rellena sus {huecos}."""
    out = _table.get(text, text) if text else text
    if not data:
        return out
    try:
        return out.format(**data)
    except (KeyError, IndexError, ValueError):  # traducción con los huecos estropeados
        return text.format(**data)


def path(code):
    return os.path.join(DIR, f"{code}.json")


def load(code):
    try:
        with open(path(code), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def set_language(code):
    """Idioma de la interfaz: es (el del código) o uno con su .json."""
    global current
    _table.clear()
    if code != "es":
        _table.update(load(code))
    current = code


def available(code):
    return code == "es" or os.path.exists(path(code))


def texts():
    """Todos los textos de la interfaz: los de T("...") y N_("...") del código."""
    found = []
    for name in SOURCES:
        try:
            with open(os.path.join(BASE_DIR, name), encoding="utf-8-sig") as f:  # alguno lleva BOM
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("T", "N_") and node.args
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                found.append(node.args[0].value)
    return list(dict.fromkeys(found))


def _protect(text):
    """Cambia los huecos {x} y los nombres que no se traducen por marcas que el
    traductor deja igual (números). Devuelve (texto, marcas)."""
    marks = []

    def mark(m):
        marks.append(m.group(0))
        return f"{len(marks) * 111}"

    names = "|".join(re.escape(k) for k in sorted(KEEP, key=len, reverse=True))
    return re.sub(r"\{\w+\}|https?://\S+|" + names, mark, text), marks


def _restore(text, marks):
    for i, m in reversed(list(enumerate(marks, start=1))):
        text = text.replace(f"{i * 111}", m, 1)
    return text


def _language_name(name, translate):
    """Un nombre de idioma suelto se traduce mal ("English" salió "changements
    climatiques"): se traduce dentro de una frase y se saca de ella."""
    got = translate(f"The language: {name}.") or ""
    if ":" not in got and "：" not in got:
        return translate(name) or name
    got = re.split(r"[:：]", got, maxsplit=1)[1].strip().rstrip(".。")
    got = re.sub(r"^(l'|l’|le |la |les |el |lo |il |der |die |das |o |a )", "", got, flags=re.I).strip()
    return got[:1].upper() + got[1:] if got else name


def generate(code, translate, progress=lambda done, total: None, names=()):
    """Crea (o completa) interfaz\\<code>.json traduciendo del inglés con
    translate(texto) -> texto. Lo que ya hay en el archivo se respeta. names:
    los textos que son nombres de idiomas."""
    english = load("en")
    out = load(code)
    todo = [t for t in texts() if t not in out]
    for i, es in enumerate(todo):
        source = english.get(es, es)
        if es in names:
            out[es] = _language_name(source, translate)
            progress(i + 1, len(todo))
            continue
        protected, marks = _protect(source)
        # Las líneas por separado (el traductor junta o pierde los saltos de línea).
        lines = [translate(line) if line.strip() else line for line in protected.split("\n")]
        got = _restore("\n".join(x or "" for x in lines), marks)
        # Si se perdió algún hueco o nombre, mejor el inglés que una frase rota.
        out[es] = got if all(m in got for m in marks) and got.strip() else source
        progress(i + 1, len(todo))
    os.makedirs(DIR, exist_ok=True)
    with open(path(code), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out
