"""
Conversación traducida en los dos sentidos.

- ÉL: lo que dice la otra persona se transcribe palabra a palabra y se muestra
  traducido a mi idioma.
- YO: lo que digo por el micrófono se transcribe, se traduce a su idioma y se
  dice en voz alta (Piper) por la salida elegida.

Modos:
- Llamada: ÉL llega por los altavoces (loopback) y YO por el micrófono. Mi voz
  traducida se manda a un micrófono virtual (p. ej. VB-Audio Virtual Cable)
  que se elige como micrófono en Teams/Zoom. Se dice frase a frase.
- Presencial: los dos entramos por el micrófono y se distingue quién habla por
  el idioma. Mi voz traducida sale por los altavoces al terminar de hablar.

Para no traducir lo que no toca: en llamada el micrófono queda libre (se puede
hablar a la vez que él) y solo se silencia, mientras habla él o suena mi voz
traducida, si se detecta que oye los altavoces (LeakDetector). En presencial
se silencia mientras suena mi voz traducida por los altavoces.
"""
import copy
import json
import os
import queue
import re
import threading
import time
from datetime import datetime

import numpy as np
from scipy.signal import fftconvolve, resample_poly

import interfaz
from eco import EchoCanceller
from interfaz import N_, T
from motores import create_engine
from traductor import (BASE_DIR, HALLUCINATIONS, MODELS_DIR, MT_MODELS, AudioHub, Capture,
                       Translator, close_stream, input_devices, to_16k)
from transcriptor import Streamer
from voz import VOICES, Speaker, output_devices

CONFIG_PATH = os.path.join(BASE_DIR, "conversacion.json")
# Catálogo de idiomas de la conversación. Un idioma puede estar aquí si tiene:
# reconocimiento (Whisper: casi todos), traductor con el inglés en los dos
# sentidos (MT_MODELS en traductor.py; así se combina con cualquier otro) y,
# para que se oiga mi voz traducida en él, una voz (VOICES en voz.py). Lo que
# falta se ve en Herramientas → Idiomas y en el desplegable ("sin voz").
# (Los nombres, en español: se traducen al mostrarlos con T.)
LANG_NAMES = {"es": N_("Español"), "en": N_("Inglés"), "fr": N_("Francés"), "de": N_("Alemán"),
              "it": N_("Italiano"), "ca": N_("Catalán"), "ru": N_("Ruso"), "zh": N_("Chino"),
              "vi": N_("Vietnamita"), "ar": N_("Árabe"), "pt": N_("Portugués")}
LANGS = list(LANG_NAMES)
# Todos los idiomas que reconoce Whisper (Herramientas → Idiomas los muestra y
# comprueba en Hugging Face si tienen traductor y voz para poder añadirlos).
WHISPER_LANGS = {
    "af": N_("Afrikáans"), "am": N_("Amárico"), "ar": N_("Árabe"), "as": N_("Asamés"), "az": N_("Azerí"), "ba": N_("Baskir"),
    "be": N_("Bielorruso"), "bg": N_("Búlgaro"), "bn": N_("Bengalí"), "bo": N_("Tibetano"), "br": N_("Bretón"),
    "bs": N_("Bosnio"), "ca": N_("Catalán"), "cs": N_("Checo"), "cy": N_("Galés"), "da": N_("Danés"), "de": N_("Alemán"),
    "el": N_("Griego"), "en": N_("Inglés"), "es": N_("Español"), "et": N_("Estonio"), "eu": N_("Euskera"), "fa": N_("Persa"),
    "fi": N_("Finés"), "fo": N_("Feroés"), "fr": N_("Francés"), "gl": N_("Gallego"), "gu": N_("Guyaratí"), "ha": N_("Hausa"),
    "haw": N_("Hawaiano"), "he": N_("Hebreo"), "hi": N_("Hindi"), "hr": N_("Croata"), "ht": N_("Criollo haitiano"),
    "hu": N_("Húngaro"), "hy": N_("Armenio"), "id": N_("Indonesio"), "is": N_("Islandés"), "it": N_("Italiano"),
    "ja": N_("Japonés"), "jw": N_("Javanés"), "ka": N_("Georgiano"), "kk": N_("Kazajo"), "km": N_("Jemer"), "kn": N_("Canarés"),
    "ko": N_("Coreano"), "la": N_("Latín"), "lb": N_("Luxemburgués"), "ln": N_("Lingala"), "lo": N_("Lao"), "lt": N_("Lituano"),
    "lv": N_("Letón"), "mg": N_("Malgache"), "mi": N_("Maorí"), "mk": N_("Macedonio"), "ml": N_("Malayalam"),
    "mn": N_("Mongol"), "mr": N_("Maratí"), "ms": N_("Malayo"), "mt": N_("Maltés"), "my": N_("Birmano"), "ne": N_("Nepalí"),
    "nl": N_("Neerlandés"), "nn": N_("Noruego nynorsk"), "no": N_("Noruego"), "oc": N_("Occitano"), "pa": N_("Panyabí"),
    "pl": N_("Polaco"), "ps": N_("Pastún"), "pt": N_("Portugués"), "ro": N_("Rumano"), "ru": N_("Ruso"), "sa": N_("Sánscrito"),
    "sd": N_("Sindi"), "si": N_("Cingalés"), "sk": N_("Eslovaco"), "sl": N_("Esloveno"), "sn": N_("Shona"), "so": N_("Somalí"),
    "sq": N_("Albanés"), "sr": N_("Serbio"), "su": N_("Sundanés"), "sv": N_("Sueco"), "sw": N_("Suajili"), "ta": N_("Tamil"),
    "te": N_("Telugu"), "tg": N_("Tayiko"), "th": N_("Tailandés"), "tk": N_("Turcomano"), "tl": N_("Tagalo"),
    "tr": N_("Turco"), "tt": N_("Tártaro"), "uk": N_("Ucraniano"), "ur": N_("Urdu"), "uz": N_("Uzbeko"),
    "vi": N_("Vietnamita"), "yi": N_("Yidis"), "yo": N_("Yoruba"), "zh": N_("Chino"), "yue": N_("Cantonés"),
}
# Idiomas añadidos desde Herramientas → Idiomas (catálogo propio, fuera del
# código): {código: {"name", "to": traductor en->código, "from": código->en,
# "voice": ruta de la voz de Piper o None}}.
USER_LANGS_PATH = os.path.join(BASE_DIR, "idiomas.json")


def register_language(code, info):
    """Da de alta un idioma en el catálogo en uso (desplegables, traductores, voz)."""
    LANG_NAMES[code] = info["name"]
    if code not in LANGS:
        LANGS.append(code)
    MT_MODELS[("en", code)] = info["to"]
    MT_MODELS[(code, "en")] = info["from"]
    if info.get("voice"):
        VOICES[code] = info["voice"]


def user_languages():
    try:
        with open(USER_LANGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_user_languages(langs):
    with open(USER_LANGS_PATH, "w", encoding="utf-8") as f:
        json.dump(langs, f, ensure_ascii=False, indent=2)


for _code, _info in user_languages().items():
    try:
        register_language(_code, _info)
    except (KeyError, TypeError):
        pass  # entrada estropeada: se ignora

# Los que no se pueden añadir sin más: por qué.
NEEDS_WORK = {
    **{c: N_("se escribe de derecha a izquierda (falta prepararlo como el árabe)")
       for c in ("fa", "he", "ur", "yi", "ps", "sd")},
    **{c: N_("no separa las palabras con espacios (hay que tratarlo como el chino)")
       for c in ("th", "lo", "my", "km", "bo", "yue")},
    "ja": N_("sin espacios, y su voz de Piper necesita componentes extra"),
    "ko": N_("su voz de Piper necesita componentes extra"),
}
# Bandera de cada idioma: franjas (horizontales o verticales) con su peso.
# Reino Unido, China y Vietnam se dibujan aparte. Windows no pinta los emojis de banderas.
FLAGS = {
    "es": ("h", [("#AA151B", 1), ("#F1BF00", 2), ("#AA151B", 1)]),
    "fr": ("v", [("#0055A4", 1), ("#FFFFFF", 1), ("#EF4135", 1)]),
    "de": ("h", [("#000000", 1), ("#DD0000", 1), ("#FFCE00", 1)]),
    "it": ("v", [("#009246", 1), ("#FFFFFF", 1), ("#CE2B37", 1)]),
    "ru": ("h", [("#FFFFFF", 1), ("#0039A6", 1), ("#D52B1E", 1)]),
    "ca": ("h", [("#FCDD09", 1), ("#DA121A", 1)] * 4 + [("#FCDD09", 1)]),  # la senyera
}


# Frase con la que se prueba la velocidad de cada modelo de Whisper (la dice la
# voz sintética en mi idioma y se transcribe).
TEST_PHRASES = {
    "es": "Mañana te mando la factura del mes pasado y luego hablamos del contrato con calma.",
    "en": "Tomorrow I will send you last month's invoice and then we can talk about the contract.",
    "fr": "Demain je t'envoie la facture du mois dernier et ensuite on parle du contrat.",
    "de": "Morgen schicke ich dir die Rechnung vom letzten Monat und dann sprechen wir über den Vertrag.",
    "it": "Domani ti mando la fattura del mese scorso e poi parliamo del contratto con calma.",
    "ca": "Demà t'envio la factura del mes passat i després parlem del contracte amb calma.",
    "ru": "Завтра я пришлю тебе счёт за прошлый месяц, а потом спокойно обсудим договор.",
    "zh": "明天我把上个月的发票发给你，然后我们再慢慢谈合同。",
    "vi": "Ngày mai tôi sẽ gửi cho bạn hóa đơn tháng trước, rồi chúng ta bàn về hợp đồng.",
    "ar": "غدًا سأرسل لك فاتورة الشهر الماضي، ثم نتحدث عن العقد بهدوء.",
    "pt": "Amanhã envio-te a fatura do mês passado e depois falamos do contrato com calma.",
}

# Herramientas → Atajos: (grupo, [(tecla o gesto, qué hace)]).
SHORTCUTS = [
    (N_("Ratón"), [
        (N_("Clic en una frase"), N_("La corriges ahí mismo, con el cursor en la palabra pulsada.")),
        (N_("Clic en zona vacía"), N_("Elige la última frase de la columna.")),
        (N_("Clic en el muñeco"), N_("Dice tu frase traducida (si está sonando, la corta).")),
        (N_("Clic en copiar"), N_("Copia la traducción de la frase al portapapeles.")),
    ]),
    (N_("Teclado al corregir o escribir"), [
        (N_("Al hablar"), N_("Con Auto scroll, lo que dices queda elegido mientras lo dices: corrígelo "
                             "con el teclado sin hacer clic. Al tocar una tecla deja de crecer.")),
        ("Enter", N_("Traduce la frase (y sigue elegida).")),
        (N_("Mayús + Enter"), N_("Traduce la frase y la dice, en tu columna (y sigue elegida).")),
        ("Esc", N_("Deshace la corrección o descarta la frase nueva, y la suelta.")),
        ("↑ / ↓", N_("Dentro de la frase; en su primera o última línea, pasa a la frase anterior o "
                     "siguiente (si la habías cambiado, se guarda y se traduce). ↓ en la última: "
                     "frase nueva.")),
        (N_("Inicio / Fin"), N_("Principio o final de la línea, sin salir de la frase.")),
        (N_("Mayús + flechas"), N_("Selecciona texto dentro de la frase.")),
    ]),
    (N_("Teclado en la columna"), [
        (N_("Escribir"), N_("Sin ninguna frase elegida (p. ej. al abrir), empieza una frase nueva al final.")),
        ("Ctrl + V", N_("Pega el portapapeles: en la frase que corriges o en una nueva.")),
        ("Ctrl + C", N_("Copia el texto seleccionado.")),
    ]),
]

# Ayuda de Herramientas → Llamadas: qué hacer para que el otro oiga mi voz traducida.
CALL_HELP = [
    (N_("1. Instala VB-Cable (una vez)"), [
        N_("Descárgalo: https://vb-audio.com/Cable/"),
        N_("Ejecuta VBCABLE_Setup_x64.exe como administrador y reinicia."),
    ]),
    (N_("2. En este programa"), [
        N_("\"Él suena por el PC\"."),
        N_("Se me oye por: tu micrófono."),
        N_("Se me escucha por: CABLE Input."),
    ]),
    (N_("3. En la llamada"), [
        N_("Micrófono: CABLE Output."),
        N_("Altavoz: tus auriculares o altavoces."),
        N_("Teams: Configuración → Dispositivos."),
        N_("Zoom: Configuración → Audio."),
        N_("WhatsApp Web: candado de la dirección → Micrófono."),
    ]),
    (N_("4. Para hablar"), [
        N_("Pulsa el muñeco de tu frase."),
    ]),
]


def flag_image(lang, w=96, h=64):
    """Bandera del idioma (imagen PIL grande: CTkImage la reduce sin perder nitidez)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "#012169")
    d = ImageDraw.Draw(img)
    if lang == "en":  # Union Jack simplificada
        d.line([(0, 0), (w, h)], fill="#FFFFFF", width=h // 5)
        d.line([(0, h), (w, 0)], fill="#FFFFFF", width=h // 5)
        d.line([(0, 0), (w, h)], fill="#C8102E", width=h // 15)
        d.line([(0, h), (w, 0)], fill="#C8102E", width=h // 15)
        d.rectangle([0, h * 2 // 6, w, h * 4 // 6], fill="#FFFFFF")
        d.rectangle([(w - h // 3) // 2, 0, (w + h // 3) // 2, h], fill="#FFFFFF")
        d.rectangle([0, h * 4 // 10, w, h * 6 // 10], fill="#C8102E")
        d.rectangle([(w - h // 5) // 2, 0, (w + h // 5) // 2, h], fill="#C8102E")
    elif lang == "zh":  # rojo con una estrella grande y cuatro pequeñas
        d.rectangle([0, 0, w, h], fill="#DE2910")
        star(d, w * 0.17, h * 0.27, h * 0.16, "#FFDE00")
        for x, y in ((0.33, 0.1), (0.4, 0.2), (0.4, 0.35), (0.33, 0.45)):
            star(d, w * x, h * y, h * 0.055, "#FFDE00")
    elif lang == "vi":  # rojo con una estrella amarilla en el centro
        d.rectangle([0, 0, w, h], fill="#DA251D")
        star(d, w / 2, h / 2, h * 0.32, "#FFFF00")
    elif lang == "pt":  # verde y rojo (2:3) con la esfera armilar amarilla en la unión
        d.rectangle([0, 0, w * 2 // 5, h], fill="#006600")
        d.rectangle([w * 2 // 5, 0, w, h], fill="#FF0000")
        r = h * 0.22
        d.ellipse([w * 2 / 5 - r, h / 2 - r, w * 2 / 5 + r, h / 2 + r], fill="#FFCC00")
    elif lang in FLAGS:
        direction, stripes = FLAGS[lang]
        total, pos = sum(n for _, n in stripes), 0
        side = h if direction == "h" else w
        for color, n in stripes:
            a, b = pos * side // total, (pos + n) * side // total
            d.rectangle([0, a, w, b] if direction == "h" else [a, 0, b, h], fill=color)
            pos += n
    else:  # idioma sin bandera dibujada (añadido después): una etiqueta con su código
        from PIL import ImageFont
        d.rectangle([0, 0, w, h], fill="#3a4250")
        try:
            font = ImageFont.truetype("seguisb.ttf", int(h * 0.55))
        except OSError:
            font = ImageFont.load_default(size=int(h * 0.55))
        d.text((w / 2, h / 2), lang.upper(), fill="#ffffff", font=font, anchor="mm")
    d.rectangle([0, 0, w - 1, h - 1], outline="#555555", width=2)  # se ve el blanco sobre fondo oscuro
    return img


def star(d, cx, cy, r, color):
    """Estrella de cinco puntas centrada en (cx, cy) y de radio r."""
    import math
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * 0.4
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    d.polygon(pts, fill=color)


def talk_icon(color, size):
    """Muñeco hablando (cabeza, hombros y ondas de voz) de size x size píxeles."""
    from PIL import Image, ImageDraw
    s = 64  # se dibuja grande y se reduce: bordes suaves
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 32, 32], fill=color)                    # cabeza
    d.ellipse([0, 38, 40, 82], fill=color)                   # hombros (se corta abajo)
    for r in (12, 21, 30):                                   # ondas de voz
        d.arc([34 - r, 20 - r, 34 + r, 20 + r], -40, 40, fill=color, width=5)
    return img.resize((size, size), Image.LANCZOS)


def gear_icon(color, size):
    """Engranaje (herramientas) de size x size píxeles."""
    import math
    from PIL import Image, ImageDraw
    s = 96
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r_out, r_in, teeth = s / 2, 44, 33, 8
    pts = []
    for i in range(teeth * 4):  # cada diente: subida, arriba, bajada, abajo
        a = 2 * math.pi * i / (teeth * 4)
        r = r_out if i % 4 in (1, 2) else r_in
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    d.polygon(pts, fill=color)
    d.ellipse([c - 14, c - 14, c + 14, c + 14], fill=(0, 0, 0, 0))  # el agujero del centro
    return img.resize((size, size), Image.LANCZOS)


def trash_icon(color, size):
    """Papelera (borrar la frase) de size x size píxeles."""
    from PIL import Image, ImageDraw
    s = 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([24, 4, 40, 10], fill=color)                  # asa
    d.rectangle([8, 10, 56, 16], fill=color)                  # tapa
    d.polygon([(12, 20), (52, 20), (48, 60), (16, 60)], fill=color)  # cubo
    for x in (24, 32, 40):                                    # ranuras
        d.line([(x, 26), (x, 54)], fill=(0, 0, 0, 0), width=4)
    return img.resize((size, size), Image.LANCZOS)


def copy_icon(color, size):
    """Dos hojas superpuestas (copiar) de size x size píxeles."""
    from PIL import Image, ImageDraw
    s = 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([4, 4, 40, 46], radius=6, outline=color, width=6)       # la de detrás
    d.rounded_rectangle([20, 18, 58, 60], radius=6, fill=(0, 0, 0, 0))         # la tapa la de delante
    d.rounded_rectangle([20, 18, 58, 60], radius=6, outline=color, width=6)
    return img.resize((size, size), Image.LANCZOS)
ME, THEM = "yo", "el"
REMOTE_HOLD = 0.4     # tras oír al otro, segundos que el micrófono sigue bloqueado
GATE_DELAY = 0.25     # retraso de cada entrada para decidir con algo de margen
VAD_WINDOW = 8192     # audio (16 kHz) que mira el detector de voz: ~0,5 s
VAD_THRESHOLD = 0.5   # probabilidad de voz de Silero VAD
VAD_HANGOVER = 0.4    # se deja pasar esto tras la última voz (pausas entre palabras)
LEAK_WINDOW = 0.5     # audio del micrófono que se compara con lo que sonó por los altavoces
LEAK_SEARCH = 0.6     # margen de retardo altavoz -> micrófono (relojes distintos)
LEAK_CORR = 0.3       # por encima, es sonido colado (voz sin relación: máx. medido 0,29)
LEAK_HOLD = 30.0      # tras una fuga, segundos que el micrófono se bloquea al sonar algo
VIRTUAL_OUT = ("cable", "voicemeeter", "virtual", "vb-audio")
EDIT_HOLD = 4.0       # tras un clic o tecla al corregir, segundos sin auto scroll en esa columna
WAVE_W, WAVE_H = 160, 26  # gráfico de la onda de cada columna (2 px por pico de 25 ms: 2 s)

DEFAULTS = {
    "modo": "llamada",        # llamada | presencial
    "yo": "es",
    "el": "en",
    "microfono": "",          # nombre del dispositivo ("" = el de por defecto)
    "salida_voz": "",
    "umbral_mic": 0.008,
    "umbral_altavoces": 0.008,
    "modelo": "base",
    "autoscroll": True,       # las columnas bajan solas a lo último que se dice
    "letra": 16,
    "interfaz": "es",         # idioma de la ventana (Herramientas → Idiomas)
    "mi_voz": "",             # voz grabada con que suena mi traducción ("" = la de Piper)
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    # Un idioma que ya no está en el catálogo (p. ej. se quitó): el de siempre.
    for key in ("yo", "el"):
        if cfg[key] not in LANGS:
            cfg[key] = DEFAULTS[key]
    if not interfaz.available(cfg["interfaz"]):
        cfg["interfaz"] = DEFAULTS["interfaz"]
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# Palabras muy frecuentes de cada idioma, para ver en qué idioma está un texto.
STOPWORDS = {
    "es": set("el la los las de del que y en un una es por con para no se lo al mi "
              "su yo tú pero como más muy este esta eso ya hay son está están".split()),
    "en": set("the and to of is we you that it are not this with for be on have has "
              "was were they he she i my your our will can do what there".split()),
    "fr": set("le la les de des du que et en un une est pour avec pas ne se ce il elle "
              "je tu nous vous ils mais comme plus très cette oui sont qui dans sur".split()),
    "de": set("der die das und ist nicht ich du wir sie es ein eine zu mit auf für den "
              "dem von aber auch noch wie sind was hat haben bin schon".split()),
    "it": set("il lo la gli le di del che e è un una per con non si io tu noi voi "
              "sono ma come più molto questo questa anche mi ci ho hai".split()),
    "ca": set("el la els les de del que i en un una és per amb no es ho al meu teu "
              "però com més molt aquest aquesta això ja hi ha són està jo".split()),
    "vi": set("và của là có không cho được trong một những các người này với tôi "
              "bạn đã sẽ thì mà rất cũng như khi".split()),
    # Solo las que lo distinguen del español (de, que, se... son de los dos).
    "pt": set("o não você vocês é os as do da dos das em um uma com mas muito isso isto "
              "eu nós eles ela ele são estão foi tem ao à pelo pela também já sim aqui".split()),
}
# Escritura de los idiomas que no usan el alfabeto latino (para distinguirlos
# sin lista de palabras: basta con ver qué letras usa el texto).
SCRIPTS = {**{c: "[Ѐ-ӿ]" for c in ("ru", "uk", "bg", "mk", "sr", "be", "kk")},  # cirílico
           "zh": "[㐀-鿿]", "ar": "[؀-ۿݐ-ݿ]"}
# Letras de las escrituras de derecha a izquierda (árabe, hebreo...).
RTL_LETTERS = "[֐-ࣿיִ-﷿ﹰ-﻿]"


def right_to_left(text):
    """¿El texto está sobre todo en una escritura de derecha a izquierda? (se
    alinea a la derecha en la columna)."""
    letters = re.findall(r"[^\W\d_]", text)
    return bool(letters) and len(re.findall(RTL_LETTERS, text)) > len(letters) / 2


def bidi(text):
    """Etiqueta extra para pintar el texto: ("rtl",) si va de derecha a izquierda."""
    return ("rtl",) if right_to_left(text) else ()


def other_language(text, lang, other):
    """¿El texto está claramente en `other` y no en `lang`? (p. ej. el vídeo
    en inglés colado en el micrófono de alguien que habla español)."""
    if lang == other:
        return False
    if lang in SCRIPTS or other in SCRIPTS:
        # Escrituras distintas: cuenta qué parte de las letras es de cada una.
        letters = re.findall(r"[^\W\d_]", text)
        if len(letters) < 3:
            return False
        def share(code):
            return (len(re.findall(SCRIPTS[code], text)) if code in SCRIPTS
                    else sum(1 for c in letters if c.isascii() or "À" <= c <= "ɏ"
                             or "Ḁ" <= c <= "ỿ")) / len(letters)
        return share(other) > 0.7 and share(lang) < 0.3
    if lang not in STOPWORDS or other not in STOPWORDS:
        return False
    words = re.findall(r"[^\W\d_]+", text.lower())  # letras de cualquier idioma; l'homme -> l, homme
    mine = sum(w in STOPWORDS[lang] for w in words)
    theirs = sum(w in STOPWORDS[other] for w in words)
    return len(words) >= 3 and theirs >= 2 and theirs > 2 * mine


# --------------------------------------------------------------------------- #
# Comprobar en Hugging Face qué tiene un idioma (Herramientas → Idiomas)
# --------------------------------------------------------------------------- #
HF_API = "https://huggingface.co/api/models"
_piper_voices = {}  # se descarga una vez: catálogo de voces de Piper
_trusted = set()    # traductores de quienes ya usamos (todos completos): una consulta


class RateLimited(Exception):
    """Hugging Face limita las consultas desde esta conexión (sin cuenta)."""
    def __init__(self, seconds):
        super().__init__(f"espera {seconds} s")
        self.seconds = seconds


def hf_get(url, **params):
    import requests
    r = requests.get(url, params=params or None, timeout=15)
    if r.status_code == 429:
        m = re.search(r"t=(\d+)", r.headers.get("RateLimit", ""))
        raise RateLimited(int(m.group(1)) if m else 60)
    return r


def ct2_translator(a, b):
    """Repositorio de Hugging Face con un traductor Opus-MT a -> b ya convertido
    a CTranslate2 (con model.bin y sus .spm), o None si no hay. Si Hugging Face
    limita las consultas, lanza RateLimited (no se da por hecho que no exista)."""
    if not _trusted:
        for author in ("ooeoeo", "michaelfeil"):
            r = hf_get(HF_API, author=author, limit=1000)
            _trusted.update(m["id"] for m in r.json()) if r.ok else None
    for repo in (f"ooeoeo/opus-mt-{a}-{b}-ct2-float16", f"michaelfeil/ct2fast-opus-mt-{a}-{b}"):
        if repo in _trusted:
            return repo

    def complete(repo):
        r = hf_get(f"{HF_API}/{repo}")
        files = {s["rfilename"] for s in r.json().get("siblings", [])} if r.ok else set()
        return {"model.bin", "source.spm", "target.spm"} <= files

    # Si no, otras conversiones con ese par exacto (comprobando que estén completas).
    r = hf_get(HF_API, search=f"opus-mt-{a}-{b}", limit=50)
    for m in r.json() if r.ok else []:
        repo = m.get("id", "")
        if (re.search(rf"opus-mt-{a}-{b}(\b|[-_])", repo) and re.search(r"ct2|ctranslate", repo, re.I)
                and "android" not in repo and complete(repo)):
            return repo
    return None


def piper_voice(code):
    """Nombre de una voz de Piper para el idioma (la de calidad más alta), o None."""
    import requests
    if not _piper_voices:
        r = requests.get("https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json", timeout=20)
        _piper_voices.update(r.json())
    order = {"high": 0, "medium": 1, "low": 2, "x_low": 3}
    found = [(order.get(v.get("quality"), 9), k) for k, v in _piper_voices.items()
             if k.split("_")[0] == code]
    return min(found)[1] if found else None


def check_language(code):
    """Lo que tiene un idioma en Hugging Face para poder conversar en él."""
    try:
        return {"to": ct2_translator("en", code), "from": ct2_translator(code, "en"),
                "voice": piper_voice(code)}
    except RateLimited as e:
        return {"error": T("Hugging Face limita las consultas: espera {s} s", s=e.seconds),
                "wait": e.seconds}
    except Exception as e:
        return {"error": T("sin conexión ({error})", error=type(e).__name__)}


def install_language(code, found, log=print):
    """Añade un idioma comprobado: descarga sus traductores (y su voz, si hay),
    lo prueba (traduce una frase de ida y vuelta y genera la voz) y, si va bien,
    lo da de alta y lo guarda en idiomas.json. Devuelve (ida, vuelta)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    from voz import VOICES_REPO
    log(T("descargando traductores…"))
    for repo in (found["to"], found["from"]):
        snapshot_download(repo, cache_dir=MODELS_DIR)
    voice = None
    if found.get("voice"):
        log(T("descargando la voz…"))
        if not _piper_voices:
            piper_voice(code)  # baja el catálogo de voces
        onnx = next(k for k in _piper_voices[found["voice"]]["files"] if k.endswith(".onnx"))
        for ext in ("", ".json"):
            hf_hub_download(VOICES_REPO, onnx + ext, cache_dir=MODELS_DIR)
        voice = onnx[:-len(".onnx")]
    info = {"name": WHISPER_LANGS.get(code, code), "to": found["to"], "from": found["from"], "voice": voice}
    register_language(code, info)
    try:
        log(T("probando…"))
        sample = "Good morning. I will send you the invoice tomorrow."
        there_t, back_t = Translator(log, "en", code), Translator(log, code, "en")
        if there_t.mt is None or back_t.mt is None:  # no vale Google: tiene que ir sin internet
            raise RuntimeError(T("el traductor no se pudo cargar"))
        there = there_t(sample)
        back = back_t(there or "")
        if not there or not back or there.strip() == sample:
            raise RuntimeError(T("la traducción de prueba salió vacía"))
        if voice:
            audio, sr = Speaker(log=lambda *_: None).synthesize(there, code)
            if len(audio) < sr * 0.3:
                raise RuntimeError(T("la voz no generó audio"))
    except Exception:
        unregister_language(code)
        raise
    langs = user_languages()
    langs[code] = info
    save_user_languages(langs)
    return there, back


def unregister_language(code):
    """Quita un idioma del catálogo en uso (sus modelos se quedan en "modelos")."""
    LANG_NAMES.pop(code, None)
    if code in LANGS:
        LANGS.remove(code)
    MT_MODELS.pop(("en", code), None)
    MT_MODELS.pop((code, "en"), None)
    VOICES.pop(code, None)


def device_index(devices, name):
    for i, n in devices:
        if n == name:
            return i
    return None  # el de por defecto


def is_virtual(name):
    return any(k in name.lower() for k in VIRTUAL_OUT)


def speaker_loopback():
    """(índice del loopback que se captura para oírle a él o None para el de
    la salida por defecto, nombre de la salida por defecto de Windows).

    Nunca un cable virtual: por ahí sale mi voz traducida hacia Teams y se
    tomaría como suya. Pasa si Windows pone el cable como salida por defecto
    (lo hace a veces al instalar VB-Cable): entonces se capturan los primeros
    altavoces de verdad."""
    import pyaudiowpatch as pa

    def pick(p):
        wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
        default = p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
        if not is_virtual(default):
            return None, default
        for d in p.get_loopback_device_info_generator():
            if not is_virtual(d["name"]):
                return d["index"], default
        raise RuntimeError(T("La salida de sonido de Windows es {device} y no hay altavoces que "
                             "capturar: elige tus altavoces en Windows.", device=default))
    return AudioHub.get().call(pick)


# --------------------------------------------------------------------------- #
# Lógica (sin ventana)
# --------------------------------------------------------------------------- #
class Gate(threading.Thread):
    """Filtro entre una captura y su Streamer. Cambia por silencio:
    - lo que no es voz humana (golpes, teclado, clics...), según Silero VAD;
    - lo que llega mientras blocked(t) es cierto (el otro habla, suena mi voz
      traducida...), siendo t el momento en que llegó el bloque.

    Retiene cada bloque GATE_DELAY segundos antes de decidir: así no se corta
    la primera sílaba (el detector tarda un poco en reconocer la voz) y sabe
    si el otro canal tuvo voz justo en ese momento (su eco llega al micrófono a
    la vez que a los altavoces, no antes)."""

    def __init__(self, src_q, dst_q, rate, blocked=lambda t: False, echo=None,
                 tap=None, leak=None):
        super().__init__(daemon=True)
        self.src_q, self.dst_q, self.rate = src_q, dst_q, rate
        self.blocked = blocked
        self.echo = echo  # EchoCanceller: quita mi voz traducida antes que nada
        self.tap = tap    # recibe todo lo que llega (altavoces -> detector de fugas)
        self.leak = leak  # ¿es sonido colado de los altavoces? (micrófono)
        self.speech = []  # momentos (monotonic) en que se detectó voz
        # Último medio segundo a 16 kHz (empieza en silencio para que el
        # detector funcione desde el primer bloque).
        self.ring = np.zeros(VAD_WINDOW, np.float32)
        self.stopped = False
        from faster_whisper.vad import get_vad_model
        self.vad = get_vad_model()

    def heard_between(self, t0, t1):
        return any(t0 <= t <= t1 for t in reversed(self.speech))

    def _is_speech(self, block):
        self.ring = np.concatenate([self.ring, to_16k(block, self.rate() or 16000)])[-VAD_WINDOW:]
        if len(self.ring) < VAD_WINDOW:
            return False
        probs = self.vad(self.ring)  # una probabilidad cada 32 ms
        return float(np.max(probs[-4:])) > VAD_THRESHOLD  # el último ~0,13 s

    def run(self):
        held = []
        while not self.stopped:
            try:
                t, block = self.src_q.get(timeout=0.05)
                now = time.monotonic()
                if self.tap is not None and block.size:
                    self.tap(block, self.rate() or 16000, t or now)
                if self.echo is not None and block.size:
                    # t: cuándo llegó el bloque a la captura (0 en las pruebas simuladas)
                    block, _ = self.echo.process(block, t or now, self.rate() or 16000)
                if block.size and np.any(block) and self._is_speech(block):
                    self.speech.append(now)
                    self.speech = [s for s in self.speech if s > now - 5]
                held.append((now, t, block))
            except queue.Empty:
                pass
            while held and time.monotonic() - held[0][0] >= GATE_DELAY:
                arrived, t, block = held.pop(0)
                voice = self.heard_between(arrived - VAD_HANGOVER, arrived + GATE_DELAY)
                if not voice or self.blocked(arrived):
                    block = np.zeros_like(block)
                elif self.leak is not None and self.leak(block, self.rate() or 16000, t or arrived):
                    block = np.zeros_like(block)  # sonido colado de los altavoces
                self.dst_q.put((t, block))


class LeakDetector:
    """¿El micrófono oye lo que suena por los altavoces? (típico con el
    micrófono de una webcam y un altavoz, o auriculares con mucho volumen).

    Se compara cada trozo de LEAK_WINDOW del micrófono con lo que ha sonado
    por los altavoces justo antes (captura de altavoces). Micrófono y altavoz
    van con relojes distintos, así que se busca en un margen amplio; con 0,5 s
    de audio una voz sin relación no pasa de ~0,3 y la fuga sí (medido con una
    webcam y un altavoz Bluetooth: 87 % de los trozos con fuga, 0 falsos)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.loop = []          # (llegada, muestras 16 kHz filtradas) de los altavoces
        self.mic = []           # lo mismo del micrófono
        self.leak_until = 0.0   # hasta cuándo se considera que el micrófono oye los altavoces
        from scipy.signal import butter
        self.sos = butter(4, [200, 4000], btype="band", fs=16000, output="sos")

    def _prep(self, block, rate):
        from scipy.signal import sosfilt
        return sosfilt(self.sos, to_16k(block, rate)).astype(np.float64)

    def add_loop(self, block, rate, t):
        with self.lock:
            self.loop.append((t, self._prep(block, rate)))
            self.loop = [x for x in self.loop if x[0] > t - 3.0]

    @staticmethod
    def _span(blocks, a, b):
        """Audio entre los instantes a y b colocando cada bloque según su llegada."""
        out = np.zeros(int((b - a) * 16000))
        for t, x in blocks:
            s = int((t - len(x) / 16000 - a) * 16000)
            lo, hi = max(s, 0), min(s + len(x), len(out))
            if hi > lo:
                out[lo:hi] = x[lo - s:hi - s]
        return out

    def check_mic(self, block, rate, t):
        """¿Este trozo del micrófono es sonido colado de los altavoces?"""
        x = self._prep(block, rate)
        with self.lock:
            self.mic.append((t, x))
            self.mic = [m for m in self.mic if m[0] > t - LEAK_WINDOW - 0.2]
            loop = list(self.loop)
            mic = list(self.mic)
        y = self._span(mic, t - LEAK_WINDOW, t)
        if np.sqrt(np.mean(y ** 2)) < 2e-3:
            return False  # el micrófono no capta nada
        w = self._span(loop, t - LEAK_WINDOW - LEAK_SEARCH, t + 0.2)
        if np.sqrt(np.mean(w ** 2)) < 1e-3:
            return False  # los altavoces no han sonado
        c = np.abs(fftconvolve(w, y[::-1], mode="valid"))
        e = np.concatenate([[0.0], np.cumsum(w ** 2)])
        norm = np.sqrt(np.maximum(e[len(y):len(y) + len(c)] - e[:len(c)], 1e-12) * np.dot(y, y))
        if float((c / norm).max()) > LEAK_CORR:
            self.leak_until = time.monotonic() + LEAK_HOLD
            return True
        return False

    def active(self):
        return time.monotonic() < self.leak_until


class Entry:
    def __init__(self, num, side, lang, target):
        self.num, self.side, self.lang, self.target = num, side, lang, target
        self.text = ""        # confirmado (blanco en el transcriptor)
        self.tentative = ""   # provisional (gris)
        self.translation = ""
        self.final = False


class Channel(threading.Thread):
    """Convierte los eventos de un Streamer en entradas de la conversación:
    las numera, las traduce sobre la marcha y, si son mías, las dice en voz alta."""

    def __init__(self, conv, streamer_q, side=None, lang=None):
        super().__init__(daemon=True)
        self.conv, self.q = conv, streamer_q
        self.side, self.lang = side, lang  # fijos (llamada) o por idioma (presencial)
        self.entry = None
        self.stopped = False

    def run(self):
        while not self.stopped:
            try:
                ev = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.handle(ev)
            except Exception as e:
                self.conv.status(T("Error: {error}", error=e))

    def handle(self, ev):
        conv, kind = self.conv, ev[0]
        if kind == "status":
            conv.status(ev[1])
        elif kind == "head":
            lang = ev[2] or self.lang
            side = self.side or (ME if lang == conv.me else THEM)
            # Lo suyo se lee en mi idioma y lo mío se dice en el suyo (si ya
            # está en ese idioma, no se traduce).
            target = conv.them if side == ME else conv.me
            self.entry = Entry(conv.next_number(), side, lang, target)
        elif kind == "words" and self.entry is not None:
            e = self.entry
            e.text += ev[1]
            e.tentative = ev[2]
            if ev[1]:
                e.translation = conv.translate(e.text, e.lang, e.target)
            conv.show(e)
        elif kind == "para" and self.entry is not None:
            e, self.entry = self.entry, None
            e.final, e.tentative = True, ""
            if not e.text.strip() or e.text.strip().lower() in HALLUCINATIONS:
                conv.ui_q.put(("discard", e.side))
                return
            if self.foreign(e):
                # En "Él suena por el PC" yo hablo mi idioma: otro idioma en mi
                # micrófono es sonido colado (el vídeo, la llamada...).
                conv.ui_q.put(("discard", e.side))
                return
            # Mi voz traducida no se dice sola: solo al pulsar el icono de la frase.
            e.translation = conv.translate(e.text, e.lang, e.target)
            conv.show(e)
            conv.write_log(e)

    def foreign(self, e):
        """¿Algo mío (llamada) que en realidad está en el otro idioma?"""
        return (e.side == ME and self.conv.mode == "llamada"
                and other_language(e.text, e.lang, e.target))


class Conversation:
    """Arranca capturas, transcripción, traducción y voz según la configuración.
    Manda a ui_q: ("entry", Entry), ("status", texto), ("speaking", número|None),
    ("ready", nombre_motor)."""

    def __init__(self, cfg, ui_q):
        self.cfg, self.ui_q = cfg, ui_q
        self.engine = None
        self.speaker = None
        self.translators = {}
        self.tr_lock = threading.Lock()
        self.num_lock = threading.Lock()
        self.number = 0
        self.parts = []          # hilos y capturas de la sesión en marcha
        self.streamers = {}      # ME/THEM -> Streamer (para los medidores)
        # El filtro aprendido se conserva entre reinicios (depende del equipo).
        self.echo = EchoCanceller()
        self.echo_on = False
        self.leak = LeakDetector()
        self.loop_device = None
        os.makedirs(os.path.join(BASE_DIR, "transcripciones"), exist_ok=True)
        self.log_path = os.path.join(
            BASE_DIR, "transcripciones",
            datetime.now().strftime("conversacion_%Y-%m-%d_%H%M.txt"))

    # -- utilidades para los canales --------------------------------------- #
    @property
    def me(self):
        return self.cfg["yo"]

    @property
    def them(self):
        return self.cfg["el"]

    @property
    def mode(self):
        return self.cfg["modo"]

    def status(self, msg):
        self.ui_q.put(("status", msg))

    def next_number(self):
        with self.num_lock:
            self.number += 1
            return self.number

    def translator(self, src, tgt):
        with self.tr_lock:
            if (src, tgt) not in self.translators:
                self.translators[(src, tgt)] = Translator(self.status, source=src, target=tgt)
            return self.translators[(src, tgt)]

    def translate(self, text, src, tgt):
        text = text.strip()
        if not text or src == tgt:
            return text
        return self.translator(src, tgt)(text) or ""

    def show(self, entry):
        # Copia: la entrada sigue cambiando mientras la ventana pinta esta versión.
        self.ui_q.put(("entry", copy.copy(entry)))

    def speak(self, entry, text):
        """Dice mi frase traducida (al pulsar su icono)."""
        if not text or self.speaker is None:
            return
        num = entry.num
        self.speaker.say(text, entry.target,
                         on_start=lambda: self.ui_q.put(("speaking", num)),
                         on_end=lambda: self.ui_q.put(("spoken", num, True)),
                         on_fail=lambda: self.ui_q.put(("spoken", num, False)),
                         tag=num)

    def stop_speaking(self, num):
        """Corta mi frase traducida num si es la que está sonando."""
        if self.speaker is not None:
            self.speaker.stop(num)

    def forget_speech(self, num):
        """La frase num se ha borrado: ni sigue sonando ni suena si estaba en cola."""
        if self.speaker is not None:
            self.speaker.forget(num)
            self.speaker.stop(num)

    def write_note(self, text):
        """Una línea suelta en el registro de la conversación (p. ej. una frase borrada)."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")

    def write_log(self, e, corrected=False):
        who = "YO" if e.side == ME else "ÉL"
        note = " (corregida)" if corrected else ""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{e.num} [{who}] [{e.lang.upper()}]{note} {e.text.strip()}\n"
                    f"    -> [{e.target.upper()}] {e.translation.strip()}\n\n")

    def levels(self):
        return {side: st.level for side, st in self.streamers.items()}

    def waves(self):
        """Lado -> picos de los últimos segundos del audio que se transcribe."""
        return {side: list(st.wave) for side, st in self.streamers.items()}

    # -- arranque ---------------------------------------------------------- #
    def load(self):
        """Carga lo que tarda (motor, traductores, voz). Una sola vez."""
        self.engine = create_engine("auto", self.cfg["modelo"], log=self.status)
        self.status(T("Cargando traductores..."))
        for a, b in ((self.me, self.them), (self.them, self.me)):
            if a != b:
                self.translator(a, b)
        self.speaker = Speaker(log=self.status)
        self.speaker.listeners.append(self._voice_started)
        self.speaker.cut_listeners.append(self.echo.cut_reference)
        self.speaker.start()
        self.speaker.preload(self.them)
        if self.cfg.get("mi_voz"):
            try:
                self.set_my_voice(self.cfg["mi_voz"])
            except Exception as e:
                self.status(T("Mi voz no está disponible ({error}): suena la voz de Piper", error=e))
        self.ui_q.put(("ready", self.engine.name))

    def set_my_voice(self, name):
        """Mi traducción suena con mi voz grabada `name` ("" = la voz de Piper).
        La primera vez carga el conversor (unos segundos)."""
        if not name:
            self.speaker.set_style(None)
            return
        import mivoz
        if not mivoz.components_ready() or name not in mivoz.profiles():
            raise RuntimeError(T("falta PyTorch o la voz «{name}»", name=name))
        self.speaker.set_style(mivoz.MyVoice(name, self.speaker.synthesize, VOICES, TEST_PHRASES,
                                             log=self.status))

    def prepare_speech(self, entry):
        """Con mi voz, la frase traducida se convierte ya (tarda ~2 s) para que
        al pulsar el muñeco suene al momento."""
        text = entry.translation.strip()
        if self.speaker is not None and entry.side == ME and text and text != "…":
            self.speaker.prepare(text, entry.target)

    def _voice_started(self, samples, out_start, rate, device_name, epoch, t0):
        """Empieza a sonar una frase de mi voz traducida: si sale por los
        altavoces que se capturan, el cancelador la tiene que restar."""
        if not self.echo_on:
            return
        if self.loop_device is None or device_name in self.loop_device:
            self.echo.add_reference(samples, out_start, rate, epoch, t0)

    def voice_audible(self):
        """¿Mi voz traducida sale por unos altavoces (y la pueden oír los micrófonos
        o la captura de altavoces)? No, si va a un micrófono virtual."""
        return not is_virtual(self.cfg["salida_voz"])

    def start(self, mic_q=None, loop_q=None):
        """Arranca la sesión. mic_q/loop_q permiten pruebas con audio simulado."""
        cfg = self.cfg
        mic_idx = device_index(input_devices(), cfg["microfono"]) if mic_q is None else None
        out_idx = device_index(output_devices(), cfg["salida_voz"])
        self.speaker.set_device(out_idx)

        def capture(mic, index, q):
            if q is not None:
                return q, None
            q = queue.Queue()
            cap = Capture(q, index, self.status, mic=mic)
            cap.start()
            cap.ready.wait(10)
            if cap.error:
                raise RuntimeError(T("No se pudo abrir el micrófono: {error}", error=cap.error) if mic
                                   else T("No se pudo abrir los altavoces: {error}", error=cap.error))
            return q, cap

        def pipeline(src_q, cap, rate, langs, threshold, blocked, side, lang, echo=None,
                     tap=None, leak=None):
            gate_q, st_q = queue.Queue(), queue.Queue()
            gate = Gate(src_q, gate_q, rate, blocked, echo, tap, leak)
            st = Streamer(self.engine, langs, gate_q, rate, threshold, st_q)
            ch = Channel(self, st_q, side, lang)
            for t in (gate, st, ch):
                t.start()
            self.parts += [x for x in (cap, gate, st, ch) if x is not None]
            return gate, st

        mq, mcap = capture(True, mic_idx, mic_q)
        mic_rate = (lambda: mcap.rate) if mcap else (lambda: 16000)
        warning = ""
        if self.mode == "llamada":
            loop_idx = None
            if loop_q is None:
                loop_idx, default_out = speaker_loopback()
                if loop_idx is not None:
                    warning = " · ⚠ " + T("la salida de Windows es {device}: pon tus altavoces "
                                          "(si no, no le oirás)", device=default_out)
            lq, lcap = capture(False, loop_idx, loop_q)
            loop_rate = (lambda: lcap.rate) if lcap else (lambda: 16000)
            holder = {}

            def mic_blocked(t):
                # Libre (como con auriculares) salvo que se haya comprobado
                # que el micrófono oye los altavoces: entonces se bloquea
                # mientras suena algo por ellos.
                if not self.leak.active():
                    return False
                other = holder["loop"].heard_between(t - REMOTE_HOLD, t + GATE_DELAY)
                return other or (self.voice_audible() and self.speaker.speaking())

            # Mi voz traducida por los altavoces no es él hablando: se resta de
            # la captura (cancelación de eco, que silencia el trozo si no puede
            # restarla con seguridad). Por un cable virtual no llega a la captura.
            use_echo = self.voice_audible()
            self.loop_device = lcap.device_name if lcap else None
            self.echo.reset_capture()  # captura nueva: su cuenta de muestras empieza de cero
            self.echo_on = use_echo

            # Él: se detecta el idioma (primero el suyo). Con el idioma fijo, si
            # hablara en otro, Whisper lo traduciría en vez de transcribirlo.
            them_langs = [self.them] + ([self.me] if self.me != self.them else [])
            holder["loop"], self.streamers[THEM] = pipeline(
                lq, lcap, loop_rate, them_langs, cfg["umbral_altavoces"],
                lambda t: False, THEM, self.them, echo=self.echo if use_echo else None,
                tap=self.leak.add_loop)
            _, self.streamers[ME] = pipeline(
                mq, mcap, mic_rate, [self.me], cfg["umbral_mic"], mic_blocked, ME, self.me,
                leak=self.leak.check_mic)
        else:
            # Presencial: un solo micrófono para los dos; decide el idioma.
            langs = [self.me] + ([self.them] if self.them != self.me else [])
            _, st = pipeline(mq, mcap, mic_rate, langs, cfg["umbral_mic"],
                             lambda t: self.voice_audible() and self.speaker.speaking(),
                             None, self.me)
            self.streamers[ME] = st
        where = T("altavoces + micrófono") if self.mode == "llamada" else T("micrófono (los dos)")
        self.status(T("Escuchando") + f" · {self.engine.name} · {where}{warning}")

    def stop(self):
        for p in self.parts:
            try:
                p.stop()
            except AttributeError:
                p.stopped = True
        # Esperar a que los dispositivos se cierren de verdad: si el programa
        # acaba con una captura abierta, Python se cierra con un error fatal.
        for p in self.parts:
            if isinstance(p, Capture):
                p.join(timeout=2)
        self.parts, self.streamers = [], {}

    def restart(self):
        self.stop()
        self.start()

    def change_model(self, modelo):
        """Cambia el modelo de Whisper: carga el nuevo (se descarga si hace
        falta) y, si va bien, sustituye al anterior y vuelve a escuchar."""
        new = create_engine("auto", modelo, log=self.status)
        self.stop()
        old, self.engine = self.engine, new
        self.cfg["modelo"] = modelo
        if old is not None:
            old.close()
        self.start()

    def test_model(self, modelo, log=print):
        """Mide un modelo de Whisper con una frase de prueba en mi idioma, dicha
        por la voz sintética: (segundos por pasada, duración de la frase, texto
        entendido). Usa el mismo tipo de motor (GPU/CPU) que el que está en uso."""
        from motores import MotorGPU
        lang = self.me
        text = TEST_PHRASES.get(lang, TEST_PHRASES["en"])
        audio, sr = self.speaker.synthesize(text, lang)
        g = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
        audio = np.concatenate([np.zeros(4000, np.float32), audio, np.zeros(4000, np.float32)])
        same = self.engine is not None and modelo == self.cfg["modelo"]
        motor = "gpu" if isinstance(self.engine, MotorGPU) else "cpu"
        eng = self.engine if same else create_engine(motor, modelo, log=log)
        try:
            eng.words(audio, lang, None)  # la primera pasada prepara el modelo
            times = []
            for _ in range(3):
                t = time.perf_counter()
                words = eng.words(audio, lang, None)
                times.append(time.perf_counter() - t)
        finally:
            if not same:
                eng.close()
        return min(times), len(audio) / 16000, "".join(w for _, _, w in words).strip()


# --------------------------------------------------------------------------- #
# Ventana
# --------------------------------------------------------------------------- #
def run_gui(overrides=None):
    import tkinter as tk
    from tkinter import ttk

    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    cfg = load_config()
    cfg.update(overrides or {})
    interfaz.set_language(cfg["interfaz"])
    ui_q = queue.Queue()
    conv = Conversation(cfg, ui_q)

    root = ctk.CTk()
    # Al cambiar el idioma de la interfaz la ventana se cierra y se vuelve a abrir.
    result = {"restart": False}

    def lang_name(code):
        return T(LANG_NAMES.get(code) or WHISPER_LANGS.get(code, code))

    def in_text(code):
        """Nombre del idioma dentro de una frase: en minúscula ("hablas español"),
        salvo en los idiomas que los escriben con mayúscula (inglés, alemán)."""
        name = lang_name(code)
        return name if interfaz.current in ("en", "de") else name.lower()

    def menu_names():
        # En los desplegables, el idioma sin voz lo dice: el otro no oiría mi traducción.
        return {c: lang_name(c) + ("" if c in VOICES else " " + T("(sin voz)")) for c in LANGS}

    names = menu_names()
    codes = {v: k for k, v in names.items()}
    DEFAULT = T("Por defecto")  # primera opción de los desplegables de dispositivos
    mics = input_devices()
    outs = output_devices()
    state = {"ready": False, "speaking": None}

    def group(parent):
        """Recuadro que junta controles relacionados dentro de una barra."""
        g = ctk.CTkFrame(parent, corner_radius=6)
        g.pack(side="left", padx=(10, 0), pady=6)
        return g

    # -- barra superior ---------------------------------------------------- #
    top = ctk.CTkFrame(root, corner_radius=0)
    top.pack(fill="x")
    # Los nombres dicen de dónde llega su voz ("Llamada"/"Presencial" confundía).
    modes = {"llamada": T("Él suena por el PC"), "presencial": T("Él está a mi lado")}
    mode_codes = {v: k for k, v in modes.items()}
    mode_btn = ctk.CTkSegmentedButton(top, values=list(modes.values()))
    mode_btn.set(modes[cfg["modo"]])
    mode_btn.pack(side="left", padx=(10, 6), pady=8)
    langs_box = group(top)  # los idiomas de los dos, juntos (en el orden de las columnas)
    ctk.CTkLabel(langs_box, text=T("Él habla")).pack(side="left", padx=(10, 0))
    them_menu = ctk.CTkOptionMenu(langs_box, width=110, values=[names[c] for c in LANGS])
    them_menu.set(names[cfg["el"]])
    them_menu.pack(side="left", padx=(6, 16), pady=5)
    ctk.CTkLabel(langs_box, text=T("Yo hablo")).pack(side="left")
    me_menu = ctk.CTkOptionMenu(langs_box, width=110, values=[names[c] for c in LANGS])
    me_menu.set(names[cfg["yo"]])
    me_menu.pack(side="left", padx=(6, 10), pady=5)
    # Lo que sale: por dónde se oye mi voz traducida.
    out_box = group(top)
    ctk.CTkLabel(out_box, text=T("Se me escucha por")).pack(side="left", padx=(10, 4))
    out_values = [DEFAULT] + [n for _, n in outs]
    out_menu = ctk.CTkOptionMenu(out_box, width=170, values=out_values, dynamic_resizing=False)
    out_menu.set(cfg["salida_voz"] if cfg["salida_voz"] in out_values else DEFAULT)
    out_menu.pack(side="left", padx=(0, 10), pady=5)
    # Lo que entra: mi micrófono.
    mic_box = group(top)
    ctk.CTkLabel(mic_box, text=T("Se me oye por")).pack(side="left", padx=(10, 4))
    mic_values = [DEFAULT] + [n for _, n in mics]
    mic_menu = ctk.CTkOptionMenu(mic_box, width=170, values=mic_values, dynamic_resizing=False)
    mic_menu.set(cfg["microfono"] if cfg["microfono"] in mic_values else DEFAULT)
    mic_menu.pack(side="left", padx=(0, 10), pady=5)
    # Herramientas (ayuda de llamadas, modelos...): al final de la barra.
    gear = ctk.CTkImage(gear_icon("#ffffff", 64), size=(16, 16))
    ctk.CTkButton(top, text="", image=gear, width=32, command=lambda: show_tools()).pack(
        side="left", padx=(10, 10), pady=8)
    # El estado (cargando, escuchando, motor, avisos) va en el título de la ventana.
    root.title(T("Cargando…"))

    # -- columnas ---------------------------------------------------------- #
    body = ctk.CTkFrame(root, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=8, pady=(8, 0))
    body.grid_columnconfigure((0, 1), weight=1, uniform="c")
    body.grid_rowconfigure(1, weight=1)
    cols = {}
    for c, (side, title) in enumerate(((THEM, T("ÉL")), (ME, T("YO")))):
        head = ctk.CTkFrame(body, fg_color="transparent")
        head.grid(row=0, column=c, sticky="ew", padx=6)
        label = ctk.CTkLabel(head, text=title, font=ctk.CTkFont(size=15, weight="bold"))
        label.pack(side="left")
        flag = ctk.CTkLabel(head, text="")  # bandera del idioma que habla: update_subtitles()
        flag.pack(side="left", padx=(8, 0))
        sub = ctk.CTkLabel(head, text="", text_color="#9aa4b2")
        sub.pack(side="left", padx=8)
        # La onda de los últimos segundos de lo que se captura (se desplaza a la
        # izquierda): una línea que pasa por el pico de cada 25 ms.
        meter = tk.Canvas(head, width=WAVE_W, height=WAVE_H, highlightthickness=0,
                          bg=ctk.ThemeManager.theme["CTk"]["fg_color"][1])
        meter.pack(side="right", pady=2)
        meter.create_line(0, WAVE_H // 2, WAVE_W, WAVE_H // 2, fill="#3a3f45")  # eje
        color = "#6fa8dc" if side == ME else "#4fc1b0"
        line = meter.create_line(0, WAVE_H // 2, WAVE_W, WAVE_H // 2, fill=color,
                                 width=1.5, smooth=True)
        # Estado del cancelador de eco (solo en ÉL): se rellena en poll().
        echo_lbl = ctk.CTkLabel(head, text="", text_color="#9aa4b2")
        echo_lbl.pack(side="right", padx=10)
        box = ctk.CTkTextbox(body, wrap="word", corner_radius=8, border_spacing=10)
        box.grid(row=1, column=c, sticky="nsew", padx=6, pady=(0, 6))
        tb = box._textbox  # tk.Text de debajo: para etiquetas con fuente propia
        tb.mark_set("live", "end-1c")
        tb.mark_gravity("live", "left")
        cols[side] = {"box": box, "tb": tb, "sub": sub, "meter": meter, "line": line,
                      "label": label, "echo": echo_lbl, "flag": flag}
    flags = {c: ctk.CTkImage(flag_image(c), size=(24, 16)) for c in LANGS}

    # -- barra inferior ---------------------------------------------------- #
    bottom = ctk.CTkFrame(root, corner_radius=0)
    bottom.pack(fill="x")
    # Mismas dos columnas que las de arriba: lo de YO queda debajo de YO.
    bar = ctk.CTkFrame(bottom, fg_color="transparent")
    bar.pack(fill="x", padx=8)
    bar.grid_columnconfigure((0, 1), weight=1, uniform="c")
    them_bar = ctk.CTkFrame(bar, fg_color="transparent")
    them_bar.grid(row=0, column=0, sticky="ew", padx=6)

    def toggle_scroll():
        cfg["autoscroll"] = bool(scroll_cb.get())
        save_config(cfg)
        if cfg["autoscroll"]:
            for side, c in cols.items():
                if editing.get("tb") is not c["tb"]:
                    c["tb"].see("end")

    # Con él marcado, las dos columnas bajan solas a lo último que se dice.
    # (Para añadir una frase a mano basta con escribir o pegar en su columna.)
    scroll_cb = ctk.CTkCheckBox(them_bar, text=T("Auto scroll"), command=toggle_scroll)
    if cfg.get("autoscroll", True):
        scroll_cb.select()
    scroll_cb.pack(side="left", pady=6)
    me_bar = ctk.CTkFrame(bar, fg_color="transparent")
    me_bar.grid(row=0, column=1, sticky="ew", padx=6)

    def font_size(delta):
        cfg["letra"] = max(10, min(40, cfg["letra"] + delta))
        apply_fonts()
        save_config(cfg)

    def clear():
        end_edit()
        for d in (entries, icons, voiced, copies, live, corr, live_tr, requested):
            d.clear()
        speak_later.clear()
        for c in cols.values():
            c["tb"].configure(state="normal")
            c["tb"].delete("1.0", "end")
            c["tb"].mark_set("live", "end-1c")

    for text, cmd in (("A+", lambda: font_size(2)), ("A−", lambda: font_size(-2)),
                      (T("Borrar"), clear)):
        ctk.CTkButton(me_bar, text=text, width=44 if len(text) < 3 else 70,
                      command=cmd).pack(side="right", padx=(8, 0), pady=6)

    def apply_fonts():
        from tkinter import font as tkfont
        n = cfg["letra"]
        num_font = ("Segoe UI", max(8, n - 7), "bold")
        # Margen izquierdo en dos columnas, con tabuladores: el número (alineado
        # a la derecha) y debajo el muñeco; a su derecha, la papelera y debajo
        # copiar. El texto empieza siempre en `indent` y las líneas que se
        # parten quedan alineadas.
        icon = state.setdefault("icon_px", int(tkfont.Font(
            family="Segoe UI", size=max(8, n - 5)).metrics("linespace") * 0.65))
        col1 = tkfont.Font(family="Segoe UI", size=max(8, n - 7), weight="bold").measure("0000") + 4
        col2 = col1 + 6
        indent = col2 + icon + 12
        big, small = ("Segoe UI", n), ("Segoe UI", max(8, n - 5))
        for side, c in cols.items():
            tb = c["tb"]
            tb.configure(tabs=(col1, "right", col2, "left", indent, "left"))
            tb.tag_configure("num", font=num_font, foreground="#6fa8dc", lmargin2=indent)
            # Número de la frase que aún se está diciendo: apagado hasta que termina.
            tb.tag_configure("num_live", font=num_font, foreground="#5f6875", lmargin2=indent)
            # Lo que se resalta (grande y blanco) es lo que me interesa leer:
            # en ÉL, su frase traducida a mi idioma; en YO, lo que digo, para
            # comprobar mientras hablo que se está entendiendo bien.
            mine = side == ME
            # lmargin2: las líneas partidas de una frase larga empiezan debajo
            # del texto, no debajo del número (Tk lo toma del primer carácter
            # de cada línea partida, que es del original).
            tb.tag_configure("orig", font=big if mine else small, lmargin2=indent,
                             foreground="#ffffff" if mine else "#9aa4b2")
            # Lo provisional (aún puede cambiar, no se corrige): gris y en cursiva,
            # bien distinto del blanco de lo confirmado.
            tb.tag_configure("tent", font=(*(big if mine else small), "italic"), lmargin2=indent,
                             foreground="#6b7380" if mine else "#5f6875")
            # Las terminadas empiezan por tabuladores (en YO, con los iconos de
            # decirla y copiarla): el texto queda igual de alineado.
            tb.tag_configure("tr", font=small if mine else big, spacing3=14,
                             foreground="#9aa4b2" if mine else "#ffffff",
                             lmargin1=0, lmargin2=indent)
            tb.tag_configure("tr_live", font=small if mine else big, spacing3=14,
                             foreground="#6f7782" if mine else "#b8bec7",
                             lmargin1=indent, lmargin2=indent)
            # Árabe (y otras de derecha a izquierda): las líneas partidas se
            # alinean a la derecha. Tk toma la alineación del primer carácter de
            # cada línea, así que el número y los iconos siguen a la izquierda.
            tb.tag_configure("rtl", justify="right")
        state["indent"] = indent

    apply_fonts()

    def update_subtitles():
        cols[THEM]["flag"].configure(image=flags[cfg["el"]])
        cols[ME]["flag"].configure(image=flags[cfg["yo"]])
        if cfg["modo"] == "llamada":
            cols[THEM]["sub"].configure(text=T("capturando los altavoces del PC"))
            cols[ME]["sub"].configure(text=T("capturando el micrófono"))
        else:
            for side, key in ((THEM, "el"), (ME, "yo")):
                cols[side]["sub"].configure(text=T("por el micrófono · se reconoce por hablar {language}",
                                                   language=in_text(cfg[key])))

    update_subtitles()

    # Ancho de la ventana: lo que ocupa la barra superior (ya con las banderas),
    # para que no se corte.
    root.update_idletasks()
    width = top.winfo_reqwidth() + 10
    root.minsize(width, 450)
    root.geometry(f"{width}x700")

    # -- pintar entradas ---------------------------------------------------- #
    entries = {}  # número -> Entry de las frases terminadas (para corregirlas)
    icons = {}    # número -> imagen del muñeco en YO (para pintarlo mientras se dice)
    from PIL import ImageTk
    from tkinter import font as tkfont
    # Algo menor que la línea de la traducción: discreto junto al número
    # (el mismo tamaño con el que apply_fonts reparte el margen).
    side_px = state["icon_px"]
    # Color del muñeco según el estado de la frase: no dicha, en cola, sonando, dicha.
    ICON_COLORS = {"off": "#6fa8dc", "queued": "#e0a458", "on": "#7ee08a", "done": "#5f6875"}
    icon_img = {st: ImageTk.PhotoImage(talk_icon(color, side_px), master=root)
                for st, color in ICON_COLORS.items()}
    voiced = {}   # número -> estado de su voz ("off" si no está)
    # Icono para copiar la traducción (gris; verde un momento al copiar).
    copy_img = {"off": ImageTk.PhotoImage(copy_icon("#9aa4b2", side_px), master=root),
                "done": ImageTk.PhotoImage(copy_icon("#7ee08a", side_px), master=root)}
    copies = {}   # número -> imagen del icono de copiar en YO
    trash_img = ImageTk.PhotoImage(trash_icon("#b07a7a", side_px), master=root)

    def delete_sentence(num):
        """Quita una frase mía de la conversación (clic en su papelera): si está
        sonando se corta, si está en cola ya no suena, y se anota en el registro."""
        e = entries.get(num)
        if e is None:
            return
        if editing.get("num") == num:
            end_edit()
        conv.forget_speech(num)
        remove_entry(cols[e.side]["tb"], num)
        voiced.pop(num, None)
        conv.write_note(f"{num} [{'YO' if e.side == ME else 'ÉL'}] (borrada)")

    def copy_translation(num):
        """Copia al portapapeles la traducción de una frase mía (clic en su icono)."""
        e = entries.get(num)
        text = e.translation.strip() if e is not None else ""
        if not text or text == "…":
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        tb, name = cols[ME]["tb"], copies.get(num)

        def show(st):
            try:
                tb.image_configure(name, image=copy_img[st])
            except Exception:
                pass  # su frase se borró o se volvió a traducir
        show("done")
        root.after(1200, lambda: show("off"))

    def set_voiced(num, st):
        voiced[num] = st
        if num in icons:
            try:
                cols[ME]["tb"].image_configure(icons[num], image=icon_img[st])
            except Exception:
                pass  # su frase se borró

    def say(num, toggle=True):
        """Clic en el muñeco de una frase mía: la pone en cola para decirla (o
        la corta si está sonando)."""
        e = entries.get(num)
        if e is None:
            return
        if voiced.get(num) != "on" and retarget(num, speak=True):
            return  # se traduce al idioma nuevo y se dice al acabar
        if voiced.get(num) != "on" and (not e.translation.strip() or e.translation.strip() == "…"):
            return
        if voiced.get(num) == "on":
            conv.stop_speaking(num)  # sonando: el clic la corta (y queda como dicha)
            if toggle:
                return
        if voiced.get(num) == "queued":
            return  # ya va a sonar: no se pone dos veces en cola
        set_voiced(num, "queued")
        conv.speak(e, e.translation.strip())

    # La frase en curso (la que se está diciendo) también se puede corregir en
    # lo ya confirmado (blanco): las palabras nuevas que lleguen van detrás.
    # El transcriptor solo añade palabras confirmadas al final, así que la
    # corrección se guarda como "este principio del texto -> este otro".
    live = {}          # lado -> última versión (Entry) de la frase en curso
    corr = {}          # número -> (texto del transcriptor que se corrigió, corrección)
    live_tr = {}       # número -> (texto corregido, su traducción)
    requested = {}     # número -> texto corregido cuya traducción ya se pidió
    speak_later = set()  # en curso con Mayús+Enter: se dicen al terminar

    def side_of(tb):
        return ME if tb is cols[ME]["tb"] else THEM

    def shown(e):
        """Texto de la frase con su corrección: lo corregido en lugar de lo que
        había y, detrás, lo que el transcriptor añadió después."""
        if e.num not in corr:
            return e.text.strip()
        raw, fixed = corr[e.num]
        rest = e.text[len(raw):] if e.text.startswith(raw) else ""
        return " ".join((fixed + " " + rest).split())

    def live_translation(e):
        """Traducción de la frase en curso; si se corrigió, la de lo corregido."""
        if e.num not in corr:
            return e.translation.strip()
        text, known = shown(e), live_tr.get(e.num)
        if known and known[0] == text:
            return known[1]
        if requested.get(e.num) != text:
            requested[e.num] = text
            num, lang, target = e.num, e.lang, e.target
            threading.Thread(target=safe(lambda: ui_q.put(
                ("live_tr", num, text, conv.translate(text, lang, target)))), daemon=True).start()
        return known[1] if known else "…"

    def live_tail(tb, e):
        """Lo provisional (gris) y la traducción provisional de la frase en curso."""
        if e.tentative:
            tb.insert("end", e.tentative.strip(), ("tent", *bidi(e.tentative)))
        tb.insert("end", "\n")
        tr = live_translation(e) or "…"
        tb.insert("end", tr + "\n", ("tr_live", *bidi(tr)))

    def draw_live(tb, e):
        tb.delete("live", "end-1c")
        tb.insert("end", f"\t{e.num}\t\t", "num_live")  # número apagado: aún se está diciendo
        # Lo confirmado lleva su número (o<n>) para poder corregirlo con un clic;
        # el espacio final separa lo provisional y cierra la zona editable.
        tb.insert("end", shown(e) + " ", ("orig", f"o{e.num}", *bidi(shown(e))))
        live_tail(tb, e)

    def update_live_edit(tb, e):
        """Frase en curso que se está corrigiendo: no se toca lo que se corrige;
        las palabras confirmadas nuevas y lo provisional se pintan detrás."""
        base = editing["raw"]
        extra = e.text[len(base):].strip() if e.text.startswith(base) else ""
        tb.delete("ed_end + 1c", "end-1c")
        if extra:
            tb.insert("end", extra + " ", ("orig", *bidi(extra)))
        live_tail(tb, e)

    def finish(tb, e):
        """Frase terminada: se pinta con su corrección (si se hizo mientras se
        decía, se vuelve a traducir entera) y con su muñeco."""
        fixed = e.num in corr
        if fixed:
            e.text = shown(e)
            known = live_tr.get(e.num)
            e.translation = known[1] if known and known[0] == e.text else "…"
        tb.delete("live", "end-1c")
        tb.insert("end", f"\t{e.num}\t", "num")
        if e.side == ME:  # la papelera, encima del icono de copiar
            name = tb.image_create("end", image=trash_img)
            for tag in ("del", f"d{e.num}"):
                tb.tag_add(tag, name)
        tb.insert("end", "\t", "num")
        tb.insert("end", e.text.strip() + " ", ("orig", f"o{e.num}", *bidi(e.text)))
        tb.insert("end", "\n")
        put_translation(tb, e.side, e.num, "end", e.translation.strip() or "…")
        tb.mark_set("live", "end-1c")
        entries[e.num] = e
        if not fixed:
            conv.prepare_speech(e)  # con mi voz: convertida ya, suena al pulsar el muñeco
        for d in (corr, live_tr, requested):
            d.pop(e.num, None)
        speak = e.num in speak_later
        speak_later.discard(e.num)
        if fixed:
            retranslate(e, changed=True, new=False, speak=speak)
        elif speak:
            say(e.num)

    def untouched_auto():
        """¿La frase elegida lo está sola (auto scroll) y aún no se ha tocado?"""
        return bool(editing.get("auto")) and not editing.get("active")

    def releasable():
        """¿La frase elegida puede soltarse para elegir lo que digo? Sí, si se
        eligió sola y no se ha tocado, o si se eligió a mano (clic, ↑/↓, Enter)
        pero hace rato que no se toca y no tiene cambios sin guardar."""
        if not editing or untouched_auto():
            return True
        idle = time.monotonic() - editing.get("active", 0) >= EDIT_HOLD
        text = " ".join(editing["tb"].get("ed_start", "ed_end").split())
        return idle and text == editing["original"]

    def follow(tb, e):
        """Con auto scroll, lo que voy diciendo queda elegido para corregirlo con
        el teclado sin hacer clic (cursor al final). Crece con cada palabra hasta
        que se toca una tecla; no quita una corrección en marcha ni con cambios."""
        if e.side != ME or not cfg.get("autoscroll", True) or not shown(e) or not releasable():
            return
        end_edit()  # la anterior elegida sola y sin tocar: se suelta tal cual
        begin_edit(tb, e.num, auto=True)

    def render(e):
        tb = cols[e.side]["tb"]
        editing_it = editing.get("tb") is tb and editing.get("live") and editing["num"] == e.num
        if e.final:
            live.pop(e.side, None)
            auto = editing_it and cfg.get("autoscroll", True) and releasable()
            resume = pause_live_edit(e) if editing_it else None
            if resume and auto:
                resume = (resume[0], None, resume[2])  # sin tocar: el cursor, al final
            finish(tb, e)
            if resume:  # se terminó de decir a media corrección: se sigue corrigiendo
                # (si estaba elegida sola, lo sigue estando: Mayús+Enter la dice)
                resume_edit(tb, e.num, *resume, auto=auto)
            else:
                follow(tb, e)  # una frase corta acaba antes de tener nada confirmado
        else:
            live[e.side] = e
            if editing_it and e.side == ME and cfg.get("autoscroll", True) and releasable():
                stop_editing()  # sin tocar: se vuelve a elegir con las palabras nuevas
                draw_live(tb, e)
                follow(tb, e)
            elif editing_it:
                update_live_edit(tb, e)
            else:
                draw_live(tb, e)
                follow(tb, e)
        # Auto scroll: se baja a lo último que se dice. Solo se espera mientras
        # se está corrigiendo de verdad en esta columna (clic o tecla hace poco):
        # una corrección abierta y olvidada no lo para.
        busy = editing.get("tb") is tb and time.monotonic() - editing.get("active", 0) < EDIT_HOLD
        if cfg.get("autoscroll", True) and not busy:
            tb.see("end")

    # -- corregir una frase en el sitio: clic donde está el error, Enter ---- #
    # El texto se edita dentro de la propia columna (mismas líneas partidas,
    # el cursor donde se hace clic). Solo se puede escribir entre las marcas
    # ed_start/ed_end, que rodean el original de la frase que se corrige.
    editing = {}  # la corrección en curso: num, caja, texto original

    def live_num(tb):
        e = live.get(side_of(tb))
        return e.num if e is not None else None

    def entry_at(tb, index):
        """Número de la frase en esa posición (original o traducción): una
        terminada o la que está en curso."""
        for tag in tb.tag_names(index):
            if tag[:1] in "ot" and tag[1:].isdigit():
                num = int(tag[1:])
                if num in entries or num == live_num(tb):
                    return num
        return None

    def in_edit(tb, index):
        return (editing.get("tb") is tb and tb.compare(index, ">=", "ed_start")
                and tb.compare(index, "<=", "ed_end"))

    def start_edit(tb, event):
        index = tb.index(f"@{event.x},{event.y}")
        if "say" in tb.tag_names(index):  # el muñeco: decir esa frase
            say(entry_at(tb, index))
            return "break"
        if "copy" in tb.tag_names(index):  # copiar su traducción
            copy_translation(entry_at(tb, index))
            return "break"
        dels = [t for t in tb.tag_names(index) if t[:1] == "d" and t[1:].isdigit()]
        if dels:  # la papelera: quitar la frase
            delete_sentence(int(dels[0][1:]))
            return "break"
        if in_edit(tb, index):
            editing["active"] = time.monotonic()
            return None  # dentro de lo que se corrige: mover el cursor, seleccionar...
        if editing:
            end_edit()  # clic fuera: se deshace la corrección a medias
            index = tb.index(f"@{event.x},{event.y}")  # pudo repintarse la frase en curso
        # La frase de esa línea (también su número, sus iconos o su traducción);
        # si el clic no da en ninguna, la última de la columna.
        num = entry_at(tb, index) or entry_at(tb, tb.index(f"{index} lineend - 1c"))
        on_text = num is not None and f"o{num}" in tb.tag_names(index)
        if num is None:
            num = last_sentence(tb)
        if num is None:
            return None  # columna vacía
        begin_edit(tb, num, index if on_text else None)
        return "break"

    def sentences(tb):
        """Números de las frases de la columna (terminadas y la que está en curso),
        en el orden en que se ven."""
        nums = [n for n in list(entries) + [live_num(tb)] if n is not None and tb.tag_ranges(f"o{n}")]
        return sorted(set(nums), key=lambda n: tuple(map(int, tb.index(f"o{n}.first").split("."))))

    def last_sentence(tb):
        """La última frase que se puede corregir (la en curso, si ya tiene algo confirmado)."""
        for n in reversed(sentences(tb)):
            if n in entries or shown(live[side_of(tb)]):
                return n
        return None

    def retarget(num, speak=False):
        """Si la frase se tradujo a un idioma que ya no es el elegido (p. ej. se
        cambió el del interlocutor), se vuelve a traducir al de ahora (y se dice,
        con speak). Devuelve si hacía falta."""
        e = entries.get(num)
        if e is None:
            return False
        want = conv.them if e.side == ME else conv.me
        if e.target == want:
            return False
        e.target = want
        set_translation(cols[e.side]["tb"], num, "…")
        retranslate(e, changed=False, new=False, speak=speak)
        return True

    def begin_edit(tb, num, at=None, auto=False):
        is_live = num not in entries
        e = live[side_of(tb)] if is_live else entries[num]
        if is_live and not shown(e):
            return  # aún no hay nada confirmado que corregir
        if not is_live:
            retarget(num)  # al pasar por ella, ya en el idioma elegido ahora
        # El original termina en un espacio que no se edita.
        tb.mark_set("ed_start", f"o{num}.first")
        tb.mark_gravity("ed_start", "left")
        tb.mark_set("ed_end", f"o{num}.last - 1c")
        tb.mark_gravity("ed_end", "right")
        # active: cuándo se tocó por última vez (0: elegida sola y sin tocar,
        # así el auto scroll no espera).
        editing.update(num=num, tb=tb, original=shown(e) if is_live else e.text.strip(),
                       live=is_live, raw=e.text, active=0 if auto else time.monotonic(), auto=auto)
        retag()
        tb.mark_set("insert", at or "ed_end")  # el cursor, donde se hizo clic
        tb.configure(insertwidth=2)  # solo se ve el cursor mientras se corrige
        if auto:
            # Sin quitarle el teclado a otra ventana del programa (Herramientas...).
            try:
                focus = root.focus_get()
            except KeyError:  # un desplegable abierto
                return
            if focus is None or focus.winfo_toplevel() is root:
                tb.focus_set()
            return
        tb.focus_set()
        tb.see("insert")

    def retag():
        """Lo que hay entre las marcas es el original de la frase (también lo recién escrito)."""
        if not editing:
            return
        tb, num = editing["tb"], editing["num"]
        for tag in ("orig", f"o{num}", "editing"):
            tb.tag_add(tag, "ed_start", "ed_end")
        # Se escribe en árabe (o se deja de escribir): cambia la alineación.
        tb.tag_remove("rtl", "ed_start", "ed_end")
        if right_to_left(tb.get("ed_start", "ed_end")):
            tb.tag_add("rtl", "ed_start", "ed_end")
        tb.tag_add("editing", "ed_end", "ed_end + 1c")  # una frase vacía también se ve

    def add_line(side):
        """Frase escrita a mano al final de una columna, lista para escribir en
        el idioma de quien habla en ella (YO: el mío; ÉL: el suyo)."""
        if not state["ready"]:
            return
        end_edit()
        lang, target = (conv.me, conv.them) if side == ME else (conv.them, conv.me)
        e = Entry(conv.next_number(), side, lang, target)
        e.final = True
        render(e)
        tb = cols[side]["tb"]
        tb.see("end")
        begin_edit(tb, e.num)

    def remove_entry(tb, num):
        """Quita una frase de la columna (una añadida a mano que se deja vacía)."""
        entries.pop(num, None)
        icons.pop(num, None)
        copies.pop(num, None)
        tb.delete(f"o{num}.first linestart", f"t{num}.last")

    def replace_orig(tb, num, text):
        tb.delete("ed_start", "ed_end")
        tb.insert("ed_start", text, ("orig", f"o{num}", *bidi(text)))

    def stop_editing():
        """Cierra la corrección en pantalla y devuelve lo que se estaba corrigiendo."""
        ed = dict(editing)
        editing.clear()
        tb = ed["tb"]
        text = " ".join(tb.get("ed_start", "ed_end").split())
        tb.tag_remove("editing", "1.0", "end")
        tb.configure(insertwidth=0)
        update_subtitles()
        return ed, text

    def pause_live_edit(e):
        """La frase en curso que se corrige se acaba de terminar: se guarda lo
        que se llevaba escrito (más lo confirmado después) y dónde estaba el cursor."""
        tb = editing["tb"]
        off = len(tb.get("ed_start", "insert")) if in_edit(tb, "insert") else None
        ed, text = stop_editing()
        extra = e.text[len(ed["raw"]):] if e.text.startswith(ed["raw"]) else ""
        return " ".join((text + " " + extra).split()), off, ed.get("speak", False)

    def resume_edit(tb, num, text, off, _speak, auto=False):
        begin_edit(tb, num, auto=auto)
        replace_orig(tb, num, text)
        retag()
        tb.mark_set("insert", f"ed_start + {min(off, len(text))}c" if off is not None else "ed_end")

    def discard_live(side):
        """La frase en curso era ruido: se quita (y su corrección, si la había)."""
        tb, e = cols[side]["tb"], live.pop(side, None)
        if editing.get("tb") is tb and editing.get("live"):
            stop_editing()
        if e is not None:
            for d in (corr, live_tr, requested):
                d.pop(e.num, None)
            speak_later.discard(e.num)
        tb.delete("live", "end-1c")

    def got_live_translation(num, text, translation):
        if num not in corr:
            return  # ya terminó (se tradujo entera) o se descartó
        live_tr[num] = (text, translation)
        for side, e in live.items():
            if e.num == num:
                render(e)

    def retranslate(e, changed, new, speak):
        num = e.num

        def run():  # fuera de la ventana: traducir tarda un poco
            ui_q.put(("edited", num, conv.translate(e.text, e.lang, e.target), changed, new, speak))
        threading.Thread(target=safe(run), daemon=True).start()

    def end_edit(commit=False, speak=False):
        if not editing:
            return
        ed, text = stop_editing()
        num, tb, original = ed["num"], ed["tb"], ed["original"]
        if ed.get("live"):
            # Frase en curso: la corrección se aplica a lo confirmado y lo que
            # llegue después se suma detrás; se traduce entera al terminar.
            if commit and text:
                corr[num] = (ed["raw"], text)
                if speak:
                    speak_later.add(num)
            e = live.get(side_of(tb))
            if e is not None and e.num == num:
                draw_live(tb, e)
            return
        e = entries.get(num)
        if e is None:
            return
        new = not original  # añadida a mano y aún sin texto
        if not commit or not text:
            if new:
                remove_entry(tb, num)  # cancelada o vacía: no queda rastro
            else:
                replace_orig(tb, num, original)  # Esc: se deja como estaba
            return
        # Enter siempre traduce y dice lo escrito, aunque no se haya cambiado.
        changed = text != original
        e.text = text
        replace_orig(tb, num, text)
        set_translation(tb, num, "…")
        retranslate(e, changed, new, speak)

    def put_translation(tb, side, num, index, text):
        """Línea de la traducción de una frase terminada; en YO empieza por los
        iconos del muñeco (la dice), bajo el número, y de copiar, bajo la papelera."""
        tags = ("tr", f"t{num}")
        # Se escribe en orden en una marca que avanza con lo que se inserta.
        tb.mark_set("tp", index)
        tb.mark_gravity("tp", "right")
        if side == ME:
            tb.insert("tp", "\t", tags)
            name = tb.image_create("tp", image=icon_img[voiced.get(num, "off")])
            for tag in (*tags, "say"):
                tb.tag_add(tag, name)
            icons[num] = name
            tb.insert("tp", "\t", tags)
            cname = tb.image_create("tp", image=copy_img["off"])
            for tag in (*tags, "copy"):
                tb.tag_add(tag, cname)
            copies[num] = cname
            tb.insert("tp", "\t", tags)
        else:
            tb.insert("tp", "\t\t\t", tags)
        tb.insert("tp", text + "\n", (*tags, *bidi(text)))
        tb.mark_unset("tp")

    def set_translation(tb, num, text):
        if not tb.tag_ranges(f"t{num}"):
            return  # la frase se borró mientras se traducía
        start, end = tb.tag_ranges(f"t{num}")
        tb.delete(start, end)
        put_translation(tb, ME if tb is cols[ME]["tb"] else THEM, num, start, text)
        # Si era la última, la marca "live" (donde empieza la frase en curso)
        # se queda delante de lo insertado y la siguiente frase lo borraría.
        if tb.compare("live", "<", f"t{num}.last"):
            tb.mark_set("live", f"t{num}.last")

    def edited(num, translation, changed, new, speak):
        e = entries.get(num)
        if e is None or not cols[e.side]["tb"].tag_ranges(f"t{num}"):
            return  # la frase se borró mientras se traducía
        if translation != e.translation and voiced.get(num) == "done":
            voiced[num] = "off"  # la traducción nueva aún no se ha dicho
        e.translation = translation
        set_translation(cols[e.side]["tb"], num, translation or "…")
        if not speak:
            conv.prepare_speech(e)  # (con Mayús+Enter se dice ya: say() la prepara)
        if changed:
            conv.write_log(e, corrected=not new)  # una añadida se anota como las dichas
        if speak and e.side == ME:
            say(num, toggle=False)  # Mayús+Enter: se dice la nueva (si sonaba la vieja, se corta)

    # Teclas que no cambian el texto: moverse, seleccionar, copiar (Ctrl+C...).
    NAV_KEYS = {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next",
                "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R"}

    def editable(tb):
        """¿Lo que cambiaría una tecla (el cursor o lo seleccionado) está dentro de
        la frase que se corrige?"""
        if editing.get("tb") is not tb:
            return False
        if tb.tag_ranges("sel"):
            return in_edit(tb, "sel.first") and in_edit(tb, "sel.last")
        return in_edit(tb, "insert")

    def key(tb, event):
        ks = event.keysym
        if editing.get("tb") is tb:
            editing["active"] = time.monotonic()  # corrigiendo: el auto scroll espera
            if ks in ("Return", "KP_Enter"):
                # Mayúsculas+Enter: además de traducirla, la dice (como su muñeco).
                # La frase sigue elegida, para seguir con ↑/↓ (Esc la suelta).
                num = editing["num"]
                end_edit(commit=True, speak=bool(event.state & 0x1))
                if tb.tag_ranges(f"o{num}"):
                    begin_edit(tb, num)
                return "break"
            if ks == "Escape":
                end_edit()
                return "break"
        if editing.get("tb") is tb and ks in ("Up", "Down") and not event.state & 0x5:
            # Flecha arriba en la primera línea de la frase (o abajo en la última):
            # a la frase anterior (o siguiente). Dentro de la frase, se mueve normal.
            first = tb.compare("insert display linestart", "<=", "ed_start")
            last = tb.compare("insert display lineend", ">=", "ed_end")
            if (ks == "Up" and first) or (ks == "Down" and last):
                go_to_neighbour(tb, -1 if ks == "Up" else 1)
                return "break"
        if ks in NAV_KEYS or (event.state & 0x4 and ks.lower() in "ca"):
            if editing.get("tb") is tb:
                tb.after_idle(clamp_cursor)  # Inicio, Fin, flechas: sin salir de la frase
            return None
        printable = bool(event.char) and event.char.isprintable() and not event.state & 0x4
        if editing.get("tb") is not tb and printable:
            # Sin corregir nada en esta columna: lo que se escribe empieza una
            # frase nueva al final (en lugar del antiguo botón "Añadir frase").
            add_line(side_of(tb))
            if editing.get("tb") is tb:
                tb.insert("insert", event.char)
                retag()
            return "break"
        # Fuera de la frase que se corrige, la columna no se puede escribir
        # (cambiaría lo que se ve sin traducir nada).
        if event.state & 0x4 or not editable(tb):
            return "break"
        selected = bool(tb.tag_ranges("sel"))
        if ks == "BackSpace" and not selected and tb.compare("insert", "<=", "ed_start"):
            return "break"
        if ks == "Delete" and not selected and tb.compare("insert", ">=", "ed_end"):
            return "break"
        if ks in ("BackSpace", "Delete") or (event.char and event.char.isprintable()):
            tb.after_idle(retag)
            return None
        return "break"  # Tab, F1...

    def go_to_neighbour(tb, step):
        """Pasa a corregir la frase anterior (step=-1) o siguiente (+1) de la
        columna. Si se había cambiado algo, se guarda y se traduce (como Enter)."""
        num = editing["num"]
        order = sentences(tb)
        i = order.index(num) if num in order else -1
        text = " ".join(tb.get("ed_start", "ed_end").split())
        changed = text != editing["original"]
        if step > 0 and i == len(order) - 1:
            # ↓ en la última: frase nueva debajo, lista para escribir (si esta no
            # es ya una nueva vacía).
            if text:
                end_edit(commit=changed)
                add_line(side_of(tb))
            return
        if step < 0 and i <= 0:
            return  # ya está en la primera
        target = order[i + step]
        end_edit(commit=changed)
        if not tb.tag_ranges(f"o{target}"):
            return
        begin_edit(tb, target)
        if not editing:
            return  # la frase en curso aún no tiene nada confirmado
        # Hacia arriba se entra por el final de la frase; hacia abajo, por el principio.
        tb.mark_set("insert", "ed_end" if step < 0 else "ed_start")
        tb.see("insert")

    def clamp_cursor():
        """Tras moverse con el teclado, el cursor (y lo seleccionado) se queda
        dentro de la frase que se corrige: Inicio va al primer carácter editable,
        no delante del número; Fin, al último."""
        if not editing:
            return
        tb = editing["tb"]
        if tb.compare("insert", "<", "ed_start"):
            tb.mark_set("insert", "ed_start")
        elif tb.compare("insert", ">", "ed_end"):
            tb.mark_set("insert", "ed_end")
        if tb.tag_ranges("sel"):
            first = "ed_start" if tb.compare("sel.first", "<", "ed_start") else "sel.first"
            last = "ed_end" if tb.compare("sel.last", ">", "ed_end") else "sel.last"
            first, last = tb.index(first), tb.index(last)
            tb.tag_remove("sel", "1.0", "end")
            if tb.compare(first, "<", last):
                tb.tag_add("sel", first, last)
        tb.see("insert")

    def paste(tb):
        if editing.get("tb") is not tb:
            add_line(side_of(tb))  # Ctrl+V sin corregir nada: el texto va a una frase nueva
        if editable(tb):
            try:
                text = " ".join(tb.clipboard_get().split())
            except Exception:
                return "break"
            if tb.tag_ranges("sel"):
                tb.delete("sel.first", "sel.last")
            tb.insert("insert", text)
            retag()
        return "break"

    def cut(tb):
        if editable(tb) and tb.tag_ranges("sel"):
            tb.clipboard_clear()
            tb.clipboard_append(tb.get("sel.first", "sel.last"))
            tb.delete("sel.first", "sel.last")
        return "break"

    for c in cols.values():
        tb = c["tb"]
        # La frase que se corrige, con fondo (pero no el margen de los números).
        tb.tag_configure("editing", background="#26384d", lmargincolor=tb.cget("background"))
        tb.configure(insertwidth=0)  # sin cursor parpadeando donde no se puede escribir
        tb.tag_raise("sel")  # lo seleccionado se sigue viendo encima
        tb.bind("<Button-1>", lambda ev, tb=tb: start_edit(tb, ev))
        tb.bind("<Key>", lambda ev, tb=tb: key(tb, ev))
        tb.bind("<<Paste>>", lambda _, tb=tb: paste(tb))
        tb.bind("<<Cut>>", lambda _, tb=tb: cut(tb))
        tb.bind("<<Clear>>", lambda _: "break")
        # Manita al pasar por una frase que se puede corregir.
        tb.tag_bind("orig", "<Enter>", lambda _, tb=tb: tb.configure(cursor="hand2"))
        tb.tag_bind("orig", "<Leave>", lambda _, tb=tb: tb.configure(cursor="xterm"))
        for tag in ("say", "copy", "del"):
            tb.tag_bind(tag, "<Enter>", lambda _, tb=tb: tb.configure(cursor="hand2"))
            tb.tag_bind(tag, "<Leave>", lambda _, tb=tb: tb.configure(cursor="xterm"))

    # -- ayuda: mi voz traducida en Teams, Zoom, WhatsApp... ---------------- #
    help_win = {}

    def call_checks():
        """(¿bien?, qué se comprueba, qué hacer si no) de la configuración actual."""
        outs_now = [n for _, n in output_devices()]
        mics_now = input_devices()
        cable = any(n.startswith("CABLE Input") for n in outs_now)
        mic = cfg["microfono"] or (mics_now[0][1] if mics_now else "")
        try:
            win_out = speaker_loopback()[1]
        except Exception:
            win_out = ""
        return [
            (cable, T("VB-Cable instalado"), T("instálalo (paso 1)")),
            (cfg["modo"] == "llamada", T("Modo: Él suena por el PC"), T("elígelo arriba")),
            (is_virtual(cfg["salida_voz"]), T("Se me escucha por: CABLE Input"), T("elígelo arriba")),
            (bool(mic) and not is_virtual(mic), T("Se me oye por: {device}", device=mic or "?"),
             T("elige tu micrófono, no el cable")),
            (bool(win_out) and not is_virtual(win_out), T("Salida de Windows: {device}", device=win_out or "?"),
             T("pon tus altavoces en Configuración → Sonido")),
        ]

    def show_tools(chapter="idiomas"):
        """Ventana de herramientas: índice de capítulos a la izquierda y el
        capítulo elegido a la derecha."""
        if help_win.get("w") is not None and help_win["w"].winfo_exists():
            help_win["show"](chapter)
            help_win["w"].focus()
            return
        w = ctk.CTkToplevel(root)
        help_win["w"] = w
        w.title(T("Herramientas"))
        w.geometry("1060x760")
        w.transient(root)
        w.grid_columnconfigure(1, weight=1)
        w.grid_rowconfigure(0, weight=1)
        menu = ctk.CTkFrame(w, width=180, corner_radius=0)
        menu.grid(row=0, column=0, sticky="ns")
        page = ctk.CTkFrame(w, fg_color="transparent")
        page.grid(row=0, column=1, sticky="nsew")
        chapters = {"idiomas": (T("Idiomas"), page_languages), "modelos": (T("Modelos"), page_models),
                    "llamadas": (T("Llamadas"), page_calls), "atajos": (T("Atajos"), page_shortcuts)}
        buttons = {}

        def show(key):
            help_win["page"] = key
            for c in page.winfo_children():
                c.destroy()
            for k, b in buttons.items():  # el capítulo elegido, resaltado
                b.configure(fg_color=("gray75", "gray25") if k == key else "transparent")
            chapters[key][1](page)
        help_win["show"] = show
        ctk.CTkLabel(menu, text=T("Herramientas"), font=ctk.CTkFont(size=15, weight="bold")).pack(
            anchor="w", padx=16, pady=(16, 10))
        for key, (title, _) in chapters.items():
            buttons[key] = ctk.CTkButton(menu, text=title, anchor="w", width=150, fg_color="transparent",
                                         hover_color=("gray70", "gray30"), command=lambda k=key: show(k))
            buttons[key].pack(padx=12, pady=2)
        show(chapter)

    def page_calls(w):
        """Capítulo "Llamadas": que el otro oiga mi voz traducida (Teams, WhatsApp...)."""
        import webbrowser
        ctk.CTkLabel(w, text=T("Que el otro oiga tu voz traducida"),
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=16, pady=(14, 6))
        box = ctk.CTkFrame(w)
        box.pack(fill="x", padx=16)
        rows = ctk.CTkFrame(box, fg_color="transparent")
        rows.pack(fill="x", padx=10, pady=(8, 4))

        def check():
            for c in rows.winfo_children():
                c.destroy()
            for ok, what, fix in call_checks():
                ctk.CTkLabel(rows, anchor="w", text=("✔  " if ok else "✘  ") + what
                             + ("" if ok else f"  →  {fix}"),
                             text_color="#7ee08a" if ok else "#e0a458").pack(fill="x")
        check()
        btns = ctk.CTkFrame(box, fg_color="transparent")
        btns.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkButton(btns, text=T("Comprobar"), width=110, command=check).pack(side="left")
        text = ctk.CTkTextbox(w, wrap="word", corner_radius=8, border_spacing=10)
        text.pack(fill="both", expand=True, padx=16, pady=12)
        t = text._textbox
        t.tag_configure("h", font=("Segoe UI", 13, "bold"), foreground="#6fa8dc", spacing1=10)
        t.tag_configure("p", font=("Segoe UI", 11), lmargin1=12, lmargin2=26, spacing1=2)
        # Las direcciones web son enlaces: subrayadas, con manita, y se abren al pulsarlas.
        t.tag_configure("link", foreground="#6fb6ff", underline=True)
        t.tag_bind("link", "<Enter>", lambda _: t.configure(cursor="hand2"))
        t.tag_bind("link", "<Leave>", lambda _: t.configure(cursor="xterm"))

        def open_link(event):
            index = t.index(f"@{event.x},{event.y}")
            for start, end in zip(*[iter(t.tag_ranges("link"))] * 2):
                if t.compare(start, "<=", index) and t.compare(index, "<", end):
                    webbrowser.open(t.get(start, end))
        t.tag_bind("link", "<Button-1>", open_link)
        for title, lines in CALL_HELP:
            t.insert("end", T(title) + "\n", "h")
            for line in lines:
                t.insert("end", "•  ", "p")
                for piece in re.split(r"(https?://\S+?)(?=[)\s]|$)", T(line)):
                    t.insert("end", piece, ("p", "link") if piece.startswith("http") else "p")
                t.insert("end", "\n", "p")
        text.configure(state="disabled")

    def page_languages(w):
        """Capítulo "Idiomas": qué se puede usar con cada idioma del catálogo y
        con qué modelos (reconocimiento, traducción con mi idioma y voz)."""
        from traductor import NOSPACE, translation_path
        me = cfg["yo"]
        title = ctk.CTkFrame(w, fg_color="transparent")
        title.pack(fill="x", padx=16, pady=(14, 2))
        ctk.CTkLabel(title, text=T("Idiomas"), font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        # Idioma de la ventana: a la derecha del título.
        ui_names = {c: lang_name(c) for c in LANGS}
        ui_menu = ctk.CTkOptionMenu(title, width=130, values=list(ui_names.values()),
                                    command=lambda name: choose_interface(
                                        next(c for c, n in ui_names.items() if n == name), ui_menu))
        ui_menu.set(ui_names.get(cfg["interfaz"], ui_names.get("es")))
        ui_menu.pack(side="right")
        ctk.CTkLabel(title, text=T("Idioma de la ventana"), text_color="#9aa4b2").pack(side="right", padx=8)
        ctk.CTkLabel(w, text=T("Lo que hace falta para conversar en cada idioma contigo, que hablas "
                               "{language}. Se eligen arriba, en \"Él habla\" y \"Yo hablo\".",
                               language=in_text(me)),
                     text_color="#9aa4b2", anchor="w", justify="left", wraplength=760).pack(
            fill="x", padx=16, pady=(0, 8))
        area = tk.Frame(w, bg=ctk.ThemeManager.theme["CTk"]["fg_color"][1])
        area.pack(fill="x", padx=16)
        # Con elementos de Tk sencillos y una lista (Treeview) para los ~90 idiomas
        # de Whisper: con los de CustomTkinter y una lista con barra de los suyos,
        # el capítulo tardaba ~6 s en abrirse.
        bg = ctk.ThemeManager.theme["CTk"]["fg_color"][1]
        font = ("Segoe UI", 10)

        def label(parent, text="", color="#dce4ee", bold=False, **kw):
            return tk.Label(parent, text=text, fg=color, bg=bg, anchor="w",
                            font=(*font, "bold") if bold else font, **kw)

        def button(parent, text, command, state="normal", color="#1f6aa5", active="#144870"):
            return tk.Button(parent, text=text, command=command, state=state, font=font, relief="flat",
                             bg=color, fg="#ffffff", activebackground=active, activeforeground="#ffffff",
                             disabledforeground="#6f7782", cursor="hand2", padx=10, pady=0, bd=0)

        mine = tk.Frame(area, bg=bg)
        heads = ("", T("Idioma"), T("Reconocimiento"),
                 T("Traducción con el {language}", language=in_text(me)),
                 T("Voz (para que te oiga)"), "")
        for col, h in enumerate(heads):
            label(mine, h, "#9aa4b2").grid(row=0, column=col, sticky="w", padx=(0, 14), pady=(0, 4))
        added = user_languages()
        for r, code in enumerate(LANGS, start=1):
            mark = " " + T("(tú)") if code == me else " " + T("(él)") if code == cfg["el"] else ""
            label(mine, image=flag_photo(code)).grid(row=r, column=0, padx=(0, 8), pady=3)
            label(mine, lang_name(code) + mark, bold=bool(mark)).grid(row=r, column=1, sticky="w", padx=(0, 14))
            label(mine, "✔ Whisper", "#7ee08a").grid(row=r, column=2, sticky="w", padx=(0, 14))
            there, back = translation_path(me, code), translation_path(code, me)
            if code == me:
                tr, color = "—", "#9aa4b2"
            elif there is None or back is None:
                tr, color = "✘ " + T("no hay traductor"), "#e07a5f"
            elif len(there) == 1 and len(back) == 1:
                tr, color = "✔ " + T("directa"), "#7ee08a"
            else:
                tr, color = "✔ " + T("a través del inglés"), "#e0a458"
            label(mine, tr, color).grid(row=r, column=3, sticky="w", padx=(0, 14))
            voice = VOICES.get(code)
            label(mine, f"✔ {voice.rsplit('/', 1)[-1]}" if voice else "✘ " + T("sin voz"),
                  "#7ee08a" if voice else "#e07a5f").grid(row=r, column=4, sticky="w", padx=(0, 14))
            if code in NOSPACE:
                label(mine, T("sin espacios"), "#9aa4b2").grid(
                    row=r, column=5, sticky="w")
            if code in added:  # añadido por ti: se puede quitar
                label(mine, T("añadido por ti"), "#9aa4b2").grid(row=r, column=5, sticky="w")
                button(mine, T("Quitar"), lambda c=code: remove_language(c), color="#7a4a4a",
                       active="#8f5555").grid(row=r, column=6, pady=2)
        mine.pack(anchor="w")
        # Los demás idiomas que reconoce Whisper: se comprueba en Hugging Face si
        # tienen traductor con el inglés (en los dos sentidos) y voz.
        others = sorted((c for c in WHISPER_LANGS if c not in LANGS), key=lang_name)
        head = ctk.CTkFrame(w, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(16, 6))
        ctk.CTkLabel(head, text=T("Otros idiomas que reconoce Whisper ({n})", n=len(others)),
                     font=ctk.CTkFont(size=14, weight="bold"), text_color="#6fa8dc").pack(side="left")
        ctk.CTkButton(head, text=T("Comprobar todos"), width=130,
                      # (los que se ven: con un filtro puesto, solo esos)
                      command=lambda: check_languages(list(tree.get_children()))).pack(side="left", padx=(14, 6))
        b_check = ctk.CTkButton(head, text=T("Comprobar"), width=100,
                                command=lambda: check_languages(list(tree.selection())))
        b_check.pack(side="left", padx=6)
        b_add = ctk.CTkButton(head, text=T("Añadir"), width=90, state="disabled",
                              command=lambda: add_language(tree.selection()[0]))
        b_add.pack(side="left", padx=6)
        style = ttk.Style(w)
        style.theme_use("clam")  # el único tema de Tk que deja cambiar los colores de la lista
        style.configure("Idiomas.Treeview", background="#2b2b2b", fieldbackground="#2b2b2b",
                        foreground="#dce4ee", rowheight=24, font=font, borderwidth=0)
        style.configure("Idiomas.Treeview.Heading", background="#333a40", foreground="#9aa4b2",
                        font=font, relief="flat")
        style.map("Idiomas.Treeview", background=[("selected", "#1f6aa5")],
                  foreground=[("selected", "#ffffff")])
        frame = tk.Frame(w, bg=bg)
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        cols = ("code", "name", "tr", "voice", "verdict")
        tree = ttk.Treeview(frame, columns=cols, show="headings", style="Idiomas.Treeview")
        for col, title, width in zip(cols, ("", T("Idioma"), T("Traducción"), T("Voz"), T("Estado")),
                                     (44, 130, 160, 190, 460)):
            tree.heading(col, text=title, anchor="w")
            tree.column(col, width=width, anchor="w", stretch=col == "verdict")
        for tag, color in (("ok", "#7ee08a"), ("warn", "#e0a458"), ("bad", "#e07a5f"), ("grey", "#9aa4b2")):
            tree.tag_configure(tag, foreground=color)
        bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        for code in others:
            tree.insert("", "end", iid=code, values=(code.upper(), lang_name(code), "", "", ""))

        # Buscar: filtra la lista por nombre o código, sin importar tildes ni mayúsculas.
        def plain(text):
            import unicodedata
            return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                           if not unicodedata.combining(c))

        def apply_filter(*_):
            words = plain(search.get()).split()
            shown = 0
            for code in others:
                hay = plain(f"{code} {lang_name(code)} {WHISPER_LANGS.get(code, '')}")
                if all(w in hay for w in words):
                    tree.move(code, "", shown)  # vuelve a la lista, en su orden
                    shown += 1
                else:
                    tree.detach(code)
            return shown

        def pick_first(_=None):
            visible = tree.get_children()
            if visible:
                tree.selection_set(visible[0])
                tree.focus(visible[0])
                tree.see(visible[0])
            return "break"

        def clear_filter(_=None):
            search.delete(0, "end")
            apply_filter()
            return "break"

        search = ctk.CTkEntry(head, width=170, placeholder_text=T("Buscar…"))
        search.pack(side="right")
        search.bind("<KeyRelease>", lambda ev: None if ev.keysym in ("Return", "Escape") else apply_filter())
        search.bind("<Return>", pick_first)
        search.bind("<Escape>", clear_filter)
        lang_view.update(tree=tree, add=b_add)
        tree.bind("<<TreeviewSelect>>", lambda _: update_add_button())
        tree.bind("<Double-1>", lambda _: check_languages(list(tree.selection())))
        for code in others:
            show_check(code)

    flag_photos = {}  # código -> bandera pequeña como imagen de Tk (para las tablas)

    def flag_photo(code):
        if code not in flag_photos:
            from PIL import Image, ImageTk
            flag_photos[code] = ImageTk.PhotoImage(flag_image(code).resize((24, 16), Image.LANCZOS),
                                                   master=root)
        return flag_photos[code]

    # código -> resultado de check_language(), "comprobando", "esperando",
    # ("adding", paso) o ("failed", error)
    lang_checks = {}
    lang_view = {}    # la lista de idiomas del capítulo y su botón Añadir

    def addable(code):
        got = lang_checks.get(code)
        return (isinstance(got, dict) and code not in NEEDS_WORK and "error" not in got
                and bool(got.get("to")) and bool(got.get("from")))

    def update_add_button():
        """Añadir: solo con un idioma elegido que se pueda añadir."""
        tree, add = lang_view.get("tree"), lang_view.get("add")
        if tree is None or not tree.winfo_exists():
            return
        sel = tree.selection()
        add.configure(state="normal" if len(sel) == 1 and addable(sel[0]) else "disabled")

    def show_check(code):
        """Pinta en la lista lo que se sabe de un idioma (si la lista sigue a la vista)."""
        tree = lang_view.get("tree")
        if tree is None or not tree.winfo_exists() or not tree.exists(code):
            return
        got, note = lang_checks.get(code), NEEDS_WORK.get(code)
        note = T(note) if note else note
        tr = voice = ""
        if isinstance(got, tuple) and got[0] == "adding":
            verdict, tag = T("añadiendo: {step}", step=got[1]), "grey"
        elif isinstance(got, tuple) and got[0] == "failed":
            verdict, tag = "✘ " + T("no se pudo añadir: {error}", error=got[1]), "bad"
        elif got is None:
            verdict, tag = ((T("necesita ajustes: {why}", why=note), "warn") if note
                            else (T("sin comprobar"), "grey"))
        elif got == "comprobando":
            verdict, tag = T("comprobando…"), "grey"
        elif got == "esperando":
            verdict, tag = T("esperando a Hugging Face…"), "grey"
        elif "error" in got:
            verdict, tag = got["error"], "bad"
        else:
            to, frm, v = got["to"], got["from"], got["voice"]
            tr = ("✔ " + T("con el inglés") if to and frm else "⚠ " + T("solo hacia el inglés") if frm else
                  "⚠ " + T("solo desde el inglés") if to else "✘ " + T("no hay traductor"))
            voice = f"✔ {v}" if v else "✘ " + T("sin voz")
            if note:
                verdict, tag = T("necesita ajustes: {why}", why=note), "warn"
            elif to and frm:
                verdict, tag = "✔ " + T("se puede añadir") + ("" if v else " " + T("(sin voz)")), "ok"
            else:
                verdict, tag = "✘ " + T("no se puede: falta traductor"), "bad"
        tree.item(code, values=(code.upper(), lang_name(code), tr, voice, verdict), tags=(tag,))
        update_add_button()

    def check_languages(codes):
        """Comprueba idiomas en Hugging Face, varios a la vez, en otro hilo."""
        codes = [c for c in codes if lang_checks.get(c) not in ("comprobando", "esperando")]
        for c in codes:
            lang_checks[c] = "comprobando"
            show_check(c)

        def one(c):
            for _ in range(4):
                got = check_language(c)
                if "wait" not in got:
                    break
                # Hugging Face pide esperar: se espera y se vuelve a intentar.
                lang_checks[c] = "esperando"
                ui_q.put(("call", lambda: show_check(c)))
                time.sleep(got["wait"] + 2)
            lang_checks[c] = got
            ui_q.put(("call", lambda: show_check(c)))

        def work():
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(2) as pool:  # pocas a la vez: Hugging Face limita sin cuenta
                list(pool.map(one, codes))
        threading.Thread(target=work, daemon=True).start()

    def add_language(code):
        """Botón Añadir: descarga, prueba y da de alta el idioma (en otro hilo)."""
        found = lang_checks.get(code)
        if not isinstance(found, dict):
            return

        def step(msg):
            lang_checks[code] = ("adding", msg)
            ui_q.put(("call", lambda: show_check(code)))

        def work():
            try:
                there, back = install_language(code, found, log=step)
            except Exception as e:
                lang_checks[code] = ("failed", str(e))
                ui_q.put(("call", lambda: show_check(code)))
                return
            lang_checks.pop(code, None)
            ui_q.put(("status", T("Añadido {language} · prueba: «{sample}»",
                                  language=lang_name(code), sample=there)))
            ui_q.put(("call", languages_changed))
        step(T("empezando…"))
        threading.Thread(target=work, daemon=True).start()

    def remove_language(code):
        """Botón Quitar (idiomas añadidos por ti). Sus modelos se quedan en "modelos"."""
        if code in (cfg["yo"], cfg["el"]):
            ui_q.put(("status", T("No se puede quitar {language}: está elegido arriba",
                                  language=lang_name(code))))
            return
        langs = user_languages()
        langs.pop(code, None)
        save_user_languages(langs)
        unregister_language(code)
        languages_changed()

    def languages_changed():
        """El catálogo ha cambiado: desplegables, banderas y capítulo Idiomas al día."""
        names.clear()
        names.update(menu_names())
        codes.clear()
        codes.update({v: k for k, v in names.items()})
        for c in LANGS:
            if c not in flags:
                flags[c] = ctk.CTkImage(flag_image(c), size=(24, 16))
        for menu, key in ((them_menu, "el"), (me_menu, "yo")):
            menu.configure(values=[names[c] for c in LANGS])
            menu.set(names[cfg[key]])
        if help_win.get("w") is not None and help_win["w"].winfo_exists() and help_win.get("page") == "idiomas":
            help_win["show"]("idiomas")

    def choose_interface(code, menu):
        """Idioma de la ventana: si no tiene su archivo de textos, se genera con
        el traductor del programa (desde el inglés); después la ventana se
        cierra y se vuelve a abrir en ese idioma."""
        if code == cfg["interfaz"]:
            return
        if busy["on"]:
            menu.set(lang_name(cfg["interfaz"]))
            return

        def apply():
            cfg["interfaz"] = code
            save_config(cfg)
            result["restart"] = True
            on_close()

        if interfaz.available(code):
            apply()
            return
        busy["on"] = True
        menu.configure(state="disabled")

        def work():
            try:
                translate = Translator(lambda *_: None, "en", code)
                interfaz.generate(code, lambda s: translate(s) or s, progress=lambda done, total: ui_q.put(
                    ("status", T("Traduciendo la interfaz… {done}/{total}", done=done, total=total))),
                    names=set(WHISPER_LANGS.values()) | set(LANG_NAMES.values()))
            except Exception as e:
                busy["on"] = False
                ui_q.put(("status", T("Error: {error}", error=e)))
                ui_q.put(("call", lambda: (menu.configure(state="normal"),
                                           menu.set(lang_name(cfg["interfaz"])))))
                return
            ui_q.put(("call", apply))
        threading.Thread(target=work, daemon=True).start()

    def page_shortcuts(w):
        """Capítulo "Atajos": lo que hace cada clic y cada tecla en las columnas."""
        ctk.CTkLabel(w, text=T("Atajos"), font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=16, pady=(14, 6))
        area = ctk.CTkScrollableFrame(w, fg_color="transparent")
        area.pack(fill="both", expand=True, padx=8, pady=(0, 10))
        for title, rows in SHORTCUTS:
            card = ctk.CTkFrame(area)
            card.pack(fill="x", padx=8, pady=5)
            ctk.CTkLabel(card, text=T(title), font=ctk.CTkFont(size=14, weight="bold"),
                         text_color="#6fa8dc", anchor="w").pack(fill="x", padx=12, pady=(8, 4))
            grid = ctk.CTkFrame(card, fg_color="transparent")
            grid.pack(fill="x", padx=12, pady=(0, 10))
            for r, (keys, what) in enumerate(rows):
                # La tecla o el gesto, como una tecla del teclado.
                ctk.CTkLabel(grid, text=T(keys), fg_color=("gray80", "gray30"), corner_radius=5,
                             font=ctk.CTkFont(weight="bold"), padx=8).grid(
                    row=r, column=0, sticky="w", padx=(0, 14), pady=3)
                ctk.CTkLabel(grid, text=T(what), anchor="w", justify="left", wraplength=600).grid(
                    row=r, column=1, sticky="w", pady=3)

    def page_models(w):
        """Capítulo "Modelos": qué modelos de IA usa ahora el programa, si están
        descargados y cuánto ocupan."""
        ctk.CTkLabel(w, text=T("Modelos que usa el programa"),
                     font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=16, pady=(14, 2))
        ctk.CTkLabel(w, text=T("Se descargan solos la primera vez que hacen falta, a la carpeta "
                               "\"modelos\". Después funcionan sin internet."), text_color="#9aa4b2",
                     anchor="w", justify="left", wraplength=640).pack(fill="x", padx=16, pady=(0, 8))
        area = ctk.CTkScrollableFrame(w, fg_color="transparent")
        area.pack(fill="both", expand=True, padx=8)
        for key, title, rows in models_info():
            card = ctk.CTkFrame(area)
            card.pack(fill="x", padx=8, pady=5)
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=14, weight="bold"),
                         text_color="#6fa8dc", anchor="w").pack(fill="x", padx=12, pady=(8, 2))
            for label, value, ok, *url in rows:
                row = ctk.CTkFrame(card, fg_color="transparent")
                row.pack(fill="x", padx=12, pady=1)
                ctk.CTkLabel(row, text=label, width=130, anchor="nw", text_color="#9aa4b2").pack(
                    side="left", anchor="n")
                color = None if ok is None else ("#7ee08a" if ok else "#e0a458")
                ctk.CTkLabel(row, text=value, anchor="w", justify="left", wraplength=520,
                             **({"text_color": color} if color else {})).pack(side="left", fill="x")
                if url and url[0]:
                    link(row, "Hugging Face ↗" if "huggingface" in url[0] else "web ↗", url[0]).pack(
                        side="left", padx=10)
            if key == "whisper":
                whisper_chooser(card)
            elif key == "tts":
                voice_chooser(card)
            ctk.CTkFrame(card, height=6, fg_color="transparent").pack()
        bottom = ctk.CTkFrame(w, fg_color="transparent")
        bottom.pack(fill="x", padx=16, pady=10)
        ctk.CTkLabel(bottom, text=T("Carpeta de modelos: {folder}  ·  {size} en total",
                                    folder=MODELS_DIR, size=mb(folder_size(MODELS_DIR))),
                     text_color="#9aa4b2").pack(side="left")
        ctk.CTkButton(bottom, text=T("Abrir carpeta"), width=110,
                      command=lambda: os.startfile(MODELS_DIR) if os.path.isdir(MODELS_DIR) else None).pack(
            side="right")

    # Herramientas → Modelos → Whisper: probar y elegir otro modelo.
    tests = {}         # modelo -> (segundos por pasada, duración, texto entendido) o mensaje
    busy = {"on": False}

    def whisper_file(modelo, gpu):
        """(repositorio, archivo o None, bytes descargados) del modelo para ese motor."""
        from motores import WHISPER_MODELS
        if gpu:
            fname = f"ggml-{WHISPER_MODELS.get(modelo, (modelo,))[0]}.bin"
            return "ggerganov/whisper.cpp", fname, file_size(fname)
        from faster_whisper.utils import _MODELS
        repo = _MODELS.get(modelo, f"Systran/faster-whisper-{modelo}")
        return repo, None, folder_size(hf_dir(repo))

    def whisper_chooser(card):
        """Tabla con los modelos de Whisper: tamaño, precisión, si está
        descargado, la prueba de velocidad en este equipo y botones Probar / Usar."""
        from motores import WHISPER_MODELS, MotorGPU
        gpu = conv.engine is None or isinstance(conv.engine, MotorGPU)
        table = ctk.CTkFrame(card, fg_color="transparent")
        table.pack(fill="x", padx=12, pady=(10, 0))
        for col, head in enumerate((T("Modelo"), T("Tamaño"), T("Precisión"), T("Estado"),
                                    T("Prueba en este equipo"), "", "")):
            ctk.CTkLabel(table, text=head, text_color="#9aa4b2", anchor="w").grid(
                row=0, column=col, sticky="w", padx=(0, 10))
        for r, (modelo, (_, size_mb, precision)) in enumerate(WHISPER_MODELS.items(), start=1):
            in_use = modelo == cfg["modelo"]
            _, _, have = whisper_file(modelo, gpu)
            ctk.CTkLabel(table, text=modelo + ("  " + T("(en uso)") if in_use else ""), anchor="w",
                         font=ctk.CTkFont(weight="bold" if in_use else "normal")).grid(
                row=r, column=0, sticky="w", padx=(0, 10))
            ctk.CTkLabel(table, text=f"{size_mb} MB" if gpu else "", anchor="e").grid(
                row=r, column=1, sticky="e", padx=(0, 10))
            ctk.CTkLabel(table, text=T(precision), anchor="w").grid(row=r, column=2, sticky="w", padx=(0, 10))
            ctk.CTkLabel(table, text="✔ " + T("descargado") if have else "↓ " + T("se descargará"),
                         text_color="#7ee08a" if have else "#9aa4b2", anchor="w").grid(
                row=r, column=3, sticky="w", padx=(0, 10))
            result = ctk.CTkLabel(table, text="", anchor="w", justify="left", wraplength=260)
            result.grid(row=r, column=4, sticky="w", padx=(0, 10))
            show_test(result, modelo)
            b_test = ctk.CTkButton(table, text=T("Probar"), width=70,
                                   command=lambda m=modelo, lab=result: run_test(m, lab))
            b_test.grid(row=r, column=5, padx=2, pady=2)
            b_use = ctk.CTkButton(table, text=T("Usar"), width=60, state="disabled" if in_use else "normal",
                                  command=lambda m=modelo: use_model(m))
            b_use.grid(row=r, column=6, padx=2, pady=2)
            repo, fname, _ = whisper_file(modelo, gpu)
            link(table, "↗", hf_url(repo, fname)).grid(row=r, column=7, padx=(8, 0))

    def show_test(label, modelo):
        got = tests.get(modelo)
        if got is None:
            label.configure(text="")
        elif isinstance(got, str):
            label.configure(text=got, text_color="#9aa4b2")
        else:
            secs, dur, heard = got
            # Cada pasada tarda casi lo mismo sea la frase larga o corta: es cada
            # cuánto se actualiza el texto mientras hablas (y lo que tarda en
            # cerrarse la frase cuando paras).
            verdict, color = ((T("fluido"), "#7ee08a") if secs <= 0.8 else
                              (T("algo más lento"), "#e0a458") if secs <= 1.5 else (T("lento"), "#e07a5f"))
            label.configure(text=verdict + " · " + T("el texto se actualiza cada {secs} s",
                                                     secs=f"{secs:.1f}".replace(".", ","))
                            + f"\n«{heard}»", text_color=color)

    def run_test(modelo, label):
        if busy["on"] or conv.speaker is None:
            return
        busy["on"] = True
        tests[modelo] = T("descargando y probando…") if not whisper_file(
            modelo, conv.engine is None or type(conv.engine).__name__ == "MotorGPU")[2] else T("probando…")
        show_test(label, modelo)

        def work():
            try:
                tests[modelo] = conv.test_model(modelo, log=lambda m: None)
            except Exception as e:
                tests[modelo] = T("error: {error}", error=e)
            busy["on"] = False
            ui_q.put(("call", lambda: refresh_models()))
        threading.Thread(target=work, daemon=True).start()

    def use_model(modelo):
        if busy["on"] or not state["ready"]:
            return
        busy["on"] = True
        tests.setdefault(modelo, T("cambiando de modelo…"))

        def work():
            try:
                conv.change_model(modelo)
                save_config(cfg)
            except Exception as e:
                ui_q.put(("status", T("No se pudo cambiar al modelo {model}: {error}", model=modelo, error=e)))
            if isinstance(tests.get(modelo), str):
                tests.pop(modelo)
            busy["on"] = False
            ui_q.put(("call", lambda: refresh_models()))
        threading.Thread(target=work, daemon=True).start()
        refresh_models()

    def refresh_models():
        if help_win.get("w") is not None and help_win["w"].winfo_exists() and help_win.get("page") == "modelos":
            help_win["show"]("modelos")

    def models_info():
        """[(clave, título, [(dato, valor, ¿bien?|None)])] de cada modelo en uso."""
        from motores import WHISPER_MODELS, MotorGPU
        from traductor import MT_MODELS, translation_path
        out = []
        # Voz -> texto
        modelo = cfg["modelo"]
        gpu = isinstance(conv.engine, MotorGPU)
        if conv.engine is None:
            motor = T("aún cargando…")
        elif gpu:
            motor = T("whisper.cpp con Vulkan en la tarjeta gráfica ({device})", device=conv.engine.name)
        else:
            motor = T("faster-whisper (CTranslate2) en {device}", device=conv.engine.name)
        repo, fname, size = whisper_file(modelo, gpu or conv.engine is None)
        out.append(("whisper", T("Reconocimiento de voz (voz → texto)"), [
            (T("Modelo"), T("Whisper {model} (multilingüe), de OpenAI · precisión {precision}", model=modelo,
                            precision=T(WHISPER_MODELS.get(modelo, ("", 0, "?"))[2])), None),
            (T("Motor"), motor, None),
            (T("Archivo"), f"{repo}" + (f" · {fname}" if fname else ""), None, hf_url(repo, fname)),
            (T("Estado"), T("descargado · {size}", size=mb(size)) if size else T("se descargará al usarlo"),
             bool(size)),
        ]))
        vad = os.path.join(os.path.dirname(__import__("faster_whisper").__file__), "assets", "silero_vad_v6.onnx")
        out.append(("vad", T("Detección de voz"), [
            (T("Modelo"), T("Silero VAD (incluido en faster-whisper)"), None,
             "https://github.com/snakers4/silero-vad"),
            (T("Estado"), T("instalado · {size}", size=mb(os.path.getsize(vad))) if os.path.exists(vad)
             else T("no encontrado"), os.path.exists(vad)),
            (T("Para qué"), T("Deja pasar solo voz humana (no golpes, teclado ni clics)."), None),
        ]))
        # Traducción: los dos sentidos de los idiomas elegidos.
        rows = []
        for src, tgt in ((cfg["yo"], cfg["el"]), (cfg["el"], cfg["yo"])):
            label = f"{lang_name(src)} → {lang_name(tgt)}"
            steps = translation_path(src, tgt)
            if steps == []:
                rows.append((label, T("mismo idioma: no se traduce"), None))
                continue
            if steps is None:
                rows.append((label, T("no hay traductor sin internet para este par"), False))
                continue
            for i, (a, b) in enumerate(steps):  # uno por paso (dos si pasa por el inglés)
                r = MT_MODELS[(a, b)]
                s = folder_size(hf_dir(r))
                via = f"  ({in_text(a)} → {in_text(b)})" if len(steps) > 1 else ""
                rows.append((label if i == 0 else "", f"{r}{via} · " + (mb(s) if s else T("se descargará al usarlo")),
                             bool(s), hf_url(r)))
        rows.append((T("Para qué"), T("Traduce cada frase sin internet (Opus-MT, de la Universidad de Helsinki); "
                                      "si no hay modelo directo, pasa por el inglés."), None))
        out.append(("mt", T("Traducción"), rows))
        # Voz sintética: la del idioma del otro (en ella se dice mi frase traducida).
        lang = cfg["el"]
        if lang in VOICES:
            voice = VOICES[lang].rsplit("/", 1)[-1]
            s = file_size(voice + ".onnx")
            vrows = [(T("Modelo"), f"Piper · {voice}", None,
                      f"https://huggingface.co/rhasspy/piper-voices/tree/main/{VOICES[lang].rsplit('/', 1)[0]}"),
                     (T("Estado"), T("descargada · {size}", size=mb(s)) if s else T("se descargará al usarla"),
                      bool(s))]
        else:
            vrows = [(T("Modelo"), T("no hay voz en {language}: él no oirá tu traducción",
                                     language=in_text(lang)), False)]
        if cfg.get("mi_voz"):  # con mi voz: el conversor de timbre
            import mivoz
            s = folder_size(hf_dir(mivoz.CONVERTER_REPO))
            vrows.append((T("Mi voz"), T("OpenVoice v2 (conversor de timbre, de MyShell)"), None,
                          hf_url(mivoz.CONVERTER_REPO)))
            vrows.append(("", T("descargado · {size}", size=mb(s)) if s else T("se descargará al usarlo"),
                          bool(s)))
        out.append(("tts", T("Voz sintética (texto → voz)"), vrows))
        return out

    # Herramientas → Modelos → Voz sintética: con qué voz suena mi frase traducida
    # (la de Piper o una mía grabada: mivoz.py) y el asistente "Nueva voz".
    voice_state = {"msg": "", "wizard": None}
    READ_TEXTS = {
        "es": "Hola, esta es una muestra de mi voz para el traductor. Mañana te mando la factura del mes "
              "pasado y luego hablamos del contrato con calma. Me parece bien quedar el jueves por la "
              "mañana, aunque si te viene mejor el viernes, también puedo. ¿Qué opinas de la propuesta? "
              "Creo que podemos cerrar el acuerdo esta misma semana, y así empezamos el proyecto nuevo "
              "cuanto antes.",
        "en": "Hello, this is a sample of my voice for the translator. Tomorrow I will send you last "
              "month's invoice, and then we can talk about the contract calmly. Thursday morning works "
              "for me, but if Friday is better for you, that's fine too. What do you think about the "
              "proposal? I believe we can close the deal this week and start the new project as soon "
              "as possible.",
    }
    REC_MAX, REC_MIN = 60, 10  # segundos de grabación

    def voice_chooser(card):
        import mivoz
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(row, text=T("Voz"), width=130, anchor="w", text_color="#9aa4b2").pack(side="left")
        piper = T("Piper (voz del idioma)")
        options = {piper: ""}
        options.update({T("Mi voz: {name}", name=n): n for n in mivoz.profiles()})
        current = cfg.get("mi_voz", "")
        menu = ctk.CTkOptionMenu(row, width=230, values=list(options), dynamic_resizing=False,
                                 command=lambda label: choose_voice(options[label]))
        menu.set(next((k for k, v in options.items() if v == current), piper))
        menu.pack(side="left")
        ctk.CTkButton(row, text=T("Nueva voz…"), width=100, command=new_voice).pack(side="left", padx=(10, 0))
        if current:
            ctk.CTkButton(row, text=T("Borrar"), width=70, fg_color="#7a4a4a", hover_color="#8f5555",
                          command=lambda: delete_voice(current)).pack(side="left", padx=(6, 0))
        note = voice_state["msg"] or (T("Tu timbre sobre la voz de Piper de cada idioma (el acento sigue "
                                        "siendo el de Piper).") if current else "")
        if note:
            ctk.CTkLabel(card, text=note, text_color="#9aa4b2", anchor="w", justify="left",
                         wraplength=620).pack(fill="x", padx=(142, 12), pady=(4, 0))

    def choose_voice(name):
        cfg["mi_voz"] = name
        save_config(cfg)
        if conv.speaker is None:  # aún cargando: load() la aplica al terminar
            refresh_models()
            return
        voice_state["msg"] = T("Cargando tu voz…") if name else ""
        refresh_models()

        def work():
            try:
                conv.set_my_voice(name)
                voice_state["msg"] = ""
            except Exception as e:
                voice_state["msg"] = T("No se pudo cargar tu voz: {error}", error=e)
                conv.speaker.set_style(None)
            ui_q.put(("call", refresh_models))
        threading.Thread(target=work, daemon=True).start()

    def delete_voice(name):
        from tkinter import messagebox
        import mivoz
        if not messagebox.askyesno(T("Borrar voz"), T("¿Borrar tu voz «{name}»? También se borra su grabación.",
                                                     name=name), parent=help_win.get("w") or root):
            return
        if cfg.get("mi_voz") == name:
            choose_voice("")
        mivoz.delete_profile(name)
        refresh_models()

    def new_voice():
        """Asistente "Nueva voz": grabar mi voz, escucharla, probarla en el idioma
        del otro y guardarla. Mientras está abierto la conversación se pausa (lo
        que grabo se traduciría y lo que suena por los altavoces se tomaría por suyo)."""
        import winsound
        import mivoz
        if voice_state["wizard"] is not None and voice_state["wizard"].winfo_exists():
            voice_state["wizard"].focus()
            return
        parent = help_win.get("w") if help_win.get("w") is not None and help_win["w"].winfo_exists() else root
        w = ctk.CTkToplevel(parent)
        voice_state["wizard"] = w
        w.title(T("Nueva voz"))
        w.geometry("740x600")
        w.transient(parent)
        paused = state["ready"]
        if paused:
            ui_q.put(("status", T("Conversación en pausa mientras creas una voz")))
            threading.Thread(target=safe(conv.stop), daemon=True).start()
        os.makedirs(mivoz.VOICES_DIR, exist_ok=True)
        tmp = os.path.join(mivoz.VOICES_DIR, "_prueba.wav")
        rec = {"on": False, "audio": None, "rate": 16000, "level": 0.0, "busy": False, "start": 0.0}

        def close():
            rec["on"] = False
            winsound.PlaySound(None, 0)
            voice_state["wizard"] = None
            w.destroy()
            if paused:
                threading.Thread(target=safe(conv.start), daemon=True).start()
            refresh_models()
        w.protocol("WM_DELETE_WINDOW", close)
        body = ctk.CTkFrame(w, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=14)

        def heading():
            for c in body.winfo_children():
                c.destroy()
            ctk.CTkLabel(body, text=T("Nueva voz"), font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w")

        def build_install():
            """Sin PyTorch: se instala aquí, una vez (bajo demanda)."""
            heading()
            ctk.CTkLabel(body, text=T("Mi voz necesita PyTorch: unos 200 MB de descarga (alrededor de 1 GB "
                                      "instalado). Se instala una sola vez, en el entorno del programa."),
                         anchor="w", justify="left", wraplength=680).pack(fill="x", pady=(8, 12))
            btn = ctk.CTkButton(body, text=T("Instalar componentes"), width=190)
            btn.pack(anchor="w")
            msg = ctk.CTkLabel(body, text="", text_color="#9aa4b2", anchor="w", justify="left", wraplength=680)
            msg.pack(fill="x", pady=10)

            def show(text, color="#9aa4b2"):
                if msg.winfo_exists():
                    msg.configure(text=text, text_color=color)

            def install():
                btn.configure(state="disabled")
                show(T("Instalando… (puede tardar unos minutos)"))

                def work():
                    try:
                        mivoz.install_components(log=lambda line: ui_q.put(("call", lambda: show(line[-160:]))))
                    except Exception as e:
                        err = str(e)
                        ui_q.put(("call", lambda: (show(T("No se pudo instalar: {error}", error=err), "#e07a5f"),
                                                   btn.configure(state="normal"))))
                        return
                    ui_q.put(("call", lambda: build_record() if w.winfo_exists() else None))
                threading.Thread(target=work, daemon=True).start()
            btn.configure(command=install)

        def build_record():
            heading()
            form = ctk.CTkFrame(body, fg_color="transparent")
            form.pack(fill="x", pady=(10, 6))
            ctk.CTkLabel(form, text=T("Nombre"), width=90, anchor="w").grid(row=0, column=0, sticky="w")
            taken = set(mivoz.profiles())
            default = next(n for n in [T("Mi voz")] + [f"{T('Mi voz')} {i}" for i in range(2, 99)]
                           if mivoz.safe_name(n) not in taken)
            name = ctk.CTkEntry(form, width=260)
            name.insert(0, default)
            name.grid(row=0, column=1, sticky="w", pady=3)
            ctk.CTkLabel(form, text=T("Micrófono"), width=90, anchor="w").grid(row=1, column=0, sticky="w")
            mic_names = [n for _, n in input_devices()]
            mic = ctk.CTkOptionMenu(form, width=420, values=mic_names or [DEFAULT], dynamic_resizing=False)
            mic.set(cfg["microfono"] if cfg["microfono"] in mic_names else (mic_names or [DEFAULT])[0])
            mic.grid(row=1, column=1, sticky="w", pady=3)
            warn = ctk.CTkLabel(form, text="", text_color="#e0a458", anchor="w")
            warn.grid(row=2, column=1, sticky="w")

            def check_mic(*_):
                warn.configure(text=T("Es un cable virtual: elige tu micrófono de verdad.")
                               if is_virtual(mic.get()) else "")
            mic.configure(command=check_mic)
            check_mic()
            me = cfg["yo"]
            text = READ_TEXTS.get(me) or TEST_PHRASES.get(me, READ_TEXTS["es"])
            ctk.CTkLabel(body, text=T("Lee este texto en voz alta, con tu tono normal ({min}–{max} s):",
                                      min=30, max=REC_MAX) if me in READ_TEXTS else
                         T("Lee esta frase y sigue hablando con naturalidad, de lo que quieras, hasta 30 s:"),
                         text_color="#9aa4b2", anchor="w").pack(fill="x", pady=(8, 2))
            tk.Label(body, text=text, fg="#dce4ee", bg=ctk.ThemeManager.theme["CTk"]["fg_color"][1],
                     font=("Segoe UI", 14), wraplength=690, justify="left", anchor="w").pack(fill="x")
            meter = tk.Canvas(body, width=690, height=10, bg="#333333", highlightthickness=0)
            meter.pack(pady=(12, 4), anchor="w")
            level = meter.create_rectangle(0, 0, 0, 10, fill="#7ee08a", width=0)
            status = ctk.CTkLabel(body, text=T("Pulsa Grabar y lee el texto."), text_color="#9aa4b2",
                                  anchor="w", justify="left", wraplength=690)
            status.pack(fill="x")
            btns = ctk.CTkFrame(body, fg_color="transparent")
            btns.pack(anchor="w", pady=(12, 0))
            b_rec = ctk.CTkButton(btns, text=T("Grabar"), width=110)
            b_play = ctk.CTkButton(btns, text=T("Escuchar"), width=110, state="disabled")
            b_test = ctk.CTkButton(btns, text=T("Probar en {language}", language=in_text(cfg["el"])),
                                   width=150, state="disabled")
            b_save = ctk.CTkButton(btns, text=T("Guardar"), width=110, state="disabled",
                                   fg_color="#2e7d4f", hover_color="#256640")
            for b in (b_rec, b_play, b_test, b_save):
                b.pack(side="left", padx=(0, 8))

            def tell(text_, color="#9aa4b2"):
                if status.winfo_exists():
                    status.configure(text=text_, text_color=color)

            def buttons():
                have = rec["audio"] is not None and not rec["on"] and not rec["busy"]
                b_rec.configure(text=T("Parar") if rec["on"] else T("Grabar") if rec["audio"] is None
                                else T("Grabar otra vez"), state="disabled" if rec["busy"] else "normal")
                for b in (b_play, b_test, b_save):
                    b.configure(state="normal" if have else "disabled")

            def record():
                import pyaudiowpatch as pa
                idx = device_index(input_devices(), mic.get())
                hub = AudioHub.get()
                dev = hub.call(lambda p: p.get_device_info_by_index(idx) if idx is not None
                               else p.get_default_input_device_info())
                rate, ch = int(dev["defaultSampleRate"]), max(1, int(dev["maxInputChannels"]))
                chunks = []

                def cb(data, frames, t, st):
                    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
                    if ch > 1:
                        a = a.reshape(-1, ch).mean(axis=1)
                    chunks.append(a)
                    rec["level"] = float(np.sqrt(np.mean(a ** 2)))
                    return (data, pa.paContinue)

                try:
                    stream = hub.call(lambda p: p.open(format=pa.paInt16, channels=ch, rate=rate, input=True,
                                                       input_device_index=dev["index"],
                                                       frames_per_buffer=rate // 10, stream_callback=cb))
                except Exception as e:
                    err = str(e)
                    rec["on"] = False
                    ui_q.put(("call", lambda: (tell(T("No se pudo abrir el micrófono: {error}", error=err),
                                                   "#e07a5f"), buttons())))
                    return
                while rec["on"] and time.monotonic() - rec["start"] < REC_MAX:
                    time.sleep(0.05)
                close_stream(stream)
                rec["on"] = False
                audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
                secs = len(audio) / rate
                if secs < REC_MIN:
                    ui_q.put(("call", lambda: (tell(T("Muy corta ({secs} s): graba al menos {min} s.",
                                                     secs=round(secs), min=REC_MIN), "#e0a458"), buttons())))
                    return
                rec.update(audio=audio, rate=rate)
                mivoz.write_wav(tmp, audio, rate)
                quiet = float(np.sqrt(np.mean(audio ** 2))) < 0.01
                ui_q.put(("call", lambda: (tell(T("Grabados {secs} s.", secs=round(secs)) + " " +
                                               (T("Suena muy bajo: acércate al micrófono o sube su volumen.")
                                                if quiet else T("Escúchala, pruébala y guárdala.")),
                                               "#e0a458" if quiet else "#7ee08a"), buttons())))

            def toggle():
                if rec["on"]:
                    rec["on"] = False
                    return
                winsound.PlaySound(None, 0)
                rec.update(on=True, start=time.monotonic())
                buttons()
                threading.Thread(target=record, daemon=True).start()

            def play():
                winsound.PlaySound(tmp, winsound.SND_FILENAME | winsound.SND_ASYNC)

            def run(job, doing):
                """Trabajo que tarda (convertir, guardar), en otro hilo."""
                rec["busy"] = True
                buttons()
                tell(doing)

                def work():
                    try:
                        after = job()
                    except Exception as e:
                        err = str(e)
                        after = lambda: tell(T("Error: {error}", error=err), "#e07a5f")
                    rec["busy"] = False
                    ui_q.put(("call", lambda: (after(), buttons() if w.winfo_exists() else None)))
                threading.Thread(target=work, daemon=True).start()

            def test():
                def job():
                    if conv.speaker is None:
                        raise RuntimeError(T("el programa aún está cargando"))
                    se = mivoz.Converter.get().extract_se(rec["audio"], rec["rate"])
                    lang = cfg["el"]
                    phrase = TEST_PHRASES.get(lang, TEST_PHRASES["en"])
                    audio, sr = conv.speaker.synthesize(phrase, lang)
                    out, osr = mivoz.MyVoice("", conv.speaker.synthesize, VOICES, TEST_PHRASES, se=se)(audio, sr, lang)
                    path = os.path.join(mivoz.VOICES_DIR, "_prueba_convertida.wav")
                    mivoz.write_wav(path, out, osr)
                    return lambda: (tell(f"«{phrase}»"), winsound.PlaySound(
                        path, winsound.SND_FILENAME | winsound.SND_ASYNC))
                run(job, T("Convirtiendo una frase a tu voz… (la primera vez descarga el conversor, ~130 MB)"))

            def save():
                label = mivoz.safe_name(name.get())
                if not label:
                    tell(T("Ponle un nombre."), "#e0a458")
                    return

                def job():
                    mivoz.save_profile(label, rec["audio"], rec["rate"])
                    return lambda: (choose_voice(label), close())
                run(job, T("Guardando tu voz…"))

            b_rec.configure(command=toggle)
            b_play.configure(command=play)
            b_test.configure(command=test)
            b_save.configure(command=save)

            def tick():
                if not w.winfo_exists():
                    return
                if rec["on"]:
                    left = REC_MAX - (time.monotonic() - rec["start"])
                    tell(T("Grabando… {secs} s (puedes parar cuando acabes el texto)",
                          secs=round(time.monotonic() - rec["start"])), "#e0a458")
                    meter.coords(level, 0, 0, int(690 * min(1.0, rec["level"] * 12)), 10)
                    if left <= 0:
                        rec["on"] = False
                else:
                    meter.coords(level, 0, 0, 0, 10)
                w.after(80, tick)
            tick()

        if mivoz.components_ready():
            build_record()
        else:
            build_install()

    def hf_url(repo, fname=None):
        """Página del modelo en Hugging Face (o la del archivo, si es uno concreto)."""
        return f"https://huggingface.co/{repo}" + (f"/blob/main/{fname}" if fname else "")

    def link(parent, text, url):
        """Etiqueta que parece un enlace y abre url en el navegador."""
        import webbrowser
        lab = ctk.CTkLabel(parent, text=text, text_color="#6fb6ff", cursor="hand2",
                           font=ctk.CTkFont(underline=True))
        lab.bind("<Button-1>", lambda _: webbrowser.open(url))
        return lab

    def hf_dir(repo):
        """Carpeta de un modelo de Hugging Face dentro de "modelos"."""
        return os.path.join(MODELS_DIR, "models--" + repo.replace("/", "--"))

    def folder_size(path):
        total = 0
        for base, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(base, f))
                except OSError:
                    pass
        return total

    def file_size(name):
        """Tamaño de un archivo descargado (busca por nombre en "modelos")."""
        for base, _, files in os.walk(MODELS_DIR):
            if name in files:
                return os.path.getsize(os.path.join(base, name))
        return 0

    def mb(n):
        return f"{n / 1e6:,.0f} MB".replace(",", ".") if n < 1e9 else f"{n / 1e9:.1f} GB".replace(".", ",")

    # -- ajustes que reinician la sesión ----------------------------------- #
    def changed(*_):
        new = {
            "modo": mode_codes[mode_btn.get()],
            "yo": codes[me_menu.get()],
            "el": codes[them_menu.get()],
            "microfono": "" if mic_menu.get() == DEFAULT else mic_menu.get(),
            "salida_voz": "" if out_menu.get() == DEFAULT else out_menu.get(),
        }
        restart = any(cfg[k] != v for k, v in new.items())
        lang_changed = new["yo"] != cfg["yo"] or new["el"] != cfg["el"]
        cfg.update(new)
        save_config(cfg)
        update_subtitles()
        if restart and state["ready"]:
            root.title(T("Reiniciando…"))
            threading.Thread(target=safe(conv.restart), daemon=True).start()
        if lang_changed:
            # Tras elegir idioma, el teclado vuelve a mi columna (para seguir con
            # ↑/↓ o escribir). Un poco después: el desplegable se queda el foco al cerrarse.
            root.after(150, focus_me)

    for w in (mode_btn, me_menu, them_menu, mic_menu, out_menu):
        w.configure(command=changed)

    def safe(fn):
        def run():
            try:
                fn()
            except Exception as e:
                ui_q.put(("status", T("Error: {error}", error=e)))
        return run

    def boot():
        conv.load()
        conv.start()

    threading.Thread(target=safe(boot), daemon=True).start()

    def focus_me():
        """El teclado a la columna YO (al abrir y tras elegir idioma): se puede
        escribir o pegar (Ctrl+V) una frase nueva sin hacer clic antes, o seguir
        con la frase que se estaba corrigiendo."""
        tb = cols[ME]["tb"]
        if editing.get("tb") is tb:
            tb.mark_set("insert", "ed_end")
        else:
            tb.mark_set("insert", "end-1c")
        tb.focus_force()

    root.after(300, focus_me)  # cuando la ventana ya se ve

    def draw_wave(c, peaks):
        """Línea que oscila arriba y abajo con la altura de cada pico, de lo más
        antiguo (izquierda) a lo último (derecha). La raíz realza la voz baja."""
        # Un punto cada 4 px con el mayor de cada 2 picos (50 ms): línea suave, no un borrón.
        n, mid, half = WAVE_W // 4, WAVE_H // 2, WAVE_H // 2 - 2
        peaks = (peaks or [])[-2 * n:]
        peaks = [0.0] * (2 * n - len(peaks)) + peaks
        pts = []
        for i in range(n):
            h = half * min(1.0, (max(peaks[2 * i], peaks[2 * i + 1]) * 4) ** 0.6)
            pts += [i * 4 + 2, mid - h if i % 2 else mid + h]
        c["meter"].coords(c["line"], *pts)

    # -- bucle de la ventana ----------------------------------------------- #
    def poll():
        try:
            while True:
                kind, *rest = ui_q.get_nowait()
                if kind == "status":
                    root.title(rest[0])
                elif kind == "ready":
                    state["ready"] = True
                elif kind == "entry":
                    render(rest[0])
                elif kind == "edited":  # frase corregida: su nueva traducción
                    edited(*rest)
                elif kind == "discard":  # el turno era ruido: se quita
                    discard_live(rest[0])
                elif kind == "live_tr":  # traducción de una frase en curso corregida
                    got_live_translation(*rest)
                elif kind == "call":  # algo hecho en otro hilo que hay que pintar aquí
                    rest[0]()
                elif kind == "speaking":
                    state["speaking"] = rest[0]
                    set_voiced(rest[0], "on")
                    cols[ME]["label"].configure(text=T("YO") + "   🔊 " + T("diciendo {n}", n=rest[0]))
                elif kind == "spoken":  # acabó de sonar (o no pudo sonar)
                    num, ok = rest
                    set_voiced(num, "done" if ok else "off")
                    if state["speaking"] == num:
                        state["speaking"] = None
                        cols[ME]["label"].configure(text=T("YO"))
        except queue.Empty:
            pass
        waves = conv.waves()
        for side, c in cols.items():
            draw_wave(c, waves.get(side))
        if conv.echo_on:
            ec = conv.echo
            if not ec.measured:
                txt = T("eco: esperando a medir")
            else:
                txt = T("eco: comprobado") if ec.validated else T("eco: aprendiendo")
                if ec.last_erle is not None:
                    txt += f" ({ec.last_erle:.0f} dB)"
        else:
            txt = ""
        cols[THEM]["echo"].configure(text=txt)
        cols[ME]["echo"].configure(
            text="⚠ " + T("el micrófono oye los altavoces: se silencia mientras suenan")
            if cfg["modo"] == "llamada" and conv.leak.active() else "",
            text_color="#e0a458")
        root.after(80, poll)

    def on_close():
        conv.stop()
        if conv.speaker is not None:
            conv.speaker.close()  # la salida de voz está siempre abierta
        if conv.engine is not None:
            conv.engine.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(80, poll)
    root.mainloop()
    return result["restart"]


def main():
    import argparse
    langs = ", ".join(f"{c} ({n.lower()})" for c, n in LANG_NAMES.items())
    ap = argparse.ArgumentParser(
        description="Conversación traducida en los dos sentidos. Sin argumentos se usan "
                    "los idiomas guardados (se pueden cambiar en la ventana).")
    ap.add_argument("--yo", choices=LANGS, help=f"idioma en que hablo yo: {langs}")
    ap.add_argument("--el", choices=LANGS, help="idioma en que habla él")
    ap.add_argument("--modo", choices=["llamada", "presencial"],
                    help="llamada: él suena por el PC; presencial: él está a mi lado")
    args = ap.parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    # Al cambiar el idioma de la interfaz, la ventana se vuelve a abrir.
    while run_gui(overrides):
        overrides = {}


if __name__ == "__main__":
    main()
