"""
Traductor en directo: captura el audio que sale por los altavoces de Windows
(loopback WASAPI), lo transcribe en inglés con Whisper y lo muestra traducido
al español en una ventana. El micrófono NO se usa, así que tu voz no aparece.

Uso:
    python traductor.py                 # arranca con el altavoz por defecto
    python traductor.py --dispositivos  # lista las salidas de audio
    python traductor.py --dispositivo 12
    python traductor.py --modelo base.en   # más rápido, menos preciso
    python traductor.py --descargar     # solo descarga el modelo y sale
"""
import argparse
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from math import gcd

import numpy as np

from interfaz import T

TARGET_SR = 16000
BLOCK_SEC = 0.1
SILENCE_SEC = 0.6      # silencio que cierra una frase
MAX_SEG_SEC = 8.0      # longitud máxima de un fragmento
MIN_SEG_SEC = 0.5      # fragmentos más cortos se descartan
PREROLL_SEC = 0.3
PARTIAL_SEC = 0.5      # cada cuánto se muestra el texto provisional mientras se habla
MT_MODELS = {  # traductores offline (Opus-MT convertidos a CTranslate2)
    ("en", "es"): "michaelfeil/ct2fast-opus-mt-en-es",
    ("es", "en"): "michaelfeil/ct2fast-opus-mt-es-en",
    **{(a, b): f"ooeoeo/opus-mt-{a}-{b}-ct2-float16" for a, b in (
        ("es", "fr"), ("es", "de"), ("es", "it"),
        ("en", "fr"), ("en", "de"), ("en", "it"),
        ("fr", "es"), ("fr", "en"), ("fr", "de"),
        ("de", "es"), ("de", "en"), ("de", "fr"), ("de", "it"),
        ("it", "es"), ("it", "en"), ("it", "fr"), ("it", "de"),
        # Idiomas añadidos después: con el inglés en los dos sentidos (así se
        # combinan con cualquier otro pasando por él) y directos con el español
        # cuando existen.
        ("en", "ru"), ("ru", "en"), ("es", "ru"), ("ru", "es"),
        ("en", "ca"), ("ca", "en"), ("ca", "es"),
        ("en", "vi"), ("vi", "en"), ("es", "vi"), ("vi", "es"),
        ("en", "zh"), ("zh", "en"),
        ("en", "ar"), ("ar", "en"), ("es", "ar"), ("ar", "es"),
    )},
    ("en", "pt"): "ooeoeo/opus-mt-tc-big-en-pt-ct2-float16",
    ("pt", "en"): "michaelfeil/ct2fast-opus-mt-ROMANCE-en",  # de cualquier lengua romance
}
# Modelos con varios idiomas de destino: hay que decirles cuál con una marca
# delante de cada frase.
MT_PREFIX = {("en", "zh"): ">>cmn_Hans<<",  # chino mandarín, caracteres simplificados
             # Árabe estándar moderno (otras variantes: arz egipcio, apc levantino...).
             ("en", "ar"): ">>ara<<", ("es", "ar"): ">>ara<<",
             ("en", "pt"): ">>por<<"}  # portugués de Portugal (>>pob<<: de Brasil)
PIVOT = "en"  # pares sin modelo directo (p. ej. fr -> it): se pasa por el inglés
# Idiomas que no separan las palabras con espacios.
NOSPACE = {"zh", "ja", "yue", "th"}


def translation_path(source, target):
    """Pasos para traducir: [] (mismo idioma), [(a, b)] (directo),
    [(a, en), (en, b)] (a través del inglés) o None (no hay traductor)."""
    if source == target:
        return []
    if (source, target) in MT_MODELS:
        return [(source, target)]
    if (source, PIVOT) in MT_MODELS and (PIVOT, target) in MT_MODELS:
        return [(source, PIVOT), (PIVOT, target)]
    return None
HALLUCINATIONS = {  # lo que Whisper "oye" en ruido o silencio
    "you", "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "bye.", "bye", ".", "okay.", "so",
    "gracias.", "gracias", "¡gracias!", "gracias por ver el video.",
    "gracias por ver el vídeo.", "¡suscríbete!", "adiós.", "chao.",
    "subtítulos realizados por la comunidad de amara.org",
    "y", "y.", "and", "and.", "eh", "eh.", "mm", "mmm", "hmm", "uh", "um",
    "obrigado.", "obrigado", "obrigada.", "obrigada", "obrigado por assistir.",
    "legendas pela comunidade amara.org",
    "ترجمة نانسي قنقر", "شكرا", "شكرا.", "شكراً", "شكراً.", "شكرا لكم", "شكرا لكم.",
    "شكرا على المشاهدة", "شكرا على المشاهدة.", "شكرا لكم على المشاهدة",
    "شكرا لكم على المشاهدة.", "اشتركوا في القناة", "اشتركوا في القناة.", "موسيقى", "[موسيقى]",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Todos los modelos (Whisper y traductor) se guardan dentro del proyecto, así la
# carpeta se puede copiar a otro ordenador sin volver a descargarlos.
MODELS_DIR = os.path.join(BASE_DIR, "modelos")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def cuda_device_count():
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count()
    except Exception:
        return 0


def load_model(name, force_cpu=False, log=print):
    from faster_whisper import WhisperModel

    if not force_cpu:
        if cuda_device_count() == 0:
            log(T("No hay tarjeta NVIDIA compatible con CUDA 12 "
                  "(las AMD e Intel no sirven para este motor). Usando CPU."))
        else:
            try:
                log(T("Probando tarjeta gráfica (CUDA)..."))
                m = WhisperModel(name, device="cuda", compute_type="float16",
                                 download_root=MODELS_DIR)
                noise = (np.random.randn(TARGET_SR) * 0.01).astype(np.float32)
                list(m.transcribe(noise, language="en", beam_size=1)[0])
                log(T("Usando tarjeta gráfica."))
                return m, "cuda"
            except Exception as e:  # sin librerías cuBLAS/cuDNN, poca memoria...
                log(T("La tarjeta NVIDIA no se pudo usar: {error}. Usando CPU.", error=e))
    # faster-whisper usa solo 4 hilos si no se indica; os.cpu_count() cuenta
    # hilos lógicos, así que la mitad aproxima los núcleos físicos.
    threads = max(4, (os.cpu_count() or 8) // 2)
    m = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads,
                     download_root=MODELS_DIR)
    return m, "cpu"


# --------------------------------------------------------------------------- #
# Captura de audio (loopback)
# --------------------------------------------------------------------------- #
def list_devices():
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        print("Salidas de audio capturables (loopback):")
        for d in p.get_loopback_device_info_generator():
            print(f"  [{d['index']}] {d['name']}  ({int(d['defaultSampleRate'])} Hz)")
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        dflt = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        print(f"\nSalida por defecto ahora mismo: {dflt['name']}")
    finally:
        p.terminate()


def input_devices():
    """[(índice, nombre)] de los micrófonos WASAPI; el primero es el de por defecto."""
    return wasapi_devices(output=False)


def wasapi_devices(output):
    """[(índice, nombre)] de entradas u salidas WASAPI (sin loopback); la de
    por defecto primero."""
    import pyaudiowpatch as pyaudio

    def list_(p):
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default = wasapi["defaultOutputDevice" if output else "defaultInputDevice"]
        key = "maxOutputChannels" if output else "maxInputChannels"
        devs = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] == wasapi["index"] and d[key] > 0 and not d.get("isLoopbackDevice"):
                devs.append((i, d["name"]))
        devs.sort(key=lambda x: x[0] != default)
        return devs

    return AudioHub.get().call(list_)


def find_loopback(p, index=None):
    import pyaudiowpatch as pyaudio
    if index is not None:
        return p.get_device_info_by_index(index)
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for lb in p.get_loopback_device_info_generator():
        if speakers["name"] in lb["name"]:
            return lb
    raise RuntimeError(
        "No encuentro el loopback del altavoz por defecto. "
        "Ejecuta con --dispositivos y elige uno con --dispositivo N."
    )


def find_microphone(p, index=None):
    import pyaudiowpatch as pyaudio
    if index is not None:
        return p.get_device_info_by_index(index)
    wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    return p.get_device_info_by_index(wasapi["defaultInputDevice"])


class AudioHub(threading.Thread):
    """Único dueño de PyAudio en el programa.

    Con WASAPI, abrir un segundo dispositivo desde otro hilo (o reabrir uno)
    falla con "Unanticipated host error". Por eso todas las aperturas y cierres
    (capturas y salida de voz) se hacen en este hilo: hub().call(fn) ejecuta
    fn(pyaudio) aquí y devuelve el resultado."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                cls._instance.start()
            return cls._instance

    def run(self):
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        while True:
            fn, result = self.jobs.get()
            try:
                result.put((fn(p), None))
            except Exception as e:
                result.put((None, e))

    def call(self, fn):
        result = queue.Queue()
        self.jobs.put((fn, result))
        value, error = result.get()
        if error is not None:
            raise error
        return value


def close_stream(stream):
    def close(_):
        try:
            stream.stop_stream()
        finally:
            stream.close()
    AudioHub.get().call(close)


class Capture(threading.Thread):
    """Mete bloques mono float32 (a la frecuencia del dispositivo) en out_q.
    mic=False: lo que suena por los altavoces (loopback); mic=True: micrófono."""

    def __init__(self, out_q, device_index, log, mic=False):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.device_index = device_index
        self.log = log
        self.mic = mic
        self.rate = None
        self.device_name = None
        self.ready = threading.Event()
        self.error = None
        self._halt = threading.Event()

    def run(self):
        import pyaudiowpatch as pyaudio
        hub = AudioHub.get()
        stream = None
        try:
            find = find_microphone if self.mic else find_loopback
            dev = hub.call(lambda p: find(p, self.device_index))
            self.rate = int(dev["defaultSampleRate"])
            self.device_name = dev["name"]
            ch = int(dev["maxInputChannels"]) or 2
            self.log(T("Escuchando: {device}", device=dev['name']))

            def cb(in_data, frame_count, time_info, status):
                a = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    a = a.reshape(-1, ch).mean(axis=1)
                self.out_q.put((time.monotonic(), a))
                return (in_data, pyaudio.paContinue)

            stream = hub.call(lambda p: p.open(
                format=pyaudio.paInt16, channels=ch, rate=self.rate,
                frames_per_buffer=int(self.rate * BLOCK_SEC),
                input=True, input_device_index=dev["index"], stream_callback=cb,
            ))
            self.ready.set()
            while not self._halt.is_set():
                time.sleep(0.2)
        except Exception as e:
            self.error = e
            self.ready.set()
        finally:
            if stream is not None:
                close_stream(stream)

    def stop(self):
        self._halt.set()


# --------------------------------------------------------------------------- #
# Segmentación por silencios
# --------------------------------------------------------------------------- #
def to_16k(audio, sr):
    if sr == TARGET_SR:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly
    g = gcd(sr, TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)


class Segmenter(threading.Thread):
    """Agrupa bloques en frases usando energía + silencios; las pasa a seg_q."""

    def __init__(self, in_q, seg_q, get_rate, threshold):
        super().__init__(daemon=True)
        self.in_q, self.seg_q = in_q, seg_q
        self.get_rate = get_rate
        self.threshold = threshold
        self.level = 0.0  # para el medidor de la ventana
        # Última instantánea de la frase en curso (el Worker la recoge cuando
        # está libre); solo se guarda la más reciente para no acumular retraso.
        self.partial = None

    def run(self):
        buf, pre = [], []
        buf_len = 0
        speaking = False
        last_voice = 0.0
        last_partial = 0.0
        while True:
            try:
                t, block = self.in_q.get(timeout=0.2)
            except queue.Empty:
                block = None
                t = time.monotonic()
            rate = self.get_rate() or 48000

            if block is not None:
                rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
                self.level = rms
                voiced = rms > self.threshold
                if voiced:
                    last_voice = t
                    if not speaking:
                        speaking = True
                        buf = list(pre)
                        buf_len = sum(len(b) for b in buf)
                if speaking:
                    buf.append(block)
                    buf_len += len(block)
                    if t - last_partial >= PARTIAL_SEC and buf_len >= PARTIAL_SEC * rate:
                        last_partial = t
                        self.partial = to_16k(np.concatenate(buf), rate)
                else:
                    pre.append(block)
                    while sum(len(b) for b in pre) > PREROLL_SEC * rate:
                        pre.pop(0)
            else:
                self.level = 0.0

            if speaking:
                silent_for = t - last_voice
                too_long = buf_len >= MAX_SEG_SEC * rate
                if silent_for >= SILENCE_SEC or too_long:
                    self.partial = None
                    last_partial = t
                    audio = np.concatenate(buf) if buf else np.zeros(0, np.float32)
                    if len(audio) >= MIN_SEG_SEC * rate:
                        self.seg_q.put(to_16k(audio, rate))
                    buf, buf_len = [], 0
                    pre = []
                    speaking = too_long and silent_for < SILENCE_SEC
                    if speaking:
                        buf = []


# --------------------------------------------------------------------------- #
# Transcripción + traducción
# --------------------------------------------------------------------------- #
class Translator:
    """Traducción offline (Opus-MT con CTranslate2); si no está disponible,
    usa Google, que puede bloquear la IP si se le hacen muchas peticiones."""

    def __init__(self, log=print, source="en", target="es"):
        self.mt = self.gt = self.pivot = None
        self.target = target
        if ((source, target) not in MT_MODELS and (source, PIVOT) in MT_MODELS
                and (PIVOT, target) in MT_MODELS):
            self.pivot = (Translator(log, source, PIVOT), Translator(log, PIVOT, target))
            return
        try:
            import ctranslate2
            import sentencepiece as spm
            from huggingface_hub import snapshot_download
            repo = MT_MODELS[(source, target)]
            try:
                path = snapshot_download(repo, cache_dir=MODELS_DIR)
            except Exception:  # sin internet: vale lo ya descargado
                path = snapshot_download(repo, cache_dir=MODELS_DIR,
                                         local_files_only=True)
            self.mt = ctranslate2.Translator(path, device="cpu", compute_type="int8",
                                             intra_threads=2)
            self.sp_src = spm.SentencePieceProcessor(model_file=os.path.join(path, "source.spm"))
            self.sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(path, "target.spm"))
            self.prefix = [MT_PREFIX[(source, target)]] if (source, target) in MT_PREFIX else []
            self.lock = threading.Lock()
            self.cache = {}  # frase -> traducción
            return
        except Exception as e:
            log(T("Traductor offline no disponible ({error}); usando Google.", error=e))
        try:
            from deep_translator import GoogleTranslator
            self.gt = GoogleTranslator(source=source, target=target)
        except Exception:
            pass

    def __call__(self, text):
        if self.pivot:
            middle = self.pivot[0](text)
            return self.pivot[1](middle) if middle else None
        if self.mt:
            try:
                # Frase a frase: Opus-MT pierde calidad (y trunca) con textos largos.
                # (En chino o japonés la frase acaba en 。！？ y no lleva espacio detrás;
                # en árabe la pregunta acaba en ؟.)
                sentences = [s for s in re.split(r"(?<=[.!?؟])\s+|(?<=[。！？])", text.strip()) if s]
                with self.lock:
                    # Al traducir sobre la marcha llega el mismo texto una y otra
                    # vez con una frase más: las ya traducidas salen de la caché.
                    if len(self.cache) > 2000:
                        self.cache.clear()
                    new = [s for s in dict.fromkeys(sentences) if s not in self.cache]
                    if new:
                        toks = [self.prefix + self.sp_src.encode(s, out_type=str) + ["</s>"] for s in new]
                        r = self.mt.translate_batch(toks, beam_size=1)
                        for s, x in zip(new, r):
                            self.cache[s] = self.sp_tgt.decode(x.hypotheses[0])
                    sep = "" if self.target in NOSPACE else " "
                    return sep.join(self.cache[s] for s in sentences)
            except Exception:
                return None
        if not self.gt:
            return None
        for _ in range(2):
            try:
                return self.gt.translate(text)
            except Exception:
                time.sleep(0.3)
        return None


class Worker(threading.Thread):
    """Transcribe las frases cerradas (seg_q) y, cuando no hay ninguna
    pendiente, la frase en curso para mostrar texto provisional."""

    def __init__(self, model, device, seg_q, segmenter, translate, ui_q):
        super().__init__(daemon=True)
        self.model, self.device = model, device
        self.seg_q, self.segmenter, self.ui_q = seg_q, segmenter, ui_q
        self.translate = translate
        self.prev = ""

    def _transcribe(self, audio, beam, vad):
        segs, _ = self.model.transcribe(
            audio, language="en", beam_size=beam, vad_filter=vad,
            condition_on_previous_text=False, without_timestamps=True,
            initial_prompt=self.prev[-200:] or None,
        )
        return " ".join(s.text.strip() for s in segs).strip()

    def run(self):
        beam = 5 if self.device == "cuda" else 2
        while True:
            try:
                audio = self.seg_q.get(timeout=0.05)
            except queue.Empty:
                self._partial()
                continue
            self.ui_q.put(("busy", self.seg_q.qsize() + 1))
            try:
                text = self._transcribe(audio, beam, True)
            except Exception as e:
                self.ui_q.put(("status", f"Error transcribiendo: {e}"))
                continue
            finally:
                self.ui_q.put(("busy", self.seg_q.qsize()))
            if not text or text.lower() in HALLUCINATIONS:
                self.ui_q.put(("partial", "", None))
                continue
            self.prev = text
            es = self.translate(text)
            self.ui_q.put(("line", datetime.now().strftime("%H:%M:%S"), text, es))

    def _partial(self):
        audio, self.segmenter.partial = self.segmenter.partial, None
        if audio is None:
            return
        try:
            text = self._transcribe(audio, 1, False)
        except Exception:
            return
        if not self.seg_q.empty():  # la frase ya se cerró: el provisional sobra
            return
        if not text or text.lower() in HALLUCINATIONS:
            return
        self.ui_q.put(("partial", text, self.translate(text)))


# --------------------------------------------------------------------------- #
# Ventana
# --------------------------------------------------------------------------- #
def run_gui(args):
    import tkinter as tk

    ui_q = queue.Queue()
    audio_q = queue.Queue()
    seg_q = queue.Queue()

    root = tk.Tk()
    root.title("Traductor en directo (inglés → español)")
    root.geometry("900x600")
    root.configure(bg="#111")

    size = {"es": args.letra, "en": max(10, args.letra - 8)}
    show_en = tk.BooleanVar(value=True)
    on_top = tk.BooleanVar(value=False)

    bar = tk.Frame(root, bg="#222")
    bar.pack(fill="x")
    status = tk.StringVar(value="Cargando modelo, espera...")
    tk.Label(bar, textvariable=status, fg="#ddd", bg="#222", anchor="w",
             font=("Segoe UI", 10)).pack(side="left", padx=8, pady=4)

    meter = tk.Canvas(bar, width=80, height=10, bg="#333", highlightthickness=0)
    meter.pack(side="left", padx=6)
    meter_bar = meter.create_rectangle(0, 0, 0, 10, fill="#4c4", width=0)

    txt = tk.Text(root, wrap="word", bg="#111", fg="#fff", insertbackground="#fff",
                  relief="flat", padx=14, pady=10)
    txt.pack(fill="both", expand=True)

    def apply_fonts():
        txt.tag_configure("es", font=("Segoe UI", size["es"]), foreground="#ffffff",
                          spacing1=10)
        txt.tag_configure("en", font=("Segoe UI", size["en"]), foreground="#8a8a8a",
                          elide=not show_en.get())
        txt.tag_configure("t", font=("Segoe UI", 9), foreground="#666",
                          elide=not show_en.get())
        # Sin traducción: el inglés se muestra una sola vez, siempre visible.
        txt.tag_configure("en_only", font=("Segoe UI", size["en"]), foreground="#8a8a8a")
        # Texto provisional de la frase que se está diciendo.
        txt.tag_configure("live", font=("Segoe UI", size["es"], "italic"),
                          foreground="#9a9a9a", spacing1=10)

    def bigger():
        size["es"] += 2; size["en"] += 1; apply_fonts()

    def smaller():
        size["es"] = max(10, size["es"] - 2); size["en"] = max(8, size["en"] - 1)
        apply_fonts()

    def toggle_top():
        root.attributes("-topmost", on_top.get())

    # Todo lo que va detrás de la marca "live" es provisional y se reemplaza.
    txt.mark_set("live", "end-1c")
    txt.mark_gravity("live", "left")

    def clear():
        txt.delete("1.0", "end")
        txt.mark_set("live", "end-1c")

    btn = dict(bg="#333", fg="#eee", relief="flat", padx=8)
    tk.Button(bar, text="A+", command=bigger, **btn).pack(side="right", padx=2, pady=3)
    tk.Button(bar, text="A−", command=smaller, **btn).pack(side="right", padx=2)
    tk.Button(bar, text="Borrar", command=clear, **btn).pack(side="right", padx=2)
    tk.Checkbutton(bar, text="Siempre encima", variable=on_top, command=toggle_top,
                   bg="#222", fg="#ddd", selectcolor="#333",
                   activebackground="#222").pack(side="right", padx=6)
    tk.Checkbutton(bar, text="Ver inglés", variable=show_en, command=apply_fonts,
                   bg="#222", fg="#ddd", selectcolor="#333",
                   activebackground="#222").pack(side="right", padx=6)
    apply_fonts()

    os.makedirs(os.path.join(BASE_DIR, "transcripciones"), exist_ok=True)
    log_path = os.path.join(
        BASE_DIR, "transcripciones",
        datetime.now().strftime("reunion_%Y-%m-%d_%H%M.txt"))

    def log(msg):
        ui_q.put(("status", msg))

    state = {"busy": 0, "listening": False, "segmenter": None}

    def startup():
        try:
            model, device = load_model(args.modelo, args.cpu, log)
        except Exception as e:
            log(f"No se pudo cargar el modelo: {e}")
            return
        log("Cargando traductor...")
        translate = Translator(log)
        cap = Capture(audio_q, args.dispositivo, log)
        cap.start()
        cap.ready.wait(10)
        if cap.error:
            log(f"Error de audio: {cap.error}")
            return
        seg = Segmenter(audio_q, seg_q, lambda: cap.rate, args.umbral)
        seg.start()
        state["segmenter"] = seg
        Worker(model, device, seg_q, seg, translate, ui_q).start()
        state["listening"] = True
        log(f"Escuchando ({device.upper()}, modelo {args.modelo}). "
            f"Transcripción en: transcripciones\\{os.path.basename(log_path)}")

    threading.Thread(target=startup, daemon=True).start()

    base_status = {"text": ""}

    def poll():
        try:
            while True:
                item = ui_q.get_nowait()
                kind = item[0]
                if kind == "status":
                    base_status["text"] = item[1]
                elif kind == "busy":
                    state["busy"] = item[1]
                elif kind == "partial":
                    _, en, es = item
                    at_bottom = txt.yview()[1] > 0.98
                    txt.delete("live", "end-1c")
                    if en:
                        txt.insert("end", (es or en) + "\n", "live")
                    if at_bottom:
                        txt.see("end")
                elif kind == "line":
                    _, hhmm, en, es = item
                    at_bottom = txt.yview()[1] > 0.98
                    txt.delete("live", "end-1c")
                    txt.insert("end", f"{hhmm}  ", "t")
                    if es:
                        txt.insert("end", en + "\n", "en")
                        txt.insert("end", es + "\n\n", "es")
                    else:
                        txt.insert("end", en + "\n\n", "en_only")
                    txt.mark_set("live", "end-1c")
                    if at_bottom:
                        txt.see("end")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{hhmm}] EN: {en}\n           ES: {es or '-'}\n\n")
        except queue.Empty:
            pass
        s = base_status["text"]
        if state["busy"]:
            s += "   · traduciendo…"
        status.set(s)
        seg = state["segmenter"]
        if seg is not None:
            lvl = min(1.0, seg.level * 8)
            meter.coords(meter_bar, 0, 0, int(80 * lvl), 10)
        root.after(100, poll)

    root.after(100, poll)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Traductor en directo inglés → español")
    ap.add_argument("--modelo", default="small.en",
                    help="tiny.en, base.en, small.en (defecto), medium.en, large-v3")
    ap.add_argument("--dispositivo", type=int, default=None,
                    help="índice de salida a capturar (ver --dispositivos)")
    ap.add_argument("--dispositivos", action="store_true", help="listar salidas y salir")
    ap.add_argument("--umbral", type=float, default=0.008,
                    help="nivel mínimo para considerar que hay voz (defecto 0.008)")
    ap.add_argument("--letra", type=int, default=24, help="tamaño de letra del español")
    ap.add_argument("--cpu", action="store_true", help="no intentar usar la GPU")
    ap.add_argument("--descargar", action="store_true",
                    help="descargar el modelo y salir (para la instalación)")
    args = ap.parse_args()

    if args.dispositivos:
        list_devices()
        return
    if args.descargar:
        print(f"Descargando modelo {args.modelo} (solo la primera vez)...")
        _, dev = load_model(args.modelo, args.cpu)
        print(f"Modelo listo ({dev}).")
        print("Descargando traductores offline...")
        print("Prueba de traducción:",
              Translator()("Good morning, how are you?") or "(sin respuesta)")
        print("Prueba de traducción:",
              Translator(source="es", target="en")("Buenos días, ¿cómo estás?")
              or "(sin respuesta)")
        return
    run_gui(args)


if __name__ == "__main__":
    main()
