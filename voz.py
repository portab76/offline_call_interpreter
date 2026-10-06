"""
Texto a voz sin internet con Piper (voces neuronales), reproducido por el
dispositivo de salida que se elija: los altavoces (conversación en persona) o
un micrófono virtual como VB-Audio Virtual Cable (llamadas: en Teams/Zoom se
elige ese cable como micrófono).

Las voces se descargan la primera vez a modelos\\ (~63 MB cada una).
"""
import os
import queue
import threading
import time

import numpy as np

from interfaz import T
from traductor import MODELS_DIR, AudioHub, close_stream, wasapi_devices

VOICES_REPO = "rhasspy/piper-voices"
# Voz por idioma (se puede cambiar por cualquiera de https://huggingface.co/rhasspy/piper-voices)
VOICES = {
    "en": "en/en_US/lessac/medium/en_US-lessac-medium",
    "es": "es/es_ES/davefx/medium/es_ES-davefx-medium",
    "fr": "fr/fr_FR/siwis/medium/fr_FR-siwis-medium",
    "de": "de/de_DE/thorsten/medium/de_DE-thorsten-medium",
    "it": "it/it_IT/paola/medium/it_IT-paola-medium",
    "ca": "ca/ca_ES/upc_ona/medium/ca_ES-upc_ona-medium",
    "ru": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
    "zh": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium",
    "vi": "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium",
    "ar": "ar/ar_JO/kareem/medium/ar_JO-kareem-medium",  # árabe estándar, acento jordano
    "pt": "pt/pt_PT/tugão/medium/pt_PT-tugão-medium",  # de Brasil: pt/pt_BR/faber/medium/pt_BR-faber-medium
}
TAIL_SEC = 0.4  # tras hablar, eco que aún puede llegar al micrófono


def voice_files(lang):
    """Descarga (la primera vez) la voz de Piper del idioma y devuelve el .onnx."""
    from huggingface_hub import hf_hub_download
    base = VOICES[lang]
    paths = []
    for ext in (".onnx", ".onnx.json"):
        try:
            paths.append(hf_hub_download(VOICES_REPO, base + ext, cache_dir=MODELS_DIR))
        except Exception:  # sin internet: vale lo ya descargado
            paths.append(hf_hub_download(VOICES_REPO, base + ext, cache_dir=MODELS_DIR,
                                         local_files_only=True))
    return paths[0]


def output_devices():
    """[(índice, nombre)] de las salidas WASAPI; la primera es la de por defecto."""
    return wasapi_devices(output=True)


class Speaker(threading.Thread):
    """Cola de frases: las sintetiza y las reproduce una tras otra.

    La salida de audio se queda abierta todo el rato (con silencio entre
    frases) y se lleva la cuenta exacta de muestras entregadas. Así cada frase
    empieza en una muestra conocida y el desfase con la captura de altavoces
    del mismo dispositivo es constante: el cancelador de eco lo mide una vez y
    ya no tiene que adivinar dónde está cada frase.

    speaking() dice si está sonando (o acaba de sonar) una frase, para no
    capturar la propia voz traducida por el micrófono o los altavoces."""

    def __init__(self, device_index=None, log=print):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.log = log
        self.q = queue.Queue()
        self.voices = {}
        self.busy_until = 0.0
        self.playing = False
        self.muted = False
        # Se les avisa al empezar a sonar cada frase con
        # (muestras mono exactas, muestra de salida en que empieza, frecuencia,
        #  nombre del dispositivo, apertura de la salida, momento time.monotonic):
        # así el cancelador de eco sabe qué restar de la captura de altavoces.
        self.listeners = []
        # Se les avisa si una frase se corta: (muestra de salida en que empezó,
        # muestras que llegan a sonar, de ellas las del fundido final, apertura).
        self.cut_listeners = []
        self.dropped = set()    # tags de frases borradas: si siguen en cola, se saltan
        self.stream = None
        self.rate = self.channels = None
        self.device_name = None
        self.epoch = 0          # cambia cada vez que se (re)abre la salida
        self.out_n = 0          # muestras entregadas desde que se abrió
        self.play_q = queue.Queue()  # (muestras, on_start, on_end, tag) listas para sonar
        self.current = None     # [muestras, posición, on_end, tag, muestra de salida inicial]
        self.lock = threading.Lock()
        # "Mi voz" (mivoz.MyVoice): se aplica a cada frase tras sintetizarla. Las
        # frases se pueden preparar antes (prepare) para que al pedirlas suenen ya.
        self.style = None
        self.ready = {}         # (texto, idioma, estilo) -> (audio, sr) ya preparados
        self.pending = {}       # lo mismo -> Event mientras se prepara
        self.ready_lock = threading.Lock()
        self.prep_q = queue.Queue()
        threading.Thread(target=self._prepare_loop, daemon=True).start()

    def say(self, text, lang, on_start=None, on_end=None, on_fail=None, tag=None):
        """on_start/on_end: al empezar y acabar de sonar; on_fail: si no llega a sonar.
        tag identifica la frase para poder cortarla (stop)."""
        if text.strip():
            self.q.put((text, lang, on_start, on_end, on_fail, tag))

    def forget(self, tag):
        """Las frases de tag que aún esperan en cola ya no sonarán (la frase se
        ha borrado); para la que suena, stop()."""
        self.dropped.add(tag)

    def stop(self, tag):
        """Corta la frase que está sonando si es la de tag (con un fundido de
        10 ms para que no chasque); la siguiente en cola empieza después."""
        with self.lock:
            if self.current is None or self.current[3] != tag:
                return
            samples, pos, _, _, start = self.current
            fade = min(int(self.rate * 0.01), len(samples) - pos)
            cut = samples[:pos + fade].copy()
            cut[pos:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            self.current[0] = cut
            for fn in self.cut_listeners:  # el cancelador de eco: la frase acaba antes
                try:
                    fn(start, pos + fade, fade, self.epoch)
                except Exception:
                    pass

    def speaking(self):
        return self.playing or time.monotonic() < self.busy_until

    def set_device(self, index):
        if index != self.device_index or self.stream is None:
            self.device_index = index
            self._open()

    def voice(self, lang):
        if lang not in self.voices:
            from piper import PiperVoice
            self.log(T("Cargando voz {voice}...", voice=VOICES[lang].rsplit('/', 1)[-1]))
            v = PiperVoice.load(voice_files(lang))
            list(v.synthesize("ok"))  # la primera síntesis es lenta
            self.voices[lang] = v
        return self.voices[lang]

    def preload(self, *langs):
        for lang in langs:
            try:
                self.voice(lang)
            except Exception as e:
                self.log(T("No se pudo cargar la voz {language}: {error}", language=lang, error=e))

    def synthesize(self, text, lang):
        chunks = list(self.voice(lang).synthesize(text))
        if not chunks:
            return np.zeros(0, np.float32), 22050
        return (np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32),
                chunks[0].sample_rate)

    # -- "Mi voz": frases preparadas --------------------------------------- #
    def set_style(self, style):
        """style(audio, sr, idioma) -> (audio, sr), o None (voz de Piper tal cual)."""
        with self.ready_lock:
            self.style = style
            self.ready.clear()

    def prepare(self, text, lang):
        """Prepara la frase en segundo plano (solo con "Mi voz": convertir tarda)."""
        if self.style is not None and text.strip():
            self.prep_q.put((text, lang))

    def _prepare_loop(self):
        while True:
            text, lang = self.prep_q.get()
            try:
                self.render(text, lang)
            except Exception:
                pass  # al pedirla se vuelve a intentar (y si falla, suena Piper)

    def render(self, text, lang):
        """Audio de la frase con la voz elegida: el preparado si ya lo está (o se
        espera a que termine de prepararse), si no se genera ahora."""
        style = self.style
        key = (text, lang, id(style))
        with self.ready_lock:
            got = self.ready.get(key)
            wait = None if got is not None else self.pending.get(key)
            mine = got is None and wait is None
            if mine:
                self.pending[key] = threading.Event()
        if got is not None:
            return got
        if wait is not None:
            wait.wait(60)
            with self.ready_lock:
                got = self.ready.get(key)
            return got if got is not None else self._make(text, lang, style)
        try:
            got = self._make(text, lang, style)
            with self.ready_lock:
                if self.style is style:
                    if len(self.ready) > 100:
                        self.ready.clear()
                    self.ready[key] = got
            return got
        finally:
            with self.ready_lock:
                self.pending.pop(key).set()

    def _make(self, text, lang, style):
        audio, sr = self.synthesize(text, lang)
        if style is None:
            return audio, sr
        try:
            return style(audio, sr, lang)
        except Exception as e:  # si mi voz falla, suena la de Piper
            self.log(T("Mi voz no está disponible ({error}): suena la voz de Piper", error=e))
            return audio, sr

    # -- salida permanente ------------------------------------------------ #
    def _open(self):
        """(Re)abre la salida en el dispositivo elegido. La abre el AudioHub
        (con WASAPI no se puede abrir desde otro hilo)."""
        import pyaudiowpatch as pa
        hub = AudioHub.get()
        self.close()

        def device(p):
            if self.device_index is None:
                wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
                return p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            return p.get_device_info_by_index(self.device_index)

        dev = hub.call(device)
        with self.lock:
            # WASAPI exige la frecuencia y los canales del dispositivo.
            self.rate = int(dev["defaultSampleRate"])
            self.channels = max(1, int(dev["maxOutputChannels"]))
            self.device_name = dev["name"]
            self.epoch += 1
            self.out_n = 0
            self.current = None
        self.stream = hub.call(lambda p: p.open(
            format=pa.paFloat32, channels=self.channels, rate=self.rate, output=True,
            output_device_index=dev["index"], stream_callback=self._callback))

    def close(self):
        if self.stream is not None:
            try:
                close_stream(self.stream)
            except Exception:
                pass
            self.stream = None

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pa
        out = np.zeros(frame_count, np.float32)
        i = 0
        with self.lock:
            while i < frame_count:
                if self.current is None:
                    try:
                        samples, on_start, on_end, tag = self.play_q.get_nowait()
                    except queue.Empty:
                        break
                    if tag is not None and tag in self.dropped:
                        continue  # su frase se borró mientras esperaba
                    start = self.out_n + i  # muestra exacta en que empieza
                    self.current = [samples, 0, on_end, tag, start]
                    self.playing = True
                    for fn in self.listeners:
                        try:
                            fn(samples, start, self.rate, self.device_name, self.epoch,
                               time.monotonic() + i / self.rate)
                        except Exception:
                            pass
                    if on_start:
                        on_start()
                samples, pos, on_end = self.current[:3]
                take = min(frame_count - i, len(samples) - pos)
                out[i:i + take] = samples[pos:pos + take]
                self.current[1] += take
                i += take
                if self.current[1] >= len(samples):
                    self.current = None
                    if self.play_q.empty():
                        self.playing = False
                        self.busy_until = time.monotonic() + TAIL_SEC
                    if on_end:
                        on_end()
            self.out_n += frame_count
        data = np.repeat(out[:, None], self.channels, axis=1) if self.channels > 1 else out
        return data.tobytes(), pa.paContinue

    def run(self):
        from scipy.signal import resample_poly
        if self.stream is None:
            try:
                self._open()
            except Exception as e:
                self.log(T("Error de voz: {error}", error=e))
        while True:
            text, lang, on_start, on_end, on_fail, tag = self.q.get()
            if tag is not None and tag in self.dropped:
                continue  # su frase se borró antes de prepararla
            if self.muted:
                if on_fail:
                    on_fail()
                continue
            try:
                audio, sr = self.render(text, lang)
                if self.stream is None:
                    self._open()
                g = np.gcd(sr, self.rate)
                out = resample_poly(audio, self.rate // g, sr // g).astype(np.float32)
                self.play_q.put((out, on_start, on_end, tag))
            except Exception as e:
                self.log(T("Error de voz: {error}", error=e))
                if on_fail:
                    on_fail()
