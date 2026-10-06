"""
Motores de reconocimiento de voz para transcriptor.py, con la misma interfaz:

    motor.lang_probs(audio)                 -> {"es": 0.9, "en": 0.1, ...}
    motor.words(audio, idioma, contexto)    -> [(inicio_s, fin_s, " palabra"), ...]

- MotorCPU: faster-whisper (CTranslate2). Usa la GPU solo si es NVIDIA (CUDA).
- MotorGPU: whisper.cpp compilado con Vulkan (carpeta motor_gpu), que funciona
  con tarjetas AMD, NVIDIA e Intel. Se arranca whisper-server.exe en segundo
  plano con el modelo cargado en la GPU y se le manda el audio por HTTP local.

En los dos se descartan los fragmentos en bucle ("What does it mean? What does
it mean?..."), que Whisper genera a veces con audio difícil.
"""
import atexit
import io
import os
import socket
import subprocess
import time
import wave
import zlib

import numpy as np

from interfaz import N_, T
from traductor import BASE_DIR, MODELS_DIR, NOSPACE, TARGET_SR, load_model

GPU_DIR = os.path.join(BASE_DIR, "motor_gpu")
GPU_EXE = os.path.join(GPU_DIR, "whisper-server.exe")
GGML_REPO = "ggerganov/whisper.cpp"
# Modelos de Whisper que se pueden elegir en la conversación (Herramientas →
# Modelos): nombre (el de faster-whisper en la CPU) -> (archivo de whisper.cpp
# para la GPU, MB de ese archivo, precisión orientativa). En la GPU se usan las
# versiones comprimidas (q5): la mitad de tamaño y más rápidas, casi igual de precisas.
WHISPER_MODELS = {
    "tiny": ("tiny-q5_1", 31, N_("baja")),
    "base": ("base", 141, N_("correcta")),
    "small": ("small-q5_1", 181, N_("buena")),
    "medium": ("medium-q5_0", 514, N_("muy buena")),
    "large-v3-turbo": ("large-v3-turbo-q5_0", 547, N_("la mejor")),
}
MAX_COMPRESSION = 2.4  # por encima, el texto es un bucle repetitivo


def is_loop(text):
    data = text.strip().encode("utf-8")
    return len(data) >= 50 and len(data) / len(zlib.compress(data)) > MAX_COMPRESSION


# --------------------------------------------------------------------------- #
# CPU (faster-whisper)
# --------------------------------------------------------------------------- #
class MotorCPU:
    def __init__(self, modelo, force_cpu=False, log=print):
        self.model, device = load_model(modelo, force_cpu, log)
        self.name = device.upper()
        self.beam = 5 if device == "cuda" else 1

    def lang_probs(self, audio):
        _, _, probs = self.model.detect_language(audio)
        return dict(probs)

    def words(self, audio, language, prompt):
        segs, _ = self.model.transcribe(
            audio, language=language, beam_size=self.beam,
            word_timestamps=True, condition_on_previous_text=False,
            temperature=0.0,  # sin reintentos: cada pasada debe ser rápida
            # Tope de tokens según la duración (se habla a ~4 tokens/s): corta
            # los bucles tipo "a little bit of a little bit of..." enseguida.
            max_new_tokens=int(len(audio) / TARGET_SR * 8) + 10,
            initial_prompt=prompt,
        )
        out = []
        for s in segs:
            if s.compression_ratio > MAX_COMPRESSION:
                break
            out += [(w.start, w.end, w.word) for w in s.words or []]
        return out

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# GPU (whisper.cpp + Vulkan)
# --------------------------------------------------------------------------- #
def _kill_with_us(proc):
    """Mete el proceso en un "job" de Windows que lo cierra cuando se cierra
    Python, aunque sea a la fuerza: así no quedan servidores ocupando la GPU."""
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                        ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.OpenProcess.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info),
                                    ctypes.sizeof(info))
        h = k32.OpenProcess(0x1F0FFF, False, proc.pid)
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(h))
        return job  # hay que conservarlo vivo mientras dure el programa
    except Exception:
        return None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ggml_model(modelo):
    """Descarga (la primera vez) el modelo en formato de whisper.cpp a modelos\\."""
    from huggingface_hub import hf_hub_download
    name = f"ggml-{WHISPER_MODELS.get(modelo, (modelo,))[0]}.bin"
    try:
        return hf_hub_download(GGML_REPO, name, cache_dir=MODELS_DIR)
    except Exception:  # sin internet: vale lo ya descargado
        return hf_hub_download(GGML_REPO, name, cache_dir=MODELS_DIR, local_files_only=True)


class MotorGPU:
    def __init__(self, modelo, log=print, threads=4):
        import requests
        self.requests = requests
        if not os.path.exists(GPU_EXE):
            raise RuntimeError(T("no está la carpeta motor_gpu"))
        log(T("Preparando modelo {model} para la GPU...", model=modelo))
        path = ggml_model(modelo)
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = os.path.join(GPU_DIR, "servidor.log")
        self.log_file = open(self.log_path, "w", encoding="utf-8", errors="replace")
        log(T("Arrancando motor GPU (Vulkan)..."))
        self.proc = subprocess.Popen(
            [GPU_EXE, "-m", path, "--host", "127.0.0.1", "--port", str(port),
             "-t", str(threads)],
            cwd=GPU_DIR, stdout=self.log_file, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.job = _kill_with_us(self.proc)
        atexit.register(self.close)
        self._wait_ready()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "using Vulkan" not in text:
            self.close()
            raise RuntimeError(T("whisper.cpp no encontró una GPU compatible con Vulkan"))
        gpu = [l.split("=", 1)[1].split("(")[0].split("|")[0].strip()
               for l in text.splitlines() if l.startswith("ggml_vulkan: 0 =")]
        self.name = f"GPU {gpu[0]}" if gpu else "GPU"
        # La primera petición prepara los programas de la GPU y tarda varios
        # segundos: mejor ahora que con la primera frase.
        log(T("Calentando la GPU..."))
        noise = (np.random.randn(TARGET_SR * 2) * 0.01).astype(np.float32)
        self.words(noise, "en", None)

    def _wait_ready(self, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            if self.proc.poll() is not None:
                raise RuntimeError(T("whisper-server se cerró (ver {file})", file=self.log_path))
            try:
                self.requests.get(self.url + "/", timeout=1)
                return
            except Exception:
                time.sleep(0.3)
        self.close()
        raise RuntimeError(T("whisper-server no respondió a tiempo"))

    @staticmethod
    def _wav(audio):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TARGET_SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return buf.getvalue()

    def _call(self, audio, **fields):
        data = {"response_format": "verbose_json", "temperature": "0",
                "temperature_inc": "0", "beam_size": "1", "best_of": "1",
                # sin "(speaking in foreign language)", "[music]", "♪"...
                "suppress_nst": "true"}
        data.update({k: str(v) for k, v in fields.items() if v is not None})
        r = self.requests.post(self.url + "/inference", timeout=60, data=data,
                               files={"file": ("a.wav", self._wav(audio), "audio/wav")})
        r.raise_for_status()
        return r.json()

    def lang_probs(self, audio):
        j = self._call(audio, language="auto", detect_language="true")
        return j.get("language_probabilities", {})

    def words(self, audio, language, prompt):
        j = self._call(audio, language=language, prompt=prompt,
                       no_language_probabilities="true")
        out, text = [], ""
        for seg in j.get("segments", []):
            text += seg.get("text", "")
            if is_loop(text):
                break
            if language in NOSPACE:
                out += char_words(seg)
                continue
            # whisper.cpp da tokens, no palabras: un token que empieza por
            # espacio abre palabra nueva; el resto se pega a la anterior.
            for tok in seg.get("words", []):
                t = tok.get("word", "")
                if not t or t.startswith(("[_", "<|")):
                    continue
                start, end = tok.get("start", 0.0), tok.get("end", 0.0)
                if out and not t.startswith(" "):
                    s, _, w = out[-1]
                    out[-1] = (s, max(end, s), w + t)
                else:
                    out.append((start, max(end, start), t))
        return out

    def close(self):
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()
        if getattr(self, "log_file", None):
            self.log_file.close()
            self.log_file = None


def char_words(seg):
    """Idiomas sin espacios (chino, japonés...): cada carácter es una "palabra"
    (así se confirma y se muestra poco a poco), con los signos de puntuación
    pegados al anterior. Los tiempos se reparten a lo largo del segmento."""
    chars = [c for c in seg.get("text", "").strip() if not c.isspace()]
    t0, t1 = seg.get("start", 0.0), seg.get("end", 0.0)
    step = (t1 - t0) / max(1, len(chars))
    out = []
    for i, c in enumerate(chars):
        s = t0 + i * step
        if out and not c.isalnum():
            ps, pe, w = out[-1]
            out[-1] = (ps, s + step, w + c)
        else:
            out.append((s, s + step, c))
    return out


def create_engine(motor, modelo, force_cpu=False, log=print):
    """motor: "auto" (GPU si se puede, si no CPU), "gpu" o "cpu"."""
    if motor != "cpu" and not force_cpu:
        try:
            return MotorGPU(modelo, log)
        except Exception as e:
            log(T("Motor GPU no disponible ({error}). Usando CPU.", error=e))
    return MotorCPU(modelo, force_cpu, log)
