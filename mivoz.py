"""
"Mi voz": la frase traducida suena con mi timbre.

Piper genera la frase en el idioma del otro (con su acento) y el conversor de
timbre de OpenVoice v2 (openvoice_lite, MIT) cambia el timbre de esa voz por el
mío. Hace falta:

- PyTorch (para el procesador): no va en requirements.txt; se instala bajo
  demanda desde Herramientas → Modelos (install_components).
- El modelo del conversor (~130 MB, myshell-ai/OpenVoiceV2): se descarga a
  modelos\\ la primera vez.
- Una "huella" de mi voz: se saca de una grabación de 30-60 s (asistente "Nueva
  voz") y se guarda en voces\\<nombre>\\ junto a la grabación.
- La huella de cada voz de Piper (la de origen): se saca una vez de unas frases
  dichas por ella y se guarda en voces\\_piper\\.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import wave

import numpy as np

from interfaz import T
from traductor import BASE_DIR, MODELS_DIR

VOICES_DIR = os.path.join(BASE_DIR, "voces")
PIPER_SE_DIR = os.path.join(VOICES_DIR, "_piper")
CONVERTER_REPO = "myshell-ai/OpenVoiceV2"
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"  # solo para el procesador: ~200 MB
TAU = 0.3  # fuerza del cambio de timbre (la de OpenVoice por defecto)


# --------------------------------------------------------------------------- #
# Componentes (PyTorch), bajo demanda
# --------------------------------------------------------------------------- #
def components_ready():
    return importlib.util.find_spec("torch") is not None


def install_components(log=print):
    """Instala PyTorch (versión para el procesador) en el entorno del programa.
    log recibe cada línea de pip. Lanza RuntimeError si falla."""
    cmd = [sys.executable, "-m", "pip", "install", "torch", "--index-url", TORCH_INDEX,
           "--progress-bar", "off", "--disable-pip-version-check"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    last = ""
    for line in proc.stdout:
        line = line.strip()
        if line:
            last = line
            log(line)
    if proc.wait() != 0:
        raise RuntimeError(last or "pip")
    importlib.invalidate_caches()


# --------------------------------------------------------------------------- #
# Voces guardadas (voces\<nombre>\muestra.wav + huella.npy)
# --------------------------------------------------------------------------- #
def profiles():
    """Nombres de mis voces guardadas."""
    try:
        names = os.listdir(VOICES_DIR)
    except OSError:
        return []
    return sorted(n for n in names if not n.startswith("_")
                  and os.path.exists(os.path.join(VOICES_DIR, n, "huella.npy")))


def safe_name(name):
    """Nombre de carpeta válido en Windows."""
    return " ".join(re.sub(r'[<>:"/\\|?*]+', " ", name).split()).strip(" .")


def profile_dir(name):
    return os.path.join(VOICES_DIR, safe_name(name))


def delete_profile(name):
    shutil.rmtree(profile_dir(name), ignore_errors=True)


def write_wav(path, audio, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def save_profile(name, audio, sr):
    """Guarda la grabación y su huella (tarda unos segundos)."""
    se = Converter.get().extract_se(audio, sr)
    folder = profile_dir(name)
    os.makedirs(folder, exist_ok=True)
    write_wav(os.path.join(folder, "muestra.wav"), audio, sr)
    np.save(os.path.join(folder, "huella.npy"), se)
    return se


def load_profile(name):
    return np.load(os.path.join(profile_dir(name), "huella.npy"))


# --------------------------------------------------------------------------- #
# Conversor de timbre (OpenVoice v2)
# --------------------------------------------------------------------------- #
class _HParams(dict):
    """Configuración del modelo con acceso por atributo (hps.data.hop_length)."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _hparams(obj):
    if isinstance(obj, dict):
        return _HParams({k: _hparams(v) for k, v in obj.items()})
    return obj


class Converter:
    """El modelo del conversor, cargado una vez para todo el programa."""
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get(cls, log=lambda *_: None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(log)
            return cls._instance

    def __init__(self, log):
        import torch
        from huggingface_hub import hf_hub_download
        from openvoice_lite.models import SynthesizerTrn
        log(T("Cargando el conversor de Mi voz..."))
        files = []
        for f in ("converter/config.json", "converter/checkpoint.pth"):
            try:
                files.append(hf_hub_download(CONVERTER_REPO, f, cache_dir=MODELS_DIR))
            except Exception:  # sin internet: vale lo ya descargado
                files.append(hf_hub_download(CONVERTER_REPO, f, cache_dir=MODELS_DIR, local_files_only=True))
        with open(files[0], encoding="utf-8") as fh:
            self.hps = _hparams(json.load(fh))
        torch.set_num_threads(max(2, min(4, (os.cpu_count() or 8) // 2)))
        self.torch = torch
        self.model = SynthesizerTrn(len(self.hps.get("symbols", [])), self.hps.data.filter_length // 2 + 1,
                                    n_speakers=self.hps.data.n_speakers, **self.hps.model)
        state = torch.load(files[1], map_location="cpu")
        self.model.load_state_dict(state["model"], strict=False)
        self.model.eval()
        self.rate = self.hps.data.sampling_rate
        self.window = torch.hann_window(self.hps.data.win_length)
        self.lock = threading.Lock()  # una conversión a la vez (todas usan el procesador)

    def _resample(self, audio, sr):
        if sr == self.rate:
            return audio.astype(np.float32)
        from scipy.signal import resample_poly
        g = np.gcd(int(sr), int(self.rate))
        return resample_poly(audio, self.rate // g, sr // g).astype(np.float32)

    def _spectrogram(self, audio):
        """Espectrograma lineal como el de OpenVoice (mel_processing.spectrogram_torch)."""
        torch, d = self.torch, self.hps.data
        y = torch.from_numpy(np.clip(audio, -1, 1)).float().unsqueeze(0)
        pad = int((d.filter_length - d.hop_length) / 2)
        y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, d.filter_length, hop_length=d.hop_length, win_length=d.win_length,
                          window=self.window, center=False, pad_mode="reflect", normalized=False,
                          onesided=True, return_complex=True)
        return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6)

    def extract_se(self, audio, sr):
        """Huella del timbre de una voz (np.ndarray [1, 256, 1])."""
        with self.lock, self.torch.no_grad():
            spec = self._spectrogram(self._resample(audio, sr))
            return self.model.ref_enc(spec.transpose(1, 2)).unsqueeze(-1).numpy()

    def convert(self, audio, sr, src_se, tgt_se, tau=TAU):
        """La voz de audio (con huella src_se) con el timbre de tgt_se: (audio, frecuencia)."""
        torch = self.torch
        with self.lock, torch.no_grad():
            spec = self._spectrogram(self._resample(audio, sr))
            lengths = torch.LongTensor([spec.size(-1)])
            out = self.model.voice_conversion(spec, lengths, sid_src=torch.from_numpy(src_se),
                                              sid_tgt=torch.from_numpy(tgt_se), tau=tau)[0][0, 0]
            return out.numpy().astype(np.float32), self.rate


_piper_se = {}  # nombre de la voz de Piper -> su huella


def piper_se(lang, voices, synth, ref_texts, audio=None, sr=None):
    """Huella de la voz de Piper del idioma (la de origen al convertir): se saca
    una vez de su frase de referencia (ref_texts[idioma]) y se guarda en
    voces\\_piper\\. Sin frase de referencia, de la propia frase (audio, sr).
    voices: {idioma: ruta de la voz de Piper}; synth(texto, idioma) -> (audio, sr)."""
    voice = voices.get(lang, lang).rsplit("/", 1)[-1]
    if voice in _piper_se:
        return _piper_se[voice]
    path = os.path.join(PIPER_SE_DIR, voice + ".npy")
    if os.path.exists(path):
        se = np.load(path)
    elif lang in ref_texts:
        ref_audio, ref_sr = synth(ref_texts[lang], lang)
        se = Converter.get().extract_se(ref_audio, ref_sr)
        os.makedirs(PIPER_SE_DIR, exist_ok=True)
        np.save(path, se)
    else:
        return Converter.get().extract_se(audio, sr)
    _piper_se[voice] = se
    return se


class MyVoice:
    """Lo que Speaker aplica a cada frase sintetizada: (audio, sr, idioma) ->
    (audio, sr) con el timbre de mi voz guardada `name` (o de la huella `se`)."""

    def __init__(self, name, synth, voices, ref_texts, log=lambda *_: None, se=None):
        self.name = name
        self.synth, self.voices, self.ref_texts = synth, voices, ref_texts
        self.conv = Converter.get(log)
        self.target = load_profile(name) if se is None else se

    def __call__(self, audio, sr, lang):
        if len(audio) < sr * 0.2:
            return audio, sr
        src = piper_se(lang, self.voices, self.synth, self.ref_texts, audio, sr)
        return self.conv.convert(audio, sr, src, self.target)
