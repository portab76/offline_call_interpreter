"""
Transcriptor en directo: captura el audio que sale por los altavoces de Windows
(loopback WASAPI) y lo muestra transcrito palabra a palabra, sin traducir.

Whisper no trabaja palabra a palabra, así que cada ~0,4 s se vuelve a
transcribir el audio acumulado de la frase en curso. Las palabras que salen
iguales en dos pasadas seguidas se dan por buenas (blanco); el resto se muestra
como provisional (gris) y puede cambiar en la siguiente pasada.

Cada párrafo lleva un número (el mismo que usa traducir_archivo.py) y se cierra
en cada pausa o, si se habla sin parar, al acabar una frase larga.

Con varios idiomas (--idiomas es,en) se detecta el idioma tras ~2 s de voz y se
vuelve a comprobar cada ~3 s; si cambia, empieza un párrafo nuevo. Cada párrafo
lleva su etiqueta [EN], [ES]... Si la detección es dudosa, se usa el primer
idioma de la lista.

Por defecto usa la GPU con whisper.cpp + Vulkan (carpeta motor_gpu, sirve para
AMD, NVIDIA e Intel) y, si no hay GPU compatible, faster-whisper en la CPU.

Uso:
    python transcriptor.py                  # inglés, modelo base.en
    python transcriptor.py --motor cpu      # forzar faster-whisper en la CPU
    python transcriptor.py --modelo small.en   # más preciso, más lento
    python transcriptor.py --idiomas es     # otro idioma (usa modelo multilingüe)
    python transcriptor.py --idiomas es,en  # varios idiomas, detección automática
    python transcriptor.py --dispositivos   # lista las salidas de audio
    python transcriptor.py --dispositivo 12
"""
import argparse
import csv
import os
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

from motores import create_engine
from interfaz import T
from traductor import BASE_DIR, HALLUCINATIONS, NOSPACE, TARGET_SR, Capture, list_devices, to_16k

STEP_SEC = 0.4        # audio nuevo necesario para lanzar otra pasada
PAUSE_SEC = 1.0       # silencio que cierra el párrafo
PREROLL_SEC = 0.5     # audio que se guarda antes de detectar voz
TRIM_SEC = 5.0        # a partir de aquí se recorta lo ya confirmado del búfer
MAX_BUF_SEC = 20.0    # búfer máximo: se confirma todo y se vacía
SPLIT_WORDS = 20      # sin pausas, se cierra el párrafo al acabar frase a partir de aquí
DETECT_SEC = 2.0      # voz necesaria para decidir el idioma de un turno
RECHECK_SEC = 3.0     # cada cuánto se vuelve a comprobar el idioma
LANG_SURE = 0.65      # probabilidad (entre los idiomas indicados) para fiarse
# Pista inicial por idioma cuando aún no hay texto anterior: en chino, una frase
# en caracteres simplificados hace que Whisper no use los tradicionales.
START_PROMPT = {"zh": "以下是普通话的句子。"}
WAVE_STEP = 0.025     # la onda de la ventana: un pico cada 25 ms...
WAVE_SEC = 3.0        # ...de los últimos 3 s
WAVE_POINTS = int(WAVE_SEC / WAVE_STEP)


def norm(word):
    return re.sub(r"[^\w']", "", word.lower())


SENTENCE_END = (".", "?", "!", "؟", "。", "？", "！")
ECHO_WORDS = 5   # tantas palabras seguidas iguales a la pista: Whisper la está repitiendo


def repeat_at_end(raw):
    """Palabras de la unidad que se acaba de repetir al final de raw (lista de
    palabras tal cual), o 0. Es repetición:
    - una frase de 3 o más palabras dicha dos veces seguidas;
    - una de 2 palabras dos veces si acaba en punto ("the country. the
      country."), si no, tres ("80 et 80 et" se respeta);
    - una palabra suelta 4 veces ("no, no, no" se respeta)."""
    seq = [norm(w) for w in raw]
    for n in range(2, min(12, len(seq) // 2) + 1):
        unit = seq[-n:]
        if unit != seq[-2 * n:-n] or len(set(unit)) == 1:  # "no no no no": palabra suelta
            continue
        if n >= 3 or raw[-1].rstrip().endswith(SENTENCE_END) or seq[-3 * n:-2 * n] == unit:
            return n
    if len(seq) >= 4 and len(set(seq[-4:])) == 1:
        return 1
    return 0


def drop_repeats(done, words):
    """Las palabras de words que no repiten lo inmediatamente anterior (done:
    lo ya confirmado). Con Whisper pequeño y habla rápida salen frases
    duplicadas que, confirmadas, acaban en la pista y se repiten cada vez más."""
    raw = [w[2] for w in done[-36:]]
    kept = []
    for w in words:
        raw.append(w[2])
        kept.append(w)
        n = repeat_at_end(raw)
        if n:
            n = min(n, len(kept))  # lo ya confirmado no se puede quitar
            del raw[-n:], kept[-n:]
    return kept


def has_repeat(raw):
    """¿Hay alguna repetición seguida (como las de repeat_at_end) en raw?"""
    return any(repeat_at_end(raw[:i]) for i in range(4, len(raw) + 1))


def echoes(words, context):
    """¿La pasada repite ECHO_WORDS palabras seguidas de la pista? (Whisper la
    copia en vez de escuchar el audio.)"""
    got = [norm(w[2]) for w in words]
    ctx = " " + " ".join(norm(w) for w in context.split()) + " "
    return any(" " + " ".join(got[i:i + ECHO_WORDS]) + " " in ctx
               for i in range(len(got) - ECHO_WORDS + 1))


def num(x):
    """Número con coma decimal (para abrir el CSV con Excel en español)."""
    return f"{x:.2f}".replace(".", ",")


class Medidor:
    """Registro para analizar la transcripción; no cambia nada de lo que hace.
    Escribe dos CSV (separados por ";"):
    - <prefijo>_pasadas.csv: una línea por pasada de Whisper.
    - <prefijo>_palabras.csv: una línea por palabra confirmada (pasa a blanco).
    Los tiempos cuentan desde que el audio llega al transcriptor (el filtro de
    voz lo retiene antes ~0,25 s más)."""

    def __init__(self, prefix):
        self.t0 = time.monotonic()
        self.files, self.out = [], {}
        for name, head in (
                ("pasadas", ["hora_s", "parrafo", "idioma", "bufer_s", "pasada_ms", "palabras_pasada",
                             "confirmadas", "grises", "motivo", "punto_no_cortado", "punto_en_gris",
                             "repetidas_quitadas", "pista_vaciada"]),
                ("palabras", ["hora_s", "parrafo", "palabra", "retraso_s", "gris_s", "motivo"])):
            f = open(f"{prefix}_{name}.csv", "w", newline="", encoding="utf-8-sig")
            self.files.append(f)
            self.out[name] = csv.writer(f, delimiter=";")
            self.out[name].writerow(head)
        self.seen = deque(maxlen=200)  # (hora, fin de lo transcrito) de cada pasada

    def now(self):
        return time.monotonic() - self.t0

    def pasada(self, para, lang, buf_s, secs, hyp, new, grey, motivo, limit, committed,
               dropped=0, cleared=""):
        if hyp:
            self.seen.append((self.now(), hyp[-1][1]))
        # ¿Había un fin de frase que no cortó? (en lo confirmado, no al final de
        # la tanda, con la frase ya por encima de SPLIT_WORDS)
        ends = [i for i, w in enumerate(new) if w[2].rstrip().endswith(SENTENCE_END)]
        missed = bool(ends) and ends[-1] != len(new) - 1 and committed >= limit
        in_grey = any(w[2].rstrip().endswith(SENTENCE_END) for w in grey)
        self.out["pasadas"].writerow([num(self.now()), para, lang or "", num(buf_s), round(secs * 1000),
                                      len(hyp), len(new), len(grey), motivo,
                                      "sí" if missed else "", "sí" if in_grey else "",
                                      dropped or "", cleared])
        self.files[0].flush()

    def palabras(self, para, words, clock, motivo):
        """words: (inicio, fin, palabra) en segundos de audio; clock: hora (de
        time.monotonic) que corresponde al segundo 0 del audio."""
        now = self.now()
        for s, e, w in words:
            heard = clock + e - self.t0
            first = next((t for t, end in self.seen if end >= e - 0.1), now)
            self.out["palabras"].writerow([num(now), para, w.strip(), num(now - heard),
                                           num(max(0.0, now - first)), motivo])
        self.files[1].flush()

    def close(self):
        for f in self.files:
            f.close()


# --------------------------------------------------------------------------- #
# Transcripción continua
# --------------------------------------------------------------------------- #
class Streamer(threading.Thread):
    """Lee bloques de audio, transcribe el búfer una y otra vez y manda a ui_q:
    ("head", número, idioma|None) al empezar cada párrafo,
    ("words", confirmadas_nuevas, provisionales) y ("para",) al cerrarlo."""

    def __init__(self, engine, languages, audio_q, get_rate, threshold, ui_q,
                 first_para=1, measure=None):
        super().__init__(daemon=True)
        # measure: prefijo de los CSV de Medidor (None: no se mide nada).
        self.meter = Medidor(measure) if measure else None
        self.clock = time.monotonic()  # hora a la que corresponde el segundo 0 del audio
        self.engine, self.languages = engine, languages
        self.multi = len(languages) > 1
        # Con un solo idioma queda fijo; con varios se decide en cada turno.
        self.language = None if self.multi else languages[0]
        self.prev_language = None
        self.last_check = 0.0      # self.now() de la última comprobación de idioma
        self.audio_q, self.get_rate = audio_q, get_rate
        self.threshold, self.ui_q = threshold, ui_q
        self.level = 0.0  # para el medidor de la ventana
        # Pico de cada trozo de WAVE_STEP s de los últimos WAVE_SEC: la onda que
        # dibuja la ventana de la conversación.
        self.wave = deque([0.0] * WAVE_POINTS, maxlen=WAVE_POINTS)
        self.stopped = False

        self.para_no = first_para  # número del próximo párrafo
        self.head_pending = True   # el párrafo aún no ha mostrado su número

        self.buf = np.zeros(0, np.float32)
        self.buf_t0 = 0.0          # tiempo (s) del primer sample del búfer
        self.total = 0             # samples recibidos desde el inicio
        self.last_voice = None     # tiempo de la última voz; None = sin voz pendiente
        self.last_pass = 0         # self.total en la última pasada
        self.committed = []        # palabras confirmadas del párrafo actual
        self.in_buf = []           # (inicio, fin, palabra, ¿se mostró?) cuyo audio sigue en el búfer
        self.commit_end = 0.0      # fin (s) de la última palabra confirmada
        self.prev_hyp = []         # provisionales de la pasada anterior
        self.recent = []           # últimas mostradas (también de párrafos anteriores): repeticiones
        # Texto ya dicho cuyo audio salió del búfer, para initial_prompt. Lo que
        # sigue en el búfer no puede ir aquí: Whisper lo daría por dicho y se
        # lo saltaría.
        self.context = ""
        self.dropped, self.cleared = 0, ""  # para Medidor: repeticiones quitadas, pista vaciada

    def now(self):
        return self.total / TARGET_SR

    # -- audio ------------------------------------------------------------- #
    def _read_audio(self):
        try:
            blocks = [self.audio_q.get(timeout=0.1)]
        except queue.Empty:
            self.level = 0.0
            self.wave.extend([0.0] * int(0.1 / WAVE_STEP))  # sin audio: la onda sigue, plana
            return
        while True:
            try:
                blocks.append(self.audio_q.get_nowait())
            except queue.Empty:
                break
        rate = self.get_rate() or 48000
        for _, raw in blocks:
            a = to_16k(raw, rate)
            rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
            self.level = rms
            step = int(TARGET_SR * WAVE_STEP)
            for i in range(0, len(a), step):
                self.wave.append(float(np.max(np.abs(a[i:i + step]))))
            self.buf = np.concatenate([self.buf, a])
            self.total += len(a)
            if rms > self.threshold:
                self.last_voice = self.now()
        self.clock = time.monotonic() - self.now()

    def _trim(self, t):
        n = int((t - self.buf_t0) * TARGET_SR)
        if n > 0:
            self.buf = self.buf[n:]
            self.buf_t0 += n / TARGET_SR
            gone = [w for w in self.in_buf if w[0] < self.buf_t0]
            self.in_buf = self.in_buf[len(gone):]
            self.context += "".join(w[2] for w in gone if w[3])  # sin las repeticiones quitadas

    # -- idioma ------------------------------------------------------------ #
    def _guess_language(self):
        """(idioma, probabilidad) entre los indicados, con la voz del búfer."""
        probs = self.engine.lang_probs(self.buf)
        p = {c: probs.get(c, 0.0) for c in self.languages}
        total = sum(p.values()) or 1.0
        best = max(p, key=p.get)
        return best, p[best] / total

    def _set_language(self, lang):
        if lang != self.prev_language:
            self.context = ""  # el texto previo en otro idioma despista a Whisper
        self.language = self.prev_language = lang
        self.last_check = self.now()

    def _detect_language(self):
        lang, p = self._guess_language()
        # Dudoso: se usa el idioma principal (el primero de la lista).
        self._set_language(lang if p >= LANG_SURE else self.languages[0])

    def _recheck_language(self):
        """Con más audio la detección es más fiable: si ahora sale otro idioma
        con claridad, lo pendiente se vuelve a transcribir en ese idioma."""
        self.last_check = self.now()
        lang, p = self._guess_language()
        if lang == self.language or p < LANG_SURE:
            return
        self.prev_hyp = []
        self._close_paragraph()
        self._trim(self.commit_end)  # lo confirmado se queda como estaba
        self._set_language(lang)

    # -- whisper ----------------------------------------------------------- #
    def _has_voice(self):
        """¿Algún trozo de 0,1 s del búfer supera el umbral de voz?"""
        n = int(TARGET_SR * 0.1)
        usable = len(self.buf) // n * n
        if usable == 0:
            return False
        rms = np.sqrt(np.mean(self.buf[:usable].reshape(-1, n) ** 2, axis=1))
        return bool(np.any(rms > self.threshold))

    def _transcribe(self):
        # Sobre silencio Whisper se inventa frases ("Y ahora, vamos a ver si...").
        # Pasa p. ej. tras recortar lo ya confirmado al final de un turno.
        if not self._has_voice():
            return []
        if self.language is None:
            self._detect_language()
        # La pista (lo último ya dicho) ayuda a Whisper, pero si trae
        # repeticiones Whisper las copia y entra en bucle: entonces se vacía.
        prompt = self.context[-200:]
        if prompt and has_repeat(prompt.split()):
            self._clear_context("repetición")
            prompt = ""
        words = self.engine.words(self.buf, self.language,
                                  prompt or START_PROMPT.get(self.language))
        words = [(self.buf_t0 + s, self.buf_t0 + e, w) for s, e, w in words
                 if norm(w)]  # fuera los "." sueltos
        if getattr(self.engine, "looped", False):
            self._clear_context("bucle")  # lo de antes del bucle sí vale
        elif prompt and echoes(words, prompt):
            # Copia la pista en vez de escuchar: la pasada no vale; la
            # siguiente, ya sin pista, escucha el audio.
            self._clear_context("eco")
            return []
        return words

    def _clear_context(self, why):
        self.context = ""
        self.cleared = why

    def _hypothesis(self):
        """Palabras de la pasada actual que van después de lo ya confirmado."""
        words = self._transcribe()
        got = [norm(w[2]) for w in words]
        done = [norm(w[2]) for w in self.in_buf]
        k = len(done)
        if got[:k] == done:
            # Lo normal: la pasada empieza repitiendo lo ya confirmado.
            hyp = words[k:]
        else:
            # Whisper cambió alguna palabra ya confirmada: se busca dónde
            # aparecen las últimas confirmadas y se sigue desde ahí; si no
            # aparecen, se usan los tiempos.
            hyp = None
            for m in (3, 2):
                tail = done[-m:]
                if len(tail) < m:
                    continue
                for j in range(len(got) - m, -1, -1):
                    if got[j:j + m] == tail:
                        hyp = words[j + m:]
                        break
                if hyp is not None:
                    break
            if hyp is None:
                hyp = [w for w in words if w[0] >= self.commit_end - 0.2]
                # Con tiempos a veces se cuela alguna palabra ya confirmada.
                done = [norm(w[2]) for w in self.committed]
                for n in range(min(5, len(done), len(hyp)), 0, -1):
                    if done[-n:] == [norm(w[2]) for w in hyp[:n]]:
                        hyp = hyp[n:]
                        break
        return hyp

    # -- párrafos ---------------------------------------------------------- #
    def _para(self):
        """Número del párrafo en curso (el del próximo si aún no ha empezado)."""
        return self.para_no if self.head_pending else self.para_no - 1

    def _measure(self, secs, hyp, new, grey, motivo):
        if self.meter:
            limit = SPLIT_WORDS * 3 if self.language in NOSPACE else SPLIT_WORDS
            self.meter.pasada(self._para(), self.language, len(self.buf) / TARGET_SR, secs, hyp,
                              new, grey, motivo, limit, len(self.committed),
                              self.dropped, self.cleared)
        self.dropped, self.cleared = 0, ""

    def _commit(self, words, motivo="acuerdo"):
        """Confirma words (su audio queda atrás) y devuelve las que se muestran:
        sin las que repiten lo inmediatamente anterior."""
        if not words:
            return []
        shown = drop_repeats(self.recent, words)
        self.recent = (self.recent + shown)[-36:]
        self.dropped += len(words) - len(shown)
        if self.meter and shown:
            self.meter.palabras(self._para(), shown, self.clock, motivo)
        self.committed += shown
        keep = {id(w) for w in shown}
        # En el búfer siguen todas (para casar con la pasada siguiente); la
        # marca dice si pasan a la pista.
        self.in_buf += [(*w, id(w) in keep) for w in words]
        self.commit_end = words[-1][1]
        return shown

    def _emit(self, new_words):
        if self.head_pending and (new_words or self.prev_hyp):
            self.ui_q.put(("head", self.para_no, self.language if self.multi else None))
            self.para_no += 1
            self.head_pending = False
        self.ui_q.put(("words", "".join(w[2] for w in new_words),
                       "".join(w[2] for w in self.prev_hyp)))

    def _close_paragraph(self):
        """Cierra el párrafo en pantalla y en el archivo; el siguiente llevará
        número nuevo."""
        self.ui_q.put(("words", "", ""))  # borra lo provisional que quedara
        if not self.head_pending:
            self.ui_q.put(("para",))
        self.committed = []
        self.head_pending = True

    def _end_paragraph(self):
        """Pausa: la última pasada se confirma entera y se empieza de cero."""
        # Sin el silencio final: sobre silencio Whisper a veces se inventa
        # frases ("y que se puede hacer.", "Thanks for watching!").
        end = int((self.last_voice + 0.3 - self.buf_t0) * TARGET_SR)
        if 0 < end < len(self.buf):
            self.buf = self.buf[:end]
            self.last_pass = 0  # hay que volver a transcribir
        t = time.monotonic()
        if self.total > self.last_pass:
            hyp = self._hypothesis()
        else:
            hyp = self.prev_hyp
        para = "".join(w[2] for w in self.committed + hyp).strip()
        if not self.committed and para.lower() in HALLUCINATIONS:
            hyp = []
        shown = self._commit(hyp, "pausa")
        self._measure(time.monotonic() - t, hyp, hyp, [], "pausa")
        self.prev_hyp = []
        self._emit(shown)
        self._close_paragraph()
        self.in_buf = []
        self.last_voice = None
        if self.multi:
            self.language = None  # el siguiente turno puede ser en otro idioma
        # Lo que queda es silencio: se vacía (el búfer pudo recortarse arriba).
        self.buf = np.zeros(0, np.float32)
        self.buf_t0 = self.now()

    def _step(self):
        self.last_pass = self.total
        t = time.monotonic()  # (la pasada incluye la comprobación del idioma, si toca)
        if (self.multi and self.language is not None
                and self.now() - self.last_check >= RECHECK_SEC):
            self._recheck_language()
        hyp = self._hypothesis()
        # Acuerdo local: se confirma el prefijo común con la pasada anterior.
        n = 0
        while (n < len(hyp) and n < len(self.prev_hyp)
               and norm(hyp[n][2]) == norm(self.prev_hyp[n][2])):
            n += 1
        new = hyp[:n]
        shown = self._commit(new)
        self._measure(time.monotonic() - t, hyp, new, hyp[n:], "acuerdo")
        self.prev_hyp = hyp[n:]
        self._emit(shown)

        # Sin pausas el párrafo no se cerraría nunca: se cierra al acabar una
        # frase larga para que tenga número propio y se pueda traducir ya.
        # (En chino o japonés cada carácter cuenta como palabra: hacen falta más.)
        limit = SPLIT_WORDS * 3 if self.language in NOSPACE else SPLIT_WORDS
        if (len(self.committed) >= limit
                and self.committed[-1][2].rstrip().endswith((".", "?", "!", "؟", "。", "？", "！"))):
            self._close_paragraph()

        length = self.now() - self.buf_t0
        if length > MAX_BUF_SEC:
            rest, self.prev_hyp = self.prev_hyp, []
            self._emit(self._commit(rest, "búfer lleno"))
            self._trim(self.commit_end)
        elif length > TRIM_SEC and self.in_buf:
            # Recorta lo ya confirmado para que cada pasada siga siendo rápida:
            # mejor tras un punto; si no hay, tras la última palabra confirmada.
            cut = self.in_buf[-1][1] if length > 2 * TRIM_SEC else None
            for w in reversed(self.in_buf):
                if w[2].rstrip().endswith((".", "?", "!", ",")):
                    cut = w[1]
                    break
            if cut:
                self._trim(cut)

    def stop(self):
        self.stopped = True

    def run(self):
        while not self.stopped:
            self._read_audio()
            if self.last_voice is None:
                # Sin voz: solo se guarda un poco de audio previo.
                self._trim(self.now() - PREROLL_SEC)
                continue
            try:
                if self.now() - self.last_voice >= PAUSE_SEC:
                    self._end_paragraph()
                elif self.language is None and len(self.buf) < DETECT_SEC * TARGET_SR:
                    continue  # aún poca voz para saber el idioma
                elif self.total - self.last_pass >= STEP_SEC * TARGET_SR:
                    self._step()
            except Exception as e:
                self.ui_q.put(("status", T("Error transcribiendo: {error}", error=e)))
        if self.meter:
            self.meter.close()


# --------------------------------------------------------------------------- #
# Ventana
# --------------------------------------------------------------------------- #
def next_paragraph_number(path):
    """Si se reabre en el mismo minuto se sigue escribiendo en el mismo archivo:
    la numeración continúa."""
    try:
        with open(path, encoding="utf-8") as f:
            nums = re.findall(r"(?m)^(\d+) ", f.read())
        return max(map(int, nums)) + 1 if nums else 1
    except OSError:
        return 1


def run_gui(args):
    import tkinter as tk

    ui_q = queue.Queue()
    audio_q = queue.Queue()

    root = tk.Tk()
    root.title(f"Transcriptor en directo ({', '.join(args.idiomas)})")
    root.geometry("900x600")
    root.configure(bg="#111")

    size = {"n": args.letra}
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
        txt.tag_configure("ok", font=("Segoe UI", size["n"]), foreground="#ffffff")
        txt.tag_configure("tent", font=("Segoe UI", size["n"]), foreground="#8a8a8a")
        txt.tag_configure("head", font=("Segoe UI", max(8, size["n"] - 8), "bold"),
                          foreground="#6fa8dc")

    def bigger():
        size["n"] += 2; apply_fonts()

    def smaller():
        size["n"] = max(10, size["n"] - 2); apply_fonts()

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
    apply_fonts()

    os.makedirs(os.path.join(BASE_DIR, "transcripciones"), exist_ok=True)
    log_path = os.path.join(
        BASE_DIR, "transcripciones",
        datetime.now().strftime("transcripcion_%Y-%m-%d_%H%M.txt"))
    first_para = next_paragraph_number(log_path)
    # Se crea ya, aunque esté vacío, para que traducir_archivo.py lo encuentre
    # desde el principio y no siga la sesión anterior.
    with open(log_path, "a", encoding="utf-8") as f:
        if first_para > 1:
            f.write("\n\n")  # por si la sesión anterior quedó a medias

    def log(msg):
        ui_q.put(("status", msg))

    state = {"streamer": None}

    def startup():
        try:
            engine = create_engine(args.motor, args.modelo, args.cpu, log)
        except Exception as e:
            log(f"No se pudo cargar el modelo: {e}")
            return
        cap = Capture(audio_q, args.dispositivo, log)
        cap.start()
        cap.ready.wait(10)
        if cap.error:
            log(f"Error de audio: {cap.error}")
            return
        st = Streamer(engine, args.idiomas, audio_q, lambda: cap.rate,
                      args.umbral, ui_q, first_para)
        st.start()
        state["streamer"] = st
        log(f"Escuchando ({engine.name}, modelo {args.modelo}). "
            f"Transcripción en: transcripciones\\{os.path.basename(log_path)}")

    threading.Thread(target=startup, daemon=True).start()

    def write(s):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(s)

    def poll():
        try:
            while True:
                item = ui_q.get_nowait()
                kind = item[0]
                if kind == "status":
                    status.set(item[1])
                    continue
                at_bottom = txt.yview()[1] > 0.98
                txt.delete("live", "end-1c")
                if kind == "head":
                    _, num, lang = item
                    label = f"{num} " + (f"[{lang.upper()}] " if lang else "")
                    txt.insert("end", label, "head")
                    txt.mark_set("live", "end-1c")
                    write(label)
                elif kind == "words":
                    _, ok, tent = item
                    if ok:
                        txt.insert("end", ok, "ok")
                        write(ok)
                    txt.mark_set("live", "end-1c")
                    txt.insert("end", tent, "tent")
                elif kind == "para":
                    txt.insert("end", "\n\n", "ok")
                    txt.mark_set("live", "end-1c")
                    write("\n\n")
                if at_bottom:
                    txt.see("end")
        except queue.Empty:
            pass
        st = state["streamer"]
        if st is not None:
            lvl = min(1.0, st.level * 8)
            meter.coords(meter_bar, 0, 0, int(80 * lvl), 10)
        root.after(50, poll)

    root.after(50, poll)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Transcriptor en directo palabra a palabra")
    ap.add_argument("--modelo", default=None,
                    help="tiny.en, base.en (defecto), small.en, medium.en; "
                         "para otros idiomas: tiny, base, small, medium, large-v3")
    ap.add_argument("--idiomas", "--idioma", dest="idiomas", default="en",
                    help="idioma o idiomas que se escuchan, separados por comas; "
                         "el primero es el principal: en (defecto), es, fr, de... "
                         "p. ej. es,en")
    ap.add_argument("--dispositivo", type=int, default=None,
                    help="índice de salida a capturar (ver --dispositivos)")
    ap.add_argument("--dispositivos", action="store_true", help="listar salidas y salir")
    ap.add_argument("--umbral", type=float, default=0.008,
                    help="nivel mínimo para considerar que hay voz (defecto 0.008)")
    ap.add_argument("--letra", type=int, default=24, help="tamaño de letra")
    ap.add_argument("--motor", default="auto", choices=["auto", "gpu", "cpu"],
                    help="auto (defecto): GPU con whisper.cpp+Vulkan si se puede "
                         "(AMD, NVIDIA, Intel), si no CPU; gpu; cpu: faster-whisper")
    ap.add_argument("--cpu", action="store_true", help="no usar ninguna GPU")
    args = ap.parse_args()

    args.idiomas = [c.strip().lower() for c in args.idiomas.split(",") if c.strip()] or ["en"]
    only_en = args.idiomas == ["en"]
    if args.modelo is None:
        args.modelo = "base.en" if only_en else "base"
    elif not only_en and args.modelo.endswith(".en"):
        args.modelo = args.modelo[:-3]  # los modelos .en solo entienden inglés
    if args.dispositivos:
        list_devices()
        return
    run_gui(args)


if __name__ == "__main__":
    main()
