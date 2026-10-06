"""
Cancelación de eco de mi voz traducida en la captura de altavoces (loopback).

Si mi voz traducida sale por los mismos altavoces que se capturan para
transcribir al otro, la captura la contiene y se tomaría como suya. La
captura de altavoces es una copia digital de lo que suena (sin sala ni
micrófono de por medio) y el programa sabe exactamente qué audio reproduce:
basta con colocarlo en su sitio y restarlo.

Colocarlo en su sitio: la salida de voz (voz.Speaker) se queda abierta todo
el rato y cuenta las muestras que entrega; aquí se cuentan las que llegan a la
captura. Como las dos van con el reloj del mismo dispositivo, el desfase entre
ambas cuentas es constante (medido: idéntico en todas las frases, con y sin
el micrófono Bluetooth abierto). Se mide una vez y cada frase queda colocada
por cuenta de muestras, sin adivinar nada aunque él esté hablando a la vez.
No se puede adivinar frase a frase por el audio: la voz es casi periódica y
la correlación tiene picos vecinos (un periodo del tono) casi igual de altos.

Restarlo: con el micrófono Bluetooth abierto (modo manos libres) Windows
altera un poco la señal (26 dB con una resta simple); un filtro FIR de 32
coeficientes, aprendido sobre la marcha, la deja en > 50 dB.

Modelo:  captura[n] = Σ_k h[k] · frase[pos + n + k - TAPS/2]  (+ la voz del otro)

Seguridad (nunca peor que sin cancelador): el bloque se silencia si
- se debería oír mi voz y el desfase aún no está medido y confirmado;
- el filtro no está comprobado (en los bloques donde solo suena mi voz se mide
  cuánto eco se le escapa: debe quedar MIN_ERLE_DB por debajo);
- un bloque dominado por mi voz encaja claramente en otro sitio (se perdieron
  muestras): entonces se vuelve a medir el desfase.
"""
import threading

import numpy as np
from scipy.signal import fftconvolve

TAPS = 32             # filtro que imita la alteración de Windows (centrado)
SEARCH_SEC = 1.0      # margen para medir el desfase alrededor de lo previsto por los tiempos
MEASURE_CORR = 0.97   # coincidencia exigida para medir el desfase (un pico vecino no llega)
MEASURE_AGREE = 2     # bloques distintos que deben dar el mismo desfase
CHECK_SEC = 0.015     # vigilancia: se busca el mejor encaje en ±15 ms...
CHECK_DOMINANT = 0.9  # ...en bloques donde mi voz domina (coincidencia ≥ esto)...
CHECK_TOL = 2         # ...y tiene que estar a ≤ 2 muestras de donde se espera
MIN_ERLE_DB = 30.0    # atenuación mínima exigida al eco; si no, se silencia el bloque
MIN_ECHO = 1e-6       # energía de eco por debajo de la cual no hay nada que restar
FORGET = 0.98         # memoria del aprendizaje del filtro
SINGLE_TALK = 0.01    # si la frase explica > 99 % del bloque, solo suena mi voz


class Reference:
    """Una frase de mi voz traducida: muestras exactas y dónde empezó a sonar."""

    def __init__(self, samples, out_start, rate, t_start):
        self.x = np.asarray(samples, dtype=np.float64)
        self.out_start, self.rate, self.t_start = out_start, rate, t_start

    def get(self, a, b):
        """x[a:b] con ceros fuera de la frase."""
        out = np.zeros(max(b - a, 0))
        lo, hi = max(a, 0), min(b, len(self.x))
        if hi > lo:
            out[lo - a:hi - a] = self.x[lo:hi]
        return out

    def taps_matrix(self, start, n):
        """Filas: frase[start + i - TAPS/2 : start + i + TAPS/2] para i en 0..n-1."""
        seg = self.get(start - TAPS // 2, start + n + TAPS // 2 - 1)
        return np.lib.stride_tricks.sliding_window_view(seg, TAPS)

    def best_lag(self, y, lo, hi):
        """Posición en [lo, hi] donde mejor encaja y, y su coincidencia (0..1)."""
        n = len(y)
        window = self.get(lo, hi + n)
        if not np.any(window) or not np.any(y):
            return None, 0.0
        c = np.abs(fftconvolve(window, y[::-1], mode="valid"))  # c[k] ~ x[lo+k:lo+k+n]·y
        e = np.concatenate([[0.0], np.cumsum(window ** 2)])
        norm = np.sqrt(np.maximum(e[n:n + len(c)] - e[:len(c)], 1e-12) * float(np.dot(y, y)))
        c = c / norm
        k = int(np.argmax(c))
        return lo + k, float(c[k])


class EchoCanceller:
    def __init__(self):
        self.lock = threading.Lock()
        self.refs = []
        self.epoch = None       # apertura de la salida de voz a la que se refiere todo
        self.in_n = 0           # muestras recibidas de la captura
        self.offset = None      # muestra de captura = muestra de salida + offset
        self.candidate = None   # [offset propuesto, bloques que lo confirman]
        self.h = np.zeros(TAPS)
        self.h[TAPS // 2] = 1.0  # al principio: copia exacta
        self.R = np.zeros((TAPS, TAPS))
        self.p = np.zeros(TAPS)
        self.validated = False  # ¿el filtro ha demostrado atenuar lo exigido?
        self.last_erle = None
        self.stats = {"cancelados": 0, "silenciados": 0}
        self.reasons = {}       # motivo -> bloques silenciados (diagnóstico)

    # -- avisos ------------------------------------------------------------- #
    def reset_capture(self):
        """La captura de altavoces se ha (re)abierto: su cuenta empieza de cero."""
        with self.lock:
            self.in_n, self.offset, self.candidate = 0, None, None
            self.refs = []

    def add_reference(self, samples, out_start, rate, epoch, t_start):
        """Lo llama voz.py cuando empieza a sonar cada frase."""
        with self.lock:
            if epoch != self.epoch:  # la salida se ha (re)abierto: otra cuenta
                self.epoch, self.offset, self.candidate = epoch, None, None
                self.refs = []
            self.refs.append(Reference(samples, out_start, rate, t_start))

    def cut_reference(self, out_start, kept, fade, epoch):
        """Lo llama voz.py si una frase se corta: solo sonaron sus `kept` primeras
        muestras, las `fade` últimas con un fundido. Si no, se restaría lo que ya
        no suena."""
        with self.lock:
            if epoch != self.epoch:
                return
            for r in self.refs:
                if r.out_start == out_start and kept < len(r.x):
                    x = r.x[:kept].copy()
                    if fade:
                        x[-fade:] *= np.linspace(1.0, 0.0, fade)
                    r.x = x

    @property
    def measured(self):
        return self.offset is not None

    # -- cada bloque de la captura de altavoces ------------------------------ #
    def process(self, block, t_end, rate):
        """block: muestras mono de la captura (todas, en orden); t_end: cuándo
        llegó (time.monotonic). Devuelve (bloque, silenciado)."""
        y = np.asarray(block, dtype=np.float64)
        n = len(y)
        with self.lock:
            cap_start = self.in_n
            self.in_n += n
            self.refs = [r for r in self.refs if not self._done(r, cap_start, t_end)]
            refs = [r for r in self.refs if r.rate == rate]
        if not refs:
            return block, False
        if self.offset is None:
            return self._measure(block, y, refs, cap_start, t_end - n / rate)

        X, used = None, []
        for r in refs:
            pos = cap_start - self.offset - r.out_start  # muestra de la frase en este bloque
            if pos >= len(r.x) + TAPS or pos + n <= -TAPS:
                continue
            Xr = r.taps_matrix(pos, n)
            X = Xr if X is None else X + Xr  # dos frases a la vez: se suman
            used.append((r, pos))
        if X is None:
            return block, False
        if self._misaligned(y, used):
            return self._silence(block, "desfase perdido")
        return self._cancel(block, y, X)

    # -- desfase -------------------------------------------------------------- #
    def _measure(self, block, y, refs, cap_start, t_start):
        """Aún sin desfase: se busca cada frase alrededor de lo previsto por los
        tiempos y se exige una coincidencia casi exacta en dos bloques."""
        silence = False
        for r in refs:
            if not (t_start < r.t_start + len(r.x) / r.rate + 0.3 and
                    t_start + len(y) / r.rate > r.t_start - SEARCH_SEC):
                continue
            if t_start + len(y) / r.rate > r.t_start - 0.2:
                silence = True  # debería sonar mi voz y aún no sé dónde: silencio
            guess = int(round((t_start - r.t_start) * r.rate))
            span = int(SEARCH_SEC * r.rate)
            pos, corr = r.best_lag(y, guess - span, guess + span)
            if pos is None or corr < MEASURE_CORR:
                continue
            cand = cap_start - (r.out_start + pos)
            if self.candidate and abs(cand - self.candidate[0]) <= 1:
                self.candidate[1] += 1
            else:
                self.candidate = [cand, 1]
            if self.candidate[1] >= MEASURE_AGREE:
                self.offset = self.candidate[0]
        if silence:
            return self._silence(block, "midiendo desfase")
        return block, False

    def _misaligned(self, y, used):
        """Si mi voz domina el bloque, su mejor encaje tiene que estar donde se
        espera. Si no (se perdieron muestras), el desfase se vuelve a medir."""
        for r, pos in used:
            span = int(CHECK_SEC * r.rate)
            best, corr = r.best_lag(y, pos - span, pos + span)
            if best is not None and corr >= CHECK_DOMINANT and abs(best - pos) > CHECK_TOL:
                self.offset, self.candidate = None, None
                return True
        return False

    def _done(self, r, cap_start, t_now):
        if self.offset is not None:
            return cap_start - self.offset - r.out_start > len(r.x) + TAPS
        return t_now > r.t_start + len(r.x) / r.rate + SEARCH_SEC + 0.5

    # -- resta ---------------------------------------------------------------- #
    def _cancel(self, block, y, X):
        echo = X @ self.h
        e_echo = float(np.dot(echo, echo))
        if e_echo < MIN_ECHO:
            return block, False
        XtX, Xty = X.T @ X, X.T @ y
        reg = 1e-9 * np.trace(XtX) / TAPS
        try:
            h_blk = np.linalg.solve(XtX + reg * np.eye(TAPS), Xty)
        except np.linalg.LinAlgError:
            return self._silence(block, "error de cálculo")
        unexplained = y - X @ h_blk
        if float(np.dot(unexplained, unexplained)) < SINGLE_TALK * float(np.dot(y, y)):
            # Solo suena mi voz: aquí se mide con fiabilidad cuánto eco se le
            # escapa al filtro en uso, y se aprende de este bloque.
            erle_before = self._erle(X, h_blk, e_echo)
            if erle_before < MIN_ERLE_DB:
                # Windows ha cambiado la alteración (p. ej. al pasar a manos
                # libres): se reaprende desde este bloque en vez de poco a poco.
                self.R, self.p = XtX.copy(), Xty.copy()
            else:
                self.R = FORGET * self.R + XtX
                self.p = FORGET * self.p + Xty
            try:
                self.h = np.linalg.solve(self.R + reg * np.eye(TAPS), self.p)
            except np.linalg.LinAlgError:
                pass
            echo = X @ self.h
            erle = self._erle(X, h_blk, float(np.dot(echo, echo)))
            self.validated, self.last_erle = erle >= MIN_ERLE_DB, erle
            if erle_before < MIN_ERLE_DB:
                return self._silence(block, "filtro reaprendido")
        elif not self.validated:
            return self._silence(block, "filtro sin comprobar")
        self.stats["cancelados"] += 1
        return (y - echo).astype(np.float32), False

    def _erle(self, X, h_blk, e_echo):
        """Atenuación (dB) del filtro en uso comparado con el ideal del bloque."""
        missed = X @ (h_blk - self.h)
        return 10 * np.log10(e_echo / max(float(np.dot(missed, missed)), 1e-15))

    def _silence(self, block, reason):
        self.stats["silenciados"] += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return np.zeros_like(block), True
