# offline call interpreter

**Intérprete de voz en tiempo real para Windows, en los dos sentidos y 100 % sin internet.**
Hablas en tu idioma, el otro te oye en el suyo, y lo que él dice lo lees traducido. Funciona en llamadas de **Teams, Zoom o WhatsApp Web** y en conversaciones cara a cara.

![Llamada español ↔ inglés](docs/capturas/llamada.png)

## Qué lo diferencia

- **Nada sale de tu PC.** Reconocimiento, traducción y voz se ejecutan en local: sin cuentas, sin suscripción y sin enviar tus conversaciones a la nube.
- **Sirve con cualquier aplicación de llamadas**, no solo con una: tu voz traducida entra en la llamada como si fuera un micrófono.
- **Tú decides qué se dice.** Ves tu frase transcrita y traducida, la corriges con el teclado si hace falta y suena al pulsar su icono.
- **Con tu propia voz.** Opcional: grabas 30 segundos y el otro te oye con tu timbre en su idioma.
- **Va en tarjetas gráficas antiguas**, también AMD e Intel (Vulkan), no solo en NVIDIA. Sin tarjeta compatible, funciona con el procesador.

## Funciones

- 11 idiomas listos (español, inglés, francés, alemán, italiano, catalán, portugués, ruso, chino, vietnamita y árabe) y se pueden añadir más desde el programa.
- Modo **llamada** (el otro suena por el PC) y modo **en persona** (los dos por el mismo micrófono).
- Corrección de cada frase en el sitio, también mientras hablas, y atajos de teclado para no soltarlo.
- Elección del modelo de reconocimiento, con prueba de velocidad en tu equipo.
- Interfaz en español, inglés o cualquier idioma del catálogo.

## Aceleración por GPU (Vulkan)

El reconocimiento de voz usa **whisper.cpp compilado con Vulkan** (`motor_gpu\`: `whisper-server.exe` + `ggml-vulkan.dll`, ya incluidos). A diferencia de la mayoría de programas de IA, que solo aceleran con NVIDIA (CUDA), funciona con **cualquier tarjeta con controlador Vulkan 1.2**, sin instalar nada aparte del controlador normal. Gasta unas 10 veces menos CPU que hacerlo con el procesador.

| Fabricante | Tarjetas admitidas |
|---|---|
| **AMD** | Radeon RX 400 y posteriores (RX 500, 5000, 6000, 7000…), Vega y gráficas integradas Ryzen. Probado en una **RX 580** |
| **NVIDIA** | GeForce GTX 900 y posteriores, RTX 20/30/40/50 |
| **Intel** | Arc, Iris Xe y gráficas integradas desde la 6.ª generación |

El modelo `base` necesita unos 300 MB de memoria de vídeo. Si no hay tarjeta compatible, el programa pasa solo al procesador (faster-whisper); con NVIDIA y CUDA 12 instalado también puede usar CUDA.

## Instalación

1. Descarga el repositorio (**Code → Download ZIP**) y descomprímelo donde lo vayas a usar.
2. Ejecuta **`1_instalar.bat`**. Prepara Python y sus paquetes (si falta Python, ofrece instalarlo).
3. Abre **`2_conversacion.bat`**. La primera vez descarga los modelos que necesita (~500 MB para español ↔ inglés).

Para que el otro te oiga en una llamada, instala [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (gratis). Los pasos para Teams, Zoom y WhatsApp Web están dentro del programa, en **Herramientas → Llamadas**.

**Requisitos:** Windows 10/11, Python 3.11 o posterior, e internet solo la primera vez que se usa cada modelo.

## Cómo funciona

```
Audio (altavoces / micrófono) → Silero VAD → Whisper → Opus-MT → Piper (+ OpenVoice para tu voz)
```

Todo son modelos abiertos que se descargan de Hugging Face y después funcionan sin conexión. El detalle de cada pantalla, los modelos y las GPU admitidas está en el [manual](docs/MANUAL.md).

## Más capturas

| En persona (español ↔ francés) | Árabe |
|---|---|
| ![En persona](docs/capturas/presencial.png) | ![Árabe](docs/capturas/arabe.png) |
| **Mi voz** | **Idiomas** |
| ![Mi voz](docs/capturas/herramientas_mi_voz.png) | ![Idiomas](docs/capturas/herramientas_idiomas.png) |

## Licencia

MIT (ver [LICENSE](LICENSE)). Los componentes de terceros conservan la suya: ver [TERCEROS.md](TERCEROS.md).
