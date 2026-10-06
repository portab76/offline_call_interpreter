# Componentes de terceros

El código de este programa es MIT (ver `LICENSE`). Usa componentes de otros, cada uno con su licencia.

## Incluidos en este repositorio

| Componente | Dónde | Licencia |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (y ggml), compilado con Vulkan | `motor_gpu\*.exe`, `motor_gpu\ggml*.dll`, `motor_gpu\whisper.dll` | MIT (`motor_gpu\LICENSE-whisper.cpp.txt`) |
| Microsoft Visual C++ Runtime | `motor_gpu\msvcp140.dll`, `vcruntime140*.dll`, `vcomp140.dll` | Redistribuible de Visual C++ (licencia de Microsoft) |
| [OpenVoice](https://github.com/myshell-ai/OpenVoice) (solo las piezas del conversor de timbre) | `openvoice_lite\` | MIT, © MyShell.ai (`openvoice_lite\LICENSE`) |

## Descargados al usarlos (no incluidos)

Se bajan de Hugging Face a `modelos\` la primera vez que hacen falta.

| Modelo | Origen | Licencia |
|---|---|---|
| Whisper (reconocimiento de voz), de OpenAI | `ggerganov/whisper.cpp` (GPU), `Systran/faster-whisper-*` (CPU) | MIT |
| Silero VAD (detección de voz) | incluido en el paquete `faster-whisper` | MIT |
| Opus-MT (traducción), de la Universidad de Helsinki | conversiones a CTranslate2 de `michaelfeil/ct2fast-opus-mt-*` y `ooeoeo/opus-mt-*` | CC-BY 4.0 |
| Voces de Piper (texto a voz) | `rhasspy/piper-voices` | Según cada voz: ver su `MODEL_CARD` en el repositorio de voces |
| OpenVoice V2 (conversor de timbre de "Mi voz") | `myshell-ai/OpenVoiceV2` | MIT |

## Paquetes de Python

Los instala `1_instalar.bat` desde PyPI (ver `requirements.txt`), cada uno con su licencia. PyTorch (BSD-3) solo se instala si se usa "Mi voz".

## No incluido

[VB-Audio Virtual Cable](https://vb-audio.com/Cable/), para que el otro oiga tu voz traducida en las llamadas, se descarga e instala aparte desde su web.
