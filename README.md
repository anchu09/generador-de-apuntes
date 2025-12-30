# Generador de Apuntes Automático

Sistema automatizado para generar apuntes a partir de grabaciones de clases online. Extrae diapositivas, transcribe el audio del profesor y genera resúmenes usando IA.

## Características

- **Detección inteligente de diapositivas**: Usa múltiples técnicas (diferencia de píxeles, detección de bordes, cambios estructurales) para detectar cambios de diapositiva, incluso con animaciones y GIFs
- **Filtrado de cámara del profesor**: Ignora la región de la cámara del profesor para evitar falsos positivos
- **Transcripción automática**: Transcribe el audio del profesor para cada diapositiva usando Whisper
- **Resúmenes con IA**: Genera resúmenes claros y concisos usando GPT-4o
- **Generación de PowerPoint**: Crea presentaciones con diapositivas originales, transcripciones y resúmenes

## Instalación

### Requisitos previos

- Python 3.8 o superior
- ffmpeg instalado en el sistema
  - macOS: `brew install ffmpeg`

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Configuración

1. Ajusta los parámetros en `config.yaml` según tus necesidades
2. Configura tu API key de OpenAI como variable de entorno:
   ```bash
   export OPENAI_API_KEY="tu-api-key-aqui"
   ```

## Uso

### Uso básico

```bash
python main.py ruta/al/video.mp4
```

### Opciones

```bash
python main.py video.mp4 -c config.yaml -o nombre_salida
```

- `-c, --config`: Ruta al archivo de configuración (default: `config.yaml`)
- `-o, --output`: Nombre para los archivos de salida (default: nombre del video)

## Estructura del Proyecto

```
generador-apuntes/
├── config.yaml              # Configuración del proyecto
├── main.py                  # Punto de entrada principal
├── requirements.txt         # Dependencias Python
├── README.md               # Este archivo
├── src/                    # Código fuente
│   ├── __init__.py
│   ├── config_loader.py    # Carga y validación de configuración
│   ├── slide_detector.py   # Detección de cambios de diapositiva
│   ├── video_processor.py  # Procesamiento de video y extracción
│   ├── transcriber.py      # Transcripción de audio
│   ├── llm_summarizer.py   # Generación de resúmenes con LLM
│   ├── powerpoint_generator.py  # Generación de PowerPoint
│   └── main.py             # Orquestación del proceso
├── output/                 # Archivos generados (se crea automáticamente)
│   └── assets/            # Diapositivas y segmentos de audio extraídos
└── .gitignore
```

## Configuración

El archivo `config.yaml` contiene todos los parámetros configurables:

### Detección de diapositivas

- `min_slide_duration`: Tiempo mínimo (segundos) que debe estar una diapositiva
- `pixel_diff_threshold`: Umbral de diferencia de píxeles
- `overall_change_threshold`: Umbral total de probabilidad de cambio

### Cámara del profesor

- `position`: Posición de la cámara (0-9, donde 0 = no hay cámara)
  - Grid 3x3: 1=esquina superior izquierda, 5=centro, 9=esquina inferior derecha
- `ignore_margins`: Ignorar márgenes negros estáticos

### LLM

- `provider`: Proveedor (openai, etc.)
- `model`: Modelo a usar (gpt-4o, etc.)
- `api_key_env`: Variable de entorno con la API key

## Proceso

1. **Detección de diapositivas**: Analiza el video frame por frame para detectar cambios
2. **Extracción**: Guarda cada diapositiva como imagen y extrae segmentos de audio
3. **Transcripción**: Transcribe el audio de cada segmento
4. **Resúmenes**: Genera resúmenes usando la diapositiva y la transcripción
5. **Generación**: Crea una tabla CSV y una presentación PowerPoint

## Salida

El proceso genera:

- **Tabla CSV**: Contiene toda la información de cada diapositiva (tiempos, rutas, transcripciones, resúmenes)
- **Presentación PowerPoint**: Diapositivas con:
  - Transcripción del profesor a la izquierda
  - Diapositiva original en el centro
  - Resumen generado a la derecha
- **Assets**: Carpeta con todas las diapositivas y segmentos de audio extraídos

## Notas

- El procesamiento puede tardar dependiendo de la duración del video
- Se requiere conexión a internet para usar la API de OpenAI
- Los costos de la API dependen del modelo y la cantidad de diapositivas

## Desarrollo

Desarrollado con Python 3.8+ y diseñado para macOS con Cursor IDE.

## Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo y modificarlo según tus necesidades.

