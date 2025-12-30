"""
Módulo para generar resúmenes usando LLM (GPT, etc.).
"""

import os
from pathlib import Path
from typing import Optional
import base64
from datetime import datetime
import traceback
import certifi

from .config_loader import LLMConfig

# Variables globales para las importaciones lazy
OpenAI = APIError = APIConnectionError = APITimeoutError = None
httpx = None


def _import_openai():
    """
    Importa openai solo cuando se necesite (lazy import).
    Esto evita que se quede colgado durante la importación del módulo.

    Returns:
        bool: True si la importación fue exitosa, False en caso contrario
    """
    global OpenAI, APIError, APIConnectionError, APITimeoutError, httpx
    if OpenAI is None:
        try:
            from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
            import httpx
        except ImportError:
            return False
    return True


class LLMSummarizer:
    """
    Genera resúmenes de diapositivas usando LLM.
    """

    def __init__(self, config: LLMConfig):
        """
        Inicializa el generador de resúmenes.

        Args:
            config: Configuración del LLM
        """
        self.config = config
        self.debug = os.getenv("LLM_DEBUG", "0") == "1"
        self.api_key = self._load_api_key()
        self._initialize_client()

    def _load_api_key(self) -> str:
        """
        Carga la API key desde config o variable de entorno.

        Returns:
            API key

        Raises:
            ValueError: Si la API key no está configurada
        """
        # Primero intentar desde el config (hardcoded)
        if self.config.api_key:
            return self.config.api_key

        # Si no está en config, buscar en variable de entorno
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            # Intentar también con el nombre directo por si acaso
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                f"API key no encontrada. Configura 'api_key' en config.yaml o la variable de entorno "
                f"{self.config.api_key_env} o OPENAI_API_KEY"
            )
        return api_key

    def _initialize_client(self):
        """Inicializa el cliente del LLM."""
        if self.config.provider.lower() == "openai":
            if not _import_openai():
                raise ImportError(
                    "OpenAI no está instalado. Instala con: pip install openai"
                )
            # Configurar httpx para usar certifi explícitamente
            # Si hay problemas de certificados SSL, se puede desactivar temporalmente
            # con la variable de entorno OPENAI_SSL_VERIFY=false (NO recomendado para producción)
            ssl_verify = os.getenv("OPENAI_SSL_VERIFY", "true").lower() != "false"

            if httpx is not None:
                if ssl_verify:
                    # Intentar usar certifi primero
                    verify_cert = certifi.where()
                else:
                    # Desactivar verificación SSL (solo para debugging)
                    print(
                        "⚠ ADVERTENCIA: Verificación SSL desactivada (OPENAI_SSL_VERIFY=false)"
                    )
                    print("  Esto es inseguro y solo debe usarse para debugging")
                    verify_cert = False

                http_client = httpx.Client(verify=verify_cert, timeout=15.0)
                self.client = OpenAI(api_key=self.api_key, http_client=http_client)
            else:
                # Fallback si httpx no está disponible
                self.client = OpenAI(api_key=self.api_key, timeout=15.0)
        else:
            raise ValueError(f"Proveedor LLM no soportado: {self.config.provider}")

    def _image_to_base64(self, image_path: str) -> str:
        """
        Convierte una imagen a base64 para enviarla al LLM.

        Args:
            image_path: Ruta a la imagen

        Returns:
            String base64 de la imagen
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_summary(
        self,
        slide_path: str,
        transcription: str,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> Optional[str]:
        """
        Genera un resumen de la diapositiva y transcripción con contexto opcional.

        Args:
            slide_path: Ruta a la imagen de la diapositiva
            transcription: Transcripción del audio del profesor
            context_before: Transcripciones de diapositivas anteriores (opcional)
            context_after: Transcripciones de diapositivas siguientes (opcional)

        Returns:
            Resumen generado o None si hay error
        """
        if not Path(slide_path).exists():
            print(f"Imagen de diapositiva no encontrada: {slide_path}")
            return None

        try:
            if self.config.provider.lower() == "openai":
                # Leer imagen
                image_base64 = self._image_to_base64(slide_path)

                # Construir el prompt con contexto si está disponible
                prompt_parts = []

                if context_before:
                    prompt_parts.append(
                        f"CONTEXTO DE DIAPOSITIVAS ANTERIORES:\n{context_before}\n\n"
                    )

                prompt_parts.append(
                    f"TRANSCRIPCIÓN DE LA DIAPOSITIVA ACTUAL (puede contener errores de transcripción, "
                    f"especialmente en nombres propios y términos técnicos):\n\n{transcription}\n\n"
                )

                if context_after:
                    prompt_parts.append(
                        f"CONTEXTO DE DIAPOSITIVAS SIGUIENTES:\n{context_after}\n\n"
                    )

                prompt_parts.append(
                    "Analiza la diapositiva y la transcripción. Genera un resumen SINTÉTICO y CONCISO que "
                    "extraiga SOLO los conceptos clave y puntos esenciales. NO hagas un parafraseo literal: "
                    "sintetiza la información de forma que sea útil para estudio rápido. "
                    "Corrige implícitamente cualquier error obvio de transcripción e infiere el significado "
                    "correcto cuando sea necesario, especialmente para nombres propios, términos técnicos y "
                    "conceptos de astrofísica. Usa el contexto proporcionado para entender mejor el flujo de "
                    "la explicación, pero mantén el resumen conciso y enfocado en lo esencial."
                )

                # Preparar mensajes
                messages = [
                    {"role": "system", "content": self.config.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "".join(prompt_parts)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ]

                # Llamar a la API
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                return response.choices[0].message.content
            else:
                raise ValueError(f"Proveedor no soportado: {self.config.provider}")

        except Exception as e:
            err_type = e.__class__.__name__
            timestamp = datetime.now().strftime("%H:%M:%S")
            slide_name = Path(slide_path).name
            print(f"[LLM][{timestamp}] Error generando resumen para {slide_name}")
            print(f"  Tipo: {err_type}")
            print(f"  Mensaje: {str(e)[:200]}")

            if hasattr(e, "status_code"):
                print(f"  Status code: {getattr(e, 'status_code')}")
            if hasattr(e, "request_id"):
                print(f"  Request ID: {getattr(e, 'request_id')}")

            # Importar openai si aún no se ha hecho (para verificar tipos de error)
            _import_openai()
            connection_errors = tuple(
                err for err in (APIConnectionError, APITimeoutError) if err is not None
            )
            if connection_errors and isinstance(e, connection_errors):
                print("  Detalle: error de conexión/timeout con la API de OpenAI.")
                print(
                    "           Verifica tu conexión a Internet o intenta nuevamente en unos segundos."
                )

            if APIError is not None and isinstance(e, APIError):
                response = getattr(e, "response", None)
                if response is not None:
                    print(f"  Respuesta de OpenAI: {response}")

            if self.debug:
                traceback.print_exc()

            return None

    def generate_summaries_batch(
        self, slide_transcription_pairs: list[tuple[str, str]]
    ) -> list[Optional[str]]:
        """
        Genera resúmenes para múltiples diapositivas.

        Args:
            slide_transcription_pairs: Lista de tuplas (ruta_slide, transcripción)

        Returns:
            Lista de resúmenes generados (None si hay error)
        """
        summaries = []
        for slide_path, transcription in slide_transcription_pairs:
            summary = self.generate_summary(slide_path, transcription)
            summaries.append(summary)
        return summaries
