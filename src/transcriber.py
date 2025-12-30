"""
Módulo para transcribir audio a texto.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import subprocess
from .config_loader import TranscriptionConfig


class Transcriber:
    """
    Transcribe segmentos de audio a texto usando Whisper u otros modelos.
    """

    def __init__(self, config: TranscriptionConfig):
        """
        Inicializa el transcriptor.

        Args:
            config: Configuración de transcripción
        """
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        """Inicializa el modelo de transcripción."""
        if self.config.model.lower() == "whisper":
            try:
                import whisper
                import torch

                # Detectar y usar GPU si está disponible (MPS para Mac, CUDA para NVIDIA)
                device = None
                if torch.backends.mps.is_available():
                    # MPS tiene problemas con float64, forzar float32
                    try:
                        # Intentar usar MPS pero con precaución
                        device = "mps"
                        print("  Usando GPU (MPS) para transcripción")
                        # Forzar float32 para evitar problemas con MPS
                        torch.set_default_dtype(torch.float32)
                    except Exception as e:
                        print(f"  Advertencia: Error con MPS ({e}), usando CPU")
                        device = "cpu"
                        torch.set_default_dtype(torch.float32)
                elif torch.cuda.is_available():
                    device = "cuda"
                    print("  Usando GPU (CUDA) para transcripción")
                else:
                    device = "cpu"
                    print("  Usando CPU para transcripción")

                # Cargar modelo en el device apropiado
                self.model = whisper.load_model("base", device=device)
                self.device = device
            except ImportError:
                raise ImportError(
                    "Whisper no está instalado. Instala con: pip install openai-whisper"
                )
        else:
            raise ValueError(
                f"Modelo de transcripción no soportado: {self.config.model}"
            )

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe un archivo de audio a texto.

        Args:
            audio_path: Ruta al archivo de audio

        Returns:
            Texto transcrito o None si hay error
        """
        if not Path(audio_path).exists():
            print(f"Archivo de audio no encontrado: {audio_path}")
            return None

        try:
            if self.config.model.lower() == "whisper":
                result = self.model.transcribe(
                    audio_path, language=self.config.language
                )
                return result["text"]
            else:
                raise ValueError(f"Modelo no soportado: {self.config.model}")
        except Exception as e:
            print(f"Error transcribiendo {audio_path}: {e}")
            return None

    def transcribe_batch(self, audio_paths: list[str]) -> list[Optional[str]]:
        """
        Transcribe múltiples archivos de audio.

        Args:
            audio_paths: Lista de rutas a archivos de audio

        Returns:
            Lista de textos transcritos (None si hay error)
        """
        return [self.transcribe(path) for path in audio_paths]

    def transcribe_video_with_timestamps(
        self,
        video_path: str,
        output_audio_path: Optional[str] = None,
        max_duration: Optional[float] = None,
    ) -> Optional[List[Tuple[float, float, str]]]:
        """
        Transcribe un video completo y devuelve la transcripción con timestamps.

        Args:
            video_path: Ruta al archivo de video
            output_audio_path: Ruta opcional donde guardar el audio extraído

        Returns:
            Lista de tuplas (start_time, end_time, text) o None si hay error
        """
        # Extraer audio del video si es necesario
        audio_path = output_audio_path
        if audio_path is None:
            audio_path = str(Path(video_path).with_suffix(".wav"))

        print(
            "Extrayendo audio del video..."
            + (" (segmento parcial)" if max_duration else "")
        )
        self._extract_audio_from_video(
            video_path, audio_path, max_duration=max_duration
        )

        if not Path(audio_path).exists():
            print(f"Error: No se pudo extraer el audio del video")
            return None

        try:
            if self.config.model.lower() == "whisper":
                print(f"Transcribiendo video completo (esto puede tardar)...")

                # Si hay error con MPS, intentar con CPU
                try:
                    result = self.model.transcribe(
                        audio_path,
                        language=self.config.language,
                        word_timestamps=True,  # Necesario para obtener timestamps
                    )
                except Exception as mps_error:
                    if "MPS" in str(mps_error) or "float64" in str(mps_error):
                        print(f"  Error con MPS, reintentando con CPU...")
                        import whisper
                        import torch

                        # Cargar modelo en CPU
                        cpu_model = whisper.load_model("base", device="cpu")
                        result = cpu_model.transcribe(
                            audio_path,
                            language=self.config.language,
                            word_timestamps=True,
                        )
                    else:
                        raise

                # Convertir a lista de segmentos con timestamps
                segments = []
                for segment in result.get("segments", []):
                    segments.append(
                        (segment["start"], segment["end"], segment["text"].strip())
                    )

                return segments
            else:
                raise ValueError(f"Modelo no soportado: {self.config.model}")
        except Exception as e:
            print(f"Error transcribiendo {video_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _extract_audio_from_video(
        self, video_path: str, output_path: str, max_duration: Optional[float] = None
    ) -> bool:
        """
        Extrae el audio de un video usando ffmpeg.

        Args:
            video_path: Ruta al video
            output_path: Ruta donde guardar el audio

        Returns:
            True si se extrajo correctamente, False en caso contrario
        """
        import shutil

        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            # Intentar ubicaciones comunes en macOS
            common_paths = [
                "/opt/homebrew/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
                "/usr/bin/ffmpeg",
            ]
            for path in common_paths:
                if Path(path).exists():
                    ffmpeg_cmd = path
                    break

        if not ffmpeg_cmd:
            print("Error: ffmpeg no encontrado. Instala con: brew install ffmpeg")
            return False

        cmd = [
            ffmpeg_cmd,
            "-i",
            video_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
        ]

        if max_duration:
            cmd.extend(["-t", f"{max_duration:.2f}"])

        cmd.extend(["-y", output_path])

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extrayendo audio: {e}")
            return False

    def split_transcription_by_intervals(
        self,
        transcription_segments: List[Tuple[float, float, str]],
        intervals: List[Tuple[float, float]],
    ) -> List[str]:
        """
        Divide una transcripción con timestamps según intervalos de tiempo.
        Evita duplicaciones asignando cada segmento a la ventana más apropiada.

        Args:
            transcription_segments: Lista de (start, end, text) de la transcripción
            intervals: Lista de (start_time, end_time) de los intervalos

        Returns:
            Lista de textos transcritos, uno por cada intervalo
        """
        result = [""] * len(intervals)

        for seg_start, seg_end, text in transcription_segments:
            # Encontrar la ventana cuyo centro está más cerca del centro del segmento
            seg_center = (seg_start + seg_end) / 2
            best_window_idx = None
            min_distance = float("inf")

            for i, (interval_start, interval_end) in enumerate(intervals):
                # Verificar si hay solapamiento
                if not (seg_end < interval_start or seg_start > interval_end):
                    interval_center = (interval_start + interval_end) / 2
                    distance = abs(seg_center - interval_center)

                    if distance < min_distance:
                        min_distance = distance
                        best_window_idx = i

            # Asignar el segmento a la mejor ventana
            if best_window_idx is not None:
                if result[best_window_idx]:
                    result[best_window_idx] += " " + text
                else:
                    result[best_window_idx] = text

        return result
