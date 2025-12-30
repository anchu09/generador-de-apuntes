"""
Módulo para procesar video y extraer frames, audio y segmentos.
"""

import cv2
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
from .config_loader import ExtractionConfig, SlideDetectionConfig
from .slide_detector import SlideDetector, SlideChange


@dataclass
class SlideSegment:
    """Representa un segmento de diapositiva con su información."""
    slide_number: int
    start_time: float
    end_time: float
    duration: float
    frame_path: str
    audio_segment_path: Optional[str] = None
    transcription: Optional[str] = None
    summary: Optional[str] = None
    num_windows: int = 0
    metadata: dict = field(default_factory=dict)


class VideoProcessor:
    """
    Procesa videos para extraer diapositivas, segmentos de audio y generar
    la tabla de segmentos.
    """

    def __init__(
        self,
        extraction_config: ExtractionConfig,
        slide_detector: SlideDetector
    ):
        """
        Inicializa el procesador de video.

        Args:
            extraction_config: Configuración de extracción
            slide_detector: Detector de cambios de diapositiva
        """
        self.config = extraction_config
        self.slide_detector = slide_detector

        # Crear directorios de salida
        self.output_dir = Path(self.config.output_dir)
        self.assets_dir = self.output_dir / self.config.assets_dir
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def process_video(
        self,
        video_path: str,
        video_name: Optional[str] = None
    ) -> List[SlideSegment]:
        """
        Procesa un video completo y extrae todas las diapositivas.

        Args:
            video_path: Ruta al archivo de video
            video_name: Nombre del video (para organizar archivos)

        Returns:
            Lista de SlideSegment con todas las diapositivas detectadas
        """
        if video_name is None:
            video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        segments: List[SlideSegment] = []
        current_slide_start = 0.0
        current_slide_frame = None
        slide_number = 0
        last_change_time = 0.0

        frame_number = 0

        print(f"Procesando video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_number / fps

            # Detectar cambio de diapositiva
            change = self.slide_detector.detect_slide_change(frame, frame_number, timestamp)

            if change is not None:
                # Si la diapositiva anterior duró lo suficiente, guardarla
                slide_duration = timestamp - last_change_time

                if slide_duration >= self.slide_detector.slide_config.min_slide_duration:
                    if current_slide_frame is not None:
                        # Guardar diapositiva
                        slide_path = self._save_slide(
                            current_slide_frame,
                            video_name,
                            slide_number
                        )

                        segment = SlideSegment(
                            slide_number=slide_number,
                            start_time=current_slide_start,
                            end_time=last_change_time,
                            duration=slide_duration,
                            frame_path=str(slide_path)
                        )
                        segments.append(segment)
                        slide_number += 1

                # Actualizar para nueva diapositiva
                current_slide_start = timestamp
                current_slide_frame = change.current_slide
                last_change_time = timestamp

            frame_number += 1

            # Mostrar progreso
            if frame_number % 100 == 0:
                progress = (frame_number / total_frames) * 100
                print(f"Progreso: {progress:.1f}% - Diapositivas detectadas: {len(segments)}")

        # Guardar última diapositiva si duró lo suficiente
        final_timestamp = total_frames / fps
        final_duration = final_timestamp - last_change_time

        if final_duration >= self.slide_detector.slide_config.min_slide_duration:
            if current_slide_frame is not None:
                slide_path = self._save_slide(
                    current_slide_frame,
                    video_name,
                    slide_number
                )

                segment = SlideSegment(
                    slide_number=slide_number,
                    start_time=current_slide_start,
                    end_time=final_timestamp,
                    duration=final_duration,
                    frame_path=str(slide_path)
                )
                segments.append(segment)

        cap.release()

        print(f"Procesamiento completado. Total diapositivas: {len(segments)}")
        return segments

    def _save_slide(
        self,
        frame: np.ndarray,
        video_name: str,
        slide_number: int
    ) -> Path:
        """
        Guarda una diapositiva como imagen.

        Args:
            frame: Frame a guardar
            video_name: Nombre del video
            slide_number: Número de diapositiva

        Returns:
            Ruta al archivo guardado
        """
        filename = f"{video_name}_slide_{slide_number:04d}.{self.config.image_format}"
        filepath = self.assets_dir / filename

        cv2.imwrite(str(filepath), frame)
        return filepath

    def extract_audio_segments(
        self,
        video_path: str,
        segments: List[SlideSegment],
        video_name: Optional[str] = None
    ) -> List[SlideSegment]:
        """
        Extrae segmentos de audio para cada diapositiva.

        Args:
            video_path: Ruta al archivo de video
            segments: Lista de segmentos de diapositivas
            video_name: Nombre del video

        Returns:
            Lista de segmentos con rutas de audio actualizadas
        """
        if video_name is None:
            video_name = Path(video_path).stem

        # Usar ffmpeg para extraer segmentos de audio
        updated_segments = []

        for segment in segments:
            audio_path = self.assets_dir / f"{video_name}_slide_{segment.slide_number:04d}.wav"

            # Comando ffmpeg para extraer segmento de audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(segment.start_time),
                '-t', str(segment.duration),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',  # Sobrescribir si existe
                str(audio_path)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                segment.audio_segment_path = str(audio_path)
            except subprocess.CalledProcessError as e:
                print(f"Error extrayendo audio para slide {segment.slide_number}: {e}")
                segment.audio_segment_path = None

            updated_segments.append(segment)

        return updated_segments

