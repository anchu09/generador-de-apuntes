"""
Módulo para dividir el video en ventanas de tiempo fijas y extraer frames.
"""

import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class VideoWindow:
    """Representa una ventana de tiempo del video."""

    window_number: int
    start_time: float
    end_time: float
    duration: float
    representative_frame: np.ndarray  # Frame promedio o representativo
    audio_segment_path: Optional[str] = None
    transcription: Optional[str] = None
    matched_slide_index: Optional[int] = None  # Índice de la diapositiva más similar
    matched_similarity: Optional[float] = None  # Similitud asociada


class VideoWindowSegmenter:
    """
    Divide el video en ventanas de tiempo fijas y extrae frames representativos
    y segmentos de audio.
    """

    def __init__(
        self,
        window_duration: float = 5.0,
        frames_to_average: int = 3,
        output_dir: Path = None,
    ):
        """
        Inicializa el segmentador de ventanas.

        Args:
            window_duration: Duración de cada ventana en segundos
            frames_to_average: Número de frames a promediar para robustez (con GIFs)
            output_dir: Directorio donde guardar segmentos de audio
        """
        self.window_duration = window_duration
        self.frames_to_average = frames_to_average
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def segment_video(
        self,
        video_path: str,
        video_name: Optional[str] = None,
        duration_fraction: Optional[float] = None,
    ) -> List[VideoWindow]:
        """
        Divide el video en ventanas de tiempo fijas.

        Args:
            video_path: Ruta al archivo de video
            video_name: Nombre del video (para organizar archivos)

        Returns:
            Lista de VideoWindow con frames representativos y rutas de audio
        """
        if video_name is None:
            video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps

        effective_duration = total_duration
        if duration_fraction is not None:
            duration_fraction = max(0.0, min(duration_fraction, 1.0))
            if duration_fraction > 0 and duration_fraction < 1.0:
                effective_duration = total_duration * duration_fraction
                print(
                    f"Procesando solo el {duration_fraction*100:.1f}% del video "
                    f"({effective_duration:.2f}s de {total_duration:.2f}s)"
                )
            elif duration_fraction <= 0.0:
                print(
                    "Advertencia: duration_fraction <= 0. Se procesará todo el video."
                )

        print(f"Procesando video: {video_path}")
        print(f"FPS: {fps}, Duración total: {total_duration:.2f}s")
        print(
            f"Ventanas de {self.window_duration}s, promediando {self.frames_to_average} frames"
        )

        windows = []
        window_number = 0
        current_time = 0.0

        while current_time < effective_duration:
            end_time = min(current_time + self.window_duration, effective_duration)
            duration = end_time - current_time

            # Extraer frame representativo (promedio de múltiples frames)
            representative_frame = self._extract_representative_frame(
                cap, fps, current_time, duration
            )

            window = VideoWindow(
                window_number=window_number,
                start_time=current_time,
                end_time=end_time,
                duration=duration,
                representative_frame=representative_frame,
                audio_segment_path=None,  # Ya no extraemos audio por ventanas
            )

            windows.append(window)
            window_number += 1
            current_time = end_time

            # Mostrar progreso
            if window_number % 10 == 0:
                progress = (current_time / total_duration) * 100
                print(f"Progreso: {progress:.1f}% - Ventanas creadas: {len(windows)}")

        cap.release()
        print(f"✓ {len(windows)} ventanas creadas")
        return windows

    def _extract_representative_frame(
        self, cap: cv2.VideoCapture, fps: float, start_time: float, duration: float
    ) -> np.ndarray:
        """
        Extrae un frame representativo optimizado para animaciones graduales.
        Estrategia: detecta el frame más completo (con más contenido visible),
        priorizando frames más tardíos que suelen tener más contenido en animaciones.

        Args:
            cap: VideoCapture abierto
            fps: Frames por segundo del video
            start_time: Tiempo de inicio de la ventana
            duration: Duración de la ventana

        Returns:
            Frame representativo (frame más completo o promedio de los mejores)
        """
        # Calcular frames a evaluar
        # Distribuir los frames a lo largo de la ventana, con más muestras hacia el final
        frame_indices = []
        for i in range(self.frames_to_average):
            # Distribución sesgada hacia el final (útil para animaciones)
            # Usar distribución cuadrática para dar más peso a frames tardíos
            progress = (i + 1) / (self.frames_to_average + 1)
            # Cuadrático: frames finales tienen más peso
            progress_squared = progress**0.7  # 0.7 da buen balance
            offset = duration * progress_squared
            frame_time = start_time + offset
            frame_number = int(frame_time * fps)
            frame_indices.append(frame_number)

        # Asegurar que siempre incluimos el frame final (más probable que tenga más contenido)
        end_frame = int((start_time + duration) * fps)
        if end_frame not in frame_indices:
            frame_indices.append(end_frame)

        # Extraer todos los frames
        frames_with_info = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Calcular "completitud" del frame
                completeness_score = self._calculate_frame_completeness(frame)
                # Bonus para frames más tardíos (útil en animaciones)
                time_position = (frame_idx - int(start_time * fps)) / max(
                    1, int(duration * fps)
                )
                time_bonus = time_position * 0.1  # Bonus del 10% para frames finales
                total_score = completeness_score + time_bonus

                frames_with_info.append(
                    {
                        "frame": frame,
                        "score": total_score,
                        "time_position": time_position,
                        "frame_idx": frame_idx,
                    }
                )

        if not frames_with_info:
            # Fallback: usar el frame del medio
            middle_frame = int((start_time + duration / 2) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            ret, frame = cap.read()
            if ret:
                return frame
            # Último recurso: primer frame disponible
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = cap.read()
            return frame

        # Ordenar por score (mayor a menor)
        frames_with_info.sort(key=lambda x: x["score"], reverse=True)

        # Estrategia híbrida: usar el frame más completo, pero promediar los top 2-3
        # para robustez contra ruido
        top_frames = frames_with_info[: min(3, len(frames_with_info))]

        if len(top_frames) == 1:
            # Solo un frame disponible
            return top_frames[0]["frame"]
        elif len(top_frames) == 2:
            # Promediar los 2 mejores
            frames_array = np.array([f["frame"] for f in top_frames], dtype=np.float32)
            averaged_frame = np.mean(frames_array, axis=0).astype(np.uint8)
            return averaged_frame
        else:
            # Promediar los 3 mejores con pesos (más peso al mejor)
            weights = [0.5, 0.3, 0.2]  # Peso decreciente
            frames_array = np.array([f["frame"] for f in top_frames], dtype=np.float32)
            # Promedio ponderado
            weighted_avg = np.zeros_like(frames_array[0], dtype=np.float32)
            for i, frame in enumerate(frames_array):
                weighted_avg += frame * weights[i]
            averaged_frame = weighted_avg.astype(np.uint8)
            return averaged_frame

    def _calculate_frame_completeness(self, frame: np.ndarray) -> float:
        """
        Calcula un score de "completitud" del frame basado en:
        - Cantidad de contenido visible (píxeles no blancos/vacíos)
        - Densidad de bordes (más bordes = más contenido)
        - Variación de píxeles (más variación = más contenido)

        Args:
            frame: Frame a evaluar

        Returns:
            Score de completitud (0-1, donde 1 es más completo)
        """
        # Convertir a escala de grises si es necesario
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        scores = []

        # 1. Contenido visible: porcentaje de píxeles no blancos/vacíos
        # Considerar píxeles "vacíos" como aquellos muy cercanos al blanco o negro puro
        # (dependiendo del fondo típico de presentaciones)
        non_white_mask = (gray < 250) & (gray > 5)  # Excluir casi blanco y casi negro
        content_ratio = np.sum(non_white_mask) / gray.size
        scores.append(content_ratio)

        # 2. Densidad de bordes: más bordes = más estructura = más contenido
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        scores.append(edge_density)

        # 3. Variación de píxeles: desviación estándar (más variación = más contenido)
        pixel_variance = np.std(gray.astype(np.float32))
        # Normalizar a 0-1 (asumiendo que 50 es una buena desviación estándar para contenido)
        variance_score = min(1.0, pixel_variance / 50.0)
        scores.append(variance_score)

        # 4. Contraste local: usar Laplacian para detectar variación local
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        local_variation = np.var(laplacian)
        # Normalizar
        local_variation_score = min(1.0, local_variation / 1000.0)
        scores.append(local_variation_score)

        # Promediar todos los scores
        completeness = np.mean(scores)
        return completeness
