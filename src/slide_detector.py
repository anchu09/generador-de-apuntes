"""
Módulo para detectar cambios de diapositiva en el video.
Utiliza múltiples técnicas de detección con probabilidades ponderadas.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from .config_loader import SlideDetectionConfig, CameraConfig


@dataclass
class SlideChange:
    """Representa un cambio de diapositiva detectado."""
    frame_number: int
    timestamp: float
    probability: float
    previous_slide: Optional[np.ndarray] = None
    current_slide: Optional[np.ndarray] = None


class SlideDetector:
    """
    Detecta cambios de diapositiva usando múltiples técnicas:
    - Diferencia de píxeles
    - Detección de bordes
    - Cambios estructurales (disposición de objetos)
    """

    def __init__(
        self,
        slide_config: SlideDetectionConfig,
        camera_config: CameraConfig
    ):
        """
        Inicializa el detector de diapositivas.

        Args:
            slide_config: Configuración de detección de diapositivas
            camera_config: Configuración de la cámara del profesor
        """
        self.slide_config = slide_config
        self.camera_config = camera_config
        self.previous_frame = None
        self.camera_region = None
        self.reference_size = None  # Tamaño de referencia para los frames

    def _get_camera_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Calcula la región de la cámara del profesor si está configurada.

        Args:
            frame: Frame del video

        Returns:
            Tupla (x, y, width, height) de la región de la cámara, o None
        """
        if self.camera_config.position == 0:
            return None

        h, w = frame.shape[:2]

        # Dividir en grid 3x3
        grid_w = w // 3
        grid_h = h // 3

        # Calcular posición en grid (1-9)
        row = (self.camera_config.position - 1) // 3
        col = (self.camera_config.position - 1) % 3

        x = col * grid_w
        y = row * grid_h

        return (x, y, grid_w, grid_h)

    def _mask_camera_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Enmascara la región de la cámara del profesor para que no cuente
        en la detección de cambios.

        Args:
            frame: Frame del video

        Returns:
            Frame con la región de la cámara enmascarada
        """
        if self.camera_region is None:
            self.camera_region = self._get_camera_region(frame)

        if self.camera_region is None:
            return frame

        masked_frame = frame.copy()
        x, y, w, h = self.camera_region
        masked_frame[y:y+h, x:x+w] = 0

        return masked_frame

    def _remove_static_margins(self, frame: np.ndarray) -> np.ndarray:
        """
        Elimina márgenes negros estáticos del frame.
        Si se eliminan márgenes, redimensiona al tamaño de referencia.

        Args:
            frame: Frame del video

        Returns:
            Frame sin márgenes estáticos (o redimensionado si es necesario)
        """
        if not self.camera_config.ignore_margins:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Detectar márgenes negros
        threshold = int(255 * self.camera_config.margin_tolerance)
        rows = np.where(np.max(gray, axis=1) > threshold)[0]
        cols = np.where(np.max(gray, axis=0) > threshold)[0]

        if len(rows) > 0 and len(cols) > 0:
            cropped = frame[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

            # Si hay un tamaño de referencia, redimensionar al tamaño original
            if self.reference_size is not None:
                h_ref, w_ref = self.reference_size
                return cv2.resize(cropped, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

            return cropped

        return frame

    def _normalize_frame_size(self, frame: np.ndarray) -> np.ndarray:
        """
        Normaliza el tamaño del frame al tamaño de referencia.

        Args:
            frame: Frame a normalizar

        Returns:
            Frame redimensionado al tamaño de referencia
        """
        if self.reference_size is None:
            return frame

        h_ref, w_ref = self.reference_size
        h_curr, w_curr = frame.shape[:2]

        if (h_curr, w_curr) != (h_ref, w_ref):
            return cv2.resize(frame, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

        return frame

    def _calculate_pixel_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calcula la diferencia de píxeles entre dos frames.

        Args:
            frame1: Frame anterior
            frame2: Frame actual

        Returns:
            Probabilidad de cambio basada en diferencia de píxeles (0-1)
        """
        # Asegurar que ambos frames tienen el mismo tamaño
        if frame1.shape[:2] != frame2.shape[:2]:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Convertir a escala de grises si es necesario
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calcular diferencia absoluta
        diff = cv2.absdiff(frame1, frame2)

        # Calcular porcentaje de píxeles diferentes
        total_pixels = frame1.size
        changed_pixels = np.count_nonzero(diff > 30)  # Threshold de diferencia

        change_ratio = changed_pixels / total_pixels

        # Normalizar según threshold
        if change_ratio >= self.slide_config.pixel_diff_threshold:
            return min(1.0, change_ratio / self.slide_config.pixel_diff_threshold)

        return change_ratio / self.slide_config.pixel_diff_threshold

    def _calculate_edge_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calcula la diferencia en la estructura de bordes entre dos frames.
        Útil para detectar cambios estructurales incluso con animaciones.

        Args:
            frame1: Frame anterior
            frame2: Frame actual

        Returns:
            Probabilidad de cambio basada en bordes (0-1)
        """
        # Asegurar que ambos frames tienen el mismo tamaño
        if frame1.shape[:2] != frame2.shape[:2]:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Convertir a escala de grises
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        # Detectar bordes con Canny
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        # Calcular diferencia
        diff = cv2.absdiff(edges1, edges2)
        changed_edges = np.count_nonzero(diff)
        total_edges = np.count_nonzero(edges1) + np.count_nonzero(edges2)

        if total_edges == 0:
            return 0.0

        change_ratio = changed_edges / total_edges

        # Normalizar
        if change_ratio >= self.slide_config.structural_change_threshold:
            return min(1.0, change_ratio / self.slide_config.structural_change_threshold)

        return change_ratio / self.slide_config.structural_change_threshold

    def _calculate_structural_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calcula la diferencia estructural (disposición de objetos) entre frames.
        Usa detección de características para identificar cambios en la disposición.

        Args:
            frame1: Frame anterior
            frame2: Frame actual

        Returns:
            Probabilidad de cambio estructural (0-1)
        """
        # Asegurar que ambos frames tienen el mismo tamaño
        if frame1.shape[:2] != frame2.shape[:2]:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Convertir a escala de grises
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2

        # Detectar características con ORB
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return 0.0

        # Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Calcular ratio de correspondencias
        total_features = min(len(des1), len(des2))
        if total_features == 0:
            return 1.0  # Sin características = cambio probable

        match_ratio = len(matches) / total_features
        structural_diff = 1.0 - match_ratio

        # Normalizar
        if structural_diff >= self.slide_config.structural_change_threshold:
            return min(1.0, structural_diff / self.slide_config.structural_change_threshold)

        return structural_diff / self.slide_config.structural_change_threshold

    def detect_slide_change(
        self,
        current_frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> Optional[SlideChange]:
        """
        Detecta si ha habido un cambio de diapositiva.

        Args:
            current_frame: Frame actual del video
            frame_number: Número de frame
            timestamp: Timestamp del frame en segundos

        Returns:
            SlideChange si se detecta un cambio, None en caso contrario
        """
        # Preprocesar frame
        processed_frame = self._remove_static_margins(current_frame)
        processed_frame = self._mask_camera_region(processed_frame)

        # Establecer tamaño de referencia en el primer frame
        if self.reference_size is None:
            self.reference_size = processed_frame.shape[:2]

        # Normalizar tamaño del frame procesado
        processed_frame = self._normalize_frame_size(processed_frame)

        # Si no hay frame anterior, guardar este y continuar
        if self.previous_frame is None:
            self.previous_frame = processed_frame.copy()
            return None

        # Asegurar que el frame anterior también está normalizado
        self.previous_frame = self._normalize_frame_size(self.previous_frame)

        # Calcular probabilidades con diferentes técnicas
        pixel_prob = self._calculate_pixel_diff(self.previous_frame, processed_frame)
        edge_prob = self._calculate_edge_diff(self.previous_frame, processed_frame)
        structural_prob = self._calculate_structural_diff(self.previous_frame, processed_frame)

        # Calcular probabilidad ponderada
        overall_prob = (
            pixel_prob * self.slide_config.pixel_diff_weight +
            edge_prob * self.slide_config.edge_detection_weight +
            structural_prob * self.slide_config.structural_weight
        )

        # Verificar si supera el threshold
        if overall_prob >= self.slide_config.overall_change_threshold:
            change = SlideChange(
                frame_number=frame_number,
                timestamp=timestamp,
                probability=overall_prob,
                previous_slide=self.previous_frame.copy(),
                current_slide=processed_frame.copy()
            )
            self.previous_frame = processed_frame.copy()
            return change

        self.previous_frame = processed_frame.copy()
        return None

