"""
Agrupa ventanas de video en segmentos basados en similitud visual de frames.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .config_loader import FrameGroupingConfig
from .video_window_segmenter import VideoWindow


@dataclass
class FrameCluster:
    cluster_id: int
    start_time: float
    end_time: float
    duration: float
    window_indices: List[int]
    frame_path: str


class FrameClusterer:
    """
    Agrupa consecutivamente las ventanas cuyos frames representativos son similares,
    generando segmentos continuos del video sin depender de las diapositivas del PDF.
    """

    def __init__(
        self,
        config: FrameGroupingConfig,
        assets_dir: Path,
        output_name: str,
        camera_position: int = 0,
    ) -> None:
        self.config = config
        self.assets_dir = assets_dir
        self.output_name = output_name
        self.threshold = config.similarity_threshold
        self.blur_kernel = (
            config.blur_kernel_size
            if config.blur_kernel_size % 2 == 1
            else max(3, config.blur_kernel_size - 1)
        )
        self.cluster_counter = 0
        self.camera_position = camera_position if 0 <= camera_position <= 9 else 0

    def cluster_windows(self, windows: List[VideoWindow]) -> List[FrameCluster]:
        clusters: List[FrameCluster] = []
        current_cluster = None

        for idx, window in enumerate(windows):
            processed_frame = self._preprocess_frame(window.representative_frame)

            if current_cluster is None:
                current_cluster = self._start_cluster(idx, window, processed_frame)
                continue

            diff = self._frame_difference(
                current_cluster["last_processed"], processed_frame
            )

            if diff <= self.threshold:
                self._extend_cluster(current_cluster, idx, window, processed_frame)
            else:
                finalized = self._finalize_cluster(current_cluster)
                if finalized is not None:
                    clusters.append(finalized)
                current_cluster = self._start_cluster(idx, window, processed_frame)

        if current_cluster is not None:
            finalized = self._finalize_cluster(current_cluster)
            if finalized is not None:
                clusters.append(finalized)

        if not clusters:
            print("⚠ No se detectaron segmentos a partir del video")
        return clusters

    def _start_cluster(
        self,
        window_idx: int,
        window: VideoWindow,
        processed_frame: np.ndarray,
    ):
        return {
            "cluster_id": None,
            "start_idx": window_idx,
            "end_idx": window_idx,
            "start_time": window.start_time,
            "end_time": window.end_time,
            "window_indices": [window_idx],
            "last_processed": processed_frame.copy(),
            "frames": [window.representative_frame.copy()],
        }

    def _extend_cluster(
        self,
        cluster,
        window_idx: int,
        window: VideoWindow,
        processed_frame: np.ndarray,
    ):
        cluster["end_idx"] = window_idx
        cluster["end_time"] = window.end_time
        cluster["window_indices"].append(window_idx)

        cluster["last_processed"] = processed_frame.copy()
        cluster["frames"].append(window.representative_frame.copy())

    def _finalize_cluster(self, cluster) -> FrameCluster | None:
        duration = cluster["end_time"] - cluster["start_time"]
        window_count = len(cluster["window_indices"])

        if (
            window_count < self.config.min_windows
            and duration < self.config.min_duration
        ):
            return None

        cluster_id = self.cluster_counter
        self.cluster_counter += 1

        mid_idx = len(cluster["frames"]) // 2
        representative_frame = cluster["frames"][mid_idx]
        frame_path = self._save_representative_frame(cluster_id, representative_frame)

        return FrameCluster(
            cluster_id=cluster_id,
            start_time=cluster["start_time"],
            end_time=cluster["end_time"],
            duration=duration,
            window_indices=cluster["window_indices"],
            frame_path=frame_path,
        )

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.camera_position:
            frame = self._mask_camera_region(frame)

        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )
        if self.blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        height, width = gray.shape[:2]
        if width > self.config.downscale_width:
            new_w = self.config.downscale_width
            new_h = int((new_w / width) * height)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return gray.astype(np.float32) / 255.0

    def _mask_camera_region(self, frame: np.ndarray) -> np.ndarray:
        """
        Enmascara (pone a negro) el cuadrante 3x3 donde está la cámara del profesor.

        position: 1-9 (de izquierda a derecha, de arriba abajo), 0 = sin cámara.
        """
        if self.camera_position == 0:
            return frame

        h, w = frame.shape[:2]
        grid_w = w // 3
        grid_h = h // 3

        row = (self.camera_position - 1) // 3
        col = (self.camera_position - 1) % 3

        x = col * grid_w
        y = row * grid_h

        masked = frame.copy()
        masked[y : y + grid_h, x : x + grid_w] = 0
        return masked

    def _frame_difference(self, ref: np.ndarray, candidate: np.ndarray) -> float:
        if ref.shape != candidate.shape:
            candidate = cv2.resize(candidate, (ref.shape[1], ref.shape[0]))
        diff = np.mean(np.abs(ref - candidate))
        return float(diff)

    def _save_representative_frame(self, cluster_id: int, frame: np.ndarray) -> str:
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.output_name}_segment_{cluster_id:04d}.{self.config.save_frame_format}"
        )
        output_path = self.assets_dir / filename
        cv2.imwrite(str(output_path), frame)
        return str(output_path)

