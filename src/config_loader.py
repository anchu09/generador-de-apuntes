"""
Módulo para cargar y validar la configuración del proyecto.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SlideDetectionConfig:
    """Configuración para la detección de cambios de diapositiva."""

    min_slide_duration: float
    pixel_diff_threshold: float
    structural_change_threshold: float
    edge_detection_weight: float
    pixel_diff_weight: float
    structural_weight: float
    overall_change_threshold: float


@dataclass
class CameraConfig:
    """Configuración de la cámara del profesor."""

    position: int  # 0-9, donde 0 = no hay cámara
    ignore_margins: bool
    margin_tolerance: float


@dataclass
class ExtractionConfig:
    """Configuración de extracción de frames y audio."""

    output_dir: str
    assets_dir: str
    frames_per_second: int
    image_format: str


@dataclass
class TranscriptionConfig:
    """Configuración de transcripción de audio."""

    language: str
    model: str
    output_format: str


@dataclass
class LLMConfig:
    """Configuración del LLM para generación de resúmenes."""

    provider: str
    model: str
    api_key_env: str
    temperature: float
    max_tokens: int
    context_slides_before: int = 2
    context_slides_after: int = 1
    system_prompt: str = ""
    api_key: Optional[str] = (
        None  # API key directa (opcional, si no se usa variable de entorno)
    )


@dataclass
class PowerPointConfig:
    """Configuración de generación de PowerPoint."""

    slide_width: int
    slide_height: int
    output_format: str
    font_size_transcription: int
    font_size_summary: int
    margin: int
    auto_font_size: bool = True
    min_font_size: int = 8
    max_font_size: int = 16
    paragraph_spacing: int = 8
    line_spacing: float = 1.2


@dataclass
class VideoConfig:
    """Configuración de procesamiento de video."""

    codec: str
    quality: str


@dataclass
class VideoWindowsConfig:
    """Configuración de ventanas de video."""

    window_duration: float
    frames_to_average: int


@dataclass
class ProcessingConfig:
    """Configuración del modo de procesamiento."""

    mode: str = "pdf_matching"
    test_fraction: float = (
        1.0  # Permite procesar solo una fracción del video para pruebas
    )


@dataclass
class MicroMergeConfig:
    """Configuración para unir micro segmentos en modo video_only."""

    enabled: bool = False
    window_threshold: int = 6
    max_total_windows: int = 24
    max_total_duration: float = 90.0
    min_segments: int = 2
    collage_max_columns: int = 3
    collage_tile_width: int = 640
    collage_tile_height: int = 360
    collage_padding: int = 8
    collage_background_color: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class FrameGroupingConfig:
    """Configuración para agrupar frames similares."""

    similarity_threshold: float
    downscale_width: int
    blur_kernel_size: int
    min_windows: int
    min_duration: float
    save_frame_format: str = "jpg"


@dataclass
class SlideMatchingConfig:
    """Configuración de matching de diapositivas."""

    use_histogram: bool
    use_structural: bool
    use_features: bool
    histogram_weight: float
    structural_weight: float
    features_weight: float
    min_similarity_threshold: float = 0.3
    # Configuración de matching robusto
    use_temporal_restrictions: bool = True
    retrocess_threshold: float = 0.65
    forward_jump_threshold: float = 0.75
    max_normal_jump: int = 5
    switch_margin: float = 0.08
    max_forward_jump: int = 4
    jump_penalty: float = 0.18
    # Configuración de suavizado temporal (legacy)
    temporal_window_ratio: float = 0.1
    enable_temporal_smoothing: bool = True
    temporal_smoothing_window: int = 10


@dataclass
class PDFConfig:
    """Configuración de procesamiento de PDF."""

    dpi: int


@dataclass
class Config:
    """Configuración completa del proyecto."""

    slide_detection: SlideDetectionConfig
    camera: CameraConfig
    extraction: ExtractionConfig
    transcription: TranscriptionConfig
    llm: LLMConfig
    powerpoint: PowerPointConfig
    video: VideoConfig
    video_windows: VideoWindowsConfig
    processing: ProcessingConfig
    micro_merge: MicroMergeConfig
    frame_grouping: FrameGroupingConfig
    slide_matching: SlideMatchingConfig
    pdf: PDFConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Objeto Config con toda la configuración cargada
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {config_path}"
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(
        slide_detection=SlideDetectionConfig(**config_dict["slide_detection"]),
        camera=CameraConfig(**config_dict["camera"]),
        extraction=ExtractionConfig(**config_dict["extraction"]),
        transcription=TranscriptionConfig(**config_dict["transcription"]),
        llm=LLMConfig(**config_dict["llm"]),
        powerpoint=PowerPointConfig(**config_dict["powerpoint"]),
        video=VideoConfig(**config_dict["video"]),
        video_windows=VideoWindowsConfig(**config_dict["video_windows"]),
        processing=ProcessingConfig(**config_dict["processing"]),
        micro_merge=_load_micro_merge_config(config_dict.get("micro_merge")),
        frame_grouping=FrameGroupingConfig(**config_dict["frame_grouping"]),
        slide_matching=SlideMatchingConfig(**config_dict["slide_matching"]),
        pdf=PDFConfig(**config_dict["pdf"]),
    )


def _load_micro_merge_config(raw_config: Optional[Dict[str, Any]]) -> MicroMergeConfig:
    """Carga la configuración de micro-merge convirtiendo valores especiales."""

    if raw_config is None:
        return MicroMergeConfig()

    config_copy = dict(raw_config)
    background_color = config_copy.get("collage_background_color")
    if isinstance(background_color, list):
        config_copy["collage_background_color"] = tuple(background_color)

    return MicroMergeConfig(**config_copy)


def validate_config(config: Config) -> bool:
    """
    Valida que la configuración sea correcta.

    Args:
        config: Objeto Config a validar

    Returns:
        True si la configuración es válida

    Raises:
        ValueError: Si la configuración no es válida
    """
    # Validar posición de cámara
    if not 0 <= config.camera.position <= 9:
        raise ValueError("La posición de la cámara debe estar entre 0 y 9")

    # Validar pesos de detección
    total_weight = (
        config.slide_detection.edge_detection_weight
        + config.slide_detection.pixel_diff_weight
        + config.slide_detection.structural_weight
    )
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(
            f"Los pesos de detección deben sumar 1.0, actual: {total_weight}"
        )

    # Validar umbrales
    if not 0 <= config.slide_detection.pixel_diff_threshold <= 1:
        raise ValueError("pixel_diff_threshold debe estar entre 0 y 1")

    if not 0 <= config.slide_detection.overall_change_threshold <= 1:
        raise ValueError("overall_change_threshold debe estar entre 0 y 1")

    return True
