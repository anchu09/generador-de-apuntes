"""
Script principal que orquesta el proceso de generación de apuntes.
"""

import argparse
import gc
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PIL import Image, ImageOps

from .config_loader import (
    Config,
    MicroMergeConfig,
    load_config,
    validate_config,
)
from .frame_clusterer import FrameClusterer, FrameCluster
from .llm_summarizer import LLMSummarizer
from .pdf_processor import PDFProcessor
from .powerpoint_generator import PowerPointGenerator
from .slide_matcher import SlideMatcher
from .transcriber import Transcriber
from .video_processor import SlideSegment
from .video_window_segmenter import VideoWindowSegmenter, VideoWindow


def _get_output_paths_from_video(video_path: str) -> tuple[Path, Path, Path]:
    """
    Extrae la estructura de la ruta del video y genera las rutas de salida.

    Estructura esperada del video:
    Mastrofisica/{semestre}/{asignatura}/clases grabadas/{video}

    Estructura de salida:
    Mastrofisica/{semestre}/{asignatura}/apuntes/tablas/{nombre_video}.csv
    Mastrofisica/{semestre}/{asignatura}/apuntes/pdf/{nombre_video}.pdf

    Returns:
        tuple: (output_base_dir, tables_dir, pdf_dir)
    """
    video_path_obj = Path(video_path).resolve()
    video_name = video_path_obj.stem

    # Parsear la estructura: Mastrofisica/{semestre}/{asignatura}/clases grabadas/{video}
    parts = video_path_obj.parts

    # Buscar el índice de "clases grabadas"
    try:
        clases_idx = parts.index("clases grabadas")
    except ValueError:
        # Si no encuentra la estructura esperada, usar la carpeta del video como base
        print(
            "⚠ No se encontró la estructura esperada 'clases grabadas', usando carpeta del video"
        )
        output_base = video_path_obj.parent.parent / "apuntes"
        tables_dir = output_base / "tablas"
        pdf_dir = output_base / "pdf"
        tables_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        return output_base, tables_dir, pdf_dir

    # Construir la ruta de salida: Mastrofisica/{semestre}/{asignatura}/apuntes
    output_base = Path(*parts[:clases_idx]) / "apuntes"
    tables_dir = output_base / "tablas"
    pdf_dir = output_base / "pdf"

    # Crear directorios si no existen
    tables_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    return output_base, tables_dir, pdf_dir


def _cleanup_assets_directory(
    assets_dir: Path,
    output_name: str,
    used_slide_paths: Optional[List[str]] = None
):
    """
    Limpia los archivos temporales de assets después de generar el PDF.
    Solo mantiene los archivos necesarios para el PDF final.

    Args:
        assets_dir: Directorio de assets
        output_name: Nombre del output (para identificar archivos a limpiar)
        used_slide_paths: Lista opcional de rutas de imágenes usadas en el PDF final
    """
    if not assets_dir.exists():
        return

    try:
        removed_count = 0

        # Limpiar archivos temporales específicos de este video
        patterns_to_remove = [
            f"{output_name}_audio.wav",
            f"{output_name}_window_*.jpg",
            f"{output_name}_segment_*.jpg",
            f"{output_name}_merged_*.jpg",
        ]

        for pattern in patterns_to_remove:
            # Convertir patrón simple a búsqueda
            if "*" in pattern:
                prefix = pattern.split("*")[0]
                for file in assets_dir.glob(f"{prefix}*"):
                    try:
                        file.unlink()
                        removed_count += 1
                    except Exception:
                        pass
            else:
                file_path = assets_dir / pattern
                if file_path.exists():
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception:
                        pass

        # Limpiar imágenes de diapositivas extraídas de PDFs que ya no se necesitan
        if used_slide_paths:
            used_paths_set = {Path(p).resolve() for p in used_slide_paths if p}
            # Buscar todas las imágenes de diapositivas de PDFs en assets_dir
            for slide_file in assets_dir.glob("*_slide_*.png"):
                slide_file_resolved = slide_file.resolve()
                # Si la imagen no está en la lista de usadas, eliminarla
                if slide_file_resolved not in used_paths_set:
                    try:
                        slide_file.unlink()
                        removed_count += 1
                    except Exception:
                        pass

        if removed_count > 0:
            print(f"✓ Limpiados {removed_count} archivos temporales de assets")
    except Exception as e:
        print(f"⚠ Error limpiando assets: {e}")


def _cleanup_memory():
    """Fuerza la liberación de memoria."""
    gc.collect()


def process_video(
    video_path: str,
    pdf_paths: list[str],
    config: Config,
    output_name: Optional[str] = None,
) -> str:
    if output_name is None:
        output_name = Path(video_path).stem

    processing_mode = getattr(config.processing, "mode", "pdf_matching").lower()
    use_pdf_matching = processing_mode != "video_only"

    print("=" * 60)
    print("GENERADOR DE APUNTES AUTOMÁTICO")
    print("=" * 60)
    print(f"Video: {video_path}")
    if pdf_paths:
        print(f"PDFs: {', '.join(pdf_paths)}")
    print(f"Nombre de salida: {output_name}")
    print(f"Modo de procesamiento: {processing_mode}")
    print()

    # Obtener rutas de salida basadas en la estructura del video
    output_base, tables_dir, pdf_dir = _get_output_paths_from_video(video_path)
    print(f"Directorio de tablas: {tables_dir}")
    print(f"Directorio de PDFs: {pdf_dir}\n")

    # Assets se siguen guardando en el directorio configurado
    output_dir = Path(config.extraction.output_dir)
    assets_dir = output_dir / config.extraction.assets_dir
    assets_dir.mkdir(parents=True, exist_ok=True)

    pdf_processor = PDFProcessor(dpi=config.pdf.dpi) if use_pdf_matching else None
    slide_matcher = (
        SlideMatcher(
            use_histogram=config.slide_matching.use_histogram,
            use_structural=config.slide_matching.use_structural,
            use_features=config.slide_matching.use_features,
            histogram_weight=config.slide_matching.histogram_weight,
            structural_weight=config.slide_matching.structural_weight,
            features_weight=config.slide_matching.features_weight,
            min_similarity_threshold=config.slide_matching.min_similarity_threshold,
            use_temporal_restrictions=config.slide_matching.use_temporal_restrictions,
            retrocess_threshold=config.slide_matching.retrocess_threshold,
            forward_jump_threshold=config.slide_matching.forward_jump_threshold,
            max_normal_jump=config.slide_matching.max_normal_jump,
            switch_margin=config.slide_matching.switch_margin,
            max_forward_jump=config.slide_matching.max_forward_jump,
            jump_penalty=config.slide_matching.jump_penalty,
            temporal_window_ratio=config.slide_matching.temporal_window_ratio,
            enable_temporal_smoothing=config.slide_matching.enable_temporal_smoothing,
            temporal_smoothing_window=config.slide_matching.temporal_smoothing_window,
        )
        if use_pdf_matching
        else None
    )
    window_segmenter = VideoWindowSegmenter(
        window_duration=config.video_windows.window_duration,
        frames_to_average=config.video_windows.frames_to_average,
        output_dir=assets_dir,
    )
    transcriber = Transcriber(config.transcription)
    ppt_generator = PowerPointGenerator(config.powerpoint)

    print("Inicializando componentes...")
    # Inicializar LLM summarizer (puede fallar si no hay API key, pero continuamos sin resúmenes)
    llm_summarizer = None
    try:
        llm_summarizer = LLMSummarizer(config.llm)
        print("✓ LLM Summarizer inicializado")
    except Exception as e:
        print(f"⚠ No se pudo inicializar LLM Summarizer: {e}")
        print("  Continuando sin generación de resúmenes")
    print("✓ Componentes inicializados\n")

    if use_pdf_matching:
        print("Extrayendo diapositivas de los PDFs...")
        all_slides = pdf_processor.extract_slides_batch(pdf_paths, assets_dir)
        print(f"✓ {len(all_slides)} diapositivas extraídas\n")
        if not all_slides:
            raise ValueError("No se encontraron diapositivas en los PDFs")
    else:
        all_slides = []
        print("Modo video_only: se omitirá el matching con PDFs.\n")

    test_fraction = getattr(config.processing, "test_fraction", 1.0) or 1.0
    duration_fraction = None
    if test_fraction is not None and 0 < test_fraction < 1.0:
        duration_fraction = test_fraction

    print("Dividiendo video en ventanas...")
    windows = window_segmenter.segment_video(
        video_path, output_name, duration_fraction=duration_fraction
    )
    print(f"✓ {len(windows)} ventanas creadas\n")

    processed_duration = windows[-1].end_time if windows else 0.0

    if use_pdf_matching and slide_matcher is not None:
        print("Asignando ventanas a las diapositivas...")
        windows = slide_matcher.match_all_windows(windows, all_slides)
        print("✓ Matching completado\n")

    print("Transcribiendo video completo...")
    audio_path = assets_dir / f"{output_name}_audio.wav"
    max_transcription_duration = processed_duration if duration_fraction else None
    transcription_segments = transcriber.transcribe_video_with_timestamps(
        video_path, str(audio_path), max_duration=max_transcription_duration
    )
    if transcription_segments is None:
        print("⚠ No se pudo transcribir el video, continuando sin transcripciones")
        transcription_segments = []
    else:
        print(f"✓ Video transcrito: {len(transcription_segments)} segmentos\n")

    print("Dividiendo transcripción por ventanas...")
    intervals = [(w.start_time, w.end_time) for w in windows]
    window_transcriptions = transcriber.split_transcription_by_intervals(
        transcription_segments, intervals
    )
    for window, transcription in zip(windows, window_transcriptions):
        window.transcription = transcription

    print(
        f"✓ Transcripciones asignadas a {len([w for w in windows if w.transcription])} ventanas\n"
    )

    try:
        if use_pdf_matching:
            result, used_slide_paths = _generate_pdf_mode_outputs(
                windows=windows,
                all_slides=all_slides,
                tables_dir=tables_dir,
                pdf_dir=pdf_dir,
                output_name=output_name,
                ppt_generator=ppt_generator,
                llm_summarizer=llm_summarizer,
            )
        else:
            result, used_slide_paths = _generate_video_only_outputs(
                windows=windows,
                assets_dir=assets_dir,
                tables_dir=tables_dir,
                pdf_dir=pdf_dir,
                output_name=output_name,
                ppt_generator=ppt_generator,
                config=config,
                llm_summarizer=llm_summarizer,
            )

        # Limpiar assets después de generar el PDF
        _cleanup_assets_directory(assets_dir, output_name, used_slide_paths)

        return result
    finally:
        # Liberar memoria explícitamente
        del windows
        del transcription_segments
        del window_transcriptions
        if use_pdf_matching and all_slides:
            del all_slides
        _cleanup_memory()


def _generate_pdf_mode_outputs(
    windows: List[VideoWindow],
    all_slides,
    tables_dir: Path,
    pdf_dir: Path,
    output_name: str,
    ppt_generator: PowerPointGenerator,
    llm_summarizer: Optional[LLMSummarizer] = None,
) -> tuple[str, List[str]]:
    slides_with_transcriptions = defaultdict(list)
    for window in windows:
        if window.matched_slide_index is not None and window.transcription:
            slides_with_transcriptions[window.matched_slide_index].append(
                {"start_time": window.start_time, "transcription": window.transcription}
            )

    for slide_idx in slides_with_transcriptions:
        slides_with_transcriptions[slide_idx].sort(key=lambda x: x["start_time"])

    slide_segments: List[SlideSegment] = []
    data_rows = []

    # Primero crear todos los slide_segments sin resúmenes
    for slide_idx, (slide_num, _, slide_path) in enumerate(all_slides):
        if slide_idx in slides_with_transcriptions:
            transcriptions = slides_with_transcriptions[slide_idx]
            valid_texts = [
                t["transcription"] for t in transcriptions if t.get("transcription")
            ]
            full_transcription = "\n\n".join(valid_texts) if valid_texts else None
            start_time = transcriptions[0]["start_time"]
            end_time = transcriptions[-1]["start_time"] + 5.0
            duration = end_time - start_time
            num_windows = len(transcriptions)
        else:
            full_transcription = None
            start_time = 0.0
            end_time = 0.0
            duration = 0.0
            num_windows = 0

        slide_segments.append(
            SlideSegment(
                slide_number=slide_num,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                frame_path=slide_path,
                transcription=full_transcription,
                summary=None,
            )
        )

    # Generar resúmenes si hay LLM summarizer disponible
    if llm_summarizer:
        print("Generando resúmenes con LLM...")
        # Obtener índices de segmentos con transcripción
        segments_with_transcription = [
            (idx, seg)
            for idx, seg in enumerate(slide_segments)
            if seg.transcription and Path(seg.frame_path).exists()
        ]

        if segments_with_transcription:
            print(
                f"  Generando {len(segments_with_transcription)} resúmenes con contexto local..."
            )
            summaries = []
            context_before_count = llm_summarizer.config.context_slides_before
            context_after_count = llm_summarizer.config.context_slides_after

            for list_idx, (seg_idx, seg) in enumerate(segments_with_transcription):
                print(
                    f"  Progreso: {list_idx + 1}/{len(segments_with_transcription)}",
                    end="\r",
                )

                # Construir contexto de diapositivas anteriores
                context_before = None
                if context_before_count > 0:
                    before_segments = []
                    for i in range(max(0, seg_idx - context_before_count), seg_idx):
                        if slide_segments[i].transcription:
                            before_segments.append(
                                f"Diapositiva {slide_segments[i].slide_number}:\n{slide_segments[i].transcription}"
                            )
                    if before_segments:
                        context_before = "\n\n".join(before_segments)

                # Construir contexto de diapositivas siguientes
                context_after = None
                if context_after_count > 0:
                    after_segments = []
                    for i in range(
                        seg_idx + 1,
                        min(len(slide_segments), seg_idx + 1 + context_after_count),
                    ):
                        if slide_segments[i].transcription:
                            after_segments.append(
                                f"Diapositiva {slide_segments[i].slide_number}:\n{slide_segments[i].transcription}"
                            )
                    if after_segments:
                        context_after = "\n\n".join(after_segments)

                summary = llm_summarizer.generate_summary(
                    seg.frame_path,
                    seg.transcription,
                    context_before=context_before,
                    context_after=context_after,
                )
                summaries.append(summary)
            print(f"\n✓ {len([s for s in summaries if s])} resúmenes generados\n")

            # Asignar resúmenes a los slide_segments correspondientes
            for list_idx, (seg_idx, seg) in enumerate(segments_with_transcription):
                if list_idx < len(summaries):
                    seg.summary = summaries[list_idx]
        else:
            print("  No hay segmentos para resumir\n")

    # Crear data_rows después de generar resúmenes
    for slide_idx, (slide_num, _, slide_path) in enumerate(all_slides):
        seg = slide_segments[slide_idx]
        data_rows.append(
            {
                "slide_number": slide_num,
                "slide_index": slide_idx,
                "start_time": seg.start_time if seg.start_time > 0 else None,
                "end_time": seg.end_time if seg.end_time > 0 else None,
                "duration": seg.duration if seg.duration > 0 else None,
                "num_windows": len(slides_with_transcriptions.get(slide_idx, [])),
                "frame_path": slide_path,
                "transcription": seg.transcription,
                "summary": seg.summary,
            }
        )

    slide_segments.sort(
        key=lambda s: (
            s.start_time if s.start_time and s.transcription else float("inf"),
            s.slide_number,
        )
    )

    table_path = tables_dir / f"{output_name}.csv"
    pd.DataFrame(data_rows).to_csv(table_path, index=False, encoding="utf-8")

    presentation_path = pdf_dir / f"{output_name}.{ppt_generator.config.output_format}"
    ppt_generator.generate_presentation(slide_segments, str(presentation_path))

    # Obtener rutas de imágenes usadas para limpieza
    used_slide_paths = [seg.frame_path for seg in slide_segments if seg.frame_path]

    print("=" * 60)
    print("PROCESO COMPLETADO")
    print("=" * 60)
    print(f"Tabla: {table_path}")
    print(f"Presentación: {presentation_path}")

    # Liberar memoria
    del slide_segments
    del data_rows
    del slides_with_transcriptions
    _cleanup_memory()

    return str(presentation_path), used_slide_paths


def _generate_video_only_outputs(
    windows: List[VideoWindow],
    assets_dir: Path,
    tables_dir: Path,
    pdf_dir: Path,
    output_name: str,
    ppt_generator: PowerPointGenerator,
    config: Config,
    llm_summarizer: Optional[LLMSummarizer] = None,
) -> tuple[str, List[str]]:
    clusterer = FrameClusterer(
        config.frame_grouping,
        assets_dir=assets_dir,
        output_name=output_name,
        camera_position=config.camera.position,
    )
    clusters = clusterer.cluster_windows(windows)

    if not clusters and windows:
        print("⚠ No se detectaron segmentos; se generará uno con todo el video.")
        fallback_path = clusterer._save_representative_frame(
            0, windows[0].representative_frame
        )
        clusters = [
            FrameCluster(
                cluster_id=0,
                start_time=windows[0].start_time,
                end_time=windows[-1].end_time,
                duration=windows[-1].end_time - windows[0].start_time,
                window_indices=list(range(len(windows))),
                frame_path=fallback_path,
            )
        ]

    slide_segments: List[SlideSegment] = []
    data_rows = []

    for cluster in clusters:
        text_blocks = [
            windows[idx].transcription.strip()
            for idx in cluster.window_indices
            if windows[idx].transcription
        ]
        full_transcription = "\n\n".join(text_blocks) if text_blocks else None

        slide_segments.append(
            SlideSegment(
                slide_number=cluster.cluster_id,
                start_time=cluster.start_time,
                end_time=cluster.end_time,
                duration=cluster.duration,
                frame_path=cluster.frame_path,
                transcription=full_transcription,
                summary=None,
                num_windows=len(cluster.window_indices),
                metadata={"source_cluster_id": cluster.cluster_id},
            )
        )

    if config.micro_merge.enabled:
        slide_segments = _merge_micro_segments(
            slide_segments,
            config.micro_merge,
            assets_dir,
            output_name,
        )
    else:
        for idx, seg in enumerate(slide_segments):
            seg.slide_number = idx

    # Generar resúmenes si hay LLM summarizer disponible
    if llm_summarizer:
        print("Generando resúmenes con LLM...")
        # Obtener índices de segmentos con transcripción
        segments_with_transcription = [
            (idx, seg)
            for idx, seg in enumerate(slide_segments)
            if seg.transcription and Path(seg.frame_path).exists()
        ]

        if segments_with_transcription:
            print(
                f"  Generando {len(segments_with_transcription)} resúmenes con contexto local..."
            )
            summaries = []
            context_before_count = llm_summarizer.config.context_slides_before
            context_after_count = llm_summarizer.config.context_slides_after

            for list_idx, (seg_idx, seg) in enumerate(segments_with_transcription):
                print(
                    f"  Progreso: {list_idx + 1}/{len(segments_with_transcription)}",
                    end="\r",
                )

                # Construir contexto de segmentos anteriores
                context_before = None
                if context_before_count > 0:
                    before_segments = []
                    for i in range(max(0, seg_idx - context_before_count), seg_idx):
                        if slide_segments[i].transcription:
                            before_segments.append(
                                f"Segmento {slide_segments[i].slide_number}:\n{slide_segments[i].transcription}"
                            )
                    if before_segments:
                        context_before = "\n\n".join(before_segments)

                # Construir contexto de segmentos siguientes
                context_after = None
                if context_after_count > 0:
                    after_segments = []
                    for i in range(
                        seg_idx + 1,
                        min(len(slide_segments), seg_idx + 1 + context_after_count),
                    ):
                        if slide_segments[i].transcription:
                            after_segments.append(
                                f"Segmento {slide_segments[i].slide_number}:\n{slide_segments[i].transcription}"
                            )
                    if after_segments:
                        context_after = "\n\n".join(after_segments)

                summary = llm_summarizer.generate_summary(
                    seg.frame_path,
                    seg.transcription,
                    context_before=context_before,
                    context_after=context_after,
                )
                summaries.append(summary)
            print(f"\n✓ {len([s for s in summaries if s])} resúmenes generados\n")

            # Asignar resúmenes a los slide_segments correspondientes
            for list_idx, (seg_idx, seg) in enumerate(segments_with_transcription):
                if list_idx < len(summaries):
                    seg.summary = summaries[list_idx]
        else:
            print("  No hay segmentos para resumir\n")

    # Crear data_rows después de generar resúmenes
    for seg in slide_segments:
        merged_from = seg.metadata.get("merged_from_ids") if seg.metadata else None
        data_rows.append(
            {
                "segment_number": seg.slide_number,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "num_windows": seg.num_windows,
                "frame_path": seg.frame_path,
                "transcription": seg.transcription,
                "summary": seg.summary,
                "merged_from_segments": (
                    ",".join(str(idx) for idx in merged_from) if merged_from else None
                ),
            }
        )

    table_path = tables_dir / f"{output_name}.csv"
    pd.DataFrame(data_rows).to_csv(table_path, index=False, encoding="utf-8")

    presentation_path = pdf_dir / f"{output_name}.{ppt_generator.config.output_format}"
    ppt_generator.generate_presentation(slide_segments, str(presentation_path))

    # Obtener rutas de imágenes usadas para limpieza
    used_slide_paths = [seg.frame_path for seg in slide_segments if seg.frame_path]

    print("=" * 60)
    print("PROCESO COMPLETADO (modo video_only)")
    print("=" * 60)
    print(f"Tabla: {table_path}")
    print(f"Presentación: {presentation_path}")

    # Liberar memoria
    del slide_segments
    del data_rows
    del clusters
    _cleanup_memory()

    return str(presentation_path), used_slide_paths


def _merge_micro_segments(
    segments: List[SlideSegment],
    merge_config: MicroMergeConfig,
    assets_dir: Path,
    output_name: str,
) -> List[SlideSegment]:
    """Fusiona segmentos con pocas ventanas en bloques más grandes."""

    if not segments:
        return segments

    merged_segments: List[SlideSegment] = []
    current_group: List[SlideSegment] = []
    group_windows = 0
    group_duration = 0.0
    group_counter = 0

    max_windows = max(0, merge_config.max_total_windows)
    max_duration = max(0.0, merge_config.max_total_duration)

    def flush_group():
        nonlocal current_group, group_windows, group_duration, group_counter
        if not current_group:
            return
        if len(current_group) >= merge_config.min_segments:
            merged_segments.append(
                _build_merged_segment(
                    current_group,
                    assets_dir,
                    output_name,
                    group_counter,
                    merge_config,
                )
            )
            group_counter += 1
        else:
            merged_segments.extend(current_group)
        current_group = []
        group_windows = 0
        group_duration = 0.0

    for seg in segments:
        segment_windows = seg.num_windows or 0
        segment_duration = seg.duration or 0.0
        is_micro = segment_windows < merge_config.window_threshold

        if is_micro:
            projected_windows = group_windows + segment_windows
            projected_duration = group_duration + segment_duration
            exceeds_windows = max_windows > 0 and projected_windows > max_windows
            exceeds_duration = max_duration > 0 and projected_duration > max_duration

            if exceeds_windows or exceeds_duration:
                flush_group()

            current_group.append(seg)
            group_windows += segment_windows
            group_duration += segment_duration
        else:
            flush_group()
            merged_segments.append(seg)

    flush_group()

    for idx, seg in enumerate(merged_segments):
        seg.slide_number = idx

    return merged_segments


def _build_merged_segment(
    group: List[SlideSegment],
    assets_dir: Path,
    output_name: str,
    group_index: int,
    merge_config: MicroMergeConfig,
) -> SlideSegment:
    """Genera un nuevo SlideSegment fusionando un grupo."""

    start_time = group[0].start_time
    end_time = group[-1].end_time
    duration = (end_time or 0.0) - (start_time or 0.0)

    transcription_parts = [
        seg.transcription.strip()
        for seg in group
        if seg.transcription and seg.transcription.strip()
    ]
    merged_transcription = (
        "\n\n".join(transcription_parts) if transcription_parts else None
    )

    total_windows = sum(seg.num_windows or 0 for seg in group)
    frame_paths = [seg.frame_path for seg in group if seg.frame_path]
    collage_path = _create_collage_image(
        frame_paths,
        assets_dir,
        output_name,
        group_index,
        merge_config,
    )

    metadata = {
        "merged_from_ids": [seg.slide_number for seg in group],
        "merged_count": len(group),
    }

    return SlideSegment(
        slide_number=group[0].slide_number,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        frame_path=collage_path or group[0].frame_path,
        transcription=merged_transcription,
        summary=None,
        num_windows=total_windows,
        metadata=metadata,
    )


def _create_collage_image(
    frame_paths: List[str],
    assets_dir: Path,
    output_name: str,
    group_index: int,
    merge_config: MicroMergeConfig,
) -> Optional[str]:
    """Crea un collage con las imágenes de los segmentos fusionados."""

    valid_paths = [Path(path) for path in frame_paths if path and Path(path).exists()]
    if not valid_paths:
        return None

    images = []
    for path in valid_paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as exc:
            print(f"⚠ No se pudo abrir la imagen {path}: {exc}")

    if not images:
        return None

    columns = max(1, min(merge_config.collage_max_columns, len(images)))
    rows = math.ceil(len(images) / columns)
    tile_width = max(1, merge_config.collage_tile_width)
    tile_height = max(1, merge_config.collage_tile_height)
    padding = max(0, merge_config.collage_padding)
    bg_color = tuple(merge_config.collage_background_color)

    width = columns * tile_width + (columns + 1) * padding
    height = rows * tile_height + (rows + 1) * padding
    collage = Image.new("RGB", (width, height), bg_color)

    resampling_method = (
        Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    )

    try:
        for idx, img in enumerate(images):
            # Calcular escala manteniendo relación de aspecto (sin recortar)
            scale = min(tile_width / img.width, tile_height / img.height)
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)

            # Redimensionar manteniendo relación de aspecto con alta calidad
            thumb = img.resize((new_width, new_height), resampling_method)

            # Crear tile con fondo blanco y centrar la imagen
            tile = Image.new("RGB", (tile_width, tile_height), bg_color)
            offset_x = (tile_width - new_width) // 2
            offset_y = (tile_height - new_height) // 2
            tile.paste(thumb, (offset_x, offset_y))

            col = idx % columns
            row = idx // columns
            x = padding + col * (tile_width + padding)
            y = padding + row * (tile_height + padding)
            collage.paste(tile, (x, y))

            img.close()
    except Exception as exc:
        print(f"⚠ Error creando collage para grupo {group_index}: {exc}")
        for img in images:
            try:
                img.close()
            except Exception:
                pass
        return None

    collage_filename = assets_dir / f"{output_name}_merged_{group_index:04d}.jpg"
    collage.save(collage_filename, "JPEG", quality=90)
    return str(collage_filename)


def main():
    parser = argparse.ArgumentParser(
        description="Generador de Apuntes Automático a partir de videos de clases y PDFs"
    )
    parser.add_argument("video", type=str, help="Ruta al archivo de video a procesar")
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=str,
        help="Ruta(s) a archivo(s) PDF con las diapositivas (opcional en modo video_only)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración (default: config.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Nombre para los archivos de salida (default: nombre del video)",
    )
    parser.add_argument(
        "--camera-position",
        type=int,
        choices=list(range(0, 10)),
        help=(
            "Cuadrante 3x3 donde está la cámara del profesor (0 = sin cámara, "
            "1-9 de izquierda a derecha y de arriba abajo)"
        ),
    )

    args = parser.parse_args()

    print("Cargando configuración...")
    config = load_config(args.config)
    validate_config(config)
    print("✓ Configuración cargada y validada\n")

    # Sobrescribir posición de cámara desde CLI si se proporciona
    if args.camera_position is not None:
        config.camera.position = args.camera_position
        print(
            f"⚙ Cámara del profesor ubicada en cuadrante 3x3: "
            f"{config.camera.position} (0 = sin cámara)"
        )

    # Verificar si se necesitan PDFs según el modo
    processing_mode = getattr(config.processing, "mode", "pdf_matching").lower()
    if processing_mode == "pdf_matching":
        if not args.pdfs:
            parser.error("En modo 'pdf_matching' se requiere al menos un archivo PDF")
    else:
        # En modo video_only, los PDFs son opcionales (se ignoran si se pasan)
        if args.pdfs:
            print("⚠ Modo 'video_only' activado: los PDFs serán ignorados\n")

    try:
        output_path = process_video(args.video, args.pdfs or [], config, args.output)
        print("\n✓ Proceso completado exitosamente")
        print(f"  Archivo generado: {output_path}")
    except Exception as exc:
        print(f"\n✗ Error durante el procesamiento: {exc}")
        raise


if __name__ == "__main__":
    main()
