"""
Módulo para hacer matching de similitud entre frames de video y diapositivas del PDF.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from .video_window_segmenter import VideoWindow


class SlideMatcher:
    """
    Encuentra la diapositiva más similar a cada frame del video usando
    múltiples técnicas de comparación visual.
    """

    def __init__(
        self,
        use_histogram: bool = True,
        use_structural: bool = True,
        use_features: bool = True,
        histogram_weight: float = 0.3,
        structural_weight: float = 0.4,
        features_weight: float = 0.3,
        min_similarity_threshold: float = 0.3,
        use_temporal_restrictions: bool = True,
        retrocess_threshold: float = 0.65,
        forward_jump_threshold: float = 0.75,
        max_normal_jump: int = 5,
        switch_margin: float = 0.08,
        max_forward_jump: int = 4,
        jump_penalty: float = 0.18,
        temporal_window_ratio: float = 0.1,
        enable_temporal_smoothing: bool = True,
        temporal_smoothing_window: int = 10,
    ):
        """
        Inicializa el matcher de diapositivas.

        Args:
            use_histogram: Usar comparación de histogramas
            use_structural: Usar comparación estructural (bordes)
            use_features: Usar detección de características (ORB)
            histogram_weight: Peso para comparación de histogramas
            structural_weight: Peso para comparación estructural
            features_weight: Peso para detección de características
            min_similarity_threshold: Umbral mínimo de similitud (0-1)
            use_temporal_restrictions: Si True, usa restricciones temporales estrictas (legacy)
            retrocess_threshold: Umbral de similitud requerido para permitir retrocesos
            forward_jump_threshold: Umbral de similitud requerido para saltos grandes hacia adelante
            max_normal_jump: Número máximo de diapositivas para considerar un salto "normal"
            temporal_window_ratio: Ratio del video para ventana temporal (solo si use_temporal_restrictions=True)
            enable_temporal_smoothing: Activar suavizado temporal
            temporal_smoothing_window: Número de ventanas para suavizado
        """
        self.use_histogram = use_histogram
        self.use_structural = use_structural
        self.use_features = use_features

        # Normalizar pesos
        total_weight = histogram_weight + structural_weight + features_weight
        self.histogram_weight = histogram_weight / total_weight
        self.structural_weight = structural_weight / total_weight
        self.features_weight = features_weight / total_weight

        # Parámetros de matching robusto
        self.min_similarity_threshold = min_similarity_threshold
        self.use_temporal_restrictions = use_temporal_restrictions
        self.retrocess_threshold = retrocess_threshold
        self.forward_jump_threshold = forward_jump_threshold
        self.max_normal_jump = max_normal_jump
        self.switch_margin = switch_margin
        self.max_forward_jump = max_forward_jump
        self.jump_penalty = jump_penalty

        # Parámetros de suavizado temporal (legacy)
        self.temporal_window_ratio = temporal_window_ratio
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.temporal_smoothing_window = temporal_smoothing_window

        # Preprocesar diapositivas (cachear características)
        self.slide_features_cache = {}
        self.slide_histograms_cache = {}

        # Para tracking temporal
        self.video_duration = None
        self.windows = None

    def match_window_to_slides(
        self,
        window: VideoWindow,
        slides: List[Tuple[int, np.ndarray, str]],
        last_assigned_slide: Optional[int] = None,
        candidate_slide_indices: Optional[List[int]] = None,
    ) -> Tuple[int, float]:
        """
        Encuentra la diapositiva más similar a un frame de ventana usando matching robusto.
        Aplica lógica contextual para permitir retrocesos y repeticiones mientras evita matches incorrectos.

        Args:
            window: Ventana de video con frame representativo
            slides: Lista de diapositivas (número, imagen, ruta)
            last_assigned_slide: Índice de la última diapositiva asignada (para lógica contextual)
            candidate_slide_indices: Lista opcional de índices de diapositivas candidatas (solo si use_temporal_restrictions=True)

        Returns:
            Tupla (índice de la diapositiva más similar, similitud). -1 si no se asigna.
        """
        frame = window.representative_frame

        # Preprocesar frame
        frame_gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )

        # Determinar qué diapositivas evaluar
        if self.use_temporal_restrictions and candidate_slide_indices is not None:
            # Modo legacy: solo buscar en candidatos temporales
            slide_indices_to_check = candidate_slide_indices
        else:
            # Modo robusto: evaluar TODAS las diapositivas
            slide_indices_to_check = range(len(slides))

        # Calcular similitud con todas las diapositivas candidatas
        all_similarities = []
        for idx in slide_indices_to_check:
            slide_num, slide_img, slide_path = slides[idx]
            similarity = self._calculate_similarity(frame, frame_gray, slide_img, idx)
            all_similarities.append((idx, similarity))

        # Ordenar por similitud (mayor a menor)
        all_similarities.sort(key=lambda x: x[1], reverse=True)

        if not all_similarities:
            return -1, 0.0

        # Obtener similitud con la diapositiva anterior (si existe)
        last_slide_similarity = None
        if last_assigned_slide is not None:
            for idx, sim in all_similarities:
                if idx == last_assigned_slide:
                    last_slide_similarity = sim
                    break

        # Obtener el mejor match
        best_idx, best_sim = all_similarities[0]

        # Aplicar lógica contextual si tenemos información de la última diapositiva asignada
        if last_assigned_slide is not None and len(slides) > 0:
            slide_jump = abs(best_idx - last_assigned_slide)

            # Caso 1: Retroceso (diapositiva anterior)
            if best_idx < last_assigned_slide:
                # Permitir retroceso solo con alta similitud
                if best_sim < self.retrocess_threshold:
                    # Buscar siguiente mejor match que no sea retroceso o que tenga mejor similitud
                    for idx, sim in all_similarities[1:]:
                        # Si encontramos una diapositiva que no es retroceso y cumple umbral mínimo
                        if (
                            idx >= last_assigned_slide
                            and sim >= self.min_similarity_threshold
                        ):
                            return idx, sim
                        # Si encontramos un retroceso con similitud muy alta, aceptarlo
                        if (
                            idx < last_assigned_slide
                            and sim >= self.retrocess_threshold
                        ):
                            return idx, sim
                    # Si no hay mejor opción y el mejor match no cumple el umbral de retroceso, no asignar
                    if best_sim < self.min_similarity_threshold:
                        return -1, best_sim
                    # Si cumple el umbral mínimo pero no el de retroceso, no asignar (muy conservador)
                    return -1, best_sim
                # Si cumple el umbral de retroceso, aceptar
                if best_sim >= self.min_similarity_threshold:
                    return best_idx, best_sim

            # Caso 2: Salto grande hacia adelante (>max_normal_jump)
            elif slide_jump > self.max_normal_jump:
                # Requerir similitud muy alta para saltos grandes
                if best_sim < self.forward_jump_threshold:
                    # Buscar mejor match cercano (dentro del rango normal)
                    for idx, sim in all_similarities:
                        if (
                            abs(idx - last_assigned_slide) <= self.max_normal_jump
                            and sim >= self.min_similarity_threshold
                        ):
                            return idx, sim
                    # Si no hay mejor opción cercana y el mejor match no cumple umbral de salto, no asignar
                    if best_sim < self.min_similarity_threshold:
                        return -1, best_sim
                    return -1, best_sim
                # Si cumple el umbral de salto grande, aceptar
                if best_sim >= self.min_similarity_threshold:
                    return best_idx, best_sim

            # Caso 3: Salto normal (dentro del rango permitido) o misma diapositiva
            else:
                if (
                    best_idx != last_assigned_slide
                    and last_slide_similarity is not None
                ):
                    # Requerir mejora suficiente para cambiar de diapositiva
                    improvement = best_sim - last_slide_similarity
                    if (
                        last_slide_similarity >= self.min_similarity_threshold
                        and improvement < self.switch_margin
                    ):
                        return last_assigned_slide, last_slide_similarity

                # Aplicar umbral mínimo normal
                if best_sim >= self.min_similarity_threshold:
                    return best_idx, best_sim

        # Si no hay información de última diapositiva pero tenemos un match fuerte, devolverlo
        if last_assigned_slide is None and best_sim >= self.min_similarity_threshold:
            return best_idx, best_sim

        # Si tenemos última diapositiva y su similitud sigue siendo aceptable, mantenerla
        if last_assigned_slide is not None and last_slide_similarity is not None:
            if last_slide_similarity >= self.min_similarity_threshold:
                return last_assigned_slide, last_slide_similarity

        # Si no hay información de última diapositiva asignada, aplicar solo umbral mínimo
        if best_sim >= self.min_similarity_threshold:
            return best_idx, best_sim

        return -1, best_sim

    def match_all_windows(
        self, windows: List[VideoWindow], slides: List[Tuple[int, np.ndarray, str]]
    ) -> List[VideoWindow]:
        """
        Asigna cada ventana a su diapositiva usando alineamiento secuencial dinámico.
        Mantiene el orden de las diapositivas y penaliza saltos grandes.
        """
        print(
            f"Asignando {len(windows)} ventanas a {len(slides)} diapositivas (alineamiento secuencial)..."
        )

        if not windows or not slides:
            return windows

        self.windows = windows
        self.video_duration = windows[-1].end_time if windows else None
        self._preprocess_slides(slides)

        print(f"  Umbral mínimo de similitud: {self.min_similarity_threshold}")
        print(f"  Máx. salto permitido: {self.max_forward_jump}")
        print(f"  Penalización por salto: {self.jump_penalty}")

        similarity_matrix = []
        for idx, window in enumerate(windows):
            sims = self._compute_window_similarities(window, slides)
            similarity_matrix.append(sims)
            if (idx + 1) % 100 == 0:
                print(
                    f"  Calculadas similitudes para {idx + 1}/{len(windows)} ventanas..."
                )

        assignments = self._assign_slides_dynamic(similarity_matrix)

        unassigned = 0
        for window_idx, slide_idx in enumerate(assignments):
            if slide_idx is None:
                windows[window_idx].matched_slide_index = None
                windows[window_idx].matched_similarity = None
                unassigned += 1
                continue

            similarity = similarity_matrix[window_idx][slide_idx]
            windows[window_idx].matched_similarity = similarity
            if similarity >= self.min_similarity_threshold:
                windows[window_idx].matched_slide_index = slide_idx
            else:
                windows[window_idx].matched_slide_index = None
                unassigned += 1

        if unassigned:
            print(f"  ⚠️ {unassigned} ventanas sin asignación (similitud < umbral)")

        if self.enable_temporal_smoothing:
            print("  Aplicando suavizado temporal...")
            windows = self._apply_temporal_smoothing(windows, slides)

        print("✓ Matching completado")
        return windows

    def _get_temporal_candidates(
        self, window_time: float, window_idx: int, total_slides: int
    ) -> List[int]:
        """
        Obtiene los índices de diapositivas candidatas basadas en restricciones temporales.

        Args:
            window_time: Tiempo de la ventana
            window_idx: Índice de la ventana
            total_slides: Número total de diapositivas

        Returns:
            Lista de índices de diapositivas candidatas
        """
        if not self.windows or self.video_duration is None or self.video_duration == 0:
            # Sin restricción temporal, devolver todas
            return list(range(total_slides))

        # Estrategia: estimar qué diapositiva debería aparecer en este tiempo
        # basado en la posición temporal relativa del video
        # Asumimos que las diapositivas aparecen en orden secuencial

        time_ratio = window_time / self.video_duration
        estimated_slide = int(time_ratio * total_slides)

        # Calcular rango de búsqueda basado en la ventana temporal
        # El rango debe ser proporcional al número de diapositivas y al ratio temporal
        range_size = max(3, int(total_slides * self.temporal_window_ratio * 2))

        # Asegurar que el rango no sea demasiado pequeño ni demasiado grande
        range_size = min(range_size, total_slides // 2)
        range_size = max(range_size, 2)

        min_slide = max(0, estimated_slide - range_size)
        max_slide = min(total_slides - 1, estimated_slide + range_size)

        candidates = list(range(min_slide, max_slide + 1))

        # Si hay ventanas ya procesadas cercanas temporalmente, usar sus asignaciones como guía
        if self.windows and window_idx > 0:
            # Buscar ventanas cercanas que ya tengan asignación
            nearby_range = min(20, window_idx)  # Buscar hasta 20 ventanas atrás
            nearby_windows = [
                w
                for w in self.windows[max(0, window_idx - nearby_range) : window_idx]
                if w.matched_slide_index is not None
            ]

            if nearby_windows:
                # Usar el rango de diapositivas asignadas en ventanas cercanas
                assigned_slides = [w.matched_slide_index for w in nearby_windows]
                if assigned_slides:
                    min_assigned = min(assigned_slides)
                    max_assigned = max(assigned_slides)
                    # Expandir el rango basado en asignaciones cercanas
                    min_slide = max(0, min(min_slide, min_assigned - 1))
                    max_slide = min(total_slides - 1, max(max_slide, max_assigned + 1))
                    candidates = list(range(min_slide, max_slide + 1))

        return candidates

    def _compute_window_similarities(
        self, window: VideoWindow, slides: List[Tuple[int, np.ndarray, str]]
    ) -> List[float]:
        """Calcula la similitud del frame de la ventana con todas las diapositivas."""
        frame = window.representative_frame
        frame_gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        )

        similarities = []
        for idx, (slide_num, slide_img, slide_path) in enumerate(slides):
            similarity = self._calculate_similarity(frame, frame_gray, slide_img, idx)
            similarities.append(similarity)
        return similarities

    def _assign_slides_dynamic(
        self, similarity_matrix: List[List[float]]
    ) -> List[Optional[int]]:
        """
        Alinea ventanas con diapositivas usando programación dinámica no decreciente.
        """
        num_windows = len(similarity_matrix)
        num_slides = len(similarity_matrix[0]) if similarity_matrix else 0

        if num_windows == 0 or num_slides == 0:
            return [None] * num_windows

        NEG_INF = -1e9
        dp = [[NEG_INF] * num_slides for _ in range(num_windows)]
        back = [[None] * num_slides for _ in range(num_windows)]

        for s in range(num_slides):
            start_penalty = s * self.jump_penalty
            dp[0][s] = similarity_matrix[0][s] - start_penalty
            back[0][s] = None

        for w in range(1, num_windows):
            for s in range(num_slides):
                best_prev = NEG_INF
                best_prev_idx = None
                min_prev = max(0, s - self.max_forward_jump)
                for prev_s in range(min_prev, s + 1):
                    jump = s - prev_s
                    penalty = jump * self.jump_penalty
                    candidate_score = dp[w - 1][prev_s] - penalty
                    if prev_s == s:
                        candidate_score += self.switch_margin
                    if candidate_score > best_prev:
                        best_prev = candidate_score
                        best_prev_idx = prev_s

                dp[w][s] = similarity_matrix[w][s] + best_prev
                back[w][s] = best_prev_idx

        assignments = [None] * num_windows
        best_final_slide = max(range(num_slides), key=lambda s: dp[-1][s])

        current_slide = best_final_slide
        for w in reversed(range(num_windows)):
            assignments[w] = current_slide
            current_slide = back[w][current_slide] if w > 0 else None

        return assignments

    def _apply_temporal_smoothing(
        self, windows: List[VideoWindow], slides: List[Tuple[int, np.ndarray, str]]
    ) -> List[VideoWindow]:
        """
        Aplica suavizado temporal para corregir asignaciones inconsistentes.
        Corrige asignaciones que están claramente fuera de lugar temporalmente.

        Args:
            windows: Lista de ventanas
            slides: Lista de diapositivas

        Returns:
            Lista de ventanas con asignaciones corregidas
        """
        window_size = self.temporal_smoothing_window
        smoothed_windows = [w for w in windows]  # Crear copia superficial

        # Primero, calcular tiempos estimados para cada diapositiva
        slide_times = {}
        for slide_idx in range(len(slides)):
            slide_times[slide_idx] = self._estimate_slide_time(
                slide_idx, windows, len(slides)
            )

        corrections = 0

        for i in range(len(windows)):
            if windows[i].matched_slide_index is None:
                continue

            current_slide = windows[i].matched_slide_index
            window_time = windows[i].start_time

            # Obtener ventanas cercanas
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(windows), i + window_size // 2 + 1)
            nearby_windows = windows[start_idx:end_idx]

            # Contar asignaciones de diapositivas en ventanas cercanas
            slide_counts = {}
            for w in nearby_windows:
                if w.matched_slide_index is not None:
                    slide_idx = w.matched_slide_index
                    slide_counts[slide_idx] = slide_counts.get(slide_idx, 0) + 1

            if not slide_counts:
                continue

            # Encontrar la diapositiva más común en el vecindario
            most_common_slide = max(slide_counts.items(), key=lambda x: x[1])[0]
            support_ratio = slide_counts[most_common_slide] / len(nearby_windows)

            # Si la asignación actual es muy diferente de las cercanas
            if most_common_slide != current_slide and support_ratio > 0.6:
                # Verificar consistencia temporal
                current_slide_time = slide_times.get(current_slide, window_time)
                most_common_time = slide_times.get(most_common_slide, window_time)

                # Si la diapositiva más común está más cerca temporalmente
                current_distance = abs(current_slide_time - window_time)
                most_common_distance = abs(most_common_time - window_time)

                # También verificar que no haya un salto muy grande en el índice de diapositiva
                slide_jump = abs(most_common_slide - current_slide)

                if most_common_distance < current_distance and slide_jump <= 3:
                    smoothed_windows[i].matched_slide_index = most_common_slide
                    corrections += 1

        if corrections > 0:
            print(
                f"  ✓ Corregidas {corrections} asignaciones temporalmente inconsistentes"
            )

        return smoothed_windows

    def _estimate_slide_time(
        self, slide_idx: int, windows: List[VideoWindow], total_slides: int
    ) -> float:
        """
        Estima el tiempo promedio en que aparece una diapositiva.

        Args:
            slide_idx: Índice de la diapositiva
            windows: Lista de ventanas
            total_slides: Número total de diapositivas

        Returns:
            Tiempo estimado
        """
        matching_times = [
            w.start_time for w in windows if w.matched_slide_index == slide_idx
        ]
        if matching_times:
            # Usar mediana en lugar de promedio para ser más robusto a outliers
            sorted_times = sorted(matching_times)
            median_idx = len(sorted_times) // 2
            if len(sorted_times) % 2 == 0:
                return (sorted_times[median_idx - 1] + sorted_times[median_idx]) / 2
            else:
                return sorted_times[median_idx]

        # Si no hay matches, estimar basado en posición temporal relativa
        # Asumiendo que las diapositivas aparecen en orden secuencial
        if self.video_duration and total_slides > 0:
            # Estimar basado en posición relativa del índice de diapositiva
            # Distribución uniforme: cada diapositiva ocupa aproximadamente
            # 1/total_slides del tiempo del video
            return (slide_idx / total_slides) * self.video_duration

        return 0.0

    def _preprocess_slides(self, slides: List[Tuple[int, np.ndarray, str]]):
        """Preprocesa todas las diapositivas para cachear características."""
        for idx, (slide_num, slide_img, slide_path) in enumerate(slides):
            if idx not in self.slide_features_cache:
                slide_gray = (
                    cv2.cvtColor(slide_img, cv2.COLOR_BGR2GRAY)
                    if len(slide_img.shape) == 3
                    else slide_img
                )

                # Cachear histograma
                if self.use_histogram:
                    hist = cv2.calcHist(
                        [slide_img],
                        [0, 1, 2],
                        None,
                        [50, 50, 50],
                        [0, 256, 0, 256, 0, 256],
                    )
                    self.slide_histograms_cache[idx] = hist

                # Cachear características
                if self.use_features:
                    orb = cv2.ORB_create()
                    kp, des = orb.detectAndCompute(slide_gray, None)
                    self.slide_features_cache[idx] = (kp, des)

    def _calculate_similarity(
        self,
        frame: np.ndarray,
        frame_gray: np.ndarray,
        slide_img: np.ndarray,
        slide_idx: int,
    ) -> float:
        """
        Calcula la similitud entre un frame y una diapositiva usando múltiples técnicas.

        Args:
            frame: Frame de video (BGR)
            frame_gray: Frame en escala de grises
            slide_img: Imagen de la diapositiva (BGR)
            slide_idx: Índice de la diapositiva

        Returns:
            Score de similitud (0-1, donde 1 es idéntico)
        """
        similarities = []

        # 1. Comparación de histogramas (rápida, buena para cambios de color)
        if self.use_histogram:
            hist_sim = self._histogram_similarity(frame, slide_img, slide_idx)
            similarities.append((hist_sim, self.histogram_weight))

        # 2. Comparación estructural (bordes, buena para cambios de contenido)
        if self.use_structural:
            struct_sim = self._structural_similarity(frame_gray, slide_img)
            similarities.append((struct_sim, self.structural_weight))

        # 3. Comparación de características (ORB, buena para cambios sutiles)
        if self.use_features:
            feat_sim = self._features_similarity(frame_gray, slide_img, slide_idx)
            similarities.append((feat_sim, self.features_weight))

        # Calcular similitud ponderada
        if not similarities:
            return 0.0

        total_similarity = sum(sim * weight for sim, weight in similarities)
        return total_similarity

    def _histogram_similarity(
        self, frame: np.ndarray, slide_img: np.ndarray, slide_idx: int
    ) -> float:
        """Compara histogramas de color."""
        # Calcular histograma del frame
        frame_hist = cv2.calcHist(
            [frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256]
        )

        # Usar histograma cacheado si está disponible
        if slide_idx in self.slide_histograms_cache:
            slide_hist = self.slide_histograms_cache[slide_idx]
        else:
            slide_hist = cv2.calcHist(
                [slide_img], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256]
            )
            self.slide_histograms_cache[slide_idx] = slide_hist

        # Comparar usando correlación (0-1, donde 1 es idéntico)
        correlation = cv2.compareHist(frame_hist, slide_hist, cv2.HISTCMP_CORREL)
        return max(0.0, correlation)  # Asegurar que sea >= 0

    def _structural_similarity(
        self, frame_gray: np.ndarray, slide_img: np.ndarray
    ) -> float:
        """Compara estructura usando detección de bordes."""
        # Convertir slide a escala de grises si es necesario
        if len(slide_img.shape) == 3:
            slide_gray = cv2.cvtColor(slide_img, cv2.COLOR_BGR2GRAY)
        else:
            slide_gray = slide_img

        # Redimensionar si tienen tamaños diferentes
        if frame_gray.shape != slide_gray.shape:
            h, w = frame_gray.shape[:2]
            slide_gray = cv2.resize(slide_gray, (w, h), interpolation=cv2.INTER_LINEAR)

        # Detectar bordes
        frame_edges = cv2.Canny(frame_gray, 50, 150)
        slide_edges = cv2.Canny(slide_gray, 50, 150)

        # Calcular similitud de bordes
        diff = cv2.absdiff(frame_edges, slide_edges)
        total_pixels = frame_edges.size
        matching_pixels = total_pixels - np.count_nonzero(diff)

        similarity = matching_pixels / total_pixels if total_pixels > 0 else 0.0
        return similarity

    def _features_similarity(
        self, frame_gray: np.ndarray, slide_img: np.ndarray, slide_idx: int
    ) -> float:
        """Compara características usando ORB."""
        # Convertir slide a escala de grises si es necesario
        if len(slide_img.shape) == 3:
            slide_gray = cv2.cvtColor(slide_img, cv2.COLOR_BGR2GRAY)
        else:
            slide_gray = slide_img

        # Redimensionar si tienen tamaños diferentes
        if frame_gray.shape != slide_gray.shape:
            h, w = frame_gray.shape[:2]
            slide_gray = cv2.resize(slide_gray, (w, h), interpolation=cv2.INTER_LINEAR)

        # Detectar características del frame
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(frame_gray, None)

        # Usar características cacheadas si están disponibles
        if slide_idx in self.slide_features_cache:
            kp2, des2 = self.slide_features_cache[slide_idx]
        else:
            kp2, des2 = orb.detectAndCompute(slide_gray, None)
            self.slide_features_cache[slide_idx] = (kp2, des2)

        if des1 is None or des2 is None:
            return 0.0

        # Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Calcular ratio de correspondencias
        total_features = min(len(des1), len(des2))
        if total_features == 0:
            return 0.0

        match_ratio = len(matches) / total_features
        return match_ratio
