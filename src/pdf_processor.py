"""
Módulo para procesar PDFs y extraer diapositivas como imágenes.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2


class PDFProcessor:
    """
    Procesa archivos PDF y extrae cada página como imagen.
    """

    def __init__(self, dpi: int = 200):
        """
        Inicializa el procesador de PDF.

        Args:
            dpi: Resolución para convertir PDF a imágenes
        """
        self.dpi = dpi

    def extract_slides(self, pdf_path: str, output_dir: Path = None) -> List[Tuple[int, np.ndarray, str]]:
        """
        Extrae todas las diapositivas del PDF como imágenes.

        Args:
            pdf_path: Ruta al archivo PDF
            output_dir: Directorio opcional donde guardar las imágenes

        Returns:
            Lista de tuplas (número_slide, imagen_numpy, ruta_archivo)
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image no está instalado. Instala con: pip install pdf2image\n"
                "También necesitas poppler: brew install poppler (macOS)"
            )

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        print(f"Extrayendo diapositivas de: {pdf_path}")

        # Convertir PDF a imágenes
        images = convert_from_path(str(pdf_path), dpi=self.dpi)

        slides = []
        for i, image in enumerate(images):
            # Convertir PIL Image a numpy array (BGR para OpenCV)
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                # Convertir RGB a BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Guardar si se especifica output_dir
            slide_path = None
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                slide_filename = f"{pdf_file.stem}_slide_{i:04d}.png"
                slide_path = output_dir / slide_filename
                cv2.imwrite(str(slide_path), img_array)

            slides.append((i, img_array, str(slide_path) if slide_path else None))

        print(f"✓ {len(slides)} diapositivas extraídas")
        return slides

    def extract_slides_batch(self, pdf_paths: List[str], output_dir: Path = None) -> List[Tuple[int, np.ndarray, str]]:
        """
        Extrae diapositivas de múltiples PDFs.

        Args:
            pdf_paths: Lista de rutas a archivos PDF
            output_dir: Directorio opcional donde guardar las imágenes

        Returns:
            Lista de tuplas (número_slide, imagen_numpy, ruta_archivo)
            Nota: El número de slide es único por PDF, pero se pueden repetir entre PDFs
        """
        all_slides = []
        for pdf_path in pdf_paths:
            slides = self.extract_slides(pdf_path, output_dir)
            all_slides.extend(slides)

        return all_slides

