from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import tempfile


class DocumentProcessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.converter = DocumentConverter()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            f"DocumentProcessor initialized with chunk_size={chunk_size}, overlap={chunk_overlap}"
        )

    def process_file(self, file_path: str, file_type: str) -> Dict[str, Any]:
        logger.info(f"Processing file: {file_path} (type: {file_type})")

        logger.debug(
            f"File path type: {type(file_path)}, path exists: {Path(file_path).exists()}"
        )

        try:
            result = self.converter.convert(source=file_path)
            logger.debug(
                f"Conversion result type: {type(result)}, has document: {hasattr(result, 'document')}"
            )

            document = result.document
            logger.debug(
                f"Document type: {type(document)}, has export_to_markdown: {hasattr(document, 'export_to_markdown')}"
            )

            full_text = document.export_to_markdown()

            chunker = HybridChunker(
                tokenizer="sentence-transformers/all-MiniLM-L6-v2",
                max_tokens=self.chunk_size,
                merge_peers=True,
            )

            chunks_iter = chunker.chunk(document)
            chunks = list(chunks_iter)

            logger.info(f"Created {len(chunks)} chunks from {file_path}")

            page_mapping = {}
            if hasattr(document, "pages"):
                for page_num, page in enumerate(document.pages, start=1):
                    try:
                        if hasattr(page, "export_to_markdown"):
                            page_text = page.export_to_markdown()
                            page_mapping[page_num] = page_text
                    except (AttributeError, TypeError) as e:
                        logger.warning(
                            f"Skipping page {page_num}: {type(page)}, error: {e}"
                        )
                        continue

            processed_chunks = []
            for idx, chunk in enumerate(chunks):
                chunk_text = chunk.text

                page_num = None
                if page_mapping:
                    for page_no, page_text in page_mapping.items():
                        if chunk_text in page_text:
                            page_num = page_no
                            break

                processed_chunks.append(
                    {
                        "chunk_index": idx,
                        "text": chunk_text,
                        "char_count": len(chunk_text),
                        "page_number": page_num,
                        "total_chunks": len(chunks),
                    }
                )

            return {
                "file_name": Path(file_path).name,
                "file_type": file_type,
                "total_chunks": len(chunks),
                "full_text": full_text,
                "chunks": processed_chunks,
                "page_mapping": page_mapping,
            }
        except Exception as e:
            logger.exception(f"Error in process_file: {e}")
            raise

    def process_uploaded_file(
        self, file_bytes: bytes, filename: str, file_type: str
    ) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file_type}"
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        try:
            result = self.process_file(tmp_path, file_type)
            result["file_name"] = filename
            return result
        finally:
            Path(tmp_path).unlink(missing_ok=True)
