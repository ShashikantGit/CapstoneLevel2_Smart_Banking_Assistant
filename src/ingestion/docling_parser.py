# =============================================================================
# src/ingestion/docling_parser.py
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Windows / PyTorch configuration
# ---------------------------------------------------------------------------
# Prevent PyTorch/TorchInductor from attempting C++ compilation.
#
# This avoids:
#   torch._inductor.exc.InductorError:
#   InvalidCxxCompiler: Compiler: cl is not found.
#
# IMPORTANT:
# These must be set BEFORE importing Docling/PyTorch.
# ---------------------------------------------------------------------------

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import base64
import io

from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------

load_dotenv()


# =============================================================================
# OpenAI Vision
# =============================================================================

def _describe_image_with_openai(img_b64: str) -> str:
    """
    Generate a searchable description of an image using an OpenAI vision model.

    The generated description is used as the text content of an image chunk
    before creating the embedding.

    Returns an empty string if the vision call fails.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            "[docling_parser] WARNING: OPENAI_API_KEY is not configured. "
            "Skipping image description."
        )
        return ""

    vision_model = os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-5.5",
    )

    try:
        vision_llm = ChatOpenAI(
            model=vision_model,
            api_key=api_key,
        )

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for document search "
                        "indexing. Include chart titles, axis labels, legend "
                        "entries, key data points, trends, numbers, tables, "
                        "and any visible text. Be specific. The description "
                        "will be used by a banking RAG bot for retrieval."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:image/png;base64,{img_b64}"
                        )
                    },
                },
            ]
        )

        response = vision_llm.invoke([message])

        content = response.content

        if isinstance(content, list):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
            ).strip()

        return str(content).strip()

    except Exception as exc:
        print(
            "[docling_parser] WARNING: Image description failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return ""


# =============================================================================
# Metadata helper
# =============================================================================

def _build_metadata(
    content_type: str,
    element_type: str,
    section: str | None,
    page_number: int | None,
    source_file: str,
    position: dict | None,
    image_base64: str | None = None,
) -> dict:
    """
    Build metadata for a parsed document element.

    The metadata structure matches the columns expected by
    src.core.db.store_chunks().
    """

    return {
        "content_type": content_type,
        "element_type": element_type,
        "section": section,
        "page_number": page_number,
        "source_file": source_file,
        "position": position,
        "image_base64": image_base64,
    }


# =============================================================================
# Table extraction
# =============================================================================

def _extract_table_text(node, doc) -> str:
    """
    Convert a Docling table into searchable plain text.

    Preferred:
        export_to_dataframe()

    Fallback:
        export_to_html()

    Final fallback:
        node.text
    """

    table_text = ""

    # -------------------------------------------------------------------------
    # Preferred method: DataFrame
    # -------------------------------------------------------------------------

    if hasattr(node, "export_to_dataframe"):
        try:
            dataframe = node.export_to_dataframe()

            if dataframe is not None and not dataframe.empty:

                rows_text: list[str] = []

                headers = [
                    str(column).strip()
                    for column in dataframe.columns
                ]

                for _, row in dataframe.iterrows():

                    pairs = []

                    for header, value in zip(headers, row):

                        value_text = str(value).strip()

                        if value_text not in (
                            "",
                            "nan",
                            "None",
                        ):
                            pairs.append(
                                f"{header}: {value_text}"
                            )

                    if pairs:
                        rows_text.append(
                            " | ".join(pairs)
                        )

                table_text = "\n".join(rows_text)

        except Exception as exc:
            print(
                "[docling_parser] WARNING: "
                f"DataFrame table extraction failed: {exc}"
            )

    # -------------------------------------------------------------------------
    # Fallback: HTML
    # -------------------------------------------------------------------------

    if not table_text and hasattr(node, "export_to_html"):

        try:
            import re

            raw_html = node.export_to_html(doc)

            table_text = re.sub(
                r"<[^>]+>",
                " ",
                raw_html or "",
            )

            table_text = re.sub(
                r"\s+",
                " ",
                table_text,
            ).strip()

        except Exception as exc:
            print(
                "[docling_parser] WARNING: "
                f"HTML table extraction failed: {exc}"
            )

    # -------------------------------------------------------------------------
    # Final fallback: raw text
    # -------------------------------------------------------------------------

    if not table_text:

        table_text = getattr(
            node,
            "text",
            "",
        )

    return table_text.strip()


# =============================================================================
# Image extraction
# =============================================================================

def _extract_image_base64(node, doc) -> str | None:
    """
    Extract a Docling picture/chart as PNG base64.

    First tries:
        node.get_image(doc)

    Then falls back to:
        node.image.pil_image
    """

    try:

        # ---------------------------------------------------------------------
        # Preferred Docling method
        # ---------------------------------------------------------------------

        if hasattr(node, "get_image"):

            pil_image = node.get_image(doc)

            if pil_image:

                buffer = io.BytesIO()

                pil_image.save(
                    buffer,
                    format="PNG",
                )

                return base64.b64encode(
                    buffer.getvalue()
                ).decode()

        # ---------------------------------------------------------------------
        # Fallback for older Docling versions
        # ---------------------------------------------------------------------

        if (
            hasattr(node, "image")
            and node.image
        ):

            pil_image = getattr(
                node.image,
                "pil_image",
                None,
            )

            if pil_image:

                buffer = io.BytesIO()

                pil_image.save(
                    buffer,
                    format="PNG",
                )

                return base64.b64encode(
                    buffer.getvalue()
                ).decode()

    except Exception as exc:

        print(
            "[docling_parser] WARNING: "
            f"Image extraction failed: {exc}"
        )

    return None


# =============================================================================
# PDF parser
# =============================================================================

def parse_document(file_path: str) -> list[dict]:
    """
    Parse a PDF using Docling.

    Returns a list of dictionaries:

        {
            "content": "...",
            "content_type": "text" | "table" | "image",
            "metadata": {
                "content_type": "...",
                "element_type": "...",
                "section": "...",
                "page_number": ...,
                "source_file": "...",
                "position": {...},
                "image_base64": "..."
            }
        }

    The returned structure is compatible with src.core.db.store_chunks().
    """

    # =========================================================================
    # Validate input file
    # =========================================================================

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    # =========================================================================
    # Configure Docling
    # =========================================================================

    print(
        f"[docling_parser] Processing PDF: {file_path}"
    )

    print(
        "[docling_parser] Using CPU accelerator."
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU,
        ),
    )

    # =========================================================================
    # Create converter
    # =========================================================================

    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        },
    )

    # =========================================================================
    # Convert PDF
    # =========================================================================

    print(
        "[docling_parser] Starting Docling conversion..."
    )

    result = converter.convert(file_path)

    doc = result.document

    print(
        "[docling_parser] Docling conversion completed."
    )

    # =========================================================================
    # Initialize parsing state
    # =========================================================================

    parsed_chunks: list[dict] = []

    current_section: str | None = None

    source_file = os.path.basename(
        file_path
    )

    # =========================================================================
    # Iterate through Docling elements
    # =========================================================================

    for item in doc.iterate_items():

        # ---------------------------------------------------------------------
        # Docling >= 2.x generally returns:
        #
        #     (node, level)
        #
        # Keep compatibility with versions returning bare nodes.
        # ---------------------------------------------------------------------

        if isinstance(item, tuple):

            node = item[0]

        else:

            node = item

        # ---------------------------------------------------------------------
        # Get element label
        # ---------------------------------------------------------------------

        label = str(
            getattr(
                node,
                "label",
                "",
            )
        ).lower()

        # ---------------------------------------------------------------------
        # Skip repeating page headers and footers
        # ---------------------------------------------------------------------

        if label in (
            "page_header",
            "page_footer",
        ):
            continue

        # =========================================================================
        # Page number and position
        # =========================================================================

        provenance = getattr(
            node,
            "prov",
            None,
        )

        page_number = None

        position = None

        if provenance:

            try:

                page_number = provenance[0].page_no

            except Exception:

                page_number = None

            try:

                bbox = provenance[0].bbox

                if bbox is not None:

                    position = {
                        "l": bbox.l,
                        "t": bbox.t,
                        "r": bbox.r,
                        "b": bbox.b,
                    }

            except Exception:

                position = None

        # =========================================================================
        # Section heading / title
        # =========================================================================

        if (
            "section_header" in label
            or label == "title"
        ):

            text = getattr(
                node,
                "text",
                "",
            )

            text = text.strip()

            if text:

                current_section = text

                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": _build_metadata(
                            content_type="text",
                            element_type=label,
                            section=current_section,
                            page_number=page_number,
                            source_file=source_file,
                            position=position,
                        ),
                    }
                )

            continue

        # =========================================================================
        # Table
        # =========================================================================

        if "table" in label:

            table_text = _extract_table_text(
                node,
                doc,
            )

            if table_text:

                parsed_chunks.append(
                    {
                        "content": table_text,
                        "content_type": "table",
                        "metadata": _build_metadata(
                            content_type="table",
                            element_type="table",
                            section=current_section,
                            page_number=page_number,
                            source_file=source_file,
                            position=position,
                        ),
                    }
                )

            continue

        # =========================================================================
        # Picture / Figure / Chart
        # =========================================================================

        if (
            "picture" in label
            or "figure" in label
            or label == "chart"
        ):

            # ---------------------------------------------------------------------
            # Extract image
            # ---------------------------------------------------------------------

            image_base64 = _extract_image_base64(
                node,
                doc,
            )

            # ---------------------------------------------------------------------
            # Extract caption
            # ---------------------------------------------------------------------

            caption = getattr(
                node,
                "text",
                "",
            ) or ""

            caption = caption.strip()

            # ---------------------------------------------------------------------
            # Generate searchable image description
            # ---------------------------------------------------------------------

            if image_base64:

                print(
                    "[docling_parser] Generating image description "
                    f"for page {page_number}..."
                )

                description = (
                    _describe_image_with_openai(
                        image_base64
                    )
                )

            else:

                description = ""

            # ---------------------------------------------------------------------
            # Determine searchable image content
            # ---------------------------------------------------------------------

            if description:

                content = description

            elif caption:

                content = caption

            else:

                content = (
                    f"[Image on page {page_number}]"
                )

            # ---------------------------------------------------------------------
            # Store image chunk
            # ---------------------------------------------------------------------

            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": _build_metadata(
                        content_type="image",
                        element_type=label,
                        section=current_section,
                        page_number=page_number,
                        source_file=source_file,
                        position=position,
                        image_base64=image_base64,
                    ),
                }
            )

            continue

        # =========================================================================
        # Normal text
        # =========================================================================

        text = getattr(
            node,
            "text",
            "",
        )

        if text:

            text = text.strip()

            if text:

                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": _build_metadata(
                            content_type="text",
                            element_type=label,
                            section=current_section,
                            page_number=page_number,
                            source_file=source_file,
                            position=position,
                        ),
                    }
                )

    # =========================================================================
    # Final result
    # =========================================================================

    print(
        "[docling_parser] Parsed elements: "
        f"{len(parsed_chunks)}"
    )

    return parsed_chunks