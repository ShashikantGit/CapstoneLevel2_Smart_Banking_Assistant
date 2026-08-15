import base64
import io
import os
import re

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


load_dotenv()


# =============================================================================
# OPENAI VISION
# =============================================================================

def _describe_image_with_openai(img_b64: str) -> str:
    """
    Generate a searchable text description for an image, figure or chart.

    The generated description is stored as the chunk content and is later
    embedded into pgvector. This allows natural-language queries to retrieve
    image/chart content.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            "[docling_parser] WARNING: OPENAI_API_KEY is not configured."
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
            temperature=0,
        )

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for document search "
                        "indexing. Include chart titles, axis labels, legend "
                        "entries, important values, numbers, trends, headings, "
                        "labels and visible text. Be accurate and concise. "
                        "This description will be used by a banking RAG system."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    },
                },
            ]
        )

        response = vision_llm.invoke([message])

        content = response.content

        if isinstance(content, list):
            parts = []

            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text = part.get("text", "")

                        if text:
                            parts.append(text)

            return " ".join(parts).strip()

        return str(content).strip()

    except Exception as exc:
        print(
            "[docling_parser] Vision description failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return ""


# =============================================================================
# IMAGE TO BASE64
# =============================================================================

def _image_to_base64(pil_image) -> str | None:
    """
    Convert a PIL image to a base64 encoded PNG string.
    """

    if pil_image is None:
        return None

    try:
        buffer = io.BytesIO()

        pil_image.save(
            buffer,
            format="PNG",
        )

        return base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

    except Exception as exc:
        print(
            "[docling_parser] Image conversion failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return None


# =============================================================================
# TABLE TO TEXT
# =============================================================================

def _table_to_text(table, doc) -> str:
    """
    Convert a Docling table to searchable plain text.

    Newer Docling versions require the `doc` argument when calling
    export_to_dataframe(), so we explicitly pass doc here.
    """

    # -------------------------------------------------------------------------
    # Preferred: DataFrame
    # -------------------------------------------------------------------------

    if hasattr(table, "export_to_dataframe"):

        try:
            df = table.export_to_dataframe(doc=doc)

            if df is not None and not df.empty:

                rows_text: list[str] = []

                headers = [
                    str(column).strip()
                    for column in df.columns
                ]

                for _, row in df.iterrows():

                    pairs: list[str] = []

                    for header, value in zip(
                        headers,
                        row,
                    ):

                        value_string = str(value).strip()

                        if value_string in (
                            "",
                            "nan",
                            "None",
                            "NaN",
                        ):
                            continue

                        pairs.append(
                            f"{header}: {value_string}"
                        )

                    if pairs:
                        rows_text.append(
                            " | ".join(pairs)
                        )

                if rows_text:
                    return "\n".join(rows_text)

        except TypeError:
            # Compatibility fallback for older Docling versions.
            try:
                df = table.export_to_dataframe()

                if df is not None and not df.empty:

                    rows_text = []

                    headers = [
                        str(column).strip()
                        for column in df.columns
                    ]

                    for _, row in df.iterrows():

                        pairs = []

                        for header, value in zip(
                            headers,
                            row,
                        ):

                            value_string = str(value).strip()

                            if value_string in (
                                "",
                                "nan",
                                "None",
                                "NaN",
                            ):
                                continue

                            pairs.append(
                                f"{header}: {value_string}"
                            )

                        if pairs:
                            rows_text.append(
                                " | ".join(pairs)
                            )

                    if rows_text:
                        return "\n".join(rows_text)

            except Exception as exc:
                print(
                    "[docling_parser] "
                    f"Legacy DataFrame extraction failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        except Exception as exc:

            print(
                "[docling_parser] "
                f"DataFrame table extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # -------------------------------------------------------------------------
    # Fallback: HTML
    # -------------------------------------------------------------------------

    if hasattr(table, "export_to_html"):

        try:
            raw_html = table.export_to_html(doc)

            text = re.sub(
                r"<[^>]+>",
                " ",
                raw_html or "",
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            if text:
                return text

        except Exception as exc:

            print(
                "[docling_parser] "
                f"HTML table extraction failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # -------------------------------------------------------------------------
    # Final fallback: text attribute
    # -------------------------------------------------------------------------

    return str(
        getattr(table, "text", "") or ""
    ).strip()


# =============================================================================
# DOCUMENT PARSER
# =============================================================================

def parse_document(file_path: str) -> list[dict]:
    """
    Parse a PDF using Docling.

    Returns:

        [
            {
                "content": "...",
                "content_type": "text|table|image",
                "metadata": {
                    "content_type": "...",
                    "element_type": "...",
                    "section": "...",
                    "page_number": 1,
                    "source_file": "...",
                    "position": {...},
                    "image_base64": "..."
                }
            }
        ]
    """

    # -------------------------------------------------------------------------
    # Validate file
    # -------------------------------------------------------------------------

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    print(
        f"[docling_parser] Processing PDF: {file_path}"
    )

    # -------------------------------------------------------------------------
    # Docling configuration
    #
    # OCR is disabled because your KB PDF contains extractable PDF content
    # and RapidOCR previously returned:
    #
    #     RapidOCR returned empty result!
    #
    # This avoids unnecessary OCR processing.
    # -------------------------------------------------------------------------

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        generate_picture_images=True,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU
        ),
    )

    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        },
    )

    # -------------------------------------------------------------------------
    # Convert PDF
    # -------------------------------------------------------------------------

    print(
        "[docling_parser] Starting Docling conversion..."
    )

    result = converter.convert(file_path)

    doc = result.document

    print(
        "[docling_parser] Docling conversion completed."
    )

    # -------------------------------------------------------------------------
    # Chunk storage
    # -------------------------------------------------------------------------

    parsed_chunks: list[dict] = []

    current_section: str | None = None

    source_file = os.path.basename(
        file_path
    )

    # -------------------------------------------------------------------------
    # Metadata helper
    # -------------------------------------------------------------------------

    def make_metadata(
        content_type: str,
        element_type: str,
        page_number,
        position,
        image_base64=None,
    ) -> dict:

        return {
            "content_type": content_type,
            "element_type": element_type,
            "section": current_section,
            "page_number": page_number,
            "source_file": source_file,
            "position": position,
            "image_base64": image_base64,
        }

    # -------------------------------------------------------------------------
    # Iterate through Docling document
    # -------------------------------------------------------------------------

    for item in doc.iterate_items():

        # Docling versions may return:
        #
        # (node, level)
        #
        # or a bare node.

        if isinstance(item, tuple):
            node = item[0]
        else:
            node = item

        label = str(
            getattr(
                node,
                "label",
                "",
            )
        ).lower()

        # ---------------------------------------------------------------------
        # Ignore repeating page headers and footers
        # ---------------------------------------------------------------------

        if label in (
            "page_header",
            "page_footer",
        ):
            continue

        # ---------------------------------------------------------------------
        # Page information
        # ---------------------------------------------------------------------

        prov = getattr(
            node,
            "prov",
            None,
        )

        page_number = None
        position = None

        if prov:

            try:
                page_number = prov[0].page_no
            except Exception:
                page_number = None

            try:

                bbox = prov[0].bbox

                if bbox is not None:

                    position = {
                        "l": bbox.l,
                        "t": bbox.t,
                        "r": bbox.r,
                        "b": bbox.b,
                    }

            except Exception:
                position = None

        # =====================================================================
        # SECTION HEADER / TITLE
        # =====================================================================

        if (
            "section_header" in label
            or label == "title"
        ):

            text = str(
                getattr(
                    node,
                    "text",
                    "",
                ) or ""
            ).strip()

            if text:

                current_section = text

                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": make_metadata(
                            content_type="text",
                            element_type=label,
                            page_number=page_number,
                            position=position,
                        ),
                    }
                )

            continue

        # =====================================================================
        # TABLE
        # =====================================================================

        if "table" in label:

            table_text = _table_to_text(
                node,
                doc,
            )

            if table_text:

                parsed_chunks.append(
                    {
                        "content": table_text,
                        "content_type": "table",
                        "metadata": make_metadata(
                            content_type="table",
                            element_type=label,
                            page_number=page_number,
                            position=position,
                        ),
                    }
                )

            continue

        # =====================================================================
        # PICTURES / FIGURES / CHARTS
        # =====================================================================

        if (
            "picture" in label
            or "figure" in label
            or label == "chart"
        ):

            image_base64 = None

            caption = str(
                getattr(
                    node,
                    "text",
                    "",
                ) or ""
            ).strip()

            # -----------------------------------------------------------------
            # Preferred image extraction
            # -----------------------------------------------------------------

            try:

                if hasattr(
                    node,
                    "get_image",
                ):

                    pil_image = node.get_image(
                        doc
                    )

                    image_base64 = _image_to_base64(
                        pil_image
                    )

            except Exception as exc:

                print(
                    "[docling_parser] "
                    f"get_image failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            # -----------------------------------------------------------------
            # Older Docling fallback
            # -----------------------------------------------------------------

            if image_base64 is None:

                try:

                    image = getattr(
                        node,
                        "image",
                        None,
                    )

                    if image:

                        pil_image = getattr(
                            image,
                            "pil_image",
                            None,
                        )

                        image_base64 = _image_to_base64(
                            pil_image
                        )

                except Exception:
                    pass

            # -----------------------------------------------------------------
            # Vision description
            # -----------------------------------------------------------------

            if image_base64:

                print(
                    "[docling_parser] "
                    f"Generating image description "
                    f"for page {page_number}..."
                )

                description = (
                    _describe_image_with_openai(
                        image_base64
                    )
                )

                if description:

                    content = description

                elif caption:

                    content = caption

                else:

                    content = (
                        f"[Image on page "
                        f"{page_number}]"
                    )

            else:

                content = (
                    caption
                    or f"[Image on page {page_number}]"
                )

            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": make_metadata(
                        content_type="image",
                        element_type=label,
                        page_number=page_number,
                        position=position,
                        image_base64=image_base64,
                    ),
                }
            )

            continue

        # =====================================================================
        # NORMAL TEXT
        # =====================================================================

        text = str(
            getattr(
                node,
                "text",
                "",
            ) or ""
        ).strip()

        if text:

            parsed_chunks.append(
                {
                    "content": text,
                    "content_type": "text",
                    "metadata": make_metadata(
                        content_type="text",
                        element_type=label,
                        page_number=page_number,
                        position=position,
                    ),
                }
            )

    # =========================================================================
    # Extraction summary
    # =========================================================================

    text_count = sum(
        1
        for chunk in parsed_chunks
        if chunk["content_type"] == "text"
    )

    table_count = sum(
        1
        for chunk in parsed_chunks
        if chunk["content_type"] == "table"
    )

    image_count = sum(
        1
        for chunk in parsed_chunks
        if chunk["content_type"] == "image"
    )

    print()
    print(
        "[docling_parser] Extraction summary:"
    )

    print(
        f"  Total chunks : {len(parsed_chunks)}"
    )

    print(
        f"  Text chunks  : {text_count}"
    )

    print(
        f"  Table chunks : {table_count}"
    )

    print(
        f"  Image chunks : {image_count}"
    )

    if not parsed_chunks:

        print(
            "[docling_parser] WARNING: "
            "No content was extracted from the PDF."
        )

    return parsed_chunks


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python -m src.ingestion.docling_parser "
            "data\\KB_Smart_Banking.pdf"
        )

        raise SystemExit(1)

    pdf_path = sys.argv[1]

    chunks = parse_document(
        pdf_path
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"Extracted {len(chunks)} chunks"
    )

    print(
        "=" * 80
    )

    for index, chunk in enumerate(
        chunks[:10],
        start=1,
    ):

        print()
        print(
            f"CHUNK {index}"
        )

        print(
            f"Type: {chunk['content_type']}"
        )

        print(
            f"Page: "
            f"{chunk['metadata'].get('page_number')}"
        )

        print(
            f"Section: "
            f"{chunk['metadata'].get('section')}"
        )

        print(
            f"Element: "
            f"{chunk['metadata'].get('element_type')}"
        )

        print(
            f"Content: "
            f"{chunk['content'][:500]}"
        )