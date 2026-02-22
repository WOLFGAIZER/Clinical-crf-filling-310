"""
Gradio Interface for Clinical CRF Filling Pipeline
Deployable on HuggingFace Spaces (GPU or CPU)

Usage:
  Local:  python app.py
  HF:     Set as main file in Space settings
"""

import os
import json
import asyncio
import tempfile
import pandas as pd
import gradio as gr

# Pipeline imports
from src.preprocess.wtts_builder import WTTSBuilder
from src.utils.data_loader import DataLoader

import google.generativeai as genai

# ---------------------------------------------------------------------------
#  CONFIG — reads from environment variables (set via HF Secrets or .env)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

# RAG imports (optional — only if dependencies installed)
RAG_AVAILABLE = False
try:
    from src.rag.embedder import WTTSEmbedder
    from src.rag.rag_pipeline import RAGCRFExtractor
    RAG_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
#  PROMPTS (same as main.py)
# ---------------------------------------------------------------------------
SKELETON_PROMPT = """
You are a Clinical Data Specialist.
Convert the Weighted Time Series (WTTS) below into a "Clinical Chronology Skeleton".

INPUT (WTTS):
{wtts_string}

INSTRUCTIONS:
1. Create a strict chronological timeline (Admission to Discharge).
2. IMPORTANT: You MUST retain the [S_xx] ID for every event you list.
3. Filter out "Routine" (Weight 0.1) events unless they indicate a status change.
4. Keep exact values (e.g., "BP 90/60", "Temp 102.5").

OUTPUT FORMAT:
[Date] [S_xx]: Event details
[Date] [S_xx]: Event details
...
"""

EXTRACTION_PROMPT = """
You are a Clinical Coding Expert.
Review the Patient Skeleton and the Valid Options for the requested items.

PATIENT SKELETON:
{skeleton}

TASK:
For each Clinical Item listed below, determine the value AND the supporting Sentence ID.
1. **Value**: Must come strictly from the "Valid Options" provided.
2. **Evidence**: Must be the specific [S_xx] ID from the skeleton that proves the value.

ITEMS TO EXTRACT & THEIR OPTIONS:
{chunk_schema_json}

OUTPUT FORMAT (JSON Object):
{{
  "item_name": {{
    "value": "Selected Option",
    "evidence": "S_xx",
    "reasoning": "Brief explanation"
  }},
  ...
}}
"""


# ---------------------------------------------------------------------------
#  LLM call
# ---------------------------------------------------------------------------
async def generate_async(prompt, model, max_retries=3, initial_delay=1):
    loop = asyncio.get_event_loop()
    for attempt in range(max_retries):
        try:
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    contents=prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json"
                    ),
                ),
            )
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                continue
        except Exception as e:
            error_message = str(e)
            if "429" in error_message or "500" in error_message:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    return {"error": f"Max retries reached - {error_message}"}
            else:
                return {"error": error_message}
    return {"error": "Failed to generate valid JSON"}


def chunk_data(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


# ---------------------------------------------------------------------------
#  Core processing function
# ---------------------------------------------------------------------------
def process_clinical_text(
    clinical_text: str,
    api_key: str,
    model_name: str,
    use_rag: bool,
    admission_time: str,
    discharge_time: str,
    progress=gr.Progress(),
):
    """Main Gradio handler — processes clinical text and returns CRF predictions."""

    if not clinical_text.strip():
        return "⚠️ Please paste clinical text or upload a file.", "", ""

    if not api_key.strip():
        return "⚠️ Please provide a Google API key.", "", ""

    # Configure Gemini
    genai.configure(api_key=api_key.strip())
    model = genai.GenerativeModel(model_name)

    # Build a synthetic patient data dict
    patient_data = {
        "document_id": "gradio_patient",
        "admission_time": admission_time or "2026-01-01",
        "discharge_time": discharge_time or "2026-01-15",
        "notes": [
            {
                "timestamp": admission_time or "2026-01-01",
                "text": clinical_text,
                "source": "user_input",
            }
        ],
    }

    builder = WTTSBuilder()

    progress(0.1, desc="Building WTTS tuples...")
    wtts_string = builder.build_wtts_string(patient_data)

    if not wtts_string.strip():
        return "⚠️ Could not extract any events from the text.", "", ""

    # ---- RAG Pipeline ----
    if use_rag and RAG_AVAILABLE:
        progress(0.3, desc="RAG: Embedding tuples...")
        embedder = WTTSEmbedder(model_name="all-MiniLM-L6-v2", device="cpu")
        extractor = RAGCRFExtractor(
            embedder=embedder,
            generate_fn=generate_async,
            top_k=15,
        )

        # We don't have target_items/valid_options from the UI,
        # so use a default set of common CRF items
        target_items = _get_default_target_items()
        valid_options = _get_default_valid_options()

        semaphore = asyncio.Semaphore(3)

        progress(0.5, desc="RAG: Retrieving & extracting...")
        result = asyncio.run(
            extractor.extract_patient(
                patient_data, builder, target_items, valid_options, semaphore, model
            )
        )

        if result and result.get("predictions"):
            predictions = result["predictions"]
        else:
            return "⚠️ RAG extraction returned no results.", wtts_string, ""

    # ---- Original Two-Pass Pipeline ----
    else:
        progress(0.3, desc="Pass 1: Generating skeleton...")
        skeleton_input = SKELETON_PROMPT.format(wtts_string=wtts_string)
        skeleton_resp = asyncio.run(generate_async(skeleton_input, model))

        skeleton_text = str(skeleton_resp)
        if isinstance(skeleton_resp, dict):
            skeleton_text = json.dumps(skeleton_resp, indent=2)

        target_items = _get_default_target_items()
        valid_options = _get_default_valid_options()

        progress(0.6, desc="Pass 2: Extracting CRF items...")
        predictions = {}
        item_chunks = list(chunk_data(target_items, 10))

        for i, chunk_items in enumerate(item_chunks):
            progress(0.6 + 0.3 * (i / max(len(item_chunks), 1)),
                     desc=f"Extracting batch {i+1}/{len(item_chunks)}...")

            chunk_schema = {
                item: valid_options.get(item, ["y", "n", "unknown"])
                for item in chunk_items
            }
            extract_input = EXTRACTION_PROMPT.format(
                skeleton=skeleton_text,
                chunk_schema_json=json.dumps(chunk_schema),
            )
            chunk_resp = asyncio.run(generate_async(extract_input, model))

            if isinstance(chunk_resp, dict) and "error" not in chunk_resp:
                predictions.update(chunk_resp)

    progress(0.95, desc="Formatting results...")

    # Format predictions for display
    results_md = _format_predictions_markdown(predictions)
    predictions_json = json.dumps(predictions, indent=2)

    progress(1.0, desc="Done!")
    return results_md, wtts_string, predictions_json


# ---------------------------------------------------------------------------
#  File upload handler
# ---------------------------------------------------------------------------
def load_from_file(file):
    """Read uploaded file (txt or parquet) and return the text content."""
    if file is None:
        return ""

    filepath = file.name if hasattr(file, "name") else str(file)

    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        text_col = next(
            (c for c in df.columns if c.lower() in ["clinical_note", "text", "body"]),
            df.columns[0],
        )
        return "\n\n---\n\n".join(df[text_col].dropna().astype(str).tolist())

    elif filepath.endswith(".jsonl"):
        texts = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                if "text" in rec:
                    texts.append(rec["text"])
        return "\n\n---\n\n".join(texts) if texts else ""

    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
#  Default CRF items (common dyspnea CRF items from the challenge)
# ---------------------------------------------------------------------------
def _get_default_target_items():
    return [
        "chronic pulmonary disease", "chronic respiratory failure",
        "chronic cardiac failure", "chronic renal failure",
        "presence of dyspnea", "improvement of dyspnea",
        "heart rate", "blood pressure", "body temperature",
        "respiratory rate", "spo2", "level of consciousness",
        "hemoglobin", "platelets", "leukocytes", "c-reactive protein",
        "creatinine", "troponin", "d-dimer",
        "ecg, any abnormality", "chest rx, any abnormalities",
        "brain ct scan, any abnormality",
        "administration of diuretics", "administration of steroids",
        "administration of bronchodilators",
        "administration of oxygen/ventilation",
        "heart failure", "pneumonia", "copd exacerbation",
        "respiratory failure", "pulmonary embolism",
        "acute coronary syndrome", "arrhythmia",
    ]


def _get_default_valid_options():
    return {item: ["y", "n", "unknown"] for item in _get_default_target_items()}


# ---------------------------------------------------------------------------
#  Format predictions as Markdown table
# ---------------------------------------------------------------------------
def _format_predictions_markdown(predictions):
    if not predictions:
        return "No predictions generated."

    lines = [
        "## 📋 CRF Predictions\n",
        "| # | CRF Item | Value | Evidence | Reasoning |",
        "|---|----------|-------|----------|-----------|",
    ]

    for i, (item, val) in enumerate(predictions.items(), 1):
        if isinstance(val, dict):
            value = val.get("value", "—")
            evidence = val.get("evidence", "—")
            reasoning = val.get("reasoning", "—")
        else:
            value = str(val)
            evidence = "—"
            reasoning = "—"

        # Color code the value
        if value.lower() == "y":
            value = "✅ Yes"
        elif value.lower() == "n":
            value = "❌ No"
        elif value.lower() == "unknown":
            value = "❓ Unknown"

        lines.append(f"| {i} | {item} | {value} | {evidence} | {reasoning} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Gradio UI
# ---------------------------------------------------------------------------
def create_app():
    with gr.Blocks(
        title="Clinical CRF Filling — RAG Pipeline",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
        css="""
        .main-header { text-align: center; margin-bottom: 1rem; }
        .results-box { min-height: 300px; }
        footer { display: none !important; }
        """,
    ) as app:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🏥 Clinical CRF Filling</h1>
            <p style="color: #666; font-size: 1.1em;">
                RAG-Enhanced Pipeline for CL4Health 2026 Challenge
            </p>
        </div>
        """)

        with gr.Row():
            # ---- Left column: Input ----
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")

                api_key_input = gr.Textbox(
                    label="Google API Key",
                    type="password",
                    placeholder="Enter your Gemini API key...",
                    value=API_KEY,
                )

                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-2.0-flash",
                    ],
                    value=DEFAULT_MODEL,
                )

                use_rag_checkbox = gr.Checkbox(
                    label="🔍 Use RAG Pipeline",
                    value=RAG_AVAILABLE,
                    info="Retrieves relevant tuples per CRF item (requires sentence-transformers + faiss)",
                    interactive=RAG_AVAILABLE,
                )

                with gr.Row():
                    admission_input = gr.Textbox(
                        label="Admission Time",
                        placeholder="2026-01-01",
                        value="2026-01-01",
                    )
                    discharge_input = gr.Textbox(
                        label="Discharge Time",
                        placeholder="2026-01-15",
                        value="2026-01-15",
                    )

                clinical_text_input = gr.Textbox(
                    label="Clinical Notes",
                    placeholder="Paste clinical text here...",
                    lines=12,
                    max_lines=30,
                )

                file_upload = gr.File(
                    label="Or Upload File (.txt, .parquet, .jsonl)",
                    file_types=[".txt", ".parquet", ".jsonl"],
                )

                submit_btn = gr.Button(
                    "🚀 Extract CRF Items",
                    variant="primary",
                    size="lg",
                )

            # ---- Right column: Output ----
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Results")

                results_output = gr.Markdown(
                    label="CRF Predictions",
                    elem_classes=["results-box"],
                )

                with gr.Accordion("🔧 WTTS Tuples (Debug)", open=False):
                    wtts_output = gr.Textbox(
                        label="Generated WTTS String",
                        lines=8,
                        interactive=False,
                    )

                with gr.Accordion("📦 Raw JSON Output", open=False):
                    json_output = gr.Code(
                        label="Predictions JSON",
                        language="json",
                    )

        # ---- Event handlers ----
        file_upload.change(
            fn=load_from_file,
            inputs=[file_upload],
            outputs=[clinical_text_input],
        )

        submit_btn.click(
            fn=process_clinical_text,
            inputs=[
                clinical_text_input,
                api_key_input,
                model_dropdown,
                use_rag_checkbox,
                admission_input,
                discharge_input,
            ],
            outputs=[results_output, wtts_output, json_output],
        )

        # Footer
        gr.Markdown("""
        ---
        <center>
        <small>
        CL4Health 2026 • CRF Filling Challenge • MIMIC-III Dataset<br>
        Built with WTTS + RAG Pipeline
        </small>
        </center>
        """)

    return app


# ---------------------------------------------------------------------------
#  Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
