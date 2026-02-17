import tiktoken
import json
from typing import List, Dict, Any

# Initialize tokenizer (standard for GPT-4/Gemini approximations)
ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))

def process_ehr_context(timeline: Dict, max_context: int, output_file: Any = None) -> List[Dict]:
    """
    Splits a patient timeline into windows if it exceeds max_context.
    For WTTS, we usually want the WHOLE timeline, so we try to fit it all.
    """
    # In the WTTS pipeline, 'timeline' might be the raw notes list or the WTTS string
    # Here we assume it's the raw dictionary with 'notes'
    
    full_text = json.dumps(timeline) # specific logic depends on your data structure
    tokens = count_tokens(full_text)
    
    # Simple pass-through if it fits
    # (The complicated TIMER logic splits long histories, but for this competition
    # we assume we are summarizing the extraction first, so we return 1 window)
    return [{
        "window_index": 0,
        "window_token_count": tokens,
        "window_percent_full": tokens / max_context,
        "start_date": timeline.get('admission_time', 'Unknown'),
        "end_date": timeline.get('discharge_time', 'Unknown'),
        "data": timeline # Pass the full data through
    }]

def create_prompt_from_timeline(window: Dict, template: str) -> str:
    """
    Injects the data into the prompt.
    For WTTS, the 'template' is constructed in the Builder, 
    so this is a helper to finalize the string.
    """
    # This function is called by predictor.py's 'general' method
    # We can just return the prompt if it was pre-formatted, 
    # or format it here if 'window' contains the raw text.
    return template.format(input=window.get('data'))

def create_persona_prompt_from_timeline(window: Dict, template: str, person_id: str):
    # Stub for predictor.py compatibility if using 'persona' mode
    prompt = create_prompt_from_timeline(window, template)
    return prompt, "General", 0