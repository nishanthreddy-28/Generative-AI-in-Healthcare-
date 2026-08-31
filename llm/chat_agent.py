"""
Chat Agent
==========
Handles extracting the 8 Pima features from conversational free-text using Gemini.
"""

import os
import json
import logging
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# The required 8 features
REQUIRED_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

class ChatExtractionResponse(BaseModel):
    extracted_features: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Dictionary of successfully extracted features from the user's message and history. Keys must be exactly from the REQUIRED_FEATURES list."
    )
    missing_features: List[str] = Field(
        default_factory=list,
        description="List of required features that are still missing."
    )
    follow_up_message: Optional[str] = Field(
        None,
        description="A friendly, conversational chatbot message asking the user to provide the missing features. If no features are missing, this should be null."
    )

chat_schema = {
    "type": "OBJECT",
    "properties": {
        "extracted_features": {
            "type": "OBJECT",
            "description": "Dictionary of numeric features extracted from the user's latest message. ONLY include keys for variables that the user explicitly mentioned in this turn. Do NOT include any other keys."
        },
        "follow_up_message": {
            "type": "STRING"
        }
    },
    "required": ["extracted_features", "follow_up_message"]
}

from groq import Groq

def _get_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


import re

def process_chat_turn(
    user_message: str,
    current_features: Dict[str, float],
    history: List[Dict[str, str]]
) -> ChatExtractionResponse:
    """
    Extract features from the user_message, merging with current_features,
    while maintaining a natural conversational flow using the chat history.
    """
    # Clean HTML tags helper to format previous messages
    def clean_html(text: str) -> str:
        # Replace line breaks/paragraphs with standard spaces
        text = re.sub(r'</?(p|div|br|li|h\d)[^>]*>', ' \n', text)
        # Strip all other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Collapse multiple spaces/newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    # Determine if all features are collected or analysis was already completed
    all_collected = len(current_features) == len(REQUIRED_FEATURES)
    
    # Check history if it mentions successful risk assessment
    has_analysis_run = False
    model_prediction = None
    model_confidence = None
    for msg in history:
        content = msg.get("content", "")
        content_lower = content.lower()
        if "completed the analysis" in content_lower or "result-card" in content_lower:
            has_analysis_run = True
            
            # Extract prediction
            if "positive class" in content_lower:
                model_prediction = "positive"
            elif "negative class" in content_lower:
                model_prediction = "negative"
                
            # Extract confidence (e.g. "80.8%")
            conf_match = re.search(r'(\d+(?:\.\d+)?)\s*%', content)
            if conf_match:
                model_confidence = conf_match.group(1) + "%"

    # Build system prompt dynamically based on the mode
    system_instructions = [
        "You are Clinicagen, a medical AI assistant that provides educational explanations based on a diabetes machine-learning prediction model and trusted medical sources.",
        "",
        "### 1. Maintain Patient Context & Stored State",
        f"You must maintain a persistent patient context. Stored clinical features: {json.dumps(current_features)}",
        "Do NOT ask the user to identify which measurement a number belongs to if the conversation context already establishes it (e.g., if you ask for pregnancies and they answer '2', associate it with pregnancies).",
        "",
        "### 2. Never Lose Previously Collected Information",
        "Before answering ANY follow-up question (e.g., 'What should I eat?', 'What should I improve?', 'Is my glucose high?', 'Why was my result positive?'), check the stored features above.",
        "Answer specifically using the available patient information. Do NOT ask the user to repeat information that has already been collected (e.g., do not say 'Please provide your BMI' if BMI is already in the stored features).",
        "",
        "### 3. Distinguish Dataset Features From Clinical Measurements",
        "The machine-learning model's output is a prediction, not a medical diagnosis. Never state 'You have diabetes'. Instead, say 'The machine-learning model classified this profile as positive' or 'The model identified this profile as resembling examples classified as positive in its training data.'",
        "Never interpret model confidence as the probability that the patient medically has diabetes. Do NOT say 'You have a 75.7% chance of having diabetes'. Instead, say 'The model produced a 75.7% confidence score for its positive classification. This is a model output and is not equivalent to a clinical probability or diagnosis.'",
        "",
        "### 4. Safety & RAG Guidance",
        "This system is for educational/research purposes only. For concerning symptoms or metrics, recommend evaluation by a healthcare professional.",
        "Use retrieved RAG context when available to support medical explanations, prioritizing authoritative sources (CDC, NIDDK, NIH, ADA).",
        ""
    ]

    if has_analysis_run or all_collected:
        system_instructions.extend([
            "STATUS: All features have been successfully collected, and the ML prediction has been performed.",
            f"Model Prediction: {model_prediction or 'positive'}",
            f"Model Confidence: {model_confidence or '80.8%'}",
            "MODE: Conversational Q&A and insight mode.",
            "DUTIES:",
            "- Do NOT ask for pregnancies, glucose, BMI, or any of the 8 variables again.",
            "- Answer user Q&A questions directly, personalizing answers using the stored features and prediction details (e.g., explaining why glucose 155 or BMI 35.2 contributed to the positive model classification, or recommending diet/activity insights based on these values)."
        ])
    else:
        missing_before = [f for f in REQUIRED_FEATURES if current_features.get(f) is None]
        next_missing = missing_before[0] if missing_before else "None"
        system_instructions.extend([
            "STATUS: Collecting variables.",
            f"STILL MISSING: {', '.join(missing_before)}",
            f"NEXT MISSING FEATURE TO ASK: {next_missing}",
            "MODE: Extraction and collection mode.",
            "DUTIES:",
            "- Extract any new features mentioned in the user's latest message.",
            "- Keep a friendly, conversational flow when collecting data, following this sequence:",
            "  1. Pregnancies",
            "  2. Glucose",
            "  3. Blood Pressure",
            "  4. Skin Thickness",
            "  5. Insulin",
            "  6. BMI",
            "  7. Diabetes Pedigree Function",
            "  8. Age",
            "- If a value has never been provided, keep it as null. Never guess or infer a missing measurement.",
            f"- Do NOT simulate or output any machine-learning model prediction results, classification labels (e.g. positive/negative), or confidence percentages in your follow_up_message if any variables are still missing. You MUST ask the user to provide the next missing feature: {next_missing}.",
            "- After extracting a variable, provide a brief, easy-to-understand educational explanation of what that measurement represents and why it may be relevant to diabetes prediction, then prompt for the next missing variable in the sequence.",
            "- If all variables are complete, confirm the collected values and transition to prediction."
        ])

    system_instructions.extend([
        "\nYou MUST return a JSON object matching this schema exactly:",
        json.dumps(chat_schema)
    ])

    system_prompt = "\n".join(system_instructions)

    # Build messages list
    messages = [{"role": "system", "content": system_prompt}]
    
    # Map last 10 messages from history to keep context windows reasonable
    for msg in history[-10:]:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        cleaned_content = clean_html(msg.get("content", ""))
        if cleaned_content:
            messages.append({"role": role, "content": cleaned_content})
            
    # Add current user message
    messages.append({"role": "user", "content": user_message})

    client = _get_client()
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        raw_text = completion.choices[0].message.content
        raw_dict = json.loads(raw_text)
        
        # Extract features from raw response
        extracted = raw_dict.get("extracted_features", {})
        
        # Merge with current features using case-insensitive and underscore-insensitive mapping
        # Only copy non-None features from current_features
        merged = {}
        for f in REQUIRED_FEATURES:
            if f in current_features and current_features[f] is not None:
                merged[f] = current_features[f]
                
        norm_map = {f.lower().replace("_", ""): f for f in REQUIRED_FEATURES}
        for k, v in extracted.items():
            norm_key = str(k).lower().replace("_", "")
            if norm_key in norm_map:
                real_key = norm_map[norm_key]
                if v is not None:
                    try:
                        # Save value as float if it's a number
                        merged[real_key] = float(v)
                    except (ValueError, TypeError):
                        pass
                    
        # Deterministically compute missing features list
        missing = [f for f in REQUIRED_FEATURES if f not in merged]
        
        # Populate final features with all required keys (missing set to None)
        final_features = {}
        for f in REQUIRED_FEATURES:
            if f in merged:
                final_features[f] = merged[f]
            else:
                final_features[f] = None
        
        follow_up = raw_dict.get("follow_up_message", "")
        if not follow_up:
            if missing:
                follow_up = "Could you please provide the missing medical details: " + ", ".join(missing)
            else:
                follow_up = "Thank you! All features collected."
        elif missing:
            # Check if LLM prematurely outputted prediction details
            has_premature_pred = any(w in follow_up.lower() for w in ["confidence", "prediction", "classified", "positive class", "negative class"])
            if has_premature_pred or "complete" in follow_up.lower():
                newly_extracted = [k for k, v in extracted.items() if v is not None and str(k).lower().replace("_", "") in norm_map]
                if newly_extracted:
                    real_key = norm_map[newly_extracted[0].lower().replace("_", "")]
                    val = extracted[newly_extracted[0]]
                    # Convert to float and format
                    try:
                        val = int(float(val)) if float(val).is_integer() else float(val)
                    except:
                        pass
                    explanation = f"Thank you. Recorded {real_key} value as {val}."
                else:
                    explanation = "Thank you."
                
                # Make the missing variable name reader-friendly
                friendly_missing = missing[0]
                if friendly_missing == "DiabetesPedigreeFunction":
                    friendly_missing = "Diabetes Pedigree Function (family history score)"
                elif friendly_missing == "BloodPressure":
                    friendly_missing = "Blood Pressure"
                elif friendly_missing == "SkinThickness":
                    friendly_missing = "Skin Thickness"
                
                follow_up = f"{explanation} We still need details for your {friendly_missing} to run the prediction model. Could you please provide that?"

        return ChatExtractionResponse(
            extracted_features=final_features,
            missing_features=missing,
            follow_up_message=follow_up
        )
        
    except Exception as e:
        logger.error("Groq chat extraction failed: %s", type(e).__name__)
        # Fallback response so the conversation doesn't die
        merged = {}
        for f in REQUIRED_FEATURES:
            if f in current_features and current_features[f] is not None:
                merged[f] = current_features[f]
        missing = [f for f in REQUIRED_FEATURES if f not in merged]
        
        final_features = {}
        for f in REQUIRED_FEATURES:
            if f in merged:
                final_features[f] = merged[f]
            else:
                final_features[f] = None
                
        return ChatExtractionResponse(
            extracted_features=final_features,
            missing_features=missing,
            follow_up_message="I'm sorry, I'm having trouble processing that. Could you please provide these missing details: " + ", ".join(missing)
        )
