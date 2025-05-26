import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import login
import json
import re
import regex
import logging
import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
import wandb

# Configuring the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

HF_TOKEN = "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"
login(HF_TOKEN)

# WANDB initialization
wandb.init(project="estonian-absa", name="EuroLLM-vLLM-run", config={
    "model": "utter-project/EuroLLM-9B",
    "use_vllm": "as library",
    "method": "ABSA with vLLM",
})

# Truncated system prompt
system_prompt = """
You are a perfectly accurate language model for Aspect-Based Sentiment Analysis (ABSA) on customer reviews of restaurants in Estonian. 

Your ONLY task is to extract (aspect term, opinion term, polarity) triplets from the given review. Return a valid JSON object in the format shown below.
General rules:
- Do NOT include any explanation, summary, reason, translation, or any natural language outside the JSON. Output only the valid JSON, nothing else.
- Aspect and opinion terms must be exact substrings from the original review (copy the words verbatim, preserve typos/spelling errors and diacritical marks).
- DO NOT include punctuation marks inside the aspect or opinion term text.
- Do NOT lemmatize, generalize, lowercase, or translate words.
- The polarity must be a separate top-level key inside each aspect object — never nested inside aspect_term or opinion_term.
- Return an empty "triplets" list if no triplets are found.
- Only return syntactically correct and parseable JSON (must be valid for Python's json.loads()).
- Escape any double quotes in text values using backslash (`\"`).
"""

# Save prompt to wandb config for tracking
wandb.config.update({"system_prompt": system_prompt})

# Load in-context examples (as string)
with open("/home/svvikhlyantseva/output_val.json", "r", encoding="utf-8") as f:
    in_context_examples = json.load(f)

examples_text = ""
for example in in_context_examples[:1]:  # Trim to 1 example
    example_json = {
        "review_text": example["review_text"],
        "triplets": [
            {
                "aspect_term": {
                    "text": triplet["aspect_term"]["text"]
                },
                "opinion_term": {
                    "text": triplet["opinion_term"]["text"]
                },
                "polarity": triplet["polarity"]
            }
            for triplet in example["triplets"]
        ]
    }
    examples_text += f"Input:\n{example['review_text']}\nOutput:\n{json.dumps(example_json, ensure_ascii=False)}\n\n"

json_schema = """
{
    "review_text": "<original review text>",
    "triplets": [
        {
            "aspect_term": {
                "text": "<exact text from review>"
            },
            "opinion_term": {
                "text": "<exact text from review>"
            },
            "polarity": "<one of: positive, negative, neutral (in English, lowercase)>"
        },
        {
            ...
    ]
}
"""

# Read test data
test_data = pd.read_csv("/home/svvikhlyantseva/modified_estonian_restaurant_reviews_test_sample.csv")
logging.info(f"{len(test_data)} test examples loaded.")
wandb.log({"num_test_examples": len(test_data)})

model_name = "utter-project/EuroLLM-9B"
SAMPLING_PARAMS = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=4096, stop=["<|im_end|>"])
llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.7)

# Prepare prompts
prompts = []
for review in tqdm(test_data["review_text"]):
    user_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Please extract aspect-based sentiment triplets from the following Estonian restaurant review and return a JSON object following the specified format.
JSON format:
{json_schema}
Examples:
{examples_text}
Here's the review text:
"{review}"<|im_end|>
<|im_start|>assistant"""
    prompts.append(user_prompt)

    # Log user prompt to wandb
    wandb.log({"last_prompt": user_prompt})

# Run generation
logging.info("Running inference with vLLM...")
outputs = llm.generate(prompts, SAMPLING_PARAMS)
logging.info(f"Got {len(outputs)} output predictions from the model.")
wandb.log({"num_model_outputs": len(outputs)})

# Parse results
results = []
num_success = 0


def clean_control_chars(s):
    """
    Removes quotes and escapes control characters inside string values in JSON.
    """
    def replacer(match):
        key, inner = match.group(1), match.group(2)
        # Remove quotes inside the value and escape control characters
        cleaned = inner.replace('"', '') \
                       .replace('\\', '\\\\') \
                       .replace('\n', '\\n') \
                       .replace('\r', '\\r') \
                       .replace('\t', '\\t')
        return f'"{key}": "{cleaned}"'

    return regex.sub(r'"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*?)"', replacer, s)


def extract_first_json_object(text: str):
    # Search for JSON objects (with nesting)
    pattern = r'\{(?:[^{}]|(?R))*\}'  # recursive regex
    candidates = regex.findall(pattern, text, regex.DOTALL)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Replace single quotes with double quotes
        candidate = re.sub(r"(?<!\\)'", '"', candidate)

        # Clean quotes and escape control characters inside string values
        cleaned_candidate = clean_control_chars(candidate)

        try:
            return json.loads(cleaned_candidate)
        except json.JSONDecodeError as e:
            print("Still invalid JSON:", e)
            continue

    return None


for review, output in zip(test_data["review_text"], outputs):
    raw_response = output.outputs[0].text.strip()
    with open('/home/svvikhlyantseva/model_raw_outputs.txt', 'a', encoding='utf-8') as f:
        f.write(raw_response + '\n' + '=' * 50 + '\n')
    try:
        json_obj = extract_first_json_object(raw_response)
        if json_obj is None:
            raise ValueError("No JSON object found in the model output.")

        # Add start and end positions
        if isinstance(json_obj.get("triplets"), list):
            for triplet in json_obj["triplets"]:
                if (
                        isinstance(triplet, dict)
                        and isinstance(triplet.get("aspect_term"), dict)
                        and isinstance(triplet.get("opinion_term"), dict)
                ):
                    aspect_text = triplet["aspect_term"].get("text")
                    opinion_text = triplet["opinion_term"].get("text")

                    if isinstance(aspect_text, str):
                        try:
                            a_start = review.index(aspect_text)
                            triplet["aspect_term"]["start"] = a_start
                            triplet["aspect_term"]["end"] = a_start + len(
                                aspect_text)
                        except ValueError:
                            pass  # aspect text not found

                    if isinstance(opinion_text, str):
                        try:
                            o_start = review.index(opinion_text)
                            triplet["opinion_term"]["start"] = o_start
                            triplet["opinion_term"]["end"] = o_start + len(
                                opinion_text)
                        except ValueError:
                            pass  # opinion text not found

        results.append(json_obj)
        num_success += 1

    except Exception as e:
        results.append({"review_text": review, "error": str(e), "raw_output": raw_response})

# Logging successful parsing
logging.info(f"{num_success}/{len(outputs)} predictions successfully parsed.")
wandb.log({"num_successfully_parsed": num_success})

# Save results
output_path = "/home/svvikhlyantseva/predicted_output_eurollm_trunc_prompt_1_shot.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
logging.info(f"Predictions saved to {output_path}.")

wandb.finish()
