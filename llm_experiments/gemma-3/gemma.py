import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.pop("XFORMERS_FORCE_DISABLE", None)
os.environ.pop("FLASH_ATTENTION_FORCE_DISABLE", None)

from huggingface_hub import login
from transformers import AutoTokenizer
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
wandb.init(project="estonian-absa", name="Gemma-vLLM-run", config={
    "model": "google/gemma-3-4b-it",
    "use_vllm": "as library",
    "method": "ABSA with vLLM",
})

# Truncated system prompt
system_prompt = """
You are a perfectly accurate language model for Aspect-Based Sentiment Analysis (ABSA) on customer reviews of restaurants in Estonian. 

Your ONLY task is to extract (aspect term, opinion term, polarity) triplets from the given review. Return a valid JSON object in the following format:
{
    "review_text": "<original review text>",
    "triplets": [
        {
            "aspect_term": "<exact text from review>",
            "opinion_term": "<exact text from review>",
            "polarity": "<one of: positive, negative, neutral (in English, lowercase)>"
        }
    ]
}

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

# Load in-context examples
with open("/home/svvikhlyantseva/output_val.json", "r", encoding="utf-8") as f:
    in_context_examples = json.load(f)

examples_messages = []
for example in in_context_examples[:1]:  # Trim to 1 example
    example_json = {
        "review_text": example["review_text"],
        "triplets": [
            {
                "aspect_term": triplet["aspect_term"]["text"],
                "opinion_term": triplet["opinion_term"]["text"],
                "polarity": triplet["polarity"]
            }
            for triplet in example["triplets"]
        ]
    }
    examples_messages.append(
        {"role": "user", "content": [{"type": "text", "text": example["review_text"]}]}
    )
    examples_messages.append(
        {"role": "assistant",
         "content": [{"type": "text", "text": json.dumps(example_json, ensure_ascii=False)}]}
    )

# Load test data
test_data = pd.read_csv("/home/svvikhlyantseva/modified_estonian_restaurant_reviews_test_sample.csv")
logging.info(f"{len(test_data)} test examples loaded.")
wandb.log({"num_test_examples": len(test_data)})

# Load model and tokenizer
model_name = "google/gemma-3-4b-it"
SAMPLING_PARAMS = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=2048, stop=["</s>"])
llm = LLM(model=model_name, gpu_memory_utilization=0.7, enforce_eager=True, dtype="auto", max_model_len=2048, max_num_seqs=1)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt formatting with chat template
prompts = []
prompt_lengths = []
for idx, review in tqdm(enumerate(test_data["review_text"]), total=len(test_data)):
    messages = [
        {"role": "system",
         "content": [{"type": "text", "text": system_prompt}]}
    ] + examples_messages + [
        {"role": "user",
         "content": [{"type": "text", "text": review}]}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

    MODEL_CONTEXT_LIMIT = 8192  # for Gemma
    if num_tokens + SAMPLING_PARAMS.max_tokens > MODEL_CONTEXT_LIMIT:
        logging.warning(f"Prompt too long ({num_tokens} tokens), skipping.")
        continue

    prompts.append(prompt)
    prompt_lengths.append(num_tokens)
    wandb.log({"prompt_length": num_tokens, "last_prompt": prompt})

# Run generation
logging.info("Running inference with Gemma via vLLM...")
outputs = llm.generate(prompts, SAMPLING_PARAMS)
logging.info(f"Got {len(outputs)} output predictions.")
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
            raise ValueError("No valid JSON found.")

        final_triplets = []
        for triplet in json_obj.get("triplets", []):
            aspect_text = triplet.get("aspect_term", "").strip()
            opinion_text = triplet.get("opinion_term", "").strip()
            polarity = triplet.get("polarity", "").lower().strip()

            try:
                a_start = review.index(aspect_text)
                a_end = a_start + len(aspect_text)
            except ValueError:
                a_start, a_end = -1, -1

            try:
                o_start = review.index(opinion_text)
                o_end = o_start + len(opinion_text)
            except ValueError:
                o_start, o_end = -1, -1

            final_triplets.append({
                "aspect_term": {"start": a_start, "end": a_end, "text": aspect_text},
                "opinion_term": {"start": o_start, "end": o_end, "text": opinion_text},
                "polarity": polarity
            })

        results.append({"review_text": review, "triplets": final_triplets})
        num_success += 1

    except Exception as e:
        results.append({"review_text": review, "error": str(e), "raw_output": raw_response})

logging.info(f"{num_success}/{len(outputs)} predictions successfully parsed.")
wandb.log({"num_successfully_parsed": num_success})

# Save results
output_path = "/home/svvikhlyantseva/predicted_output_gemma_trunc_prompt_1_shot.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
logging.info(f"Predictions saved to {output_path}.")

wandb.finish()
