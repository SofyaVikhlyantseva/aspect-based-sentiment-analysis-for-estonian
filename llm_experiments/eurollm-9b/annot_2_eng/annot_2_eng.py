import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
- Return an empty "triplets" list if no triplets are found.

Aspect terms are single or multiword terms naming particular aspects (features/attributes) of the restaurant (i.e., “toit” ‘food’, “teenindus” ‘service’, “atmosfäär” ‘atmosphere’, names of dishes).
For example, in “Kõik oli tasemel: atmosfäär, teenindus, toit”, the aspect terms are “atmosfäär”, “teenindus” and “toit”.
These words and phrases should be sufficiently specific, such that they can be isolated from the text of the review and, for example, placed in the “People often mention” section of the site (“Often mentioned .../ Often praised.../ Often scolded..."). Abstract (vague) or overly individual concepts (for example: “kogemus” ‘experience’, “õhtu” ‘evening’ or “iga tukk” ‘every bite’) are not aspect terms, because taken in isolation, they would not give a clear idea of the restaurant and would look inappropriate in the “People often mention" section of the site.

What should be annotated as aspect term:
- Nouns or nominal phrases explicitly mentioning aspects. Aspects can be in different morphological cases, this is normal. For example, in “Super maitsvad söögid, joogid ning meeldiv ja kiire teenindus”, the aspect terms are “söögid” ‘dishes’, “joogid” ‘drinks’, “teenindus” ‘service’. Notice that in (i), one of the aspect terms is “soojas salatis” ‘in a warm salad’ as a whole, not just “salatis”: here “soojas” ‘in a warm’ is not a positive opinion term, but an element of the dish name ‘warm salad’. In (ii), the aspect term is the entire phrase “kuulus parmesanikeras keerutatud truhlipasta” ‘famous truffle paste wrapped in a ball of parmesan cheese’, i.e. the noun with all its dependents should be annotated.
i. Tellitud pastas ning soojas salatis olid mereainete karpide tükid
ii. Näiteks valmistas suuri pettumusi kuulus parmesanikeras keerutatud trühvlipasta
If an aspect term appears more than once in a review text, all its occurrences should be annotated.
If there are no aspect terms, return an empty list.

What should NOT be annotated as aspect term:
- implicit aspect terms, i.e., aspect terms that are not explicitly mentioned, but can be inferred from adjectives or other expressions. For example, sentence “Väga maitsev!” can be thought of as referring to an implicit aspect term “toit” ‘food’, but you shouldn’t annotate anything in this sentence;
- pronouns (e.g., “kõik” ‘all’, “kogu” ‘whole, entire’, “see” ‘this, it, that’), even if they refer to an aspect. For example, in “Mulle meeldis toit, see oli hämmastav!” ‘I loved the food, it was amazing!’, “see” ‘it’ should not be annotated;
- references not to a certain aspect but to the target entity (the restaurant the review is about) as a whole. For example, in “On minu lemmik restoran”, “restoran” should not be annotated as aspect term;
- mentions of other entities (geographical locations in particular, e.g., “Tallinnas”);
- names of the restaurants.
Notice that if a nominal phrase fits the definition of an aspect term, but there is no term expressing an opinion related to it, it should not be annotated as an aspect term. For example, in “Tellisin joogi”, “Tellisin” ‘ordered’ is just factual information, not an assessment/opinion, therefore “joogi” ‘drink' should not be annotated as an aspect term.

Opinion terms are words or phrases expressing evaluation, opinion etc. on a certain aspect. In review texts, they can be expressed in different parts of speech:
- adjectives/participles (for example: ülehinnatud ‘overrated') or groups of adjectives/participles and their dependents (in particular, adverbs). For example, in “Väga armas ja hubane öhkond ning meeldiv öhkond teenindus”, “öhkond” ‘atmosphere’ and “teenindus” ‘service’ are aspect terms, and “Väga armas” ‘very nice', “hubane” ‘cozy’ and “meeldiv” ‘pleasant' are opinion terms related to them;
- verbs or groups of verbs and their dependents. For example, in “Näiteks valmistas suuri pettumusi kuulus parmesanikeras keerutatud trühvlipasta”, “valmistas suuri pettumusi” ‘made great disappointments’ should be annotated as an opinion term;
- predicative nominal phrases. For example, in “Baklažaan on peamise menüü jaoks kummaline valik”, the opinion term is “peamise menüü jaoks kummaline valik” ‘a strange choice for the main menu'.
Notice that if words or phrases expressing opinions are connected via conjunctions (e.g., “ja”/”ning” ‘and’), then the conjunctions do not need to be annotated, and each word or phrase should be isolated and annotated separately. For example, in “Väga armas ja hubane öhkkond ning meeldiv teenindus”, the opinion terms related to the aspect term “öhkkond” ‘atmosphere‘ are “Väga armas” 'very nice’ and “hubane” ‘cozy' separately, rather than the combined “Väga armas ja hubane” entirely.
When listing words or phrases expressing opinions without coordinating conjunctions (i.e., separated by commas), each of them should also be annotated as a separate opinion term.
BUT: if, when splitting words or phrases expressing opinions connected by a conjunction, it becomes unclear what was meant, then they should be combined into one opinion term with an embedded conjunction. For example, in (iii), you should not break the participles “läbimõeldud” ‘thought out’ and “teostatud” ‘executed’, connected by the conjunction “ja”, because “nii hästi” ‘so well’ refers to both the first and second participles, and “teostatud” ‘executed' just by itself ceases to be an opinion term.
iii. Kõik oli nii hästi läbimõeldud ja teostatud: alates vastuvõtust restorani uksel, kogu menüü ja joogid sinna juurde, väga mõnus jalutuskäik söökikordade keskel, teenindajate suhtlus, atmosfäär…
Note also that in (iii), the opinion term “nii hästi läbimõeldud ja teostatud” ‘so well thought out and executed’ refers to each of the aspect terms “vastuvõtust restorani uksel” ‘(from) reception at the door of the restaurant’, “menüü” ‘menu’, “joogid” ‘drinks’, “jalutuskäik söökordade keskel” ‘walking in the middle of a meal', “teenindajate suhtlus” ‘communication of waiters', “atmosfäär” ‘atmosphere'. Thus, several opinion terms can relate to one aspect term, and, conversely, one opinion can be related to several aspect terms at once, this is normal.
However, if there is any opinion in the review, but it is expressed regardless of the aspect (such as in “Ilus ja maitsev!” ‘It's beautiful and delicious!’), then it should not be annotated as an opinion term.
That is, the term (aspect or opinion) must be associated with at least one opposite term (opinion or aspect, respectively).
Copulas (i.e., form of the verb ‘to be’, e.g., third-person singular “on” in the present tense or third-person singular “oli” in the past tense) should not be included in opinion terms, unless they are parts of embedded aspect terms (such as in “Meeldis ka see, et kokale on jäetud loominguline vabadus”, where “meeldis” ‘liked’ is the opinion term, and the entire construction “kokale on jäetud loominguline vabadus” ‘the chef is given creative freedom’ naming what the author of the review liked is the aspect term).

Sentiment polarity (a relationship between an aspect term and an opinion term) can be positive or negative if the reviewer expresses a positive or negative, respectively, attitude, opinion, evaluation, emotion, or feeling etc. towards a particular aspect.
Neutral polarity should be assigned in cases where expressions with a neutral sentiment are used, e.g., “mõõdukas” ‘moderate’, “midagi ebatavalist” ‘nothing unusual’, etc.
Please note that, when determining the polarity, it is necessary to rely precisely on the context. For example, in “Igas roogi tükis oli midagi hapukat, rammusat, krõmpsu ja imelist”, opinion terms “hapukat” ‘sour’, “rammusat” ‘fatty (saturated)’, “krõmpsu” ‘crunchy’ and “imelist” ‘amazing' should be considered positive in this context (although in general these adjectives are not necessarily positive, for example, something can be unpleasantly sour or crispy because it's burnt).
"""

# Save prompt to wandb config for tracking
wandb.config.update({"system_prompt": system_prompt})

# Load in-context examples (as string)
with open("/home/svvikhlyantseva/output_val.json", "r", encoding="utf-8") as f:
    in_context_examples = json.load(f)

examples_text = ""
for example in in_context_examples[:2]:  # Trim to 2 examples
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
    examples_text += f"Input:\n{example['review_text']}\nOutput:\n{json.dumps(example_json, ensure_ascii=False)}\n\n"

json_schema = """
{
    "review_text": "<original review text>",
    "triplets": [
        {
            "aspect_term": "<exact text from review>",
            "opinion_term": "<exact text from review>",
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
SAMPLING_PARAMS = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=900)
llm = LLM(model=model_name, dtype="float16", gpu_memory_utilization=0.7)

tokenizer = AutoTokenizer.from_pretrained(model_name)

MODEL_CONTEXT_LIMIT = 4096  # for EuroLLM

# Prepare prompts
prompts = []
prompt_lengths = []  # for logging
for review in tqdm(test_data["review_text"]):
    user_prompt = f"""
{system_prompt}

JSON format:
{json_schema}

Examples:
{examples_text}

Here's the review text:
"{review}"
"""
    num_tokens = len(tokenizer.encode(user_prompt, add_special_tokens=False))
    prompt_lengths.append(num_tokens)

    if num_tokens + SAMPLING_PARAMS.max_tokens > MODEL_CONTEXT_LIMIT:
        logging.warning(f"Prompt too long ({num_tokens} tokens), skipping.")
        continue

    prompts.append(user_prompt)

    # Log prompt length and user prompt to wandb
    wandb.log({"prompt_length": num_tokens})
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

        final_triplets = []
        for triplet in json_obj.get("triplets", []):
            aspect_text = triplet.get("aspect_term", "").strip()
            opinion_text = triplet.get("opinion_term", "").strip()
            polarity = triplet.get("polarity", "").lower().strip()

            # Add start and end positions
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
                "aspect_term": {
                    "start": a_start,
                    "end": a_end,
                    "text": aspect_text
                },
                "opinion_term": {
                    "start": o_start,
                    "end": o_end,
                    "text": opinion_text
                },
                "polarity": polarity
            })

        formatted_result = {
            "review_text": review,
            "triplets": final_triplets
        }

        results.append(formatted_result)
        num_success += 1

    except Exception as e:
        results.append({"review_text": review, "error": str(e), "raw_output": raw_response})

# Logging successful parsing
logging.info(f"{num_success}/{len(outputs)} predictions successfully parsed.")
wandb.log({"num_successfully_parsed": num_success})

# Save results
output_path = "/home/svvikhlyantseva/eurollm_annot_2_eng.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
logging.info(f"Predictions saved to {output_path}.")

wandb.finish()
