# ===[ CONFIG ]===

PROMPT_VERSION = 'v1'
N_EXAMPLES  = 4
PROMPT_PATH = 'data/alpaca_libre_prompt_{{version}}.txt'
SEED_PATH   = 'data/seed_tasks.jsonl' # from Self-Instruct paper
RANDOM_SEED = None # 2501
TEMPERATURE = 0.7

# ===[ SEEDS ]===

import random
import json

seed_tasks = [json.loads(line) for line in open(SEED_PATH)]
random.seed(RANDOM_SEED)

def random_seed_tasks(n):
    return random.sample(seed_tasks, n)

# ===[ PROMPT ]===

import jinja2
def render(text, **kw):
    return jinja2.Template(text).render(**kw)

def get_prompt(version, n_examples):
    examples = random_seed_tasks(n_examples)
    template = open(PROMPT_PATH.replace('{{version}}',version)).read()
    return render(template, tasks=examples, enumerate=enumerate, randint=random.randint, len=len)

###

from pprint import pprint

from ai_bricks.api import openai
model = openai.model('gpt-3.5-turbo', temperature=TEMPERATURE) # API_KEY from OPENAI_KEY env variables

prompt = get_prompt(PROMPT_VERSION, N_EXAMPLES)
print(prompt)

resp = model.complete(prompt, n=4)
pprint(resp)
print(resp['cost'], resp['rtt'], resp['usage'])

# examples=4 n=2 total=2400 prompt=800 output=1600
# examples=4 n=3 total=6000 prompt=800 output=5200
# examples=4 n=4 total=4500 prompt=700 output=3800
# examples=4 n=4 total=7400 prompt=900 output=6500

# TODO: handle resp
