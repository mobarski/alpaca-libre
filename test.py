from pprint import pprint

# ===[ CONFIG ]=====================================================================================

PROMPT_VERSION = 'v1'
N_EXAMPLES  = 4
N_TASKS     = 10 # 20
PROMPT_PATH = 'data/alpaca_libre_prompt_{{version}}.txt'
SEED_PATH   = 'data/seed_tasks.jsonl' # from Self-Instruct paper
RANDOM_SEED = None # 2501
TEMPERATURE = 0.7

# ===[ SEEDS ]======================================================================================

import random
import json

seed_tasks = [json.loads(line) for line in open(SEED_PATH)]
random.seed(RANDOM_SEED)

def random_seed_tasks(n):
    return random.sample(seed_tasks, n)

# ===[ PROMPT ]=====================================================================================

import jinja2
def render(text, **kw):
    "Render a template with the given variables."
    return jinja2.Template(text).render(**kw)

def get_prompt(version, n_examples):
    "Get the prompt with the given version and number of example tasks."
    examples = random_seed_tasks(n_examples)
    template = open(PROMPT_PATH.replace('{{version}}',version)).read()
    return render(template, tasks=examples, n_tasks=N_TASKS,
                  enumerate=enumerate, randint=random.randint, len=len)

# ===[ STATUS ]=====================================================================================

import re

def qa_status(text):
    "Get quality assurance status for the given instruction."
    blacklist = ["image","images","graph","graphs","picture","pictures","file","files",
                 "map","maps","draw","plot","go to","video","audio","music",
                 "flowchart","diagram",]
    blacklist_re = re.compile(r'\b(' + '|'.join(blacklist) + r')\b', re.IGNORECASE)
    if blacklist_re.search(text):
        return 'blacklist'
    #
    bad_start_re = re.compile('^\s*(?:write a program|[^a-z0-9])', re.IGNORECASE)
    if bad_start_re.search(text):
        return 'bad start'
    #
    n_words = len(text.split())
    if n_words <= 3:
        return 'too short'
    if n_words > 150:
        return 'too long'
    return 'ok'

# ===[ PARSING ]====================================================================================

output_re = re.compile(r'(\d+)\nInstruction: (.*)\nInput: (.*)\nOutput: (.*)', re.MULTILINE)

def parse_one_task(text):
    "Parse one task from the output, returning (id, instruction, input, output)."
    groups = output_re.findall(text)
    if not groups:
        return '','ERROR',text,''
    return groups[0]

def parse_all_tasks(text):
    "Parse all tasks from the output, returning a list of dicts."
    xxx = re.split('# TASK ', text)[1:]
    zzz = [parse_one_task(x) for x in xxx]
    vvv = [{'instruction':x[1], 'input':x[2], 'output':x[3], 'status':qa_status(x[1])} for x in zzz]
    pprint(vvv)

# ===[ MAIN ]=======================================================================================

from ai_bricks.api import openai
model = openai.model('gpt-3.5-turbo', temperature=TEMPERATURE) # API_KEY from OPENAI_KEY env var

prompt = get_prompt(PROMPT_VERSION, N_EXAMPLES)
print(prompt) # XXX

resp = model.complete(prompt, n=1)
pprint(resp) # XXX
print(resp['cost'], resp['rtt'], resp['usage'])

# examples=4 n=2 total=2400 prompt=800 output=1600
# examples=4 n=3 total=6000 prompt=800 output=5200
# examples=4 n=4 total=4500 prompt=700 output=3800
# examples=4 n=4 total=7400 prompt=900 output=6500

# TODO: handle resp

for text in resp['texts']:
    parse_all_tasks(text)
