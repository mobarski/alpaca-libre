from pprint import pprint

"""
# FILE SECTIONS:
- CONFIG     - configuration variables
- SEEDS      - tasks from the Self-Instruct paper
- PROMPT     - prompt generation
- STATUS     - quality assurance status
- PARSING    - model output parsing
- SIMILARITY - similarity calculation
- OUTPUT     - output file generation
- MAIN       - main loop + show stats
"""

# ===[ CONFIG ]=====================================================================================

N_EXAMPLES  = 4 # number of example tasks to include in the prompt
N_TASKS     = 20 # total number of tasks (including examples) per ???
N_COMPLETE  = 2 # number of completions to request from GPT-3
N_TURNS     = 40 # number of shots to api
TEMPERATURE = 1.0 # do not use 0 when n=1 or you will get duplicates
SIMILARITY_THRESHOLD = 0.7 # similarity threshold for filtering
RANDOM_SEED = None
PROMPT_PATH = 'data/alpaca_libre_prompt_v1.txt' # modified Alpaca prompt
SEED_PATH   = 'data/seed_tasks.jsonl' # from Self-Instruct paper
OUTPUT_PATH = 'data/alpaca_libre_tasks_v1.jsonl'

# ===[ SEEDS ]======================================================================================

import random
import json

seed_tasks = [json.loads(line) for line in open(SEED_PATH)]
random.seed(RANDOM_SEED)

def random_seed_tasks(n):
    "Get n random *seed tasks*."
    return random.sample(seed_tasks, n)

# ===[ PROMPT ]=====================================================================================

import jinja2
def render(text, **kw):
    "Render a template with the given variables."
    return jinja2.Template(text).render(**kw)

def get_prompt(n_examples):
    "Get the prompt with the given version and number of example tasks."
    examples = random_seed_tasks(n_examples)
    template = open(PROMPT_PATH).read()
    return render(template, tasks=examples, n_tasks=N_TASKS,
                  enumerate=enumerate, randint=random.randint, len=len)

# ===[ STATUS ]=====================================================================================

import re

def qa_status(text):
    "Get quality assurance status for the given instruction. Anything other than 'ok' is bad."
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
    raw_tasks = re.split('# TASK ', text)[1:]
    parsed_tasks = [parse_one_task(x) for x in raw_tasks]
    tasks = [{'instruction':x[1], 'input':x[2],
              'output':x[3], 'status':qa_status(x[1])} for x in parsed_tasks]
    return tasks

# ===[ SIMILARITY ]=================================================================================

import time
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def check_similarity(tasks):
    "Check similarity between tasks. Set status to 'too similar' if needed. WARNING: quadratic complexity."
    t0 = time.time()
    instructions = [task['instruction'] for task in tasks]
    instructions_tokens = [scorer._tokenizer.tokenize(inst) for inst in instructions]
    for i, task in enumerate(tasks):
        tokens = instructions_tokens[i]
        similarity = [rouge_scorer._score_lcs(tokens, x).fmeasure for x in instructions_tokens]
        task['similarity'] = round(list(sorted(similarity, reverse=True))[1], 3)
        if task['status'] == 'ok' and task['similarity'] > SIMILARITY_THRESHOLD:
            task['status'] = 'too similar'
    dt = time.time() - t0
    print('check_similarities dt', dt) # XXX

# ===[ OUTPUT ]=====================================================================================

def save_tasks(tasks, label=''):
    "Save tasks to a JSONL file."
    path = OUTPUT_PATH
    if label:
        path = path.replace('.jsonl', f'-{label}.jsonl')
    with open(path, 'a') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')

# ===[ MAIN ]=======================================================================================

from tqdm import tqdm
from collections import Counter
from ai_bricks.api import openai
model = openai.model('gpt-3.5-turbo', temperature=TEMPERATURE) # API_KEY from OPENAI_KEY env var

tasks = []
stats = Counter()

# MAIN LOOP

for i in tqdm(range(N_TURNS)):
    prompt = get_prompt(N_EXAMPLES)
    resp = model.complete(prompt, n=N_COMPLETE)

    stats['cost'] += resp['cost']
    stats['rtt']  += resp['rtt']
    stats['completion_tokens'] += resp['usage']['completion_tokens']
    stats['prompt_tokens'] += resp['usage']['prompt_tokens']
    stats['total_tokens'] += resp['usage']['total_tokens']

    for text in resp['texts']:
        parsed_tasks = parse_all_tasks(text)
        save_tasks(parsed_tasks, label='tmp') # in case of crash
        tasks.extend(parsed_tasks)

check_similarity(tasks)
save_tasks(tasks)

# SHOW STATS

cnt = Counter()
cnt.update([task['status'] for task in tasks])
cnt_ok = cnt.get('ok', 0)
usd_per_task = (stats['cost']/cnt_ok) if cnt_ok else None
sec_per_task = (stats['rtt']/cnt_ok)  if cnt_ok else None

pprint(cnt)
print(resp['cost'], resp['rtt'], resp['usage'])
print('usd_per_ok_task', usd_per_task)
print('sec_per_ok_task', sec_per_task)
