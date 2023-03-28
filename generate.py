from pprint import pprint

# TODO: parallelize similarit check / replace with vector index

"""
# FILE SECTIONS:
- CONFIG     - configuration variables
- SEEDS      - tasks from the Self-Instruct paper
- PROMPT     - prompt generation
- STATUS     - quality control status
- PARSING    - model output parsing
- SIMILARITY - similarity calculation
- OUTPUT     - output file generation
- MAIN       - main loop + show stats
"""

# ===[ CONFIG ]=====================================================================================

# TASKS_PER_TURN = N_COMPLETE * (N_TASKS-N_EXAMPLES)
# TOTAL_TURNS    = N_ROUNDS * N_TURNS

N_EXAMPLES  = 4 # number of example tasks to include in the prompt
N_TASKS     = 20 # total number of tasks (including examples) per completion
N_COMPLETE  = 2 # number of completions to request from GPT-3
N_ROUNDS    = 12 # number of rounds to run
N_TURNS     = 60 # number of shots to api per round
N_WORKERS   = 12 # number of parallel workers
TEMPERATURE = 1.0 # do not use 0 when n=1 or you will get duplicates
SIMILARITY_THRESHOLD = 0.7 # similarity threshold for filtering
RANDOM_SEED = None # !!! not compatible with parallel workers !!!
PROMPT_PATH = 'data/alpaca_libre_prompt_v1.txt' # modified Alpaca prompt
SEED_PATH   = 'data/seed_tasks.jsonl' # from Self-Instruct paper
OUTPUT_PATH = 'data/output/alpaca_libre_tasks_v4.jsonl'

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

# variations of <nooutput> and N/A 
nooutput_re = re.compile(r'"?\s*(?:[<]?\s*no[ _-]*output\s*[>]?|\s*N\s*/\s*A\s*)\s*"?', re.IGNORECASE)

def qc_status(instruction, input, output):
    "Get quality control status for the given task. Anything other than 'ok' is bad."
    instruction_status = qc_status_instruction(instruction)
    if instruction_status != 'ok':
        return instruction_status
    input_status = qc_status_input(input)
    if input_status != 'ok':
        return input_status
    output_status = qc_status_output(output)
    if output_status != 'ok':
        return output_status
    return 'ok'

def qc_status_instruction(text):
    "Get quality control status for the given instruction. Anything other than 'ok' is bad."
    if text=='ERROR':
        return 'error'
    #
    blacklist = ["image","images","graph","graphs","picture","pictures","file","files",
                 "map","maps","draw","plot","go to","video","audio","music","flowchart","diagram",]
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

def qc_status_input(text):
    "Get quality control status for the given input. Anything other than 'ok' is bad."
    blacklist = ['http:','https:']
    blacklist_re = re.compile(r'\b(' + '|'.join(blacklist) + r')', re.IGNORECASE)
    if blacklist_re.search(text):
        return 'blacklist'
    return 'ok'

def qc_status_output(text):
    "Get quality control status for the given output. Anything other than 'ok' is bad."
    if text.strip() == '':
        return 'empty'
    if nooutput_re.search(text.strip()):
        return 'nooutput'
    return 'ok'

# ===[ PARSING ]====================================================================================

task_re = re.compile(r'(\d+)\s*\nInstruction:(.*)\nInput:(.*)\nOutput:(.*)', re.MULTILINE|re.DOTALL)
# variations of <noinput> and N/A 
noinput_re = re.compile(r'"?\s*(?:[<]?\s*no[ _-]*input\s*[>]?|\s*N\s*/\s*A\s*)\s*"?', re.IGNORECASE)

def parse_one_task(text):
    "Parse one task from the output, returning (id, instruction, input, output)."
    groups = task_re.findall(text)
    if not groups:
        return '','ERROR',text,''
    id,inst,input,output = groups[0]
    inst = re.sub('^ ','', inst)
    input = re.sub('^ ','', input)
    output = re.sub('^ ','', output)
    input = "" if noinput_re.search(input.strip()) else input
    return id,inst,input,output

def parse_all_tasks(text):
    "Parse all tasks from the output, returning a list of dicts."
    raw_tasks = re.split('# TASK ', text)[1:]
    parsed_tasks = [parse_one_task(x) for x in raw_tasks]
    tasks = [{'instruction':x[1], 'input':x[2],
              'output':x[3], 'status':qc_status(x[1],x[2],x[3])} for x in parsed_tasks]
    return tasks

# ===[ SIMILARITY ]=================================================================================

import time
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def check_similarity(tasks):
    "Check similarity between tasks. Set status to 'too similar' if needed. WARNING: quadratic complexity."
    instructions = [task['instruction'] for task in tasks]
    instructions_tokens = [scorer._tokenizer.tokenize(inst) for inst in instructions]
    for i, task in tqdm(enumerate(tasks), total=len(tasks),
                        desc='similarity', ncols=80, position=2):
        tokens = instructions_tokens[i]
        similarity = [rouge_scorer._score_lcs(tokens, x).fmeasure for x in instructions_tokens]
        task['similarity'] = round(list(sorted(similarity, reverse=True))[1], 3)
        if task['status'] == 'ok' and task['similarity'] > SIMILARITY_THRESHOLD:
            task['status'] = 'too similar'

# ===[ OUTPUT ]=====================================================================================

def save_tasks(tasks, label=''):
    "Save tasks to a JSONL file."
    path = OUTPUT_PATH
    if label:
        path = path.replace('.jsonl', f'-{label}.jsonl')
    with open(path, 'a') as f:
        for task in tasks:
            t = {k:task[k] for k in['status','instruction','input','output']}
            t['other'] = {k:task[k] for k in task if k not in ['status','instruction','input','output']}
            f.write(json.dumps(t) + '\n')

def save_resp(resp, label):
    path = OUTPUT_PATH.replace('.jsonl', f'-{label}.jsonl')
    with open(path, 'a') as f:
        f.write(json.dumps(resp) + '\n')

# ===[ MAIN ]=======================================================================================

from tqdm import tqdm
from retry import retry
from multiprocessing import Pool
from collections import Counter
from ai_bricks.api import openai
model = openai.model('gpt-3.5-turbo', temperature=TEMPERATURE) # API_KEY from OPENAI_KEY env var

# MAIN LOOP

@retry(tries=6, delay=1, backoff=2)
def worker(prompt):
   return model.complete(prompt, n=N_COMPLETE)

def main_loop():
    stats = Counter()
    tasks = []
    #
    t0 = time.time()
    pool = Pool(N_WORKERS)
    for _ in tqdm(range(N_ROUNDS), total=N_ROUNDS,
                  desc='rounds', ncols=80, position=0):
        batch = []
        prompts = [get_prompt(N_EXAMPLES) for _ in range(N_TURNS)]
        for resp in tqdm(pool.imap_unordered(worker, prompts), total=N_TURNS,
                         desc='turns', ncols=80, position=1):
            save_resp(resp, label='raw-resp') # for debugging / crash recovery
            stats['cost'] += resp['cost']
            stats['rtt']  += resp['rtt']
            stats['completion_tokens'] += resp['usage']['completion_tokens']
            stats['prompt_tokens'] += resp['usage']['prompt_tokens']
            stats['total_tokens'] += resp['usage']['total_tokens']

            for text in resp['texts']:
                parsed_tasks = parse_all_tasks(text)
                batch.extend(parsed_tasks)
                tasks.extend(parsed_tasks)

        check_similarity(batch)
        save_tasks(batch)
    total_time = time.time() - t0

    # SHOW STATS

    cnt = Counter()
    cnt.update([task['status'] for task in tasks])
    cnt_ok = cnt.get('ok', 0)

    print()
    pprint(cnt)
    print()
    print(f'TOTAL TASKS: {len(tasks)}')
    print(f'STATUS=OK: {cnt_ok} -> {100*cnt_ok/len(tasks):.1f}%')
    print(f'TASKS per round: {N_TURNS * N_COMPLETE * (N_TASKS-N_EXAMPLES)}')
    print()
    print(f'TOTAL TOKENS: {stats["total_tokens"]}')
    print(f'TOTAL TIME: {total_time:.1f}s')
    print(f'TOTAL RTT: {stats["rtt"]:.1f}s')
    print(f'TOTAL COST: ${stats["cost"]:.3f}')
    print()
    print(f'RTT SECONDS per ok task: {stats["rtt"]/cnt_ok:.2f}s')
    print(f'SECONDS per ok task: {total_time/cnt_ok:.2f}s')
    print(f'TOKENS per RTT second: {stats["total_tokens"]/stats["rtt"]:.1f}')
    print(f'TOKENS per second: {stats["total_tokens"]/total_time:.1f}')
    print()
    print(f'USD per 1k ok tasks: ${1000*stats["cost"]/cnt_ok:.3f}')

if __name__ == '__main__':
    main_loop()
