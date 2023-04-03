import re
import json
from tqdm import tqdm

def export_as_json(path_in, path_out):
    with open(path_in, "r") as f:
        data_in = [json.loads(line) for line in tqdm(f)]
    data_out = []
    for doc in tqdm(data_in):
        if doc['status']!='ok': continue
        data_out.append({k:doc[k] for k in ['instruction', 'input', 'output']})
    with open(path_out, "w") as f:
        json.dump(data_out, f, indent=4)

def convert_anthropics_to_json(path_in, path_out, char_limit=None):
    with open(path_in, "r") as f:
        data_in = [json.loads(line) for line in tqdm(f)]
    data_out = []
    multi_turn = 0
    total = 0
    for doc in tqdm(data_in):
        if 'chosen' not in doc: continue
        selected = doc['chosen']
        if 'http' in selected: continue # no URLs, seriously!
        dialog = re.split('\n\n(?:Human|Assistant):', selected)
        assert(dialog[0]=='')
        del dialog[0]
        doc = {'instruction': dialog[0].strip(), 'input': '', 'output': dialog[1].strip()} # only one turn
        data_out.append(doc)
        if len(dialog)>2:
            multi_turn += 1
        total += 1
    with open(path_out, "w") as f:
        json.dump(data_out, f, indent=4)
    print(f"multi_turn/total: {multi_turn}/{total}")

if __name__ == "__main__":
    #export_as_json("data/output/work/alpaca_libre_tasks_v4.jsonl","data/output/alpaca_libre_ok_tasks_v4.json")
    convert_anthropics_to_json("data/external/rlhf_harmless_base.jsonl","data/output/rlhf_harmless_base.json")
    convert_anthropics_to_json("data/external/rlhf_helpful_base.jsonl","data/output/rlhf_helpful_base.json")
    convert_anthropics_to_json("data/external/rlhf_helpful_online.jsonl","data/output/rlhf_helpful_online.json")
    #convert_anthropics_to_json("data/external/rlhf_helpful_rejection_sampled.jsonl","data/output/rlhf_helpful_rejection_sampled.json")
