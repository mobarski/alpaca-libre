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


if __name__ == "__main__":
    export_as_json("data/output/work/alpaca_libre_tasks_v4.jsonl","data/output/alpaca_libre_ok_tasks_v4.json")

