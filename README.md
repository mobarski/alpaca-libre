# Alpaca Libre

Small research project - how much it would cost to create Alpaca-like dataset using slightly different approach. **All data byproducts are CC0-licensed**.

Remember that developing a model based on data **you** generated via model API might violate the terms of service of the model API provider.

![alpaca on the Altiplano grasslands with the Statue of Liberty in the background](assets/alpaca-libre-cover.jpg)

# Usage

1. Clone the repo:
`git clone https://github.com/mobarski/alpaca-libre && cd alpaca-libre`
2. Install required python modules:
`pip install -r requirements.txt`
3. View / edit generate.py
4. Set API_KEY:
`export OPENAI_KEY=...`
5. Run the script:
`python3 generate.py`

# Attribution

- `data/seed_tasks.jsonl` - is from the Self-Instruct paper
- `data/alpaca_libre_prompt_v1.txt` - is from the Alpaca paper (with slight modfification)

# Output

The output file (`data/alpaca_libre_tasks_v1.jsonl`) is in the jsonl format.
It contains one task (json object) per line.
Each task object has the following items:
- status - anything other than 'ok' is bad
- instruction
- input
- output
- other

# References

GitHub repos:
- https://github.com/tatsu-lab/stanford_alpaca
- https://github.com/yizhongw/self-instruct
- https://github.com/orhonovich/unnatural-instructions

Papers:
- https://crfm.stanford.edu/2023/03/13/alpaca.html
- https://arxiv.org/abs/2212.10560
- https://arxiv.org/abs/2212.09689

