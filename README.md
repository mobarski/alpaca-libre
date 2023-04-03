# Alpaca Libre

🦙🗽 Small research project - how much it would cost to create Alpaca-like dataset, with 50k+ demonstrations, using slightly different approach. **All data byproducts are CC0/MIT-licensed**.

🔥 The project also contains 100k+ MIT-licensed demonstrations from [Anthropics HH-RLHF repo](https://github.com/anthropics/hh-rlhf) - converted into "Alpaca compatible format".

👉 [Follow me](https://twitter.com/KerbalFPV) on Twitter for news and updates.

🚫 Remember that releasing a model based on data **you** generated via model API might violate the Terms of Service of the model API provider.

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

Files in the `data/output` directory are in the same format as original Alpaca dataset.

Files in the `data/output/work` directory are in the .jsonl format and:

- contain one task (JSON object) per line,

- contain also tasks that failed quality checks (status!='ok')

  - these tasks might be marked as 'ok' after manual inspection

- each task object has the following items:

  - status - anything other than 'ok' is bad

  - instruction - instruction part of the prompt

  - input - input part of the prompt

  - output - expected output

  - other - dictionary for other information (similarity, etc)


# References

GitHub repos:
- https://github.com/tatsu-lab/stanford_alpaca
- https://github.com/yizhongw/self-instruct
- https://github.com/orhonovich/unnatural-instructions
- https://github.com/gururise/AlpacaDataCleaned
- https://github.com/anthropics/hh-rlhf

Papers:
- https://crfm.stanford.edu/2023/03/13/alpaca.html
- https://arxiv.org/abs/2212.10560
- https://arxiv.org/abs/2212.09689
- https://arxiv.org/abs/2204.05862


# Changelog

- **0.4.2**
  - MIT-licensed demonstrations from [Anthropics HH-RLHF repo](https://github.com/anthropics/hh-rlhf)
    - 104k human preferred responses from the train datasets:
      - 41k harmless
      - 42k helpful
      - 21k helpful-online
- **0.4.1**
  - v4 dataset converted into the same format as original Alpaca
  - jsonl dataset moved into work dir
- **0.4**
  - grouping turns into rounds
  - basic input quality check
  - better `<noinput>` handling
  - `<nooutput>` handling
  - retry with backoff on API error
  - progressbars
  - fixed: typos in Alpaca prompt
  - fixed: whitespace handling after task number
- **0.3**
  - parallel main loop
  - better cli output
  - output format change (everythig not essential is placed in the "other" object)
  - basic output quality check
  - fixed: multiline input/output handling
  - fixed: no initial space / empty section handling
  - fixed: `<noinput>`
