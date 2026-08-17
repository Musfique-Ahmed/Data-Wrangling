# Dataset

## Source

This project does **not** ship with a file named `data_130.csv` from the
reference notebook `gpt_4o.ipynb`. The reference notebook reads:

```python
df = pd.read_csv('/content/MyDrive/MyDrive/iccit26/data_130.csv')
```

The closest artefact supplied in this repository is the file
`banglish hate speech dataset - Sheet3.csv` placed at the project root
by the user. This file is **copied** to `data/data_130.csv` (no
modification) and treated as the experiment dataset.

## Shape

```
Rows:               312
Columns (input):    Sentences
Unique texts:      312
Null texts:        0
Duplicates:        0
```

The Banglish hate-speech sample used in the reference notebook
(`awamiligara ture pun marche?`) is present in this file at row index 3.

## Column mapping

The loader (`src/dataset.py`) reads column `Sentences` and renames it
in-memory to `Text`. The output CSV column is `Text`. The original
column is never written back to the source file.

## Schema difference vs reference

The reference notebook reads `data_130.csv` which (per the notebook's
`df.drop(['GPT','Gemini','Deepseek'], axis=1)` line) **did** contain
prediction columns from previous GPT/Gemini/Deepseek runs. The CSV
supplied for this experiment has those columns stripped and contains
only `Sentences`.

## Implication for evaluation

Because there is no `GPT / Gemini / Deepseek` column in the supplied
CSV, there is **no ground-truth label** in the dataset. Phase 7 will
therefore report prediction distributions, invalid rates, failure rates,
latency, throughput, energy, and CO₂ — **not** Accuracy / F1.

If you want strict 130-row fidelity and a ground-truth label, please
provide the original `data_130.csv` referenced by `gpt_4o.ipynb`. This
project will continue to ship with the 312-row supplied file unless
explicitly told otherwise.