# 3-gram Language Model (NumPy)

A basic implementation of a trigram language model built from scratch using Python and NumPy.  
The project trains on a text corpus, builds unigram/bigram/trigram statistics, and supports:

- Sentence generation (`eval.py`)  
- Next word prediction (`predict_one.py`)  
- REST API serving with FastAPI (`api.py`)  

## Usage

Train in Jupyter Notebook:
```bash
jupyter notebook notebooks/3gram_model.ipynb
Run evaluation / prediction:

python3 models/eval.py the quick 20
python3 models/predict_one.py the quick


Start API:

uvicorn api:app --reload --port 8000

Example Output
Input: the quick
Generated: the quick brown fox jumps over the lazy dog ...


Future Work: Add Kneser–Ney smoothing, sampling with temperature, and top-k decoding.


---
