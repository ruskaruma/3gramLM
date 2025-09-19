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
```
Run evaluation / prediction:
```bash
python3 models/eval.py the quick 20
python3 models/predict_one.py the quick
```

Start API:
```bash
uvicorn api:app --reload --port 8000
```

Example Output:
```bash
    Input: the quick
    Generated: the quick brown fox jumps over the lazy dog ...
```

Future Work: Add Kneser–Ney smoothing, sampling with temperature, and top-k decoding.

## Contributing

Contributions, issues, and feature requests are welcome!  
If you’d like to improve this project:

1. Open an **Issue** describing the bug, feature, or enhancement.  
2. Fork the repository and create a new branch.  
3. Open a **Pull Request (PR)** with your changes.  

## License

This project is licensed under the [Apache License 2.0](LICENSE).  

© 2025 Ruskaruma. Licensed under the Apache License, Version 2.0 (the "License")  
You may not use this project except in compliance with the License.  
---
