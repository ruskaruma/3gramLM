# models/api.py
import os,pickle
from collections import defaultdict,Counter
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# robust model path: model file in same dir as this script
HERE=os.path.dirname(__file__)
MODEL=os.path.join(HERE,'trigram_model_full.pkl')
if not os.path.exists(MODEL):
    raise RuntimeError('model not found: {}'.format(MODEL))

with open(MODEL,'rb') as f:
    m=pickle.load(f)

vocab=m.get('vocab',[])
v2i=m.get('v2i',{})
i2v=m.get('i2v',{})
bigram_next=m.get('bigram_next',defaultdict(Counter))
bigram2=m.get('bigram2',defaultdict(Counter))
unigrams=m.get('unigrams',Counter())

def predict_next(w1,w2):
    bn=(w1,w2)
    if bn in bigram_next and bigram_next[bn]:
        return max(bigram_next[bn].items(), key=lambda x:x[1])[0]
    if bigram2.get(w2):
        return max(bigram2[w2].items(), key=lambda x:x[1])[0]
    return max(unigrams.items(), key=lambda x:x[1])[0] if unigrams else '</s>'

def generate(start,maxlen=20):
    w1,w2=start
    out=[w1,w2]
    for _ in range(maxlen):
        nxt=predict_next(w1,w2)
        out.append(nxt)
        if nxt=='</s': break
        w1,w2=w2,nxt
    return ' '.join(out)

app=FastAPI()

class Pair(BaseModel):
    w1: str
    w2: str
    maxlen: Optional[int] = 20
    mode: Optional[str] = 'gen'

@app.post("/predict")
def predict(payload: Pair):
    w1=payload.w1.lower(); w2=payload.w2.lower()
    if payload.mode=='next':
        return {"next": predict_next(w1,w2)}
    return {"generation": generate((w1,w2), maxlen=payload.maxlen or 20)}
