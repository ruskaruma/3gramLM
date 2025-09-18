#code for models/predict_one.py
import os, sys, pickle
from collections import defaultdict, Counter

MODEL=os.path.join('models','trigram_model_full.pkl')
if not os.path.exists(MODEL):
    print('model not found:', MODEL); sys.exit(1)

with open(MODEL,'rb') as f:
    m=pickle.load(f)

bigram_next=m.get('bigram_next', defaultdict(Counter))
bigram2=m.get('bigram2', defaultdict(Counter))
unigrams=m.get('unigrams', Counter())

def predict_next(w1,w2):
    bn=(w1,w2)
    if bn in bigram_next and bigram_next[bn]:
        return max(bigram_next[bn].items(), key=lambda x:x[1])[0]
    if bigram2.get(w2):
        return max(bigram2[w2].items(), key=lambda x:x[1])[0]
    return max(unigrams.items(), key=lambda x:x[1])[0] if unigrams else '</s>'

#CLI single next prediction
if len(sys.argv)>=3:
    w1=sys.argv[1].lower(); w2=sys.argv[2].lower()
    print(predict_next(w1,w2))
else:
    print('usage: python models/predict_one.py <word1> <word2>')
    print('example: python models/predict_one.py the quick')
