#code for models/eval.py
import os, sys, pickle
from collections import defaultdict, Counter
MODEL=os.path.join('models','trigram_model_full.pkl')
if not os.path.exists(MODEL):
    print('model not found:', MODEL); sys.exit(1)
with open(MODEL,'rb') as f:
    m=pickle.load(f)

#safe loads and fallbacks
vocab=m.get('vocab',[])
v2i=m.get('v2i',{})
i2v=m.get('i2v',{})
bigram_next=m.get('bigram_next', defaultdict(Counter))
bigram2=m.get('bigram2', defaultdict(Counter))
unigrams=m.get('unigrams', Counter())

#greedy next word prediction
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
        if nxt=='</s>': break
        w1,w2=w2,nxt
    return ' '.join(out)

#Command Line Interface
if len(sys.argv)>=3:
    w1=sys.argv[1].lower(); w2=sys.argv[2].lower()
    maxlen = int(sys.argv[3]) if len(sys.argv)>3 else 20
    print(generate((w1,w2), maxlen=maxlen))
else:
    print('usage: python models/eval.py <word1> <word2> [maxlen]')
    print('example: python models/eval.py the quick 25')
