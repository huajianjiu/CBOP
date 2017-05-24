from EwcWord2vec import *

brown = BrownCorpus()
text8 = Text8Corpus("../Corpus/text8")

model = EwcWord2vec(brown)
model.train(text8)