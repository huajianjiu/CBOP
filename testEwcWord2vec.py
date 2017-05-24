from EwcWord2vec import EwcWord2vec
from gensim.models.word2vec import Text8Corpus

text8 = Text8Corpus("../Corpus/text8")
ukwac = Text8Corpus("../ukWac/ukwac_subset_10M_processed")

model = EwcWord2vec(text8, workers=20)
model.train(ukwac)