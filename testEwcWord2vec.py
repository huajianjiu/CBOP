from EwcWord2vec import *

text8 = Text8Corpus("../Corpus/text8")
ukwac = Text8Corpus("../ukWac/ukwac_subset_10M_processed")

model = EwcWord2vec(text8)
model.train(ukwac)