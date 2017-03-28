time ./pp_word2vec -train text8 -output vectors_cbop_ppdb2 -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 3.5
./compute-accuracy vectors_cbop_ppdb2.bin < questions-words.txt
