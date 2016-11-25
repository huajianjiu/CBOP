time ./pp_word2vec -train text8 -output t00.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 0.0
./compute-accuracy t00.bin < questions-words.txt > text8_t00_qw.txt
python compute-wordsim.py t00.bin.txt WS353.csv > 535_t00.txt

time ./pp_word2vec -train text8 -output t10.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 1.0
./compute-accuracy t10.bin < questions-words.txt > text8_t10_qw.txt
python compute-wordsim.py t10.bin.txt WS353.csv > 535_t10.txt

time ./pp_word2vec -train text8 -output t20.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 2.0
./compute-accuracy t20.bin < questions-words.txt > text8_t20_qw.txt
python compute-wordsim.py t20.bin.txt WS353.csv > 535_t20.txt

time ./pp_word2vec -train text8 -output t30.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 3.0
./compute-accuracy t30.bin < questions-words.txt > text8_t30_qw.txt
python compute-wordsim.py t30.bin.txt WS353.csv > 535_t30.txt

time ./pp_word2vec -train text8 -output t40.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 4.0
./compute-accuracy t40.bin < questions-words.txt > text8_t40_qw.txt
python compute-wordsim.py t40.bin.txt WS353.csv > 535_t40.txt

time ./pp_word2vec -train text8 -output t50.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 5.0
./compute-accuracy t50.bin < questions-words.txt > text8_t50_qw.txt
python compute-wordsim.py t50.bin.txt WS353.csv > 535_t50.txt

time ./pp_word2vec -train text8 -output t60.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 6.0
./compute-accuracy t60.bin < questions-words.txt > text8_t60_qw.txt
python compute-wordsim.py t60.bin.txt WS353.csv > 535_t60.txt

time ./pp_word2vec -train text8 -output t70.bin -cbow 1 -size 200 -window 8 -negative 25 -sample 1e-4 -threads 20 -binary 1 -iter 15 -dropout 1 -threshold 7.0
./compute-accuracy t70.bin < questions-words.txt > text8_t70_qw.txt
python compute-wordsim.py t70.bin.txt WS353.csv > 535_t70.txt

