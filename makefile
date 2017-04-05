CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
PYTHONFLAGS = -I/usr/include/python2.7 -lpython2.7

all: word2vec cbop_word2vec compute-accuracy

word2vec : pp_word2vec.c
	$(CC) pp_word2vec.c -o pp_word2vec $(CFLAGS)
wsd : ppwsd_w2v.c
	$(CC) ppwsd_w2v.c -o ppwsd_w2v $(CFLAGS) $(PYTHONFLAGS) -g -ggdb
wsd_1thread : ppwsd_w2v_1thread.c
	$(CC) ppwsd_w2v_1thread.c -o ppwsd_w2v_1thread $(CFLAGS) $(PYTHONFLAGS) -g -ggdb
cbop_word2vec : cbop_word2vec.c
	$(CC) cbop_word2vec.c -o cbop_word2vec $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh
debug : cbop_word2vec.c
	$(CC) cbop_word2vec.c -o cbop_debug $(CFLAGS) -g

clean:
	rm -rf pp_word2vec cbop_word2vec compute-accuracy cbop_debug ppwsd_w2v ppwsd_w2v_1thread
