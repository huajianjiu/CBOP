CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: word2vec compute-accuracy

word2vec : pp_word2vec.c
	$(CC) pp_word2vec.c -o pp_word2vec $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh
debug : pp_word2vec.c
	$(CC) pp_word2vec.c -o pp_debug $(CFLAGS) -g

clean:
	rm -rf pp_word2vec compute-accuracy pp_debug
