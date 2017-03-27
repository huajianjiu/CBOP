import numpy as np
import sys
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

word_vector = {}
human_rank = []
machine_rank = []
total = 0
unseen = 0

def get_vector(w):
    try:
        return word_vector[w]
    except KeyError:
        return np.zeros(200)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        return np.zeros(200)

if __name__ == "__main__":
    if len(sys.argv)<3:
        print "Usage: python compute-wordsim.py <WORD_VECTOR_TXT_FILE> <TEST_DATASET> \n \
        where FILE contains human assigned similar scores. I will get the sim via word embeddings and output the spearman correlation"
        exit()
    # read word vectors
    with open(sys.argv[1], "r") as f:
        word_vector_lines = f.readlines()
        for word_vector_line in word_vector_lines:
            if len(word_vector_line.split(" ")) < 3: continue
            word_vector_line=word_vector_line.split(" ")
            word_vector[word_vector_line[0]] = np.array(word_vector_line[1:-1], dtype="float32")
    # read test set and calculate similarity
    with open(sys.argv[2], "r") as f:
        test_lines = f.readlines()
        total = len(test_lines)
        for test_line in test_lines:
            if len(test_line.split(",")) < 3: continue
            w1 = test_line.split(",")[0]
            w2 = test_line.split(",")[1]
            w1_vec = get_vector(w1)
            w2_vec = get_vector(w2)
            if w1_vec.all() == 0 or w2_vec.all() == 0: 
                unseen += 1
                continue
            else:   
                machine_rank.append(float(1.0) - cosine(w1_vec, w2_vec))
                human_rank.append(test_line.split(",")[2].strip())

    rho, pval = spearmanr(np.array(human_rank, dtype="float32"), \
                            np.array(machine_rank, dtype="float32"))
    print "unseen: "+str(unseen)+"/"+str(total)
    print rho