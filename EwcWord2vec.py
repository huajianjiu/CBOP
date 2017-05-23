from gensim.models.word2vec import *

# TODO: check whether the ewc train_sg_pair and train_cbow_pair is used.
# TODO: if no, override the cython functions in the gensim implementation

# Override the train_batch_sg, train_batch_cbow, to make them use the train_xx_pair in this module.
# Plain Python. Cython ones in the future.

FAST_VERSION = -1
MAX_WORDS_IN_BATCH = 10000


def train_batch_sg(model, sentences, alpha, work=None):
    """
    Update skip-gram model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

            # now go over all words from the (reduced) window, predicting each one in turn
            start = max(0, pos - model.window + reduced_window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                # don't train on the `word` itself
                if pos2 != pos:
                    train_sg_pair(model, model.wv.index2word[word.index], word2.index, alpha)
        result += len(word_vocabs)
    return result


def train_batch_cbow(model, sentences, alpha, work=None, neu1=None):
    """
    Update CBOW model by training on a sequence of sentences.

    Each sentence is a list of string tokens, which are looked up in the model's
    vocab dictionary. Called internally from `Word2Vec.train()`.

    This is the non-optimized, Python version. If you have cython installed, gensim
    will use the optimized version from word2vec_inner instead.

    """
    result = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in model.wv.vocab and
                       model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            l1 = np_sum(model.wv.syn0[word2_indices], axis=0)  # 1 x vector_size
            if word2_indices and model.cbow_mean:
                l1 /= len(word2_indices)
            train_cbow_pair(model, word, word2_indices, l1, alpha)
        result += len(word_vocabs)
    return result


def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):
    # train_sg_pair with ewc loss
    print ("ewc train_sg_pair")
    if context_vectors is None:
        context_vectors = model.wv.syn0
    if context_locks is None:
        context_locks = model.syn0_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    lock_factor = context_locks[context_index]

    neu1e = zeros(l1.shape)

    if model.hs:
        # work on the entire tree at once, to push as much work into numpy's C routines as possible (performance)
        l2a = deepcopy(model.syn1[predict_word.point])  # 2d matrix, codelen x layer1_size
        fa = expit(dot(l1, l2a.T))  # propagate hidden -> output
        ga = (1 - predict_word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[predict_word.point] += outer(ga, l1)  # learn hidden -> output
            if model.fisher_syn1 is not None:
                print ("add ewc gradient for syn1")
                model.syn1[predict_word.point] += float(0) - \
                                                  alpha * model.lam * outer(model.fisher_syn1[predict_word.point],
                                                                            (l2a - model.star_syn1[predict_word.point]))
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [predict_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
            if model.fisher_syn1neg is not None:
                print ("add ewc gradient for syn1neg")
                model.syn1neg[word_indices] += float(0) - \
                                               alpha * model.lam * outer(model.fisher_syn1neg[word_indices],
                                                                         (l2b - model.star_syn1neg[word_indices]))
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        if model.fisher_syn0 is not None:
            print ("add ewc gradient for syn0")
            star_l1 = model.star_syn0[context_index]
            l1 += lock_factor * \
                  (neu1e - alpha * model.lam * outer(model.fisher_syn0[context_index],
                                                     (l1 - star_l1)))
        else:
            l1 += neu1e * lock_factor  # learn input -> hidden (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e


def train_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)
    # train_cbow_pair with ewc loss
    print ("ewc train_cbow_pair")
    if model.hs:
        l2a = model.syn1[word.point]  # 2d matrix, codelen x layer1_size
        fa = expit(dot(l1, l2a.T))  # propagate hidden -> output
        ga = (1. - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
            if model.fisher_syn1 is not None:
                print ("add ewc gradient for syn1")
                model.syn1[word.point] += float(0) - \
                                          alpha * model.lam * outer(model.fisher_syn1[word.point],
                                                                    (l2a - model.star_syn1[word.point]))
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = expit(dot(l1, l2b.T))  # propagate hidden -> output
        gb = (model.neg_labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if learn_hidden:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
            if model.fisher_syn1neg is not None:
                print ("add ewc gradient for syn1neg")
                model.syn1neg[word_indices] += float(0) - \
                                               alpha * model.lam * outer(model.fisher_syn1neg[word_indices],
                                                                         (l2b - model.star_syn1neg[word_indices]))
        neu1e += dot(gb, l2b)  # save error

    if learn_vectors:
        # learn input -> hidden, here for all words in the window separately
        if not model.cbow_mean and input_word_indices:
            neu1e /= len(input_word_indices)
        if model.fisher_syn0 is not None:
            print ("add ewc gradient for syn0")
            for i in input_word_indices:
                model.wv.syn0[i] += model.syn0_lockf[i] * \
                                    (neu1e - alpha * model.lam * outer(model.fisher_syn0[i],
                                                                       (model.wv.syn0[i] - model.star_syn0[i])))
        else:
            for i in input_word_indices:
                model.wv.syn0[i] += neu1e * model.syn0_lockf[i]

    return neu1e


def fisher_batch_sg(model, sentences, alpha, work=None):
    # TODO: remember to average the fisher information
    pass


def fisher_batch_cbow(model, sentences, alpha, work=None, neu1=None):
    # TODO: remember to average the fisher information
    pass


def fisher_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):
    pass


def fisher_cbow_pair(model, word, input_word_indices, l1, alpha, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)
    pass


class EwcWord2vec(Word2Vec):
    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3,
                 seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5,
                 null_word=0, trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, lam=15,
                 vocabfile="fullwiki_vocab.txt"):
        self.vocabfile = vocabfile
        self.lam = lam
        self.fisher_syn0 = None
        self.fisher_syn1 = None
        self.fisher_syn1neg = None
        self.star_syn0 = []
        self.star_syn1 = []
        self.star_syn1neg = []
        super(EwcWord2vec, self).__init__(sentences, size, alpha, window, min_count, max_vocab_size, sample, seed,
                                          workers, min_alpha, sg, hs, negative, cbow_mean, hashfxn, iter, null_word,
                                          trim_rule, sorted_vocab, batch_words)

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        # need the sentence count so call the original one
        super(EwcWord2vec, self).scan_vocab(sentences=sentences, progress_per=progress_per, trim_rule=trim_rule)
        # scan external vocab and update raw_vocab
        logger.info("collecting all words and their counts from external vocab file")
        vocab = self.raw_vocab
        with open(self.vocabfile, "r") as f:
            for line in f.readlines:
                vocab[line.split(" ")[0]] = line.split(" ")[1]
        self.raw_vocab = vocab

    def train(self, sentences, total_examples=None, total_words=None,
              epochs=None, start_alpha=None, end_alpha=None,
              word_count=0,
              queue_factor=2, report_delay=1.0):
        # The train override the train of Word2Vec and will be called in the super().__init__
        # print ("train!")
        # print (hasattr(self, "syn1"))
        # get fisher information at the end
        # hope the overrided _do_train_job will be used
        trained_word_count = \
            super(EwcWord2vec, self).train(sentences, total_examples=total_examples, total_words=total_words,
                                           epochs=epochs, start_alpha=start_alpha, end_alpha=end_alpha,
                                           word_count=word_count,
                                           queue_factor=queue_factor, report_delay=report_delay)

        # start of the part for ewc:
        # get star parameters
        self.star_syn0 = self.wv.syn0
        if hasattr(self, "syn1"):
            self.star_syn1 = self.syn1
        if hasattr(self, "syn1neg"):
            self.star_syn1neg = self.syn1neg
        self.calculate_fisher(sentences)

        return trained_word_count

    def calculate_fisher(self, sentences):
        # TODO: calculate fisher information using the frequent words.
        # TODO: for example, if power is 0.75, the corpus size is 1000, the most 750 frequent words are used.
        # TODO: sample for one_class(A word) or for all words
        # TODO: when sample one class, use the distribution the same as the the ns, i.e. w.r.t. tf*power
        # the vocab is sorted if this is no revision of the gensim implementation. May 2017.
        # to cover the whole parameters, similar with training, accumulate fisher information
        # Notice: use the all samples to cover the whole embedding layer as more as possible

        job_tally = 0
        # TODO: worker loop. accumulate the squared gradient of log likelihood

        # TODO: divide the accmulated squared log likelihood with the total

        pass

    def _do_train_job(self, sentences, alpha, inits):
        # override it by a clone to use the train_batch_xx in this module
        """
        Train a single batch of sentences. Return 2-tuple `(effective word count after
        ignoring unknown words and sentence length trimming, total word count)`.
        """
        print("ewc do_train_job")
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += train_batch_sg(self, sentences, alpha, work)
        else:
            tally += train_batch_cbow(self, sentences, alpha, work, neu1)
        return tally, self._raw_word_count(sentences)

    def _do_fisher_job(self, sentences, alpha, inits):
        print("ewc do_fisher_job")
        work, neu1 = inits
        tally = 0
        if self.sg:
            tally += fisher_batch_sg(self, sentences, alpha, work)
        else:
            tally += fisher_batch_cbow(self, sentences, alpha, work, neu1)
        return tally, self._raw_word_count(sentences)
