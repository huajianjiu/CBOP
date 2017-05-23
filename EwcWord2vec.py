from gensim.models.word2vec import *

# TODO: check whether the ewc train_sg_pair and train_cbow_pair is used.

# Override the train_batch_sg, train_batch_cbow, to make it use the train_xx_pair functions defined here.
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


def fisher_batch_sg(model, sentences, work=None):
    # accumulate fisher information
    # remember to report accumulate times
    word_count = 0
    accum_count = 0
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
                    accum_count += fisher_sg_pair(model, model.wv.index2word[word.index], word2.index)
        word_count += len(word_vocabs)
    return word_count, accum_count

def fisher_batch_cbow(model, sentences, work=None, neu1=None):
    # TODO: accumulate fisher information
    # TODO: remember to average the fisher information
    # TODO: remember to report accumulate times
    word_count = 0
    accum_count = 0

    return word_count, accum_count

def fisher_sg_pair(model, word, context_index, learn_vectors=True, learn_hidden=True,
                   context_vectors=None, context_locks=None):
    # TODO: accumulate fisher information
    # TODO: remember to report accumulate times
    accum_count = 0

    return accum_count


def fisher_cbow_pair(model, word, input_word_indices, l1, learn_vectors=True, learn_hidden=True):
    neu1e = zeros(l1.shape)
    # TODO: accumulate fisher information
    # TODO: remember to report accumulate times
    accum_count = 0

    return accum_count


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
        self.calculate_fisher(sentences, total_examples=total_examples, total_words=total_words)

        return trained_word_count

    def calculate_fisher(self, sentences, total_examples=None, total_words=None,
                         word_count=0,
                         queue_factor=1, report_delay=1.0):
        # calculate fisher information using all of the samples to max cover the embedding layer and hidden layer.
        # the vocab is sorted if this is no revision of the gensim implementation. May 2017.
        # to cover the whole parameters, similar with training, accumulate fisher information
        # Notice: use the all samples to cover the whole embedding layer as more as possible

        job_tally = 0

        print("Start to accumulate Fisher information.")

        logger.info("Start to accumulate Fisher information.")

        # worker loop(consumer) and job producer. accumulate the squared gradient of log likelihood
        def worker_loop():
            """Accumulate Fisher Information, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences = job
                accum, tally, raw_tally = self._do_fisher_job(sentences, (work, neu1))
                progress_queue.put((len(sentences), accum, tally, raw_tally))  # report back progress and accum_count
                jobs_processed += 1
            logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    logger.debug(
                        "queueing job #%i (%i words, %i sentences)",
                        job_no, batch_size, len(job_batch))
                    job_no += 1
                    job_queue.put(job_batch)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
                logger.debug(
                    "queueing job #%i (%i words, %i sentences)",
                    job_no, batch_size, len(job_batch))
                job_no += 1
                job_queue.put(job_batch)

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 3) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, accum_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
                logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, accums, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            accum_count += accums
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                if total_examples:
                    # examples-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% examples, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * example_count / total_examples, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                else:
                    # words-based progress %
                    logger.info(
                        "PROGRESS: at %.2f%% words, %.0f words/s, in_qsize %i, out_qsize %i",
                        100.0 * raw_word_count / total_words, trained_word_count / elapsed,
                        utils.qsize(job_queue), utils.qsize(progress_queue))
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "Accumulate Fisher Information on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warning(
                "under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay"
            )

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warning(
                "supplied example count (%i) did not equal expected count (%i)", example_count, total_examples
            )
        if total_words and total_words != raw_word_count:
            logger.warning(
                "supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words
            )

        self.clear_sims()

        # divide the accmulated squared log likelihood with accum_count
        self.fisher_syn0 /= accum_count
        self.fisher_syn1 /= accum_count
        self.fisher_syn1neg /= accum_count

        return accum_count

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

    def _do_fisher_job(self, sentences, inits):
        print("ewc do_fisher_job")
        work, neu1 = inits
        tally = 0
        accum = 0
        if self.sg:
            word_count, accum_count = fisher_batch_sg(self, sentences, work)
            tally += word_count
            accum += accum_count
        else:
            word_count, accum_count = fisher_batch_cbow(self, sentences, work, neu1)
            tally += word_count
            accum += accum_count
        return accum, tally, self._raw_word_count(sentences)
