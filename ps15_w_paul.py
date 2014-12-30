"""
Generate n-gram models for two corpora:
    Brown, Sci-Fi Category
    Reuters, Coffee Category
"""

import sys
import getopt
import nltk
import imp
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist
from nltk.probability import ELEProbDist
import nltk.corpus

def build_ngram_model(words, n, gamma = 0.2, percent = 0.75):
    est = lambda fdist, bins: LidstoneProbDist(fdist, gamma)

    split = int(len(words) * percent)
    test = words[split:]
    train = words[:split]

    lm = NgramModel(n, test, estimator=est)
    return lm, test, train

def get_corpus_words(corpus_name, ids=None):
    corpus_method = getattr(nltk.corpus, corpus_name)

    if ids:
        return corpus_method.words(categories = ids)
    else:
        return corpus_method.words()

def print_result(result):
    for i, model in enumerate(result):
        print model

        print("{:<20}{:<}"
                .format("Perplexity on test:",
                model[0].perplexity(model[2])))

        print("{:<20}{:<}"
                .format("Generate 15 words:",
                " ".join(model[0].generate(15))))

def main(argv):
    # parse command line options
    try:
        # list of argv from below,
        # string of one letter options e.g. -h, -w
        # list of long options, e.g. --help
        opts, args = getopt.getopt(argv, "hw", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

    brown_words = get_corpus_words("brown", "science_fiction")
    reuters_words = get_corpus_words("reuters", "coffee")
    brown_models, reuters_models = [], []

    for i in range(3):
        brown_models.append( build_ngram_model(brown_words, i+1) )
        reuters_models.append( build_ngram_model(reuters_words, i+1) )

    print "Brown Corpus, Science Fiction"
    print_result(brown_models)
    print "Reuters Corpus, Coffee"
    print_result(reuters_models)


if __name__ == "__main__":
    # first argv is the script name, which we don't care about
    # so just pass the rest of it to main
    main(sys.argv[1:])
