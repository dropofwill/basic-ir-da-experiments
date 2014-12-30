"""
Corpus excercise.

-a Finds all words that occur at least 3 times in the Brown Corpus

-b "String to translate" converts argument to Pig Latin
"""

import sys
import re
import getopt
from nltk import FreqDist
from nltk.corpus import brown
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.tokenize.simple import SpaceTokenizer


def get_freq_words(freq=3, external_corpus=None):
    """
    freq: 3, how many times does a word occur to be kept
    external_corpus: None, use all of the Brown corpus by default
        Expects corpus to be pre-processed into a list of words already.
    """

    # Default to using the entire brown corpus
    if (external_corpus is None):
        corpus_words = brown.words()
    else:
        corpus_words = external_corpus

    # Basic normalization
    corpus_words = [word.lower() for word in corpus_words]

    fd = FreqDist(corpus_words)

    result = []
    for w, f in fd.items():
        if f >= freq:
            result.append(w)

    #print fd.items()
    return result

def eng_word_to_pig_latin(word, append="ay"):
    """
    Takes a given word and converts it to pig latin
    word: the string to convert
    append: "ay", the ending string to append
    """

    # Verbose regex to capture the leading syllables of the word
    initial_consonants = r'''
    \b                                         # Beginning of word
    (
    [BCDFGHJKLMNPRSTVWXYZbcdfghjklmnprstvwxyz] # First consonants, include y
    [BCDFGHJKLMNPRSTVWXZbcdfghjklmnprstvwxz]?  # Optional consonants, exclude qy
    |(qu|Qu|qU|QU)                             # Treat qu as a consonant
    )
    '''

    front = re.search(initial_consonants, word, re.VERBOSE)

    # Re-slice the string and append ending
    if front:
        pl_word = word[front.end():] + word[front.start():front.end()] + append
    else:
        pl_word = word + append

    # handle capitalization
    if (word.istitle()):
        pl_word = pl_word.title()

    return pl_word

def eng_text_to_pig_latin(text, append="ay", tokenizer=PunktWordTokenizer):
    """
    Tokenize a given text and convert each word to pig latin
    text: the string to convert
    append: "ay", the ending string to append

    returns: pig latin string
    """

    text = [w for w in tokenizer().tokenize(text)]
    text = [eng_word_to_pig_latin(w, append) for w in text]
    return " ".join(text)

def main(argv):
    # parse command line options
    try:
        # list of argv from below,
        # string of one letter options e.g. -h, -w
        # list of long options, e.g. --help
        opts, args = getopt.getopt(argv, "hab", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

        if o in ("-a"):
            freq_words = get_freq_words()
            print """Finding tokens that occur more
                 than 3 times in the Brown Corpus"""
            print ", ".join(freq_words)
            print "Found",len(freq_words), "types that occur more than 3 times in the Brown Corpus"
            sys.exit(0)

        if o in ("-b"):
            print "Converting to Pig Latin..."
            if len(args):
                print eng_text_to_pig_latin(" ".join(args))
            else:
                print __doc__
            sys.exit(0)

    if not len(args):
        print __doc__
        sys.exit(0)

if __name__ == "__main__":
    # first argv is the script name, which we don't care about
    # so just pass the rest of it to main
    main(sys.argv[1:])
