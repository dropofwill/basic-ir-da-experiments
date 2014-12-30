"""
Levensthein Distance in NLTK

By default prints out the edit distance between the word
pairs from the Problem Set plus two pairs of my own.

Optionally pass two strings to compute their edit distance as well.
"""

import sys
import getopt
from nltk.metrics.distance import edit_distance

# Word pairs from the Problem Set and two of my choosing
word_pairs = [
    ('rain', 'shine'),
    ('RIT', 'MIT'),
    ('Google', 'Yahoo'),
    ('Democrats', 'Republicans'),
    ('Computational Linguistics', 'NLP'),
    ('Spanish', 'English'),
    ('Robots', 'Rboots'),
    ('their', 'there')
]

def lev_dist(word1, word2):
    return edit_distance(word1, word2)

def all_lev_dist():
    for w1, w2 in word_pairs:
        print("{:<30}{:<15}{:<}{:<}"
                .format(w1, w2, "Levensthein Distance: ", lev_dist(w1, w2)))

def main(argv):
    # parse command line options
    try:
        # list of argv from below,
        # string of one letter options e.g. -h, -w
        # list of long options, e.g. --help
        opts, args = getopt.getopt(argv, "h", ["help", "all"])

    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        if o in ("-a", "--all"):
            all_lev_dist()
            sys.exit(0)

    if len(args) >= 2:
        print lev_dist(args[0], args[1])

if __name__ == "__main__":
    # first argv is the script name, which we don't care about
    # so just pass the rest of it to main
    main(sys.argv[1:])
