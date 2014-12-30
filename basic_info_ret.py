"""
Text pre-processing and basic information retrieval

Takes a url as an argument

-w or --wiki flag for Wikimedia specific retrieval

-p or --punct flag for punctuation as tokens mode
"""

import sys
import getopt
import urllib2
import nltk
import re
from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktWordTokenizer

def get_html(url):
    return urllib2.urlopen(url).read()

def get_text(html, wiki_mode=False):
    soup = BeautifulSoup(html)
    if (not wiki_mode):
        text = soup.body.get_text(" ")
    else:
        # A wiki media specific id for content
        soup = soup.find(id="mw-content-text")

        # Remove everything after references from text
        after_reference = False
        non_content_tags = []
        children = soup.findChildren()

        for child in children:
            if 'reflist' in  child.get('class', []):
                after_reference = True
            if after_reference:
                non_content_tags.append(child)

        [ref.decompose() for ref in non_content_tags]
        text = soup.get_text(" ")
    return text

def get_tokens(text, isalpha=True):
    """
    Pass a text to tokenize with Punkt
    Optionally ignore punctuation tokens
    """
    tokens = []
    for w in PunktWordTokenizer().tokenize(text):
        if isalpha:
            if w.isalpha():
                tokens.append(w.lower())
        else:
            tokens.append(w.lower())
    return tokens

def remove_citations_from(text):
    """
    Remove Wiki style citations from a
    pre tokenized text
    """
    citations = re.compile(r'(\[\s.*?\s\])', re.DOTALL)
    text = citations.sub('', text)
    return text

def get_mean_token_len(tokens):
    token_len, num = 0.0, 0.0

    for i, w in enumerate(tokens):
        token_len += len(w)
        num += 1

    return token_len / num

def get_types(tokens):
    return set(tokens)

def get_bigrams(tokens):
    bigrams = set([bi for bi in nltk.bigrams(tokens)])
    return bigrams

def get_longest_word_len(tokens):
    return max(len(w) for w in tokens)

def get_words_of_len(tokens, length):
    return [w for w in tokens if len(w) == length]

def get_fdist(tokens):
    return nltk.FreqDist(tokens)

def get_most_common(fdist_items, n=5, isalpha=True):
    if isalpha:
        most_common = [w for w, c in fdist_items if w.isalpha()]
    else:
        most_common = [w for w, c in fdist_items]
    return most_common[:n]

def percentage(n1, n2):
    return float(n1)/float(n2)

def type_difference(inclusive_set, exclusive_set):
    return inclusive_set - exclusive_set

def print_results(results):
    """
    Pretty print the resultsults dictionary
    """

    print("{:<40}{:>}"
        .format("Cardinality of tokens",
        len(results["tokens"])))

    print("{:<40}{:>}"
        .format("Cardinality of types:",
        len(results["types"])))

    print("{:<40}{:>}"
        .format("Cardinality of set of bigrams:",
        len(results["bigrams"])))

    print("{:<40}{:>}"
        .format("Mean token length in chars:",
        results["mean_token_len"]))

    print("{:<40}{:>}"
        .format("Length of longest tokens:",
        results["longest_word_len"]))

    print("{:<40}{:>}"
        .format("Set of longest tokens:",
        ", ".join(results["longest_words"])))

    print("{:<40}{:>}"
        .format("Top 5 most frequent tokens:",
        ", ".join(results["most_common"])))

    print("{:<40}{:>}"
        .format("Cardinality of hapax legomena:",
        len(results["hapaxes"])))

    print("{:<40}{:>.2%}"
        .format("Percentage of types that are hapaxes:",
        percentage(len(results["hapaxes"]), len(results["types"]))))

    print("{:<40}{:>.2%}"
        .format("Percentage of English types in use:",
        percentage(len(results["types"]), len(results["eng_types"]))))

    print("{:<40}{:>}"
        .format("Cardinality of unknown types:",
        len(type_difference(results["types"], results["eng_types"]))))

def get_stats(url, wiki_mode=False, punct_mode=False):
    """
    Get linguistic statistics about a given url
    """
    res = dict()
    text = get_text(get_html(url), wiki_mode)

    if wiki_mode:
        text = remove_citations_from(text)

    #print text

    if not punct_mode:
        res["tokens"] = get_tokens(text)
    else:
        res["tokens"] = get_tokens(text, not punct_mode)

    res["types"] = get_types(res["tokens"])
    res["bigrams"] = get_bigrams(res["tokens"])

    res["eng_tokens"] = [w.lower() for w in nltk.corpus.words.words()]
    res["eng_types"] = get_types(res["eng_tokens"])

    res["mean_token_len"] = get_mean_token_len(res["tokens"])
    res["longest_word_len"] = get_longest_word_len(res["types"])
    res["longest_words"] = get_words_of_len(res["types"],
                                            res["longest_word_len"])

    res["fdist"] = get_fdist(res["tokens"])
    res["fdist_items"] = res["fdist"].items()
    res["hapaxes"] = res["fdist"].hapaxes()

    if not punct_mode:
        res["most_common"] = get_most_common(res["fdist_items"])
    else:
        res["most_common"] = get_most_common(res["fdist_items"],
                                                isalpha = not punct_mode)
    return res


def main(argv):
    wiki_mode, punct_mode = False, False
    # parse command line options
    try:
        # list of argv from below,
        # string of one letter options e.g. -h, -w
        # list of long options, e.g. --help
        opts, args = getopt.getopt(argv, "hpw", ["help", "wiki", "punct"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

        if o in ("-w", "--wiki"):
            print "Using wiki mode..."
            wiki_mode = True

        if o in ("-p", "--punct"):
            print "Using punctuation as tokens mode..."
            punct_mode = True

    if len(args):
        given_url = args[0]
        stats = get_stats(given_url, wiki_mode, punct_mode)
        print_results(stats)
        # Plot cumulative frequent distribution
        stats["fdist"].plot(30, cumulative=True)
    else:
        print __doc__
        sys.exit(0)

if __name__ == "__main__":
    # first argv is the script name, which we don't care about
    # so just pass the rest of it to main
    main(sys.argv[1:])
