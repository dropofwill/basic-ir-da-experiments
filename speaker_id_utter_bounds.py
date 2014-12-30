"""
Challenge Problem, Speaker Identification and Utterance Segmentation

Takes a path to a text file as input
Returns a tab seperated file

To save redirect to a tsv/csv file

python ps16_w_paul.py some_file.txt > some_file.tsv
"""

import os
import sys
import getopt
import csv
import nltk
from nltk.tokenize.simple import SpaceTokenizer
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
import nltk.probability
import pprint
pp = pprint.PrettyPrinter(indent=4)

def generate_tokens_and_bounds(sents):
    tokens = []
    bounds = set()
    offset = 0
    for sent in sents:
        tokens.extend(sent)
        offset += len(sent)
        #print len(sent), offset
        bounds.add(offset-1)

    #for b in bounds:
        #print tokens[b]

    return tokens, bounds

def basic_features(tokens, i):
    """
    Words that are around utterance boundaries
    """
    next1 = tokens[i+1] if i+1 < len(tokens) else "</s>"
    next2 = tokens[i+2] if i+2 < len(tokens) else "</s>"
    prev1 = tokens[i-1] if i-1 >= 0 else "<s>"
    prev2 = tokens[i-2] if i-2 >= 0 else "<s>"

    return {"next-word": next1,
            "word-after-next": next2,
            "prev-word": prev1,
            "word-before-prev": prev2 }

def disfluency_w_bounds_features(tokens, i, bounds_set,
        disfluency_set=["{SL}", "{LS}", "{NS}"]):
    """
    Whether disfluencies exist around utterance boundaries
    """
    bounds_array = sorted(list(bounds_set))
    #print sorted(bounds_array)
    sent_len = None

    for j, b in enumerate(bounds_array):
        # if current token is within the bounds do:
        if bounds_array[j] <= i and bounds_array[j+1] > i:
            #print bounds_array[j], i, bounds_array[j+1]
            sent_len =  bounds_array[j+1] - bounds_array[j]
            #print sent_len
            break

    bound_tokens = { "next1": tokens[i+1] if i+1 < len(tokens) else "</s>",
                    "next2": tokens[i+2] if i+2 < len(tokens) else "</s>",
                    "prev1": tokens[i-1] if i-1 >= 0 else "<s>",
                    "prev2": tokens[i-2] if i-2 >= 0 else "<s>",
                    "sent-len-greater-100": sent_len > 100 }

    features = {}
    disfluencies = disfluency_set

    # Set default features to false
    for k in bound_tokens:
        for dis in disfluencies:
            features[str(k) + dis] = False

    for k in bound_tokens:
        for dis in disfluencies:
            if bound_tokens[k] == dis:
                features[str(k) + dis] = True

    return features

def disfluency_features(tokens, i, disfluency_set=["{SL}", "{LS}", "{NS}"]):
    bound_tokens = { "next1": tokens[i+1] if i+1 < len(tokens) else "</s>",
        "next2": tokens[i+2] if i+2 < len(tokens) else "</s>",
        "prev1": tokens[i-1] if i-1 >= 0 else "<s>",
        "prev2": tokens[i-2] if i-2 >= 0 else "<s>",
        "prev-is-same": tokens[i-1] == tokens[i] if i-1 >= 0 else False,
        "next-is-same": tokens[i+1] == tokens[i] if i+1 < len(tokens) else False}

    features = {}
    disfluencies = disfluency_set

    # Set default features to false
    for k in bound_tokens:
        for dis in disfluencies:
            features[str(k) + dis] = False

    for k in bound_tokens:
        for dis in disfluencies:
            if bound_tokens[k] == dis:
                features[str(k) + dis] = True

    return features

def segment_tokens(tokens, classifier,
        feature_selector=lambda tokens, i: disfluency_features(tokens, i)):
    begin = 0
    sents = []
    for i, word in enumerate(tokens):
        if classifier.classify(feature_selector(tokens, i)) == True:
            sents.append(tokens[begin:i+1])
            begin = i+1
    if begin < len(tokens):
        sents.append(tokens[begin:])
    return sents


def read_csv(path):
    output = []
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            sent = [w for w in SpaceTokenizer().tokenize(row[1])]
            output.append(sent)
    return output

def read_boundary_corpus(input_dir):
    corpus = []
    for f in os.listdir(input_dir):
        if f.startswith("AS1"):
            input_path = os.path.join(input_dir, f)
            corpus += read_csv(input_path)
    return corpus

def naive_bayes(features, tokens):
    size = int(len(features) * 0.8)
    train, test = features[size:], features[:size]
    classifier = nltk.NaiveBayesClassifier.train(train)
    #print len(train), len(test)
    #print nltk.classify.accuracy(classifier, test)
    #print nltk.classify.accuracy(classifier, train)
    #classifier.show_most_informative_features()

    #pp.pprint(segment_tokens(tokens[:size], classifier))
    return classifier

def svc(train, train_labels, test, test_labels):
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import metrics

    classifier = LinearSVC(penalty="l1", dual=False, C=2)

    vectorizer = TfidfVectorizer(analyzer="word")
    tfidf = vectorizer.fit(train)
    X_train = tfidf.transform(train)

    classifier.fit(X_train, train_labels)
    X_test = tfidf.transform(test)

    prediction = classifier.predict(X_test)
    score = metrics.accuracy_score(test_labels, prediction)
    #print score

    return classifier, tfidf

# did as badly as bag of words
def decision_tree(features, tokens):
    size = int(len(features) * 0.4)
    train, test = features[size:], features[:size]
    classifier = nltk.DecisionTreeClassifier.train(train)
    ##print len(train), len(test)
    #print nltk.classify.accuracy(classifier, test)
    #print nltk.classify.accuracy(classifier, train)
    classifier.pseudocode()

    return classifier


def find_boundaries(file_path):
    sents = read_boundary_corpus("./transcripts/")

    real_file_content = open(file_path).read()
    real_tokens = SpaceTokenizer().tokenize(real_file_content)

    tokens, bounds = generate_tokens_and_bounds(sents)

    # disfluencies plus some words that seem to start utterances
    mix_features = [
            (disfluency_features(tokens, i,
                ["{SL}", "{LS}", "{BR}", "{NS}", "UM", "UH", "WELL", "OK",
                    "ALRIGHT", "I", "YEAH", "YES", "HM"]),
            (i in bounds))
            for i in range(1, len(tokens) -1)]
    nb_mix_class = naive_bayes(mix_features, tokens)

    #print "\n"
    #print "Naive Bayes with Mix real"
    output = segment_tokens(real_tokens, nb_mix_class)
    utterances = []
    for i, sent in enumerate(output):
        utterance = " ".join(sent)
        utterances.append(sent)

    return utterances

def edit_tokens (input_words):
    """
    Merge "{", ".+", "}" into one token
    """
    length = len(input_words)
    for i, w in enumerate(input_words):
        if (w == "{"):
            if (i+2 < length):
                if (input_words[i+2] == "}"):
                    input_words[i:i+3] = [''.join(input_words[i:i+3])]
    return input_words


def prepare_split(facilitator_file, participant_file, reader):
    fac_words = [w for w in reader.words(facilitator_file) if w != "sp"]
    par_words = [w for w in reader.words(participant_file) if w != "sp"]
    fac_words = edit_tokens(fac_words)
    par_words = edit_tokens(par_words)

    fac_labels = ['facilitator' for w in fac_words]
    par_labels = ['participant' for w in par_words]

    return fac_words + par_words, fac_labels + par_labels


def identify_speakers(dir_path):
    reader = CategorizedPlaintextCorpusReader(dir_path,
                    r'.*\.txt', cat_pattern=r'.+_.+_(.*)\.txt')
    facilitator_files = reader.fileids(categories='facilitator')
    participant_files = reader.fileids(categories='participant')
    #print facilitator_files, participant_files

    train_data, train_labels = prepare_split(facilitator_files[0], participant_files[0], reader)
    test_data, test_labels = prepare_split(facilitator_files[1], participant_files[1], reader)

    svc_vect_class = svc(train_data, train_labels, test_data, test_labels)
    return svc_vect_class

def main(argv):
    # parse command line options
    try:
        # list of argv from below,
        # string of one letter options e.g. -h, -w
        # list of long options, e.g. --help
        opts, args = getopt.getopt(argv, "h", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)

    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

    utterances = find_boundaries(args[0])
    identity_classifier, vectorizer = \
    identify_speakers('./ps16_dev_data/plain_text/AS1_split_fac_part_plain_text')

    p_overall, f_overall = 0, 0
    for utterance in utterances:
        utterances_transformed = vectorizer.transform(utterance)
        prediction = identity_classifier.predict(utterances_transformed)

        p_val, f_val = 0, 0
        for p in prediction:
            if str(p) == "participant":
                p_val += 1
            if str(p) == "facilitator":
                f_val += 1

        if p_val > f_val:
            identity = "participant"
            p_overall += 1
        else:
            identity = "facilitator"
            f_overall += 1

        print identity, "\t", " ".join(utterance)


if __name__ == "__main__":
    # first argv is the script name, which we don't care about
    # so just pass the rest of it to main
    main(sys.argv[1:])
