"""
Generate Speaker Classified sorted transcripts from the Praat textgrids
"""

import sys
import re
import os
import getopt
import tgt
import csv
import operator
import pprint
import copy
from nltk.tokenize.simple import SpaceTokenizer
pp = pprint.PrettyPrinter(indent=4)

def get_tg(filename):
    textgrid = tgt.io.read_textgrid(filename)
    #print textgrid.get_tier_names()

    if textgrid.has_tier("facilitator"):
        fac = textgrid.get_tier_by_name("facilitator")

    if textgrid.has_tier("participant"):
        par = textgrid.get_tier_by_name("participant")

    for tier_name in textgrid.get_tier_names():
        if str(tier_name) != "facilitator" and str(tier_name) != "participant":
            textgrid.delete_tier(tier_name)

    #if textgrid.has_tier("phone"):
        #textgrid.delete_tier("phone")

    #if textgrid.has_tier("task"):
        #textgrid.delete_tier("task")
    #print textgrid.get_tier_names()
    #overlap = tgt.util.get_overlapping_intervals(fac, par)
    #print overlap
    return textgrid

def write_tg_select(tg, txt):
    """
    Experimental, didn't get it to work
    """
    table = tgt.io.export_to_table(tg, separator="\t")
    #print table
    table = table.split('\n')
    table_csv = list(csv.reader(table, delimiter="\t"))

    # don't need to know what type of tier it is
    for row in table_csv:
        del row[1]

    # sp don't exist in the the plain text files
    data = copy.deepcopy(table_csv)
    cur_i = 0
    for i, row in enumerate(table_csv):
        sp_r = re.search(r'sp', row[3])
        sym_r = re.search('[\*\+\-]*', row[3])
        if sp_r:
            #print data[i][3]
            data[i][3] = re.sub(sp_r.group(0), "", row[3])
            print data[i][3]

        #print data[i][3]
        data[i][3] = ''.join(char for char in data[i][3] if char.isalpha())
        #print data[i][3]

    data2 = copy.deepcopy(data)
    for row in data:
        if row[3] == "":
            data2.remove(row)

    #pp.pprint(data2)

    fac_data, par_data = [], []
    for row in data2:
        if row[0] == "facilitator":
            fac_data.append(row)
        if row[0] == "participant":
            par_data.append(row)

    file_content = open(txt).read()
    tokens = SpaceTokenizer().tokenize(file_content)
    for i, token in enumerate(tokens):
        print token
        tokens[i] = ''.join(char for char in token if char.isalpha())
        print token

    #print tokens
    #pp.pprint(fac_data)
    #pp.pprint(par_data)

    fac_i, par_i, trans_i = 0, 0, 0

    for token in tokens:
        #print token, fac_data[fac_i], par_data[par_i]
        if fac_i < len(fac_data):
            if token == fac_data[fac_i][3]:
                fac_i += 1
        if par_i < len(par_data):
            if token == par_data[par_i][3]:
                par_i += 1

    #print fac_i, len(fac_data) - 1
    #print par_i, len(par_data) - 1
    #print len(tokens), len(fac_data) + len(par_data)


def write_tg_sort(tg, filename="./test_tg.csv"):
    table = tgt.io.export_to_table(tg, separator="\t")
    #print table
    table = table.split('\n')
    table_csv = list(csv.reader(table, delimiter="\t"))

    # don't need to know what type of tier it is
    for row in table_csv:
        del row[1]

    #print table_csv[0]
    # remove table header
    del table_csv[0]
    #print table_csv[0]

    # sp don't exist in the the plain text files
    data = [row for row in table_csv if row[3] != "sp"]

    # make sure it sorts correctly
    convert_cells_to_floats(data)

    # sort by starting time
    sorted_table = sorted(data, key=operator.itemgetter(1))
    #print sorted_table

    export = []
    prev_speaker = None
    iterator = -1
    # combine adjacent speech rows into a string
    for row in sorted_table:
        if row[0] == prev_speaker:
            speech = " " + str(row[3])
            export[iterator][1] += speech
        else:
            # Silence on it's own doesn't tell us anything
            if row[3] != "{SL}" and row[3] != "sp":
                iterator += 1
                # print timestamps
                #time = str(row[1]) + " - " + str(row[2])
                export.append([row[0], row[3]])
                prev_speaker = row[0]
            else:
                print "{SL} or sp on its own"

    # move header row back to top
    #export.insert(0, export.pop())

    with open(filename, 'w') as out:
        writer = csv.writer(out, delimiter='\t', quotechar='\"')
        for row in export:
            writer.writerow(row)

def process_dir_tg(input_dir, output_dir):
    for f in os.listdir(input_dir):
        if f.endswith("TextGrid") and f.startswith("AS1"):
            print f
            input_path = os.path.join(input_dir, f)
            tg = get_tg(input_path)
            output_filename = f[:-8] + "tsv"
            output_path = os.path.join(output_dir, output_filename)
            write_tg_sort(tg, output_path)

def process_file_tg(input_tg, input_stream):
    tg_file = get_tg(input_tg)
    return write_tg_select(tg_file, input_stream)

def convert_cells_to_floats(csv_cont):
    """
    Converts cells to floats if possible
    (modifies input CSV content list).
    """

    for row in range(len(csv_cont)):
        for cell in range(len(csv_cont[row])):
            try:
                csv_cont[row][cell] = float(csv_cont[row][cell])
            except ValueError:
                pass

process_dir_tg("ps16_dev_data/TextGrids/", "./transcripts/")
#process_file_tg("ps16_dev_data/TextGrids/AS1_T1_Stereo.TextGrid", "ps16_dev_data/plain_text/AS1_T1_Stereo.txt")

#tg = get_tg("./ps16_dev_data/TextGrids/AS1_T1_Stereo.TextGrid")
#write_tg(tg)
