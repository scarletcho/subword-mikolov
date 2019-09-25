# -*- coding: utf-8 -*-

'''
subword_german.py
~~~~~~~~~

This script converts infrequent words into subwords in a given corpus in German,
following the method introduced in Mikolov, et al.(2012).

Usage:  $ python subword.py <corpus-filename> <W-parameter> <S-parameter>

Yejin Cho (ycho@utexas.edu)


Reference:
    Mikolov, et al. (2012), Subword Language Modeling with Neural Networks

Last updated: 2019-09-25

'''

import time
import numpy as np
import re
import sys

def readtext(fname):
    with open(fname, encoding="utf-8") as f:
        print('Currently reading text from file...')
        txt = f.readlines()
    print('Lines in text file is read as a list variable')
    return txt


def readtext2word(fname):
    startTime = time.time()
    wordlist = []
    with open(fname, encoding="utf-8") as f:
        cnt = 0
        for chunk in f:
            wrd = chunk.split()
            cnt += 1
            for w in wrd:
                wordlist.append(w)
    print("Read %d lines: resulting dictionary is %d elements long. Took %f seconds" % (cnt, len(wordlist),time.time()-startTime))

    return wordlist


def filter_by_frequency(wordlist, proportion):
    startTime = time.time()
    un, ind, inv, cou = np.unique(wordlist, return_index=True, return_inverse=True, return_counts=True)
    print("Unigram has %d unique words, Inverse indices are %d long. Took %f seconds" % (len(un), len(inv), time.time() - startTime))

    # this sorts the counts from high to low
    sorted_cou_ind = np.argsort(cou)[::-1]

    # sets the proportion of items to keep
    # proportion = 0.9
    nfreq = int(np.round(len(sorted_cou_ind) * proportion))

    # print out the counts and words for the nfreq most common unigrams
    print(sorted_cou_ind[0:nfreq], un[sorted_cou_ind[0:nfreq]])

    # Save frequent/infrequent as a separate array
    freq_ind = sorted_cou_ind[0:nfreq]
    freq_un = un[sorted_cou_ind[0:nfreq]]

    infreq_ind = sorted_cou_ind[nfreq:]
    infreq_un = un[sorted_cou_ind[nfreq:]]

    return freq_ind, freq_un, infreq_ind, infreq_un, ind, un, inv


def subword_split(infreq_unigram):
    subword_stack = []

    # Split all the remaining infrequent words into subwords
    for wrd_id in range(0, len(infreq_unigram)):
        wrd = infreq_unigram[wrd_id]
        vowels = ['a','e','i','o','u','ä','ö','ü']
        subwrd_tmp = []
        string = ''
        for i in range(0, len(wrd)):
            string += wrd[i]
            # Simply append single letter word as one subword
            if len(wrd) == 1:
                subwrd_tmp.append(string)
                string = ''

            # From second letter ...
            if i+1 != 1:
                # If stacked string has at least two letters and ends with vowel
                if wrd[i] in vowels and len(string) > 1:
                    subwrd_tmp.append(string)
                    string = ''
                # If reached the end of word
                elif i+1 == len(wrd):
                    if len(string) == 1:
                        subwrd_tmp[-1] += string
                    else:
                        subwrd_tmp.append(string)
                    string = ''

        subwrd = []
        for k in range(0, len(subwrd_tmp)):
            # If current subword is longer than 4 letters
            if len(subwrd_tmp[k]) >= 4:
                # Save first 2 letters as one subword
                subwrd.append(subwrd_tmp[k][0:2])
                # Save the rest as another subword
                subwrd.append(subwrd_tmp[k][2:])
            else:
                subwrd.append(subwrd_tmp[k])

        # print('------------------------------')
        # print('Word input: ')
        # print('\'' + wrd + '\'\n')
        # print('Subword output: ')
        # print(subwrd)
        # print('------------------------------')

        subword_stack.append(subwrd)

    subword_idx = []
    for p in range(0, len(subword_stack)):
        item = subword_stack[p]
        if len(item) > 1:
            subword_idx.append(p)
            buffer = ''
            for subitem in item:
                if subitem is not item[-1]:
                    buffer += subitem + '+ '
                else:
                    buffer += subitem + ':'
            subword_stack[p] = [buffer]

    return subword_stack, subword_idx


def character_split(infreq_unigram):
    subword_stack = []

    # Split all the remaining infrequent words into subwords
    for wrd_id in range(0, len(infreq_unigram)):
        wrd = infreq_unigram[wrd_id]
        wrd = re.sub('\+$', '', wrd)
        subwrd = ''

        for id in range(0, len(wrd)): # for id letter in wrd:
            letter = wrd[id]
            # prev = wrd[id-1]
            if id is not len(wrd)-1:
                if id > 0 and wrd[id-1] == '+':
                    continue
                elif id < len(wrd)-1 and wrd[id+1] is ':':
                    subwrd += letter
                elif letter is not '+':
                    subwrd += letter + '+ '

            if id is len(wrd)-1:
                if letter is ':':
                    subwrd += letter
                elif letter is '+':
                    subwrd += ':'
                else:
                    subwrd += letter + ':'

        if subwrd[-1] == ' ':
            subwrd = subwrd[:-1]

        # print('------------------------------')
        # print('Word input: ')
        # print('\'' + wrd + '\'\n')
        # print('Subword output: ')
        # print(subwrd)
        # print('------------------------------')

        subword_stack.append(subwrd)

    return subword_stack

def mikolov_subword(fname, w, s):
    # Add 1 to parameter W and S, in order to compensate for <eos> token
    # (which is added for convenient recovery of newlines and removed before returning final output text)
    # w += 1
    # s += 1

    print('# [STEP 0] Read corpus as wordlist')
    startTime = time.time()
    # txt = readtext(fname)
    wordlist1 = readtext2word(fname)


    print('# [STEP 1] Keep W most frequent words')
    freq1_ind, freq1_un, infreq1_ind, infreq1_un, full_ind1, full_un1, full_inv1 = filter_by_frequency(wordlist1, w)


    print('# [STEP 2] The first subword split')
    subword_stack, subword_idx = subword_split(infreq1_un)
    print('Flattening subword stack...')
    subword_stack_flat = [y for x in subword_stack for y in x]


    print('# [STEP 3] Replace infrequent words into designed subwords (initial split)')
    full_un1 = full_un1.tolist()
    full_inv1 = full_inv1.tolist()
    infreq1_ind = infreq1_ind.tolist()

    for k in subword_idx:
        # print(k)
        # print(full_un1[infreq1_ind[k]])
        # print(subword_stack_flat[k])
        full_un1[infreq1_ind[k]] = subword_stack_flat[k]
        # print(full_un1[infreq1_ind[k]])


    print('# [STEP 4] Keep S most frequent items')
    wordlist2 = []
    wordlist2.extend(np.array(full_un1)[full_inv1])

    wordlist2_flat = []
    for i in range(0, len(wordlist2)):
        wordlist2_flat.extend(wordlist2[i].split())

    freq2_ind, freq2_un, infreq2_ind, infreq2_un, full_ind2, full_un2, full_inv2 = filter_by_frequency(wordlist2_flat, s)


    print('# [STEP 5] The second subword split')
    subwrd_stack_char = character_split(infreq2_un)
    print("Final subword generation completed. Took %f seconds" % (time.time() - startTime))


    print('# [STEP 6] Replace infrequent words into designed subwords (final split)')
    full_un2 = full_un2.tolist()
    full_inv2 = full_inv2.tolist()
    infreq2_ind = infreq2_ind.tolist()

    for k in range(0, len(infreq2_ind)):
        full_un2[infreq2_ind[k]] = subwrd_stack_char[k]


    print('# [STEP 7] Concatenate subwords as regular text')
    txt = " ".join(np.array(full_un2)[full_inv2])
    txt = re.sub(' \+: ', ' ', txt)
    txt = re.sub('(?<= [a-z]): ', '+ ', txt)
    txt = re.sub(' ?\<eos\> ?', '\n', txt)
    print(txt)

    # print('\n========================================================================')
    # print('=> FINAL SUBWORD-LEVEL TEXT OUTPUT:')
    # print(txt)
    # print('========================================================================\n')

    return txt


if __name__ == '__main__':

    fname = "deu.words.train" # sys.argv[1]
    w = 0.2 # int(sys.argv[2])
    s = 1.0 # int(sys.argv[3])
    
    subwordtxt = mikolov_subword(fname, w, s)

    wordlist = []
    for item in subwordtxt:
        wordlist.extend(item.split())

    un, ind, inv, cou = np.unique(wordlist, return_index=True, return_inverse=True, return_counts=True)
    print("Unigram has %d unique items, Inverse indices are %d long." % (len(un), len(inv)))
    print(un)

    with open('sub_' + fname, 'w', encoding="utf-8") as f:
        for line in subwordtxt:
            f.write('{}'.format(line))
