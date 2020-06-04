__author__ = '{Alfonso Aguado Bustillo}'

from numpy import array


def split(string):
    return [char for char in string]


def one_hot_encoding(data):

    data_encoded = []

    for index, dna_sequence in data.items():
        nucleotides = array(split(dna_sequence))  # get dna sequence

        one_hot = []

        for nucleotide in nucleotides:
            if nucleotide == 'A':
                one_hot.append([1., 0., 0., 0.])
            elif nucleotide == 'T':
                one_hot.append([0., 1., 0., 0.])
            elif nucleotide == 'C':
                one_hot.append([0., 0., 1., 0.])
            elif nucleotide == 'G':
                one_hot.append([0., 0., 0., 1.])

        one_hot = array(one_hot).flatten()  # flatten to have one-dimension encoded sequence
        data_encoded.append(one_hot)

    data_encoded = array(data_encoded)

    return data_encoded


def binary_encoding(data, nucleotide="A"):

    data_encoded = []

    for index, dna_sequence in data.items():
        nucleotides = array(split(dna_sequence))  # get dna sequence

        binary = []

        for nucleotide in nucleotides:
            if nucleotide == 'A':
                binary.append([1.])
            else:
                binary.append([0.])

        binary = array(binary).flatten()  # flatten to have one-dimension encoded sequence
        data_encoded.append(binary)

    data_encoded = array(data_encoded)

    return data_encoded
