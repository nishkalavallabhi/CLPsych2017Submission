
'''
Add Lexicon features (based on sentiment lexicons online) to the data file.

input files:
trainingdata-withaddedfeatures.csv
testdata-withaddedfeatures-renamednewcatfeatures.csv

outputfiles with added features need to be created.
'''

import nltk
from gensim import utils

def loadDict(file_path):
    my_dict = {}
    fh = open(file_path)
    for line in fh:
        word = line.strip().split("\t")[0]
        sentiment = line.strip().split("\t")[1]
        my_dict[word] = sentiment

    fh.close()
    return my_dict

#features to get: fraction of positive/negative sentiment words from the dict in this text. denominator: num words in text that exist in dict.
#for Bingliu, mpqa files.
def getFractionSentimentFeatures(text,dictionary):
    words = text
    num_words = len(words)
    positive = 0
    negative = 0
    denominator = 1
    for word in words:
        if word.lower().isalpha() and word.lower() in dictionary.keys():
            sentiment = dictionary[word.lower()]
            if sentiment == "positive":
                positive +=1
            elif sentiment == "negative":
                negative +=1
    return (float)(positive)/denominator, (float)(negative)/denominator

#Since sentiments are numeric here, words > +1 are positive sentiment, < -1 are negative sentiment. in between are neutral which we ignore for now.
#Other than this, calculation is the same as above.
def getNumericSentimentFeatures(text,dictionary):
    words = text
    positive = 0
    negative = 0
    denominator = 1
    for word in words:
        if word.lower().isalpha() and word.lower() in dictionary.keys():
            sentiment = float(dictionary[word.lower()])
            if sentiment > 1.0:
                positive +=1
            elif sentiment < -1.0:
                negative +=1
            denominator +=1
    return (float)(positive)/denominator, (float)(negative)/denominator

def appendNewFeatures(input_path, output_path):
    hamilton = loadDict("../resources/hamilton-sentiment.tsv") #Numeric sentiment
    bingliu = loadDict("../resources/bingliu.tsv") #positive-negative sentiment
    mpqa = loadDict("../resources/mpqa.tsv")
    fh = open(input_path)
    header = fh.readline().strip()
    new_header = header + ",bingpos,bingneg,mpqapos,mpqaneg,hamiltonpos,hamiltonneg"

    fw = open(output_path, "w")
    fw.write(new_header)
    fw.write("\n")

    for line in fh:
        text = nltk.word_tokenize(utils.to_unicode(line.split('"')[1]).strip().lower())
        bingpos,bingneg = getFractionSentimentFeatures(text, bingliu)
        mpqapos,mpqaneg = getFractionSentimentFeatures(text,mpqa)
        hamiltonpos,hamiltonneg = getNumericSentimentFeatures(text,hamilton)
        result = [bingpos, bingneg, mpqapos, mpqaneg, hamiltonpos, hamiltonneg]
        towrite = line.strip() + "," + ",".join(str(item) for item in result)
        fw.write(towrite)
        fw.write("\n")
    fw.close()
    fh.close()


def main():
    #input_path = "../DataFiles/trainingdata-withaddedfeatures.csv"
    #output_path = "../DataFiles/trainingdata-withaddedfeatures-withlexiconfeatures.csv"

    input_path = "../DataFiles/testdata-withaddedfeatures-renamednewcatfeatures.csv"
    output_path = "../DataFiles/testdata-withaddedfeatures-renamednewcatfeatures-withlexiconfeatures.csv"
    appendNewFeatures(input_path,output_path)

if __name__ == '__main__':
    main()
