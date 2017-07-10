"""
Hamilton et.al. (2016) EMNLP paper " Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora"
has a resource that has historical sentiment lexicons for thousands of words and adjectives.
I am taking the 2000s decade sentiment scores from both "frequent words" and "adjectives" section, and creating one unified
file for my use which will have a word and its average sentiment from both lexicons (if it exists in both text files).
Resource URL: https://nlp.stanford.edu/projects/socialsent/
"""


input1 = "../resources/2000s-freq.tsv"
input2 = "../resources/2000s-adj.tsv"
output = "../resources/hamilton-sentiment.tsv"

my_final_dict = {}

def loadFile(file_path, final_dict):
    fh = open(file_path)
    count = 0
    common = 0
    for line in fh:
      try:
        word = line.split("\t")[0]
        sentiment = float(line.split("\t")[1].strip())
        if word in final_dict.keys():
#            print(word, "exists in both lists!")
            common +=1
            val = final_dict[word]
            val = (val + sentiment)/2.0
            final_dict[word] = val
        else:
            final_dict[word] = float(sentiment)
        count += 1
      except:
          print(line)
    fh.close()
    print(str(count), "lines in this file", str(file_path))
    print(str(common), "lines in both files")


loadFile(input1,my_final_dict)
print(len(my_final_dict))

loadFile(input2,my_final_dict)
print(len(my_final_dict))

print(my_final_dict["fundamental"])

fw = open(output,"w")
for key,value in my_final_dict.items():
    fw.write(key + "\t" + str(value))
    fw.write("\n")
fw.close()




