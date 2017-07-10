

file_path = "/Users/sowmya/Downloads/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"
output_path = "../resources/mpqa.tsv"
#file structure: type=strongsubj len=1 word1=wonderous pos1=adj stemmed1=n priorpolarity=positive

fh = open(file_path)
fw = open(output_path, "w")
for line in fh:
    try:
        word = line.strip().split(" ")[2].split("=")[1]
        polarity = line.strip().split(" ")[5].split("=")[1]
        fw.write(word + "\t" + polarity)
        fw.write("\n")
    except:
        print("Exception!", line)
fw.close()
fh.close()
