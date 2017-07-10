
input1 = "../../resources/bingliu-positive.txt"
input2 = "../../resources/bingliu-negative.txt"
output = "../../resources/bingliu.tsv"

my_final_dict = {}

def loadFile(file_path, final_dict):
    fh = open(file_path)
    sentiment = file_path.split("-")[1].split('.txt')[0]
    for line in fh:
      try:
        word = line.strip()
        if not word in final_dict.keys():
            final_dict[word] = sentiment
        else:
            print(word, "seems to exist in both lexicons!!")
      except:
          print(line)
    fh.close()

loadFile(input2,my_final_dict)
print(len(my_final_dict))

loadFile(input1,my_final_dict)
print(len(my_final_dict))

fw = open(output,"w")
for key,value in my_final_dict.items():
    fw.write(key + "\t" + str(value))
    fw.write("\n")
fw.close()
