
#Get the textonly version of the xml files to build doc2vec models.
import os
import bs4

#folder_path = "/Users/sowmya/Downloads/clpsych17-data/clpsych17-test/posts"
folder_path = "/Users/sowmya/Downloads/clpsych17-data/data/training/posts"

output_file = "bulk-forvectormodels-fromtraining.txt"
fw = open(output_file, "w")

files = os.listdir(folder_path)
print("total size: ", str(len(files)))

count = 0
for file in files:
    file_path = os.path.join(folder_path,file)
    soup = bs4.BeautifulSoup(open(file_path, "rb").read().decode("utf-8", errors="ignore").lower(),'lxml-xml')
    try:
        body = bs4.BeautifulSoup(soup.body.get_text(),'lxml')
        post_content = body.get_text().strip().replace("\n"," ").replace('"',"\\'")
        fw.write(post_content)
        fw.write("\n")
        print("Wrote for: ", file)
        count += 1
    except:
        continue
        #print("Something wrong with file: ", file)

fw.close()

print("Done. Wrote content for: ", str(count), "files into", output_file)

#Notes: Try  sfsfssf