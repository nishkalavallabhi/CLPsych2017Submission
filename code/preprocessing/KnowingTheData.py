import bs4
import os
import pprint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#This is for training data
'''
dir_path = "/Users/sowmya/Downloads/clpsych17-data/data/training/posts/"
author_categories_path = "/Users/sowmya/Downloads/clpsych17-data/data/training/author_rankings.tsv"
post_categories_path = "/Users/sowmya/Downloads/clpsych17-data/data/training/labels.tsv" #second field is labels.
'''

#This is for test data:
dir_path = "/Users/sowmya/Downloads/clpsych17-data/clpsych17-test/posts/"
author_categories_path = "/Users/sowmya/Downloads/clpsych17-data/clpsych17-test/user-rankings.tsv"
post_categories_path = "/Users/sowmya/Downloads/clpsych17-data/clpsych17-test/test_ids.tsv"

output_path = "testdata.csv"

def process_files(dir_path, post_mappings, output_path):
    files = os.listdir(dir_path)
    fw = open(output_path, "w")
    header = "postid,post_category,board_type,author_type,author_id,author_ranking,post_emoticon,post_content"
    fw.write(header)
    fw.write("\n")
    analyzer = SentimentIntensityAnalyzer()
    for file in files:
      if file.replace(".xml","").replace("post-","") in post_mappings:
        #print(file)
        print(file)
        file_path = os.path.join(dir_path,file)
        soup = bs4.BeautifulSoup(open(file_path, "rb").read().decode("utf-8", errors="ignore").lower(),'lxml-xml')
        try:
            post_id = soup.message['href'].split("/")[3]

            if soup.find_all("board"):
                board_type = soup.find_all("board")[0]['href'].split("/")[3]
            else:
                board_type = "Unknown"
                print("board_type unknown for: ", file_path)
            if soup.find_all("author"):
                author_type = soup.author['type']
            else:
                author_type = "Unknown"
                print("author_type unknown for: ", file_path)
          #  print(author_type)

            author_id = soup.author['href'].split("/")[3]
            body = bs4.BeautifulSoup(soup.body.get_text(),'lxml')
            post_content = body.get_text().strip().replace("\n"," ").replace('"',"\\'")
            if body.find_all("img"):
                imgs = body.find_all("img")[0]
                if imgs.find_all("class"):
                    post_emoticon = imgs.img['class']
            else:
                post_emoticon = "none"
          #  print(post_content)
          #  print(post_emoticon)
            post_category = post_mappings[post_id]
            author_ranking = author_mappings[author_id]
            output_string = ",".join([post_id,post_category,board_type,author_type,author_id,author_ranking,post_emoticon,'"'+post_content+'"'])
            if post_category == 'green':
                print(analyzer.polarity_scores(post_content))
            fw.write(output_string)
            fw.write("\n")
        except:
            print("could not read stuff for: ", file)
    fw.close()

def get_features(post_content):
    num_questionmarks = 0
    num_exclamations = 0
    num_nonwords = 0
    analyzer = SentimentIntensityAnalyzer()
    trial_sentence = "i don't know how long i can even keep myself together before i'm screwed."
    print(analyzer.polarity_scores(trial_sentence))

def load_post_categories(file_path):
    post_mappings = {}
    fh = open(file_path)
    for line in fh:
        splits = line.split("\t")
        post_mappings[splits[0]] = splits[1]
    fh.close()
    return post_mappings

#Since test data does not come with categories, I am keeping "green" for everything.
def load_post_categories_testdata(file_path):
    post_mappings = {}
    fh = open(file_path)
    for line in fh:
        post_id = line.strip()
        post_mappings[post_id] = "green"
    fh.close()
    return post_mappings

def load_author_categories(file_path):
    author_mappings = {}
    fh = open(file_path)
    for line in fh:
        splits = line.split("\t")
        author_mappings[splits[0]] = splits[1].strip()
    fh.close()
    return author_mappings

#get_features("some conteont")

#For training data
'''
post_mappings = load_post_categories(post_categories_path)
author_mappings = load_author_categories(author_categories_path)
process_files(dir_path,post_mappings,output_path)
'''

#For testing:
post_mappings = load_post_categories_testdata(post_categories_path)
author_mappings = load_author_categories(author_categories_path)
process_files(dir_path,post_mappings,output_path)

print(post_mappings)
print(author_mappings)