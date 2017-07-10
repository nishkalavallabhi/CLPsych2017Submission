from textblob import TextBlob
from statistics import mean, median


'''
get: num. words per post, num sentences per post, num. exclamations, num. question marks
get: mean, median, min, sum of sentence level sentiment acc. to TextBlob
get: 

removed posts: 135293, 135314, 136007, 138166, 139420 - somehow text was empty.

'''

def getFeatures(filepath, outputpath):
  fh = open(filepath)
  fw = open(outputpath, "w")

  header = fh.readline()     
  header = header +  ",num_sent,wordspersen,min_sentiment,sum_sentiments,mean_sentiments,median_sentiments,percent_pos,percent_neg,percent_neutral,num_qm,num_ex" 
  fw.write(header)
  fw.write("\n")

  for line in fh:
    text = line.split('"')[1]
    if text == "":
        text = "happy" #Usually these texts which are returning only empty strings contain only emoticons. There were 5 or 6 such empty texts in training data as well, and all were green. So, just doing this for now.
    print("post id", line.split(",")[0])
    features = getSentiment(text)
    output = line.strip()
    for feature in features:
       output += "," + str(feature)
    fw.write(output)
    fw.write("\n")
  fw.close()
  fh.close()


def getSentiment(text):
   blob = TextBlob(text)
   sentiments = []
   num_neg = 0
   num_pos = 0
   num_neu = 0
   num_qmarks = text.count("?")
   num_excl = text.count("!")
   for sentence in blob.sentences:
       pol = sentence.sentiment.polarity
       sentiments.append(pol)

       if pol > 0:
         num_pos += 1
       elif pol < 0:
         num_neg += 1
       else:
         num_neu += 1
   return [len(blob.sentences),round((float)(len(blob.words)/len(blob.sentences)),2), min(sentiments), round(sum(sentiments),2), 
round(mean(sentiments),2), round(median(sentiments),2), round((float)(num_pos)/len(blob.sentences),2), 
round((float)(num_neg)/len(blob.sentences),2), round((float)(num_neu)/len(blob.sentences),2), num_qmarks, num_excl]


def testing():
	print("Amber example")
	print(getSentiment("@blithe    thank you for your awesome advice - i feel much better. i had one of those mood swings where you feel like nothing can be achieved and all the self esteem drops and stuff. i am over it - i felt like i overreacted but i guess it got to do with the mood. thank you for your suggestions and advice, the websites were useful, thank you."))

	print("Green example")
	print(getSentiment("@distanceperson    your name is actually very cute!   i know how you feel - many people in most of my classes tease about my name and they even created a song out of my name which i found amusing  school is like that, we cannot run away from teasing. my advice is to ignore them, show them that you don't care, just smile and move on - i found out myself that silence and patience is the answer.    unfortunately, we cannot runaway from insults. don't take any insults in, hear from your ear and let it out of your ear at once. don't be sad - do not bother explain it to them that it is annoying you, peope who tease usually tease more when they find out that it annoys the person they are annoying.   like everyone says, don't be ashamed of your name. your name is cool (and cute!) and don't take those insults in.    i don't know if this post made you feel better but i hope it did xd   you're awesome btw <3"))

	print("Red example")
	print(getSentiment("she told me to leave her alone... i dont know what to do.... i dont want to mingle around i just want her back... thats the only thing that will make me feel better... shes to fsr away and busy with exams she will not leave her studies. we loved talking to each other about tandom topics.. family gossip and planning our future and lots of flirting... i used to wake her up every day and we used to sleep in the phone together...i dont know if i can talk to her... i dont want to risk losing her.. without  her... the job is no longer important anymore... i havent slept since yesterday and i cant stop crying"))

	print("Crisis example")
	print(getSentiment("neg: feeling a bit fragile today. kinda like there is too much going on inside my head to fully understand. wanting to just live my life with only me running it for once. feeling controlled by my parents, feeling like i can't stand up to them and tell them to back off, to lay off the expectations, to lay off me wanting to do thiings for them, mum especially. i'm just getting really tired of never being able to have time alone for myself to figure things out myself. wanting to move out already! pos: there is a lot going on inside my head right now, i am going to try some journaling later tonight - when i lay on the ground and throw out millions of coloured pens and textas to expres myself! i'm not sure on the whole parent thing, i'll think of some strategiest o try in my journalling. i am looking at moving out at the end of the year. i need that time to find my feet.   neg: felt really anxious when running into a parent from the last centre i did prac at (she's also a psychologist). part of me felt though like talking, but it had been so long since i last spoke to her i had no clue what to say. i also didn't feel like talking. i wanted to go home and be alone.   pos: part of me felt like talking because it was so easy to talk to her while on prac. i did end up talking to her, she updated me on her daughters condition they were trying to figure out when i was there. it was a good chat. by the end i understood why i had that inclination to stop and chat.    neg: now i'm once again confused. do i try to gain contact with her again, or do i try with pref #2 who i emailed over easter? i'm feeling all so blood confused about the whole damn thing. i just don't know why it all has to be so hard!? i don't even know what i want. it feels like talking to her will be harder than i thought, maybe i over simplified it in my head? maybe i'm over-complicating it now? what am i to do? why do i alway have to question everything???  pos: i can always try tomorrow, see which one i end up calling first? i have an aptointment with eheadsapce sometime this week, i can talk it over with them. i can do some journalling to see what i think would be best for me. given everything right now.   neg: feeling really numb right now. kind of just wanting to sef-medicate. nothing seems to feel normal anymore. i feel like each day i start in a pit of despair, and that's where i end up in the end, never managing to escapse the sameness of my pit of despair - nothing feels good anymore. pos: i'm aware the urge to self-medicate. i can manage the outcome of this.  i'm becoming aware of some of these feelings running around inside me, thus meaning i can try to work out some strategies to cope through them.   neg: not feeling like i have any coping strategies anymore, i remember writing a list ages ago, but i don't know what happened to it. nothing i am doing is working. pos: i am going to look at some factsheets, and make a bank of coping strategies tonight. it can be a project of types. i miss projects."))

#This is just to test that sentiment analyyer from TextBlob
#testing()

getFeatures("trainingdata-noaddnlfeatures.csv","trainingdata-withaddedfeatures.csv")
getFeatures("testdata-noaddlfeatures.csv","testdata-withaddedfeatures.csv")