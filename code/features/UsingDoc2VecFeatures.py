from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import imblearn #To use sampling methods to over or under sample the data.
import numpy
import pandas as pd
import nltk

import matplotlib.pyplot as plt
import itertools


'''
Things to change each time we build a new model.
build_doc2vec_model() - need to change the saving path, other settings in this.
loading the model in create_feature_labels_array should have the right path.
fitOneHotEncodings(numpy.array(big_vector),50,51) - those two arguments need to change according to doc2vec model dimensionality

Notes: what affects classification - dimensionality of vector does not do much beyond 30 or so. balanced class_weight has good impact
'''

def build_doc2vec_model():
    # Creating labeled sentences from training data
    sentences = TaggedLineDocument('bulk-total.txt')
    model = Doc2Vec(alpha=0.1, size=30, window=10, min_count=5, dm=0, dbow_words=1, iter=10)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=81863,epochs=10)
    model.save('../models/clpsych-30dim-large.d2v')

def load_doc2vec_model(path):
    loaded_model = Doc2Vec.load(path)
    return loaded_model

def fitOneHotEncodings(nparray,dim1,dim2):

    array_30 = ['getting_help', 'intros', 'feedback_suggestion', 'toughtimes_hosted_chats', 'something_not_right', 'mancave', 'everyday_life_stuff']#,'aceweek', 'games', 'getting_real_sessions']
    array_31 = ['Super frequent scribe', 'Post Mod', 'Star contributor', 'Frequent scribe',
                 'Uber contributor', 'Mod', 'Mod Squad', 'Youth Ambassador', 'Special Guest Contributor', 'Rookie scribe', 'Casual scribe', 'Frequent Visitor', 'Visitor', 'Rookie']#, 'Builder', 'Super star contributor']

    le = LabelEncoder()

    le.fit(array_31)
    new1 = le.transform(nparray[:,dim2])
    nparray[:,dim2] = new1

    le.fit(array_30)
    new2 = le.transform(nparray[:,dim1])
    nparray[:,dim1] = new2

    encoder = OneHotEncoder(categorical_features=[dim1,dim2])
    encoder.fit(nparray)
    return encoder.transform(nparray).toarray()

#Using the unsupervised doc2vec model to even infer training vectors.
def create_feature_labels_array(path):
    fh = open(path)
    fh.readline() #This line has only the header.
    text_vector = []
    labels = []
    loaded_model = Doc2Vec.load('../models/clpsych-30dim-large.d2v')
    ids = []

    for line in fh:
        text = nltk.word_tokenize(utils.to_unicode(line.split('"')[1]).strip().lower())
        other_features_part1 = line.split(',')
        ids.append(other_features_part1[0])
        other_features_part2 = line.split('"')[2].strip().split(',')
        temp = [other_features_part1[2], other_features_part1[5]] #These become: doc2vec dim +1, doc2vec dim +2
        temp.extend(float(e) for e in other_features_part2[1:])
        #print(temp)
        cat = line.split(',')[1]
        loaded_model.random.seed(0)
        temp_vec = []
        temp_vec.extend(loaded_model.infer_vector(text).tolist())
        temp_vec.extend(temp)
        #print(temp_vec)
        text_vector.append(temp_vec)
      #  print(numpy.concatenate((text_vector,temp),0))
       # text_vector.append(temp)
        labels.append(cat)
    fh.close()

    return text_vector, labels, ids

#Store the feature files for some other use
def write_to_csv(text_vector,text_cats, dimension, output_path):
    fh = open(output_path,"w")
    fh.write(getheader(dimension))
    fh.write("\n")

    for i in range(0,len(text_vector)):
        line = ",".join(str(e) for e in text_vector[i]) + "," + text_cats[i]
        fh.write(line)
        fh.write("\n")
    fh.close()

def getheader(num_rows):
    header = ""
    for i in range(1,num_rows+1):
        header += "dim"+str(i)+","
    header += "post_category"
    return header

def doExperiments(train_vector,train_labels): #test_vector,test_labels):

    k_fold = StratifiedKFold(10)
    classifiers = [LogisticRegression(C=0.1, max_iter=500), LogisticRegression(C=0.1, class_weight='balanced', max_iter=500),
                   SVC(kernel='linear', probability=True),SVC(kernel='linear', class_weight='balanced', probability=True),
                   SGDClassifier(class_weight='balanced'), RidgeClassifier(class_weight='balanced'), Perceptron(class_weight='balanced'), MLPClassifier()]
    #classifiers = [MLPClassifier(max_iter=500)]
    #RandomForestClassifer(), GradientBoostClassifier()
    #Not useful: SVC with kernels - poly, sigmoid, rbf.

    for classifier in classifiers:
        print(classifier)
        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
        print(cross_val)
        print(sum(cross_val)/float(len(cross_val)))
        print(confusion_matrix(train_labels, predicted, labels=["green","amber","red","crisis"]))

'''
   classifiers = [LogisticRegression(C=0.1, max_iter=500), LogisticRegression(C=0.1, class_weight='balanced', max_iter=500),
                   SVC(kernel='linear', probability=True),SVC(kernel='linear', class_weight='balanced', probability=True),
                   SGDClassifier(class_weight='balanced'), RidgeClassifier(class_weight='balanced'), Perceptron(class_weight='balanced'), MLPClassifier()]
'''

def doExperimentsWithNeighbors(train_vector,train_labels):
    k_fold = StratifiedKFold(10)
    classifiers = [KNeighborsClassifier(5, weights='uniform'), KNeighborsClassifier(5, weights='distance'),KNeighborsClassifier(10, weights='uniform'), KNeighborsClassifier(10, weights='distance')]
    #    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')

    for classifier in classifiers:
        print(classifier)
        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
        print(cross_val)
        print(sum(cross_val)/float(len(cross_val)))
        print(confusion_matrix(train_labels, predicted, labels=["green","amber","red","crisis"]))

def generateTestPredictions(train_vector,train_labels,test_vector,test_labels,test_ids):
     classifiers = [LogisticRegression(C=0.1, class_weight='balanced', max_iter=500), SVC(kernel='linear', class_weight='balanced', probability=True),RidgeClassifier(class_weight='balanced') ]
     for classifier in classifiers:
         output_path = str(classifier)[0:5] + "-predictions.tsv"
         fh = open(output_path, "w")
         classifier.fit(train_vector,train_labels)
         predictedLabels = classifier.predict(test_vector)
         fh.write("194634"+"\t"+"green") #I made a mistake and missed the first file in the test set while doing feature estimation. Just keeping the majority category for that.
         fh.write("\n")
         for i in range(0,len(test_ids)):
            fh.write(test_ids[i]+"\t"+predictedLabels[i].strip())
            fh.write("\n")
         print("Wrote predictions using: ", classifier)
         fh.close()

#Save feature vectors to use somewhere.
def saveFeatureVectors(feature_file,cats,output_path):
    fw = open(output_path,"w")
    header_len = len(feature_file[0])
    header = ",".join(str(l) for l in range(1,header_len+1)) + "," + "post_category"
    fw.write(header)
    fw.write("\n")
    for index in range(0,len(feature_file)):
        fw.write(",".join(str(l) for l in feature_file[index]) + "," + cats[index])
        fw.write("\n")
    fw.close()


def main():

   # build_doc2vec_model()
   # print("new  doc2vec model built")

    text_vector,text_cats, train_ids = create_feature_labels_array("../DataFiles/trainingdata-withaddedfeatures-withlexiconfeatures.csv")
    test_vector,test_cats,test_ids = create_feature_labels_array("../DataFiles/testdata-withaddedfeatures-renamednewcatfeatures-withlexiconfeatures.csv")

    print("training data dimensions")
    print(len(text_vector))
    print(len(text_vector[0]))

    print("test data dimensions")
    print(len(test_vector))
    print(len(test_vector[0]))

    #Doing this because one hot encodings are throwing errors if i do train and test data separately as some stuff from training does not exist in test and vice-versa.
    #For this time, I using the test file where categories that do not exist in training are renamed to those that do. So, vice-versa part does not hold.
    big_vector = []
    big_vector.extend(text_vector)
    big_vector.extend(test_vector)

    print("big vector dimensions")
    print(len(big_vector))
    print(len(big_vector[0]))


    #dim1 is N if N is the dimensionality of embedding space being used to get features.. dim2 is N+1
    big_vector = fitOneHotEncodings(numpy.array(big_vector),30,31)

    #We can do over-sampling here too.
    text_vector = big_vector[0:len(text_vector)]
    test_vector = big_vector[len(text_vector):]

    print("post_transformation")
    print(len(text_vector))
    print(len(text_vector[0]))

    print("post_transformation")
    print(len(test_vector))
    print(len(test_vector[0]))

    saveFeatureVectors(text_vector,text_cats,"trainingdata-forresampling-allfeatures.csv")
    saveFeatureVectors(test_vector,test_cats,"testdata--forresampling-allfeatures.csv")

    #write_to_csv(text_vector,text_cats,len(text_vector[0]),"doc2vec-30dim-withaddedfeatures.csv")
    #doExperiments(text_vector,text_cats)
    #doExperimentsWithNeighbors(text_vector,text_cats)
    #generateTestPredictions(text_vector,text_cats,test_vector,text_cats,test_ids)


if __name__ == '__main__':
    main()