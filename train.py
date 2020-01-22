import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

def train():
    print("Downloading data")
    train = pd.read_csv('train.csv')
    train.drop(['id', 'title', 'author'],axis=1, inplace=True)
    train.dropna(inplace=True)
    train['label'].replace(0,'Real',inplace=True)
    train['label'].replace(1,'Fake',inplace=True)
    X_train,X_test, y_train, y_test = train_test_split(train['text'], train['label'], test_size = 0.33, random_state = 53)
    model =  make_pipeline(CountVectorizer(), PassiveAggressiveClassifier())

    print("Training model")
    model.fit(X_train , y_train)

    print("Saving model")
    with open('model.pickle','wb') as file:
        pickle.dump(model,file)

    with open('y_train.pickle','wb') as file:
        pickle.dump(y_train,file)

#if __name__ ==" __main__":
#    print("function is doing:")
train()
