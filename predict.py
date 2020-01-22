import pickle
import sys
def predict_news(document):
    with open('y_train.pickle', 'rb') as file:
        y_train = pickle.load(file)

    with open('model.pickle', 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict([document])
    return prediction[0]

if __name__ == '__main__':
    try:
        document = sys.argv[1]
        print("Arguement taken.")
    except Exception as e:
        print("Argument required")
    else:
        prediction =  predict_news(document)
        print(prediction)
