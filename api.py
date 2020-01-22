import flask
from flask import request
from flask_cors import CORS
from flask import jsonify
from predict import predict_news
#these are basic libraries

app = flask.Flask(__name__)
app.config['DEBUG'] = True
CORS(app)



@app.route('/predict', methods=['GET'])
def api():
    document = request.headers['news']
    print("Your news is "+ document)
    prediction = predict_news(document)
    return prediction
if __name__ == '__main__':
    app.run(host='0.0.0.0', port= 5000)
