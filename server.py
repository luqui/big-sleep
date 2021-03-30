import flask

api = flask.Flask(__name__)

@api.route('/progress.png', methods=['GET'])
def get_progress():
    return flask.send_file('progress.png', mimetype='image/png')

@api.route('/client.html', methods=['GET'])

def get_client():    return flask.send_file('client.html', mimetype='text/html')

if __name__ == '__main__':
    api.run(host='0.0.0.0', port=5555)
