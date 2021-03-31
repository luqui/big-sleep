import flask

api = flask.Flask(__name__)
api.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@api.route('/progress.png', methods=['GET'])
def get_progress():
    return flask.send_file('progress.png', mimetype='image/png')

@api.route('/client.html', methods=['GET'])
def get_client():
    return flask.send_file('client.html', mimetype='text/html')

@api.route('/prompt.json', methods=['GET'])
def get_prmpt():
    return flask.send_file('PROMPT.json', mimetype='text/json')

@api.route('/set_prompt', methods=['POST'])
def set_prompt():
    contents = flask.json.dumps(flask.request.json)
    with open('PROMPT.json', 'w') as fh:
        fh.write(contents)
    return flask.Response()


if __name__ == '__main__':
    api.run(host='0.0.0.0', port=5555)
