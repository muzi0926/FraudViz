import flask

import demo

app = flask.Flask(__name__)


class Result:
    def __init__(self, code, data):
        self.code = code
        self.data = data


@app.route('/getData', methods=['GET'])
def get_data():
    try:
        year_begin = flask.request.args.get("yearBegin", type=int)
        year_end = flask.request.args.get("yearEnd", type=int)
        file_name = demo.run_model(year_begin, year_end)
        result = Result(200, file_name)
        return flask.jsonify(result.__dict__)
    except Exception as e:
        print(e)
        result = Result(500, "error")
        return flask.jsonify(result.__dict__)


@app.route('/getFinal', methods=['GET'])
def get_final():
    try:
        file_name = 'Final_20200117.csv'
        result = Result(200, file_name)
        return flask.jsonify(result.__dict__)
    except Exception as e:
        print(e)
        result = Result(500, "error")
        return flask.jsonify(result.__dict__)


if __name__ == '__main__':
    app.run(debug=True)
