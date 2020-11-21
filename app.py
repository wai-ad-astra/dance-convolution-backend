from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/collect_sample')
def collect_sample():
    return 'heya'


# whether or not called directly
if __name__ == '__main__':
    app.run()

