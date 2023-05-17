from flask import Flask
app = Flask(__name__)

@app.route("/predict")
def home():
  return '<h1>Hello, World!</h1>'

# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()