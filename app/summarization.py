# -*- coding: UTF-8 -*-
"""
hello_urlvar: Using URL Variables
"""
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, world!'

@app.route('/hello/<username>')  # URL with a variable
def hello_username(username):    # The function shall take the URL variable as parameter
    return 'Hello, {}'.format(username)

@app.route('/hello/<int:userid>')  # Variable with type filter. Accept only int
def hello_userid(userid):          # The function shall take the URL variable as parameter
    return 'Hello, your ID is: {:d}'.format(userid)

if __name__ == '__main__':
    app.run(debug=True)  # Enable reloader and debugger