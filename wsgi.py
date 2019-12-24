from dotenv import load_dotenv, find_dotenv

from flask import Flask 

load_dotenv(find_dotenv())

from strainer import main as application

if __name__ == '__main__':
    Flask.run(application)