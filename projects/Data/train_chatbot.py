import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pandas as pd


from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import en_core_web_sm
nlp = en_core_web_sm.load()

def train_mathbot(df,pattern_col,equation_col,name_of_bot):
    questions,answers = df[pattern_col].tolist(),df[equation_col].tolist()
    q_and_a = []
    for question,answer in zip(questions,answers):
        q_and_a.append(question)
        q_and_a.append(answer)

    bot = ChatBot(name_of_bot)
    bot = ChatBot(name_of_bot, read_only = True, 
              preprocessors=['chatterbot.preprocessors.convert_to_ascii', 
                             'chatterbot.preprocessors.unescape_html',
                             'chatterbot.preprocessors.clean_whitespace'],
             logic_adapters = [
                 {
                     'import_path': 'chatterbot.logic.BestMatch',
                     'default_response': 'Sorry, I am unable to process your request. Please try again, or contact us for help.',
                     'maximum_similarity_threshold': 0.90
                 }
             ])
    trainer = ListTrainer(bot)
    trainer.train(
        q_and_a
    )
    trainer.train([
        "Thank you!",
        "You're most welcome!",
        "Thanks!",
        "Of course!",
    ])

    return bot

def run_chatbots(bot,bot2):
    name=input("Please enter your Name: ")
    email = input("Please enter your Email ID: ")
    print("Welcome to the ChatBot Service for Pseudonymous ENT! How can I help you? (Enter 'Bye' to exit)")
    while True:
        request=input(name+':')
        if request=='Bye' or request =='bye':
            print('ChatBot: It was great talking to you! Bye!')
            break
        else:
            choice = input("Select between i or d:-")
            if choice=="i":
                request=input(name+':')
                response=bot.get_response(request)
                print('ChatBot:',response)
            elif choice=="d":
                request=input(name+':')
                response=bot2.get_response(request)
                print('ChatBot:',response)



    

if __name__=="__main__":
    path = r"C:\Users\gprak\Downloads\Data\df_i.xlsx"
    df_i = pd.read_excel(path,engine="openpyxl")
    bot = train_mathbot(df_i,"patterns","integration","MathBot")

    path = r"C:\Users\gprak\Downloads\Data\df_t.xlsx"
    df_t = pd.read_excel(path,engine="openpyxl")
    bot2 = train_mathbot(df_t,"patterns","differentiation","MathBot2")
    run_chatbots(bot,bot2)