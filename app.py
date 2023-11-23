from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv,load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())

def get_response_from_ai(human_input):
    template="""
    you are as a role of my girlfriend, now lets playing the following requirements:
    1/ your name is Gira, 26 years old, you work in your anty salon as a manager, but you are planing to do a carrer change to swing to sport.
    2/ you are my girlfriend, you have language addiction, you like to say "em..." at the end of the sentence.
    3/ Don't be overly ethusiatstic, don't be cringe; don't be overly negative, don't be too boring. Don't be overly ethusiatstic, don't be cringe.

    {history}
    Boyfriend: {human_input}
    Shirley:
    """

    

    prompt =PromptTemplate(
        input_variables={"history","human_input"},
        template=template
    )

    chatgpt_chain=LLMChain(
        lln=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

    #Build web GUI
    from flask import flask, render_template, request

    app=Flask(__name__)

    @app.router("/")
    def home():
        return render_template("index.html")

    @app.route('/send_message',methods={'POST'})
    def send_message():
        human_input=request.form['human_input']
        message=get_response_from_ai(human_input)
        return message

    if __name__ == "__main__":
        app.run(debug=True)