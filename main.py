import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, abort
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from afinn import Afinn
import sys
from analysis import analyse

app = Flask(__name__)

# New Code
session = { 'data': [86,114,106,106,107,111] }
color = "powderblue"
bar_color = "green"
# New Code End

@app.route("/")
def index():
    return render_template('start.html')

@app.route("/why")
def why():
    return render_template("why.html")

@app.route("/whyAbs")
def whyAbs():
    return render_template("whyAbs.html")

@app.route("/whySent")
def whySent():
    return render_template("whySent.html")

@app.route("/whySwear")
def whySwear():
    return render_template("whySwear.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/textInput")
def textInput():
    return render_template("textInput.html")


@app.route('/results', methods=['POST'])
def submit():

    global color
    global bar_color

    depression_mark = 0

    name = request.form["name"]
    day = request.form["day"]
    previousDay = request.form["previousDay"]
    time = request.form["time"]
    future = request.form["future"]

    answers = [name, day, previousDay, time, future]
    answers = [x.strip() for x in answers if x.strip()]
    print(answers)

    file = open("./answers.txt", "w")
    file.writelines("\n".join(answers))
    file.close()

    absolute_self_centrism, rel_self_centrism, rel_swear_counter, \
            rel_absolute_counter, final_afinn_score, sentiment = analyse("answers.txt")

    first_person = "You referenced to yourself " + str(rel_self_centrism * 100) + "% of time. "
    if rel_self_centrism > 0.05:
        first_person += "So, you are quite focused on yourself."
        depression_mark += 1
    else:
        first_person += "Good news: that corresponds to the average use of first person pronouns."

    absolute = str(round(rel_absolute_counter * 100, 2)) + "% of words were absolute. "
    # Threshold taken from al-Mosaiwi(2018)
    if rel_absolute_counter > 0.012:
        absolute = absolute + "You seem to interpret your experiences in a very black or white kind of way. " \
                              "It is possible that you see all your experiences as either perfect or terrible. "
        depression_mark += 1
    else:
        absolute = absolute + "Good news: that corresponds to the average use of absolute words."

    swearSentence = str(round(rel_swear_counter * 100, 2)) + "% of words were swear words. "
    if rel_swear_counter > 0.002:
        swearSentence += "You are cursing much more than an average person. " \
                        "Perhaps you are in a negative state of mind or angry at something."
        depression_mark += 1
    else:
        swearSentence += "Good news: you are cursing not very much."

    if final_afinn_score <= 0:
        afinnString = "The afinn score is " + str(final_afinn_score) + \
                      ". That means that you seem to be thinking very negatively."
        depression_mark += 1
    else:
        afinnString = "The afinn score is " + str(final_afinn_score) + \
                      ". You're seeing your experiences in a neutral to positive view."

    SentimentString = "The sentiment analysis score is " + str(sentiment)
    if sentiment > 0.5:
        SentimentString += ". Good news: you are thinking rather positively."
    else:
        SentimentString += ". That means you are thinking rather negatively."
        depression_mark += 1

    mark = "Your overall depression mark is " + str(depression_mark)
    depr_sentence = "The greater is your overall depression mark, " \
                    "the more likely you have depression symptoms. " \
                    "0 means you have none. 5 means most probably you should go to the therapist."

    session['data'] = [final_afinn_score, sentiment, rel_swear_counter, rel_absolute_counter, rel_self_centrism,
                       depression_mark]

    relation = depression_mark / 5 * 100

    if depression_mark >= 3:
        color = "#726E6D"
        bar_color = "#8C001A"
    else:
        color = "#6AFB92"
        bar_color = "#FA69D1"

    return render_template("results.html", sentimen_analysis=SentimentString, absolute=absolute,
                           afinn=afinnString, swear=swearSentence, first_person_pronouns=first_person,
                           speaker_name=name, color=color, depression_mark=depression_mark, relation=relation,
                           bar_color=bar_color)

# 'You entered: {}'.format(request.form['text'])


@app.route("/compare")
def compare():
    data_you = session['data']
    return render_template('compare.html', data_you=data_you)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)

