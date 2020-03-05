# Depression Detection project

Here is the source code for the Depression Detection Project (Course "Affective Computing", WiSe 19/20)

Depression is a very serious medical illness that may lead to severe outcomes, such as mental breakdown and even suicide. However, it is not only very serious, but also quite common today. It is even often the case that people are depressed without ever realising it and, therefore, do not seek help and treatment from a professional. This study aims at helping to discover symptoms of depression based on what a person says and what a person writes. The main focus lies on text analysis with different methods, such as analysis of the vocabulary a person uses as well as sentiment analysis. The proposed algorithm was tested on the data sets manually collected from Reddit and Shakespeare’s tragedy Hamlet. Also we realised our algorithm as a web application in order to give an opportunity to all the people to test themselves, which could be a first step on the path to recovery.

More detailes are to be found in the [paper](https://github.com/agsedova/depression_detection/blob/master/Depression_recognition_Schinke_Sedova.pdf).

# Text Analysis

## [main.py](https://github.com/agsedova/depression_detection/blob/master/main.py)

The main file to run the application.

## [analysis.py](https://github.com/agsedova/depression_detection/blob/master/analysis.py)
The main sript for the text input; is called from main.py. 
Can be also called separately on its own:

    analysis.py <path to the input file with text to be analysed>
    analysis.py data\reddit_neg

## [Data](https://github.com/agsedova/depression_detection/tree/master/data)

Manually collected different data which we use for testing our app (for more detailed information please look at the paper)

## [Templates](https://github.com/agsedova/depression_detection/tree/master/templates), [static](https://github.com/agsedova/depression_detection/tree/master/static), [images](https://github.com/agsedova/depression_detection/tree/master/images)
Files for the front-end. 

## [Sentiment_analysis](https://github.com/agsedova/depression_detection/tree/master/Sentiment_analysis)
Here are all the files which are needed to perform the sentiment analysis with LSTM.

**data** : here are the data which we used as training, development and tests set as well as data which we used for demonstrating our app during the final presentation (answers.txt)

**sent_train.py** : training the LSTM Model

**sent_predict.py** : sentiment prediction with the trained model on the new data

You don't need to train the model from scratch - we have already done it! If you want to use it, please download it [here](https://www.icloud.com/iclouddrive/07l-mKo0NRemSlP5AQQY__HgQ#trained_model) and put the file in [Sentiment_analysis folder](https://github.com/agsedova/depression_detection/tree/master/Sentiment_analysis).

# Voice Analysis

Also we tried to do the voice analysis. Our scripts are in folder [audio_analysis]
(https://github.com/agsedova/depression_detection/tree/master/audio_analysis)

