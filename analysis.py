########################################################################################################################
# Filename      : analysis.py
# Autor         : Anastasiia Sedova
# Version       : 1.0
# Datum               : Jan 21 2020
# Beschreibung  : main file for analysis of user input
#
# Aufruf: analysis.py input_path
#
# Input
# input_path : path to .txt file where the person's answers are stored
#
#
# Output
# speaker_name : number of times person says his/her own name
# first_person_pronouns : number of personal pronouns
# pronouns : number of other pronouns
# bsolute_counter : number of absolute words
# swear_counter : number of swear words
# afinn_score : afinn score
# sentiment : sentiment score 
########################################################################################################################

from afinn import Afinn
from Sentiment_analysis.sent_predict import predict
# from pos_and_proper_names_counter import POSAndProperNames
import argparse
import nltk
from nltk import word_tokenize
import nltk.data
import re
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def calculate_absolute_words(words):
    """
    Looking for absolute words. Source: https://kddidit.com/2015/04/20/grammar-absolute-words/
    """
    absolute = ("absolute", "absolutely", "astounding", "all", "always", "awful", "awfully", "complete", "completely",
                "dead", "deadly", "gigantic", "gigantically", "entire", "entirely", "fatally", "impossibly",
                "infinitely", "irrevocably", "giant", "entirely", "eternal", "eternally", "every", "everybody",
                "everyone", "exhausted", "fantastic", "fatal", "final", "finite", "full", "fully", "impossible",
                "infinite", "irrevocable", "hideous", "hideously", "horrible", "horribly", "never", "nobody", "no one",
                "none", "overwhelmed", "perfect", "perfectly", "single", "supreme", "starving", "terrible", "terribly",
                "terrifying", "total", "totally", "ultimate", "ultimately", "unique", "uniquely", "wonderful",
                "wonderfully")
    absolute_counter = len([x for x in words if x in absolute])
    return absolute_counter


def calculate_afinn_score(sentence):
    afinn = Afinn(language='en')
    afinn_score = afinn.score(sentence)
    return afinn_score


def calculate_swear_words(words):
    """
    Looking for swear words. Source: https://en.wiktionary.org/wiki/Category:English_swear_words
    """
    swear = ("arse", "ass", "asshole", "bitch", "bastard", "bollocks", "bugger", "child-fucker", "crap", "cunt", "damn",
             "effing", "frigger", "fuck", "goddamn", "godsdamn", "hell", "horseshit", "motherfucker", "nigga", "nigger",
             "shit", "shitty", "shitass", "slut", "piss", "prick", "twat", "piss", "pissed")
    swear_counter = len([x for x in words if x in swear])
    return swear_counter


def word_counting(sentence, speaker_name):

    speaker_name_count = 0
    first_person_pronouns = 0
    other_pronouns = 0

    # tagging
    tagged_token = nltk.pos_tag(word_tokenize(sentence))

    for token in tagged_token:

        # NN = noun, singular 'desk', NNS = noun plural	'desks', NNPS = proper noun, plural	'Americans'
        # if token[1] in ["NN", "NNS", "NNPS"]:
        #     noun += 1
        if token[0] == speaker_name:
            speaker_name_count += 1

        if token[0] in ["i", "me", "myself" "my", "mine"]:
            first_person_pronouns += 1

        # PRP = personal pronoun (I, he, she), PRP$ = possessive pronoun (my, his, hers)
        if token[1] in ["PRP", "PRP$"] and token[0] not in ["i", "me", "myself" "my", "mine"]:
                other_pronouns += 1

    return speaker_name_count, first_person_pronouns, other_pronouns

    def print_results():

        # print("Nouns: " + str(noun))
        # print("Adjectives: " + str(adjective))
        # print("Verbs: " + str(verb))
        # print("The number of adverbs is " + str(adverb))
        print("Other pronouns: " + str(pronouns))
        print("Pronounce with references to one " + str(first_person_pronouns))
        print("The proper name of the speaker: " + str(speaker_name))
        print("The other names speaker says: " + str(other_names))


def analyse(input_path):

    answers_sentences = []      # list of sentences represented as strings
    answers_words = []      # list of all words containing in all answers

    # read input data
    with open(input_path) as file:
        for paragraph in file:
            if paragraph != "":
                sentences = nltk.sent_tokenize(paragraph)           # split the paraphraph into sentences
                for sentence in sentences:
                    # preprocessing: all letters lower, exclude numbers and punctuations marks
                    clear_sentence = preprocessing(sentence)
                    answers_sentences.append(clear_sentence)    # add sentence to the whole list of sentences
                    tokenized_sentence = word_tokenize(clear_sentence)  # tokenization
                    answers_words.extend(tokenized_sentence)
    print(len(answers_words))
    file.close()

    speaker_name = answers_words[0]         # the first answer is always the proper name of the speaker
    answers_words = answers_words[1:]
    sentences = sentences[1:]

    # POS and proper nouns counter
    total_speaker_name = -1       # to eliminate the first answer on the question "What's your name"
    total_first_person_pronouns = 0
    total_other_pronouns = 0

    total_afinn_score = 0

    # calculate number of first-person pronouns
    for sentence in answers_sentences:

        # calculate the POS + name
        sent_speaker_name, sent_first_person_pronouns, sent_other_pronouns = word_counting(sentence, speaker_name)
        total_speaker_name += sent_speaker_name
        total_first_person_pronouns += sent_first_person_pronouns
        total_other_pronouns += sent_other_pronouns

        # Calculate the afinn score
        total_afinn_score += calculate_afinn_score(sentence)

    total_speaker_name = max(0, total_speaker_name)

    absolute_self_centrism = total_speaker_name + total_first_person_pronouns
    rel_self_centrism = round(absolute_self_centrism / len(answers_words), 3)

    final_afinn_score = round(total_afinn_score / len(answers_sentences), 3)

    # calculate the number of absolute words
    absolute_absolute_counter = calculate_absolute_words(answers_words)
    rel_absolute_counter = round(absolute_absolute_counter/len(answers_words), 3)

    # calculate the number of swear words
    absolute_swear_counter = calculate_swear_words(answers_words)
    rel_swear_counter = round(absolute_swear_counter/len(answers_words), 3)

    # do sentiment analysis
    sentiment = predict("Sentiment_analysis/trained_model.pt", input_path)

    print("Afinn score: " + str(final_afinn_score))

    print("A speaker says his/her own name: " + str(total_speaker_name))
    print("References to oneself:", total_first_person_pronouns, "of", len(answers_words), "in total.")
    print("Use of other pronouns:", total_other_pronouns, "of", len(answers_words), "in total.")

    print("Swear Words:", absolute_swear_counter, "of", len(answers_words), "in total.")
    print("Absolute Words:", absolute_absolute_counter, "of", len(answers_words), "in total.")

    print("Sentiment is: " + str(sentiment))

    return absolute_self_centrism, rel_self_centrism, rel_swear_counter, rel_absolute_counter, final_afinn_score, sentiment


def preprocessing(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[!,.\-()\n]", "", sentence)
    clear_sentence = re.sub("'", " ", sentence)
    return clear_sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Text file containing person's answers on the questions.")
    args = parser.parse_args()

    analyse("data/" + args.input_file)

