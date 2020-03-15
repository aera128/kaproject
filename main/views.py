from django.http import HttpResponseRedirect
from django.shortcuts import render
from main.form import KAForm
from joblib import load
import nltk_download
from nltk_download.tokenize import sent_tokenize, word_tokenize


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


def index(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = KAForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            response = form.cleaned_data['raw_text']
            crf = load('main/model.joblib')
            sentences = sent_tokenize(response)

            data = list()
            for s in sentences:
                data.append(nltk_download.pos_tag(word_tokenize(s)))
            X = [sent2features(s) for s in data]

            response = crf.predict(X)

            html = '<p class="text-justify">'
            entities = '<table class="table">' \
                       '<thead>' \
                       '<tr>' \
                       '<th scope="col">#</th>' \
                       '<th scope="col">Entity</th>' \
                       '<th scope="col">Tag</th>' \
                       '</tr>' \
                       '</thead>' \
                       '<tbody>'
            count = 0
            for i, s in enumerate(sentences):
                words = word_tokenize(s)
                for j in range(len(response[i])):
                    if response[i][j] != "O" and response[i][j] != "O\r\n":
                        count += 1
                        html += '<span class="badge badge-pill badge-primary">' + words[j] + ' - ' + response[i][
                            j] + '</span> '
                        entities += '<tr>' \
                                    '<th scope="row">' + str(count) + '</th>' \
                                                                 '<td>' + words[j] + '</td>' \
                                                                                     '<td>' + response[i][j] + '</td>'
                    else:
                        html += words[j] + " "
            html += "</p>"
            entities += '</tbody>' \
                        '</table>'
            return render(request, 'main/ka.html', {
                'form': form,
                'response': html,
                'entities': entities
            })

    # if a GET (or any other method) we'll create a blank form
    else:
        form = KAForm()
    return render(request, 'main/ka.html', {'form': form})
