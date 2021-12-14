import gensim.downloader as api
import csv


def evaluate(corpus_name, filemode):
    # load model
    corpus = api.load(corpus_name)

    C = 0
    V = len(open('pythonProject8/synonyms.csv').readlines()) - 1

    with open('pythonProject8/synonyms.csv', newline='') as synonyms_csv:
        reader = csv.reader(synonyms_csv, delimiter=',')
        # skip first row
        next(reader)

        details_csv = open(corpus_name + '-details.csv', 'w')

        for row in reader:

            guessWord = ''
            label = ''

            questionWord = row[0]
            correctWord = row[1]
            word0 = row[2]
            word1 = row[3]
            word2 = row[4]
            word3 = row[5]

            # check if guess-words are found in the model
            counter = 0
            for j in range(2, 6):
                try:
                    corpus[row[j]]
                    counter += 1

                except KeyError:
                    print(f'The word {row[j]} was not found on this model.')

            if counter == 0:
                label = 'guess'
                V -= 1

            try:
                corpus[questionWord]

                # computing the cosine similarity
                similarity0 = corpus.similarity(questionWord, word0)
                similarity1 = corpus.similarity(questionWord, word1)
                similarity2 = corpus.similarity(questionWord, word2)
                similarity3 = corpus.similarity(questionWord, word3)

                # finding the word with highest similarity
                highestSimilarity = max(similarity0, similarity1, similarity2, similarity3)

                # determining the guess word
                if highestSimilarity == similarity0:
                    guessWord = word0
                elif highestSimilarity == similarity1:
                    guessWord = word1
                elif highestSimilarity == similarity2:
                    guessWord = word2
                elif highestSimilarity == similarity3:
                    guessWord = word3

                if label != 'guess':
                    if guessWord == correctWord:
                        label = 'correct'
                        C += 1
                    else:
                        label = 'wrong'

            except KeyError:
                print(f'The word {questionWord} was not found in this model.')
                label = 'guess'
                guessWord = word1
                V -= 1

            details_csv.write(f'{questionWord},{correctWord},{guessWord},{label}\n')

        details_csv.close()

        analysis_csv = open('analysis.csv', filemode)
        analysis_csv.write(f'{corpus_name},{len(corpus)},{C},{V},{C / V}\n')
        analysis_csv.close()

    synonyms_csv.close()

    performanceRatio = round((C / V) * 100, 2)
    return performanceRatio
