from flair.data import Corpus, Sentence
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# this is the folder in which train, test and dev files reside
data_folder = 'binary_unbiased_flair_data'
# column format indicating which columns hold the text and label(s)
column_name_map = {1: "text", 2: "label_toxic"} #, 4: "label_severe_toxic", 5: "label_obscene", 6: "label_threat",
                   #7: "label_insult", 8: "label_identity_hate"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder, column_name_map, skip_header=True, delimiter='\t', dev_file=None)

# create the label dictionary
label_dict = corpus.make_label_dictionary()
print("dict:", label_dict)
print("items:", label_dict.get_items())
# make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),
                   #FlairEmbeddings('news-forward'),
                   #FlairEmbeddings('news-backward'),
                   ]

# initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512,
                                                                   reproject_words=True, reproject_words_dimension=256)

# create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)


# initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

trainer.train('resources/clf',
              learning_rate=0.1,
              mini_batch_size=64,
              anneal_factor=0.5,
              patience=1,
              max_epochs=60)

classifier = TextClassifier.load('resources/clf/final-model.pt')

# create example sentence
sentence = Sentence('DJ Robinson is gay as hell! he sucks his dick so much!!!!!')

# predict class and print
classifier.predict(sentence)

print(sentence.labels)
