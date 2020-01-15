from pathlib import Path
from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import WordEmbeddings, BytePairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# this is the folder in which train, test and dev files reside
data_folder = 'binary_unbiased_flair_data'
# column format indicating which columns hold the text and label(s)
column_name_map = {1: "text", 2:"label_toxic"} #, 4: "label_severe_toxic", 5: "label_obscene", 6: "label_threat",
                   #7: "label_insult", 8: "label_identity_hate"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(data_folder, column_name_map, skip_header=True, delimiter='\t', dev_file=None)

# create the label dictionary
label_dict = corpus.make_label_dictionary()
print("dict:", label_dict)
print("items:", label_dict.get_items())
# make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),
                   BytePairEmbeddings('en')
                   ]

# initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512,
                                                                   reproject_words=True, reproject_words_dimension=256)

# create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, multi_label=False)
# load checkpoint
checkpoint = 'resources/clf/checkpoint.pt'
trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)

# resume training
trainer.train('resources/clf',
              learning_rate=0.025,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=3,
              max_epochs=40,
              monitor_test=True,
              checkpoint=True
              )