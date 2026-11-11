
from gensim.models import KeyedVectors
import os

print("Loading Word2Vec vectors...")

wv = KeyedVectors.load_word2vec_format(
    "word2vec_vectors.txt",
    binary=False
)

print("Vocabulary:", len(wv))
print("Vector size:", wv.vector_size)

print("Saving Word2Vec model...")

wv.save(
    "word2vec.kv",
    separately=["vectors"]
)

print("Word2Vec model saved successfully!")
