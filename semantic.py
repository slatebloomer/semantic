import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
word4 = nlp("paw")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print(word4.similarity(word1))
print(word4.similarity(word2))

# It's interesting  that "banana" is far more similar to "monkey" than to "cat". This is because it recognises
# that monkeys eat bananas so there is an extra relationship there beyond just eg being alive. Cool. I added 'paw'
# to see what happened because cats are better known for having "paws" than monkeys but a "monkey paw" is a
# superstitious thing so there is some link there

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

# I find this really interesting - I would have perhaps thought that "where did my dog go" is semantically more
# similar to "hello, there is my car" but thinking about it again I can totally see why it isn't. This is a very
# useful example of why NLP can see patterns that perhaps humans can't instantly spot


# Note about the example.py file using the simpler model: wow! That's really interesting. The simpler model is
# far less able to pick up similarities so  the figures are lower. I got this error message, which is very
# informative and interesting

# UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the
# Doc.similarity method will be based on the tagger, parser and NER, which may not give useful
# similarity judgements. This may happen if you're using one of the small models, e.g.
# `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors.
# You can always add your own word vectors, or use one of the larger models instead if available.

