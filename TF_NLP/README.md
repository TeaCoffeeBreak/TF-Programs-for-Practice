# Tensorflow NLP programs
Text is one of the category of data that are widely used for processing. Natural language processing is field of creating programs that can understand and generate text. Neural Networks are one of the architectures that are used to solve these types of problems. There are several preprocessing tasks that are needed to perform before passing text data to the model. Tensorflow provides different APIs for doing these preprocessing task. Tokenizer is one of such api that is useful in generating tokens.

#### General steps to follow while solving NLP based problems:
1. Split dataset in training and validation.
2. Tokenize both training and validation data. While tokenizing select appropriate vocab size.
3. Pad tokenized sequences with appropriate padding parameters.
4. Build model with embedding layer.
5. Compile and fit model on training data (Use callbacks).
6. Validate model on validation data.

#### Following are the different datasets used:
1. IMDB Reviews: This dataset contains 50000 movie reviews with labels good or bad. It is available in tensorflow datasets.
2. Irish Lyrics: This dataset has irish poems with more than 13000 words. It used for text generation using word based encoding
3. Shakespeare: This dataset contains Shakespeare's play scripts with more than 202,000 words. This dataset is used for text generation using character based encoding.
4. Url classification: This is a classification dataset that can be used to classify if url is malicious or not. It contains more than 400,000 urls with labels good or bad.