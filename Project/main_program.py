#### IMPORTS ####
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score as acc

############    FUNCTIONS    ############

def encode(X_ds, y_ds):
    nl_questions = []
    nl_tokens = []
    sql_queries = []
    sql_tokens = []
    max_query_len = 0
    max_question_len = 0

    # Insert start and end characters to the target dictionary
    sql_tokens.append("@")
    sql_tokens.append("&")

    for index,_ in enumerate(X_ds.values.tolist()):
        nl_questions.append(X_ds.iloc[index])
        sql_queries.append(y_ds.iloc[index])
        # Determine max length of input and output training instances.
        if len(X_ds.iloc[index]) > max_question_len:
            max_question_len = len(X_ds.iloc[index])
        if len(y_ds.iloc[index]) > max_query_len:
            max_query_len = len(y_ds.iloc[index])
        # Insert tokens into "dictionary" if not present yet.
        for token_nl in X_ds.iloc[index]:
            if token_nl not in nl_tokens:
                nl_tokens.append(token_nl)
        for token_sql in y_ds.iloc[index]:
            if token_sql not in sql_tokens:
                sql_tokens.append(token_sql)

    # Sort the dictionaries.
    sql_tokens = sorted(sql_tokens)
    nl_tokens = sorted(nl_tokens)
    nl_tokens_index = dict()
    sql_tokens_index = dict()

    for index, token in enumerate(nl_tokens):
        nl_tokens_index[token] = index
    for index, token in enumerate(sql_tokens):
        sql_tokens_index[token] = index

    # Create square and populate them with the dictionary indexes encoding.
    enc_in_data = np.zeros((len(nl_questions), 
                                max_question_len), dtype="float32")
    dec_in_data = np.zeros((len(nl_questions), 
                                max_query_len), dtype="float32")
    dec_out_data = np.zeros((len(nl_questions), 
                                    max_query_len), dtype="float32")

    # Iterate through both lists of NL questions and SQL queries.
    for i, (question, query) in enumerate(zip(nl_questions, sql_queries)):
        # Iterate through each token of the NL question and mark the corresponding 
        # space in the square with the corresponding dictionary index.
        for j, token in enumerate(question):
            enc_in_data[i, j] = nl_tokens_index[token]
        # Iterate through each token of the SQL question and follow the same 
        # procedure for both decoders keeping in mind that the output decoder is 
        # goes 1-timestep ahead.
        for j, token in enumerate(query):
            dec_in_data[i, j] = sql_tokens_index[token]
            if j > 0:
                dec_out_data[i, j - 1] = sql_tokens_index[token]

    return enc_in_data, dec_in_data, dec_out_data, nl_questions, sql_queries, len(nl_tokens), len(sql_tokens), nl_tokens_index, sql_tokens_index, max_query_len

def preprocess(file_path, list_filter_keywords):
    df = pd.read_json(open(file_path, "r", encoding="utf8"))

    list_filtered = []
    add_flag = False
    for index, tokens in enumerate(df["query_toks"].values.tolist()):
        for keyword in list_filter_keywords:
            if keyword in tokens:
                add_flag = True
        if add_flag:
            list_filtered.append(df.iloc[index])
        add_flag = False

    df_filtered = pd.DataFrame(list_filtered)
    y_df = df_filtered["query_toks"].copy()
    X_df = df_filtered["question_toks"].copy()

    for query in y_df:
        # Start
        query[:0] = ["@"]
        # End
        query.append("&")

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, 
                                                    random_state=50)
    enc_in_data, dec_in_data, dec_out_data, nl_q, sql_q, num_nl_tok, num_sql_tok, nl_dict, sql_dict, max_query_len = encode(X_train, y_train)
    enc_in_data_t, dec_in_data_t, dec_out_data_t, nl_q_t, sql_q_t, num_nl_tok_t, num_sql_tok_t, nl_dict_t, sql_dict_t, max_query_len_t = encode(X_test, y_test)

    train_data = [enc_in_data, dec_in_data, dec_out_data, nl_q, sql_q, num_nl_tok, num_sql_tok, nl_dict, sql_dict, max_query_len]
    test_data = [enc_in_data_t, dec_in_data_t, dec_out_data_t, nl_q_t, sql_q_t, num_nl_tok_t, num_sql_tok_t, nl_dict_t, sql_dict_t, max_query_len_t]
    return train_data, test_data

def decode_sequence(seq, max_query_len_test, sql_index, rev_sql_index):
  states_value = encoder_model.predict(seq)
  target_seq = np.zeros((1,1))
  target_seq[0, 0] = sql_index['@']

  stop_flag = False
  translated_query = []
  while not stop_flag:
    decoder_output, state_h, state_c = decoder_model.predict([target_seq] 
                                                             + states_value)
    sampled_token_dict_index = np.argmax(decoder_output[0, -1, :])
    sampled_token = rev_sql_index[sampled_token_dict_index]
    translated_query.append(sampled_token)

    if sampled_token == '&' or len(translated_query) > max_query_len_test:
      stop_flag = True

    target_seq = np.zeros((1,1))
    target_seq[0, 0] = sampled_token_dict_index

    states_value = [state_h, state_c]
  return translated_query

############            MAIN CODE           ############
# Pre-process the file specified by the provided path.
# The SQL queries that include the keywords provided in the list will be included after
#   the filtration.
train_data, test_data = preprocess("train_spider.json", ["avg","min","max","count","sum"])

#### Define hyper-parameters
batch_size = 10  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.
num_samples = len(train_data[3])  # Number of samples to train on.

#### Build model
# Define encoder
encoder_inputs = Input(shape=(None,))
embedded_encoder = Embedding(train_data[5], latent_dim)(encoder_inputs)
_, state_h, state_c = LSTM(latent_dim, return_state=True, 
                        dropout=0.25)(embedded_encoder)
encoder_states = [state_h, state_c]
# Define decoder, using the context info from encoder as initial state
decoder_inputs = Input(shape=(None,))
embedded_decoder = Embedding(train_data[6], latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, 
                    return_state=True, dropout=0.25)
decoder_outputs, _, _ = decoder_lstm(embedded_decoder, 
                                    initial_state=encoder_states)
decoder_dense = Dense(train_data[6], activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define and compile model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss="sparse_categorical_crossentropy",
            metrics="accuracy")
# Train model
model.fit([ train_data[0], train_data[1]],
            train_data[2],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)

#### Inference
# Set up the architecture
encoder_model = Model(encoder_inputs, encoder_states)
decoder_states_inputs = [Input(shape=(latent_dim,)),Input(shape=(latent_dim,))]
decoder_outputs, state_h, state_c = decoder_lstm(
    embedded_decoder, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

encoder_input_test =    test_data[0]
max_query_len_test =    test_data[9]
nl_token_index =        train_data[7]
sql_token_index =       train_data[8]
nl_questions_test =     test_data[3]
sql_queries_test =      test_data[4]
# Reverse the dictionary to decode the predictions
rev_sql_index = dict((i, char) for char, i in sql_token_index.items())
for seq_index in range(5):
    # Attempt inference and decoding with several sequences of the test set.
    decoded_query = decode_sequence(encoder_input_test[seq_index : seq_index + 1],
                                    max_query_len_test, sql_token_index, rev_sql_index)
    print("\n-")
    print("Input NL question:", nl_questions_test[seq_index])
    print("Expected SQL query:", sql_queries_test[seq_index][1:-1]) # Except start/end chars
    print("Decoded query:", decoded_query[:-1]) # Except end char