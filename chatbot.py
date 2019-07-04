import pandas as pd
import nltk
import tensorflow as tf
import numpy as np
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

nltk.download('punkt')

def preprocess_data(df):
  token_sents = []
  seq_data = []
  
  for index, row in df.iterrows():
    # remove spaces before and after, since they affect the actual sentences
    question = row["Question"].strip()
    answer = row["Answer"].strip()
    # append to our seq_data list
    seq_data.append([question, answer])
    # prepare tokens for word2vec
    token_sents += [[w.lower() for w in nltk.word_tokenize(question)]]
    token_sents += [[w.lower() for w in nltk.word_tokenize(answer)]]
    # add our symbols for other tokens 
    # we just need to be able to turn '_p_' and '_u_' into a word vector,
    # the actual embedding shouldn't matter.
    token_sents += [['_p_', '_u_']]
  return seq_data, token_sents

def load_wv_model():
  # nice and easy with gensim
  return FastText.load("data/word_embeddings.model")

# util for dealing with OOV
def get_vector(word, model):
  try:
    return model[word]
  except KeyError:
    return model['_u_']

# also making our batches here
def question_to_vec(question, model, max_q_length):
  q = [w.lower() for w in nltk.word_tokenize(question)]
  
  for _ in range(max_q_length-len(q)):
    q.append('_p_')
  return np.array([get_vector(w, model) for w in q])

def answer_to_vec(answer, answer_set):
  # remove multiples
  for i, a in enumerate(answer_set):
    if a == answer:
      return np.eye(len(answer_set))[i]
    
def answer_to_index(answer, answer_set):
  # remove multiples
  for i, a in enumerate(answer_set):
    if a == answer:
      return i
    
def make_batch(seq_data, wv_model, max_q_length=None, answer_set=None):
    input_batch = []
    output_batch = []
    target_batch = []
    
    # the maximum length of any question, required for padding
    if max_q_length is None:
      max_q_length = max([len(seq[0]) for seq in seq_data])
    # the list of possible unique answers
    # we must sort since set gives non-deterministic ordering
    if answer_set is None:
      answer_set = sorted(list(set([s[1] for s in seq_data] + ['_e_', '_b_', '_p_', '_u_'])))
    
    for seq in seq_data:
        input_data = question_to_vec(seq[0], wv_model, max_q_length)
        output_data = [answer_to_vec('_b_', answer_set)]
        output_data += [answer_to_vec(seq[1], answer_set)]
        
        # Output of decoder cell (Actual result), Add '_E_' at the end of the sequence data
        target = [answer_to_index(seq[1], answer_set)]
        target += [answer_to_index('_e_', answer_set)]
        # Add to our batches
        input_batch.append(input_data)
        output_batch.append(output_data)
        target_batch.append(target)

    return input_batch, output_batch, target_batch

def build_seq2seq_model(model_name, n_input, n_output, lr=0.001, n_hidden=256):
  with tf.variable_scope(model_name):

    # encoder/decoder shape = [batch size, time steps, input size]
    enc_input = tf.placeholder(tf.float32, [None, None, n_input])
    dec_input = tf.placeholder(tf.float32, [None, None, n_output])

    # target shape = [batch size, time steps]
    targets = tf.placeholder(tf.int64, [None, None])
    
    output_keep_prob = tf.placeholder_with_default(tf.constant(0.5, dtype=tf.float32), ())


    # encoder cell
    with tf.variable_scope('encoder'):
      fwd_enc_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
      fwd_enc_cell = tf.nn.rnn_cell.DropoutWrapper(fwd_enc_cell, output_keep_prob=output_keep_prob)
      
      bk_enc_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
      bk_enc_cell = tf.nn.rnn_cell.DropoutWrapper(bk_enc_cell, output_keep_prob=output_keep_prob)

      enc_outs, enc_stats = tf.nn.bidirectional_dynamic_rnn(
          fwd_enc_cell,
          bk_enc_cell,
          enc_input,
          dtype=tf.float32)
      enc_outputs = tf.add(enc_outs[0], enc_outs[1])
      enc_states = tf.add(enc_stats[0], enc_stats[1])
    # decoder cell, with attention
    with tf.variable_scope('decoder'):
      # attention mech
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
          num_units=n_hidden,
          memory=enc_outputs
      )
      # feed attention into GRU cell
      dec_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
      # wrap attention wrapper
      dec_cell = tf.contrib.seq2seq.AttentionWrapper(
          dec_cell,
          attention_mechanism,
          attention_layer_size=n_hidden
      )
      dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=output_keep_prob)

      # we need to wrap the initial state so it can be passed into the rnn
      initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=tf.shape(enc_states)[0])
      initial_state = initial_state.clone(cell_state=enc_states)
      outputs, dec_states = tf.nn.dynamic_rnn(dec_cell,
                                              dec_input,
                                              initial_state=initial_state,
                                              dtype=tf.float32)

    seq_model = tf.layers.dense(outputs, n_output, activation=None)
    
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=seq_model, labels=targets))

    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    
    # for eval
    prediction = tf.argmax(seq_model, 2)
    
    # bundle everything so it is easier to handle later on
    output_dict = {
        "cost": cost,
        "optimizer": optimizer,
        "enc_input": enc_input,
        "dec_input": dec_input,
        "targets": targets,
        "predictions": prediction,
        "n_input": n_input,
        "n_output": n_output,
        "dropout_keep_prob": output_keep_prob
    }

    return output_dict
  
def train(sess, total_epochs, feed_dict, optimizer, cost):
  for epoch in range(total_epochs):
      
    _, loss = sess.run([optimizer, cost], feed_dict=feed_dict)
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
  print('Training completed')

def load_seq2seq_model(output_batch_c, output_batch_f, output_batch_p):
  tf.reset_default_graph()
  input_size = 300
  # get output sizes for each model
  output_size_comic = len(output_batch_c[0][0])
  output_size_friend = len(output_batch_f[0][0])
  output_size_professional = len(output_batch_p[0][0])
  # build our models
  comic_model = build_seq2seq_model("comic", input_size, output_size_comic)
  friend_model = build_seq2seq_model("friend", input_size, output_size_friend)
  professional_model = build_seq2seq_model("professional", input_size, output_size_professional)
  # put them all in a list
  models = [comic_model, friend_model, professional_model]
  # initialisation
  init_op = tf.global_variables_initializer()
  # restore session from these
  saver = tf.train.Saver()
  sess = tf.Session()

  saver.restore(sess, 'data/chatbot')
  
  return models, sess
  
def answer(sess, sentence, vec_model, seq_model, answer_set):
  seq_data = [sentence, '_u_']

  # n_input = max q length
  input_batch, output_batch, target_batch = make_batch(
      [seq_data], vec_model.wv, seq_model["n_input"], answer_set)
  

  result = sess.run(seq_model["predictions"],
                    feed_dict={
                      seq_model["enc_input"]: input_batch,
                      seq_model["dec_input"]: output_batch,
                      seq_model["targets"]: target_batch,
                      seq_model["dropout_keep_prob"]: 1.0
                    })

  # convert index number to actual token 
  decoded = [answer_set[i] for i in result[0]]
        
  # Remove anything after '_e_'        
  if "_e_" in decoded:
    end = decoded.index('_e_')
    translated = ' '.join(decoded[:end])
  else :
    translated = ' '.join(decoded[:])
    
  return translated

def get_response(sess, wv_model, models, answer_sets, question, mode):
  if mode == "comic":
    seq_model = models[0]
    answer_set = answer_sets[0]
  elif mode == "friend":
    seq_model = models[1]
    answer_set = answer_sets[1]
  elif mode == "professional":
    seq_model = models[2]
    answer_set = answer_sets[2]
  return answer(sess, question, wv_model, seq_model, answer_set)

# deprecated: we're using an API aproach so a bit diff
def log_print(f, string):
  f.write(string + '\n')
  print(string)

def chat(log_name, sess, wv_model, models, answer_sets):
  # default session is friend
  mode = "friend"
  seq_model = models[1]
  answer_set = answer_sets[1]
  with open(log_name, 'w') as f:
    while True:
      question = input("You: ")
      f.write(f"You: {question}\n")
      if question[:13] == "!personality ":
        if question[13:] == "comic":
          log_print(f, "Switched to comic personality!")
          mode = "comic"
          seq_model = models[0]
          answer_set = answer_sets[0]
        elif question[13:] == "friend":
          log_print(f, "Switched to friend personality!")
          mode = "friend"
          seq_model = models[1]
          answer_set = answer_sets[1]
        elif question[13:] == "professional":
          log_print(f, "Switched to professional personality!")
          mode = "professional"
          seq_model = models[2]
          answer_set = answer_sets[2]
        else:
          log_print(f, "You entered an invalid personality. The only options are 'comic', 'personality', or 'friend'.")
        continue
      elif question.lower() == "!end":
        log_print(f, "Ended conversation.")
        return
          
      ans = answer(sess, question, wv_model, seq_model, answer_set)
      log_print(f, f"Chatbot: {ans}")
      
def load_chatbot():
  # load data in dataframes
  df_comic = pd.read_csv('data/qna_chitchat_the_comic.tsv', sep="\t")
  df_friend = pd.read_csv('data/qna_chitchat_the_friend.tsv', sep="\t")
  df_professional = pd.read_csv('data/qna_chitchat_the_professional.tsv', sep="\t")

  # run our preprocessing on each dataset
  seq_data_comic, _ = preprocess_data(df_comic)
  seq_data_friend, _ = preprocess_data(df_friend)
  seq_data_prof, _ = preprocess_data(df_professional)
  
  # load word embedding model
  wv_model = load_wv_model()
  
  # setup for seq2seq model
  _, output_batch_c, _ = make_batch(seq_data_comic, wv_model.wv)
  _, output_batch_f, _ = make_batch(seq_data_friend, wv_model.wv)
  _, output_batch_p, _ = make_batch(seq_data_prof, wv_model.wv)
  
  models, sess = load_seq2seq_model(output_batch_c, output_batch_f, output_batch_p)

  # sort our answer lists for each model to avoid set() non-determinism
  # this makes the seq_data match the preprocessing that happens in make_batch
  answer_sets = [
    sorted(list(set([s[1] for s in seq_data_comic] + ['_e_', '_b_', '_p_', '_u_']))),
    sorted(list(set([s[1] for s in seq_data_friend] + ['_e_', '_b_', '_p_', '_u_']))),
    sorted(list(set([s[1] for s in seq_data_prof] + ['_e_', '_b_', '_p_', '_u_'])))
  ]

  # return all our bits and bobs
  return sess, wv_model, models, answer_sets

if __name__ == "__main__":
  sess, wv_model, models, answer_sets = load_chatbot()
  print(get_response(sess, wv_model, models, answer_sets, "Hello!", "friend"))
