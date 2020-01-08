# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import pickle

import tensorflow as tf

import metrics

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")

flags.DEFINE_string(
    "gpu_device", "0",
    "Which gpu device to use")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_data(cls, input_file):
    """Reads data file"""
    data = [] #[([w1,w2,...,wn],[l1,l2,...,ln])]
    words = []
    labels = []
    with open(input_file) as f:
      for line in f:
        line = line.strip()

        # one sentence finished
        if len(line) < 1:
          l = ' '.join(labels)
          w = ' '.join(words)
          data.append((l,w))

          words, labels = [], []
          continue
        # print(line)
        if len(line.split(' ')) < 2:
          continue
        word = line.split(' ')[0]
        label = line.split(' ')[1]
        words.append(word)
        labels.append(label)

    return data




class EEProcessor(DataProcessor):
  """Processor for the ACE 2005 data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, "train.data.en.new.txt")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, "dev.data.en.new.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_data(os.path.join(data_dir, "test.data.en.new.txt")), "test")

  def get_labels(self):
    """See base class."""
    # return ['B-Acquit', 'B-Appeal', 'B-Arrest-Jail', 'B-Attack', \
    # 'B-Be-Born', 'B-Charge-Indict', 'B-Convict', 'B-Declare-Bankruptcy', \
    # 'B-Demonstrate', 'B-Die', 'B-Divorce', 'B-Elect', 'B-End-Org', \
    # 'B-End-Position', 'B-Execute', 'B-Extradite', 'B-Fine', 'B-Injure', \
    # 'B-Marry', 'B-Meet', 'B-Merge-Org', 'B-Nominate', 'B-Pardon', 'B-Phone-Write', \
    # 'B-Release-Parole', 'B-Sentence', 'B-Start-Org', 'B-Start-Position', 'B-Sue', \
    # 'B-Transfer-Money', 'B-Transfer-Ownership', 'B-Transport', 'B-Trial-Hearing', \
    # 'I-Acquit', 'I-Appeal', 'I-Arrest-Jail', 'I-Attack', 'I-Be-Born', 'I-Charge-Indict', \
    # 'I-Convict', 'I-Declare-Bankruptcy', 'I-Demonstrate', 'I-Die', 'I-Divorce', 'I-Elect', \
    # 'I-End-Org', 'I-End-Position', 'I-Execute', 'I-Extradite', 'I-Fine', 'I-Injure', 'I-Marry', \
    # 'I-Meet', 'I-Merge-Org', 'I-Nominate', 'I-Pardon', 'I-Phone-Write', 'I-Release-Parole', 
    # 'I-Sentence', 'I-Start-Org', 'I-Start-Position', 'I-Sue', 'I-Transfer-Money', 'I-Transfer-Ownership', \
    # 'I-Transport', 'I-Trial-Hearing', 'O','[CLS]','[PAD]']

    return ['Acquit', 'Appeal', 'Arrest-Jail', 'Attack', 'Be-Born', 'Charge-Indict', \
    'Convict', 'Declare-Bankruptcy', 'Demonstrate', 'Die', 'Divorce', 'Elect', 'End-Org', \
    'End-Position', 'Execute', 'Extradite', 'Fine', 'Injure', 'Marry', 'Meet', 'Merge-Org', 
    'Nominate', 'O', 'Pardon', 'Phone-Write', 'Release-Parole', 'Sentence', 'Start-Org', \
    'Start-Position', 'Sue', 'Transfer-Money', 'Transfer-Ownership', 'Transport', 'Trial-Hearing', '[CLS]','[PAD]']

  def _create_examples(self, data, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, sentence) in enumerate(data):
      label, text = sentence[0], sentence[1]
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(text)
      label = tokenization.convert_to_unicode(label)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """
  Converts a single `InputExample` into a single `InputFeatures`.
  example:[Jim,Hen,##son,was,a,puppet,##eer]
  labels:[I-PER,I-PER,X,O,O,O,X]

  """

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  with open(FLAGS.middle_output+"/label2id.pkl",'wb') as w:
    pickle.dump(label_map,w)


  textlist = example.text_a.split(' ')
  labellist = example.label.split(' ')
  tokens = []
  labels = []

  for i,(word, label) in enumerate(zip(textlist, labellist)):
    token = tokenizer.tokenize(word)
    tokens.extend(token)
    for i, _ in enumerate(token):
      labels.append(label)
      # if i == 0:
      #   labels.append(label)
      # else:
      #   if label[0] == 'B':
      #     labels.append('I' + label[1:])
      #   else:
      #     labels.append(label)


  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[:(max_seq_length-1)]
    labels = labels[:(max_seq_length-1)]

  ntokens = []
  segment_ids = []
  label_ids = []
  ntokens.append("[CLS]")
  segment_ids.append(0)
  label_ids.append(label_map["[CLS]"])
  for i, token in enumerate(tokens):
    ntokens.append(token)
    segment_ids.append(i)
    label_ids.append(label_map[labels[i]])

  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  mask = [1]*len(input_ids)

  # padding
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    mask.append(0)
    segment_ids.append(0)
    label_ids.append(label_map['[PAD]'])
    ntokens.append("[PAD]")
  assert len(input_ids) == max_seq_length
  assert len(mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length
  assert len(ntokens) == max_seq_length  
  
  if ex_index < 3:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in ntokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))      

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=mask,
    segment_ids=segment_ids,
    label_ids=label_ids,
  )

  return feature, ntokens, label_ids



def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """
  Convert a set of `InputExample`s to a TFRecord file.


  """

  writer = tf.python_io.TFRecordWriter(output_file)

  batch_tokens = []
  batch_labels = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 5000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

    batch_tokens.extend(ntokens)
    batch_labels.extend(label_ids)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()

  return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

# def hidden2tag(hiddenlayer,numclass):
#   """
#   进行线性变换
#   """
#   linear = tf.keras.layers.Dense(numclass,activation=None)
#   return linear(hiddenlayer)

# def crf_loss(logits, labels, mask, num_labels, mask2len):
#   """

#   """
#   with tf.variable_scope("crf_loss"):
#     trans = tf.get_variable(
#       name = "transition",
#       shape = [num_labels, num_labels],
#       initializer = tf.contrib.
#       )



def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""

  # Bert
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.

  # shape = (batch size, seq_length, embedding_size)
  output_layer = model.get_sequence_output()

  # output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout

      # shape = (batch size, seq_length, embedding_size)
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    # shape = (batch size * seq_length, embedding_size)
    output_layer = tf.reshape(output_layer, [-1, hidden_size])
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    # shape = (batch size * seq_length, num_labels)
    logits = tf.nn.bias_add(logits, output_bias)
    # shape = (batch size * seq_length,）
    labels = tf.reshape(labels, [-1])

    mask = tf.cast(input_mask, dtype=tf.float32)
    # shape = (batch size * seq_length, num_labels)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)

    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.

    # shape = (batch size * seq_length, num_labels)
    probabilities = tf.math.softmax(logits, axis=-1)
    # shape = (batch size * seq_length,)
    predict = tf.math.argmax(probabilities, axis=-1)

    return (loss, logits, predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    print('shape of input_ids', input_ids.shape)

    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, logits, predicts) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    # 加载bert模型
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None

    # 训练状态
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    # eval状态
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(label_ids, logits, num_labels, mask):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels-1, weights=mask)

        return {
          "confusion_matrix": cm
        }

        # accuracy = tf.metrics.accuracy(
        #     labels=label_ids, predictions=predictions, weights=is_real_example)
        # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        # return {
        #     "eval_accuracy": accuracy,
        #     "eval_loss": loss,
        # }

      eval_metrics = (metric_fn,
                      [label_ids, logits, num_labels, input_mask])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predicts,
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      # "cola": ColaProcessor,
      # "mnli": MnliProcessor,
      # "mrpc": MrpcProcessor,
      # "xnli": XnliProcessor,
      "ee": EEProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  # get labels list
  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)

    # 训练多少步，每一步训练数据是一个batch
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

    with open(output_eval_file, 'w') as f:
      tf.logging.info("***** Eval results *****")
      confusion_matrix = result["confusion_matrix"]
      p,r,f=metrics.calculate(confusion_matrix, len(label_list) - 1)
      tf.logging.info("***************************************")
      tf.logging.info("**************P = %s*************************", str(p))
      tf.logging.info("**************R = %s*************************", str(r))
      tf.logging.info("**************F = %s*************************", str(f))

    # with tf.gfile.GFile(output_eval_file, "w") as writer:
    #   tf.logging.info("***** Eval results *****")
    #   for key in sorted(result.keys()):
    #     tf.logging.info("  %s = %s", key, str(result[key]))
    #     writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    with open(FLAGS.middle_output+'/label2id.pkl', 'rb') as f:
      label2id = pickle.load(f)
      id2label = {v:k for k,v in label2id.items()}

    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    batch_tokens, batch_labels = file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")


    print_predictions(output_predict_file, result, batch_tokens, batch_labels, id2label)
    # Writer(output_predict_file,result,batch_tokens,batch_labels,id2label)

    # output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    # with tf.gfile.GFile(output_predict_file, "w") as writer:
    #   num_written_lines = 0
    #   tf.logging.info("***** Predict results *****")
    #   for (i, prediction) in enumerate(result):
    #     probabilities = prediction["probabilities"]
    #     if i >= num_actual_predict_examples:
    #       break
    #     output_line = "\t".join(
    #         str(class_probability)
    #         for class_probability in probabilities) + "\n"
    #     writer.write(output_line)
    #     num_written_lines += 1
    # assert num_written_lines == num_actual_predict_examples

def print_predictions(output_predict_file, result, batch_tokens, batch_labels, id2label):
  """
  对test的预测结果进行打印，用于后续的分析
  """
  with open(output_predict_file, 'w') as f:
    for i, prediction in enumerate(result):
      token = batch_tokens[i]
      predict = id2label[prediction]
      true_label = id2label[batch_labels[i]]
      if token!="[PAD]" and token!="[CLS]" and true_label!="X":
        if predict=="X" and not predict.startswith("##"):
          predict = "O"
        line = "{}\t{}\t{}\n".format(token,true_label,predict)
        f.write(line)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_device
  tf.app.run()
