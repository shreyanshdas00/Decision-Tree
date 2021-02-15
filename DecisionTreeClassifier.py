import pandas as pd
import numpy
import matplotlib.pyplot as plt
import copy

class Question:

  def __init__(self, feature_type, feature, value):

    self.__feature_type = 'Categorical' if not feature_type else 'Numerical' #specifies type of value to be compared
    self.__feature = feature #feature name whose value is to be compared
    self.__value = value #value on which the split is to be made

  def __str__(self):
    
    return str(self.__feature) + (' == ' if self.__feature_type == 'Categorical' else ' >= ') + str(self.__value)
    
  def partition(self, data):
    
    left_child_data = right_child_data = None
    if self.__feature_type == 'Categorical':
      try:
        right_child_data = data[data[self.__feature] == self.__value]
      except:
        pass
      try:
        left_child_data = data.drop(right_child_data.index)
      except:
        pass
    else:
      try:
        right_child_data = data[data[self.__feature] >= self.__value]
      except:
        pass
      try:
        left_child_data = data.drop(right_child_data.index)
      except:
        pass
    return left_child_data, right_child_data


class DecisionNode:

  __max_leaves = -1

  def __init__(self, data, labels, max_depth = -1, max_leaves = -1, min_child_weight = 0, gini_impurity = None):

    self.__max_depth = max_depth
    self.__min_child_weight = min_child_weight
    DecisionNode.__max_leaves = max_leaves
    self.__is_leaf = True
    self.__data = data
    self.__labels = labels
    self.__data_size = len(self.__data)
    self.__question = None
    self.__gini_impurity = self.get_gini_impurity(self.__labels) if gini_impurity is None else gini_impurity
    self.__left_child = None
    self.__right_child = None
    self.__predictions = list(zip(list(self.__labels['Label'].unique()), [(len(self.__labels[self.__labels['Label'] == unique_label]) / self.__data_size) for unique_label in list(self.__labels['Label'].unique())]))
  
  def get_question(self):

    return self.__question

  def is_leaf(self):

    return self.__is_leaf
  
  def get_predictions(self):

    return self.__predictions
  
  def go_left(self):

    return self.__left_child
  
  def go_right(self):

    return self.__right_child
  
  def get_gini_impurity(self, data_chunk):

    data_chunk_size = len(data_chunk)
    gini_impurity = 1
    for class_label in data_chunk['Label'].unique():
      gini_impurity -= ((len(data_chunk[data_chunk['Label'] == class_label])/data_chunk_size)**2)
    return gini_impurity * data_chunk_size/self.__data_size
  
  def get_information_gain(self, left_child_gini_impurity, right_child_gini_impurity):

    return self.__gini_impurity - (left_child_gini_impurity + right_child_gini_impurity)
  
  def get_best_split(self):

    if self.__max_depth == 1 or self.__gini_impurity == 0 or self.__max_leaves == 1:
      self.__is_leaf = True
      return
    else:
      max_information_gain = 0
      best_question = None
      left_child_data = None
      left_child_labels = None
      left_child_gini_impurity = None
      right_child_data = None
      right_child_labels = None
      right_child_gini_impurity = None
      for feature in list(self.__data.columns):
        curr_information_gain = 0
        curr_question = None
        curr_left_child_data = None
        curr_left_child_labels = None
        curr_left_child_gini_impurity = None
        curr_right_child_data = None
        curr_right_child_labels = None
        curr_right_child_gini_impurity = None
        done = {}
        thresholds = sorted(list(copy.deepcopy(self.__data[feature])))
        feature_type_imposed_limit = 0 if type(thresholds[0]) is numpy.str_ or type(thresholds[0]) is str else 1
        for index in range(len(thresholds) - feature_type_imposed_limit):
          if feature_type_imposed_limit:
            value = (thresholds[index] + thresholds[index+1])/2
          else:
            value = thresholds[index]
          if value in done:
            continue
          curr_question = Question(feature_type_imposed_limit, feature, value)
          curr_left_child_data, curr_right_child_data = curr_question.partition(self.__data)
          done[value] = 1
          if len(curr_left_child_data) < self.__min_child_weight or len(curr_right_child_data) < self.__min_child_weight:
              curr_information_gain = 0
          else:
              curr_left_child_labels = self.__labels.loc[curr_left_child_data.index]
              curr_right_child_labels = self.__labels.drop(curr_left_child_data.index)
              curr_left_child_gini_impurity = self.get_gini_impurity(curr_left_child_labels)
              curr_right_child_gini_impurity = self.get_gini_impurity(curr_right_child_labels)
              curr_information_gain = self.get_information_gain(curr_left_child_gini_impurity, curr_right_child_gini_impurity)  
          if curr_information_gain > max_information_gain:
            max_information_gain = curr_information_gain
            best_question = curr_question
            left_child_data = curr_left_child_data
            left_child_labels = curr_left_child_labels
            left_child_gini_impurity = curr_left_child_gini_impurity
            right_child_data = curr_right_child_data
            right_child_labels = curr_right_child_labels
            right_child_gini_impurity = curr_right_child_gini_impurity
      if max_information_gain == 0:
        DecisionNode.__max_leaves -= 1
      else:
        DecisionNode.__max_leaves -= 1
        self.__is_leaf = False
        self.__question = best_question
        self.__left_child = DecisionNode(left_child_data, left_child_labels, self.__max_depth -1, DecisionNode.__max_leaves, self.__min_child_weight, left_child_gini_impurity)
        self.__right_child = DecisionNode(right_child_data, right_child_labels, self.__max_depth -1, DecisionNode.__max_leaves, self.__min_child_weight, right_child_gini_impurity)
        
  def predict_direction(self, data):

    return self.__question.partition(data)


class DecisionTreeClassifier:

  def __init__(self, max_depth = -1, max_leaves = -1, min_child_weight = 0):#, gamma = -1):

    self.__max_depth = max_depth
    self.__max_leaves = max_leaves
    self.__min_child_weight = min_child_weight
    #self.__gamma = gamma
    self.__root = None

  def fit(self, train_data, train_labels, cross_validation_data, cross_validation_labels):

    train_labels = pd.DataFrame(copy.deepcopy(list(train_labels)), columns = ['Label'])
    train_labels.index = train_data.index
    cross_validation_labels = pd.DataFrame(copy.deepcopy(list(cross_validation_labels)), columns = ['Label'])
    cross_validation_labels.index = cross_validation_data.index
    self.__root = DecisionNode(train_data, train_labels, self.__max_depth, self.__max_leaves, self.__min_child_weight)
    self.train(self.__root, train_data, train_labels, cross_validation_data, cross_validation_labels, [[self.evaluate_loss(train_data, train_labels), self.evaluate_accuracy(train_data, train_labels)]], [[self.evaluate_loss(cross_validation_data, cross_validation_labels), self.evaluate_accuracy(cross_validation_data, cross_validation_labels)]])

  def train(self, decision_node, train_data, train_labels, cross_validation_data, cross_validation_labels, loss_training = [], loss_cross_validation = [], is_first = True):

    decision_node.get_best_split()
    if decision_node.get_question() is not None or is_first:
      loss_training.append([self.evaluate_loss(train_data, train_labels), self.evaluate_accuracy(train_data, train_labels)])
      loss_cross_validation.append([self.evaluate_loss(cross_validation_data, cross_validation_labels), self.evaluate_accuracy(cross_validation_data, cross_validation_labels)])
      fig, axes = plt.subplots(1, 2,  sharex=True, figsize=(14,7))
      fig.clf()
      loss_plot = fig.add_subplot(1, 2, 1)
      loss_plot.plot([loss for loss, accuracy in loss_training], label = 'Training Loss')
      loss_plot.plot([loss for loss, accuracy in loss_cross_validation], label = 'Cross_Validation Loss')
      loss_plot.legend(loc='upper left')
      accuracy_plot = fig.add_subplot(1, 2, 2)
      accuracy_plot.plot([accuracy for loss, accuracy in loss_training],label='Training Accuracy')
      accuracy_plot.plot([accuracy for loss, accuracy in loss_cross_validation], label = 'Cross_Validation Accuracy')
      accuracy_plot.legend(loc='upper left')
      plt.suptitle('Loss and Accuracy Plots')
      plt.show()
      print("Node added:", decision_node.get_question())
      print("Loss on training set: %.6f" %loss_training[-1][0])
      print("Loss on cross validation set: %.6f" %loss_cross_validation[-1][0])
      print("Accuracy on training set: %.6f" %loss_training[-1][1])
      print("Accuracy on cross validation set: %.6f" %loss_cross_validation[-1][1], end ='\n\n')
      self.train(decision_node.go_left(), train_data, train_labels, cross_validation_data, cross_validation_labels, loss_training, loss_cross_validation, False)
      self.train(decision_node.go_right(), train_data, train_labels, cross_validation_data, cross_validation_labels, loss_training, loss_cross_validation, False)

  def predict(self, data):

    predictions = self.prediction_call(data, self.__root).sort_values('Index Map')
    predictions.index = list(predictions['Index Map'])
    return predictions.drop(columns = ['Index Map'])

  def prediction_call(self, data, decision_node):

    if decision_node.is_leaf():
      return pd.DataFrame(list(zip(list(data.index),list([decision_node.get_predictions()])*len(data))), columns = ['Index Map', 'Prediction'])
    left, right = decision_node.predict_direction(data)
    if left is not None and right is not None:
      return pd.concat([self.prediction_call(left, decision_node.go_left()), self.prediction_call(right, decision_node.go_right())], axis=0)
    elif left is None:
      return self.prediction_call(left, decision_node.go_right())
    elif right is None:
      return self.prediction_call(right, decision_node.go_left())
    else:
      pass
  
  def evaluate_loss(self, data, data_labels):

    data_labels = pd.DataFrame(copy.deepcopy(data_labels), columns = ['Label'])
    data_labels.index = data.index
    data_labels = list(data_labels.sort_index()['Label'])
    probability_predictions = list(self.predict(data)['Prediction'])
    predicted_probabilities = [predictions[[int(prediction[0] == data_label) for prediction in predictions].index(max([int(prediction[0] == data_label) for prediction in predictions]))][1] for predictions, data_label in zip(probability_predictions, data_labels)]
    return sum([((1 - predicted_probability)**2) for predicted_probability in predicted_probabilities]) / len(data)

  def evaluate_accuracy(self, data, data_labels):

    data_labels = pd.DataFrame(copy.deepcopy(data_labels), columns = ['Label'])
    data_labels.index = data.index
    data_labels = list(data_labels.sort_index()['Label'])
    probability_predictions = list(self.predict(data)['Prediction'])
    class_predictions = [predictions[[prediction[1] for prediction in predictions].index(max([prediction[1] for prediction in predictions]))][0] for predictions in probability_predictions]
    return 100 * sum([int(class_prediction == data_label) for class_prediction, data_label in zip(class_predictions, data_labels)])/len(data)
