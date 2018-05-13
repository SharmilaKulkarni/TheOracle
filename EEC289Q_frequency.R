# Code considering only the frequency of the words appearing as a feature for classification
# File name: EEC289Q_frequency.R
# Last modified: May 13th 2018

# libraries we'll need
library(tidytext)
library(tidyverse)
library(glue)
library(dplyr)

# Read in our data
train <- read.csv("train.csv", sep=",")

# Separating the data set into 90% traning and 10% testing
texts<-sample_frac(train, 0.9)
sid<-as.numeric(rownames(texts)) 
test<-texts[-sid,]

# split the data by author
grp_author <- group_by(texts, author)

# Frequency calculation

author_freq_words <-  texts %>%
  group_by(author) %>% # group by author
  select(text) %>% # grab only the Sentence column
  mutate(text = as.character(text)) %>% # convert them to characters
  unnest_tokens(words, text) %>% # tokenize
  count(words) %>% # frequency by token (by author)
  bind_tf_idf(words, author, n) # normalized frequency

### Use this to make guesses about who wrote a given sentence

# One way to guess authorship is to use the joint probabilty that each 
# author used each word in a given sentence.

  # joint probability with padding
  joint_prob <- function(author, testSentence, tf_idf){

  # get the term frequencytestSentence
  freq <- inner_join(author_freq_words, testSentence)   # get the term frequency
  # select just the target author
  grp_author <- freq[freq$author == author,]
  
  # number of returned terms 
  returnedTerms <- dim(grp_author)[1]
  
  # add a very small amount for every term in the test sentence we didn't see
  # in our training corpus
  if(length(testSentence$words) < dim(grp_author)[1]){
    # making the smoothing term very low reflects the idea that we think it's
    # unlikely that the author would use this term, since we haven't seen them 
    # use it before
    smoothingTerm <- (length(testSentence$words) - returnedTerms) * 0.000001
  } else {
    # since we're taking the product, making the smoothing term 1 won't
    # change our results
    smoothingTerm <- 1
  }
  
  # return probaility
  return(prod(c(grp_author$tf, smoothingTerm)))
}
n <-0 

for (j in 1:nrow(test)){
# first, let's start with a test sentence
testSentence <- as_data_frame(list(text = as.character(test$text[j])))

# tokenize it
preProcessedTestSentence <- unnest_tokens(testSentence, words, text)

# get the frequency for each word by author
testProbailities <- author_freq_words[author_freq_words$words %in% preProcessedTestSentence$words,]

# empty variable to put our predictions in
est_author <- NULL

# Joint probability for each author
for(i in levels(texts$author)){
  est_author <- c(est_author, joint_prob(i, preProcessedTestSentence, author_freq_words))
}

# and the author is
levels(test$author)[which.max(est_author)]

if(test$author[j] == levels(test$author)[which.max(est_author)]){n=n+1}
}
pred_accuracy <- n*100/nrow(test)
