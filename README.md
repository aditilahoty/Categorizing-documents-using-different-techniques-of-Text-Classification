# Categorizing-documents-using-different-techniques-of-Text-Classification

Text classification is an example of Machine Learning (ML) in the form of Natural Language Processing (NLP). By classifying text , we are aiming to assign one or more classes or categories to a document, making it easier to manage and sort. This is especially useful for publishers, news sites, blogs or anyone  who deals with a lot of content.
                                               
A more detailed look at real world document classification-

1. DATASET:  The quality of the tagged dataset is by far the most important component of a statistical NLP classifier. Our
             dataset consists of 2507 short research paper titles, largely technology related. They have been
              pre-classified manually into 5 categories conferences which are INFOCOM, ISCAS, SIGGRAPH, VLDB,WWW .
              Then,divided dataset into train and test for training and predicting the model respectively. 

2. PRE-PROCESSING:  Real world data is often incomplete, inconsistent and lacking in certain behaviour and is likely to contain
                    many errors, hence data preprocessing is necessary. In this Data goes through a series of steps:
                      
                      ● Removing stop words: Stopwords are commonly used words (such as “the”, “a”, “an”, “in”) and they play no role in classification. These are provided in the corpus of NLTK library.
                      
                      ● Removing all other characters like punctuations except alphabets
                      
                      ● Converting all text to lowercase for easy analysis
                      
                      ● Append all preprocessed data into a list

3. TEXT-VECTORIZATION:  Document representation is an important part of the classification process. Each document must be
                         processed into a form that a classifier can operate on and still preserve as much as the original information as possible.
                         In order to perform machine learning on text, we need to transform our documents (text) into vector
                          representations such that we can apply numeric machine learning. This process is called feature extraction or more simply vectorization,
                          and is an essential step for text classification.

Here we use a approach named TF-IDF, it stands for Term Frequency-Inverse Document Frequency which basically tells the importance of the word in the corpus or dataset. TF-IDF contains two concept Term Frequency(TF) and Inverse Document Frequency(IDF).

Term Frequency : It is defined as how frequently the word appears in the document or corpus. As each sentence is not the same length so it may be possible a word appearing in a long sentence occurs more time as compared to word appearing in a shorter sentence. 

Inverse Document Frequency: Inverse Document frequency is another concept which is used for finding
out the importance of the word. It is based on the fact that less frequent words are more informative and
important.

TF-IDF : It is basically a multiplication between TF values and IDF values . It basically reduces the values
of common words that are used in different documents. The most important words have more TF-IDF,
while most frequent words are not that important having less TF-IDF value.

4. FEATURE SELECTION:  The document term matrix (from Tf-Idf) is usually very high dimensional and sparse. It can create issues for
                       machine learning algorithms during the learning phase. Therefore, we have used a feature selection technique to reduce dimensionality of the     dataset.The benefits of performing feature selection before modeling your data are:
                       
1) Avoid Overfitting : Less redundant data gives performance boost to the model and results in less opportunity to make decisions based on noise
2) Reduces Training Time : Less data means that algorithms train faster
One of the most common method used for text data is Chi square test ,hence we have used it for feature selection of our data.

The chi square test is used in statistics to test the independence of two events. More precisely, In feature
selection we use chi square test to test whether the occurrence of a specific term(word/feature) and the
occurrence of a specific class(label) are independent. This test thus is used to determine the best features
for a given dataset by determining the features on which the output class label is most dependent on.
For each feature in the dataset, the χ2 score is calculated and then ordered in descending order according
to the χ2 score value. The higher the value of χ2 score, the more dependent the output label is on the
feature and higher the importance the feature has on determining the output.

  Chi-square score is given by-
  
          X2 = (Observed frequency – expected frequency)2  / (expected frequency)
          
 5. CLASSIFICATION : The following classifiers were used to predict the class/category of labels-
 
● RANDOM FOREST- Random forest is a meta estimator that fits a number of decision tree classifiers on various
                 sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
                    Random forests (RF) construct many individual decision trees at training after selecting a random
                  sample from the given dataset. Predictions from all trees are pooled to make the final prediction.
                  The reason we chose RF is because it is considered a highly accurate and robust method as the
                number of decision trees participating in the process is high. Moreover, it does not suffer from the
           overfitting problem and takes the average of all the predictions, which cancels out the biases.
                The algorithm is considered suitable for classification problems 
                
                
● SUPPORT VECTOR MACHINE- Support Vector Machine, abbreviated as SVM can be used for both regression and classification
                        tasks. But, it is widely used in classification objectives.In the SVM algorithm, we plot each data item
                          as a point in n-dimensional space (where n is number of features you have) with the value of each
                       feature being the value of a particular coordinate. Then, we perform classification by finding the
                     hyper-plane that differentiates the two classes very well. Here, we have used SVM with linear kernel
                     and have tried to compare the results with those of Random Forest.

The main objective is to segregate the given dataset in the best possible way. The distance between the either nearest points is known as the margin. The objective is to select a hyperplane with the maximum possible margin between support vectors in the given dataset. SVM searches for the maximum marginal hyperplane in the following steps:

    1. Generate hyperplanes which segregate the classes in the best way. Left-hand side figure showing three hyperplanes black, 
    blue and orange. Here, the blue and orange have higher classification error, but the black is separating the two classes correctly.
    
    2. Select the right hyperplane with the maximum segregation from the either nearest data points as shown in the right-hand side figure.
    
The SVM algorithm is implemented in practice using a kernel. A kernel transforms an input data space into the required form.A linear kernel can be used as a normal dot product of any two given observations. The product between two vectors is the sum of the multiplication of each pair of input values.


6. CROSS VALIDATION: Cross Validation is a technique which involves reserving a particular sample of a dataset on which we do
                      not train the model. Later, we test our model on this sample before finalizing it. Here, we have used
                     Stratified K fold cross validation. Stratification is the process of rearranging the data to ensure each fold is a good representative of the
                     whole. For example in a binary classification problem where each class comprises 50% of the data, it is best to arrange the data such that in every                       fold, each class comprises around half the instances.

StratifiedKFold is a variation of KFold. First, StratifiedKFold shuffles your data, after that splits the data into
n_splits parts and Done. Now, it will use each part as a test set. Note that it only and always shuffles data one time before splitting.
Then, we find values of precision,recall and fscore of test data for predicting accuracy of both classifiers:

    ● The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability   of     the classifier not to label as positive a sample that is negative.

    ● The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the   classifier to find all the positive samples.

    ● The Fscore/F1-Score can be interpreted as a weighted harmonic mean of the precision and recall, where an Fscore reaches its best value at 1 and worst score at 0.The F score can provide a more realistic measure of a test’s performance by using both precision and recall.
    

7. RESULT:

       CLASSIFIER       PRECISION  RECALL  F-SCORE
    
       Random forest       0.7093   0.6837   0.6716
    
       SVM                0.8088   0.7998   0.7934
       
 Hence,SVM classifier worked well on our dataset than randomforest.
 
 
                                                                   THANK YOU
