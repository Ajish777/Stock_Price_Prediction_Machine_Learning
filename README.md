
# Stock_Price_Prediction_Machine_Learning

Machine learning proves immensely helpful in many industries in automating tasks that earlier required human labor one such application of ML is predicting whether a particular trade will be profitable or not.

In this project, I have predicted a signal that indicates whether buying a particular stock will be helpful or not by using ML.

# Importing Libraries

Python libraries make it very easy for us to handle the data and perform typical and complex tasks with a single line of code.

* Pandas – This library helps to load the data frame in a 2D array format and has multiple functions to perform analysis tasks in one go.

* Numpy – Numpy arrays are very fast and can perform large computations in a very short time.

* Matplotlib/Seaborn – This library is used to draw visualizations.

* Sklearn – This module contains multiple libraries having pre-implemented functions to perform tasks from data preprocessing to model development and evaluation.

* XGBoost – This contains the eXtreme Gradient Boosting machine learning algorithm which is one of the algorithms which helps us to achieve high accuracy on predictions.

# Importing Dataset

The dataset I have used here to perform the analysis and build a predictive model is Tesla Stock Price data. I have taken OHLC(‘Open’, ‘High’, ‘Low’, ‘Close’) data from 1st January 2010 to 31st December 2017 which is for 8 years for the Tesla stocks.

I have attached the CSV file but you can also download the CSV file from: https://www.kaggle.com/datasets/timoboz/tesla-stock-data-from-2010-to-2020 

```
df = pd.read_csv('/content/Tesla.csv')
df.head()
```

* Output:

![Capture1](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/4a1c8b5b-24fe-41e3-8364-782796259661)

From the first five rows, we can see that data for some of the dates is missing the reason for that is on weekends and holidays Stock Market remains closed hence no trading happens on these days.

```
df.shape
```

* Output:

```
(1692, 7)
```

From this, we got to know that there are 1692 rows of data available and for each row, we have 7 different features or columns.

```
df.describe()
```
*Output:

![Capture2](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/df939756-517a-4fa9-a482-ab3c0ab8a9ec)

```
df.info()
```
* Output:

![Capture3](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/c20aeaa5-1337-4bbb-953d-3a2a3c0a49a7)

# Exploratory Data Analysis

EDA is an approach to analyzing the data using visual techniques. It is used to discover trends, and patterns, or to check assumptions with the help of statistical summaries and graphical representations. 

While performing the EDA of the Tesla Stock Price data I have analyzed how prices of the stock have moved over the period of time and how the end of the quarters affects the prices of the stock.

```
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
```
* Output:

![Capture4](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/27ee0060-ddda-4c66-b954-b93914ba0b92)

The prices of tesla stocks are showing an upward trend as depicted by the plot of the closing price of the stocks.

```
df.head()
```
* Output:

![Capture5](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/9c4d5bc7-366e-4aa6-9de5-1d9a9528ecf7)

The data in the ‘Close’ column and that available in the ‘Adj Close’ column is the same. let’s check whether this is the case with each row or not.

```
df[df['Close'] == df['Adj Close']].shape
```
* Output:
```
(1692, 7)
```

From here we can conclude that all the rows of columns ‘Close’ and ‘Adj Close’ have the same data. So, having redundant data in the dataset is not going to help so, I'll drop this column before further analysis.

```
df = df.drop(['Adj Close'], axis=1)
```
Now drawing the distribution plot for the continuous features given in the dataset.

Before moving further let’s check for the null values if any are present in the data frame.

```
df.isnull().sum()
```
* Output:

![Capture6](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/06023022-19b1-400b-b566-de591952f732)

This implied that there are no null values in the data set provided.

```
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
```

* Output:

![Capture7](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/c91d82df-a9d1-465c-808b-86b8be4b07d9)

In the distribution plot of OHLC data, we can see two peaks which means the data has varied significantly in two regions. And the Volume data is left-skewed.

```
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
```
* Output:

![Capture8](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/808a056f-c144-4694-b39d-0f6f90eff957)

From the above boxplots, we can conclude that only volume data contains outliers in it but the data in the rest of the columns are free from any outlier.

# Feature Engineering

Feature Engineering helps to derive some valuable features from the existing ones. These extra features sometimes help in increasing the performance of the model significantly and certainly help to gain deeper insights into the data.

```
splitted = df['Date'].str.split('/', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

df.head()
```

* Output:

![Capture9](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/44234ea5-6f4f-44e6-998a-cb4323928e13)

Now we have three more columns namely ‘day’, ‘month’ and ‘year’ all these three have been derived from the ‘Date’ column which was initially provided in the data.

```
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()
```
* Output:

![Capture10](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/cc9fcbe6-2f37-4d2c-b6bf-ca5985994570)

A quarter is defined as a group of three months. Every company prepares its quarterly results and publishes them publicly so, that people can analyze the company’s performance. These quarterly results affect the stock prices heavily which is why we have added this feature because this can be a helpful feature for the learning model.

```
data_grouped = df.groupby('year').mean()
plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
plt.subplot(2,2,i+1)
data_grouped[col].plot.bar()
plt.show()
```
* Output:

![Capture11](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/b51b0467-df60-4d6b-8726-71502aaeb25e)

From the above bar graph, we can conclude that the stock prices have doubled from the year 2013 to that in 2014.

```
df.groupby('is_quarter_end').mean()
```
* Output:

![Capture12](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/1cfb5d31-48b3-4adf-b1fe-682773faea4f)

Here are some of the important observations of the above-grouped data:

* Prices are higher in the months which are quarter end as compared to that of the non-quarter end months.

* The volume of trades is lower in the months which are quarter end.

```
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
```
* Output:

Above I have added some more columns which will help in the training of our model. I have added the target feature which is a signal whether to buy or not we will train our model to predict this only. But before proceeding let’s check whether the target is balanced or not using a pie chart.

```
plt.pie(df['target'].value_counts().values, 
		labels=[0, 1], autopct='%1.1f%%')
plt.show()
```
* Output:

![Capture13](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/0f10d377-f093-4d39-8d01-1dceca2fba4d)

When we add features to our dataset we have to ensure that there are no highly correlated features as they do not help in the learning process of the algorithm.

```
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
```
* Output:

![Capture14](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/04c7da6e-88ee-4a5a-8f14-88dedbf0385f)

From the above heatmap, we can say that there is a high correlation between OHLC that is pretty obvious, and the added features are not highly correlated with each other or previously provided features which means that we are good to go and build our model.

# Data Splitting and Normalization

```
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
```
* Output:
```
(1522, 3) (170, 3)
```

After selecting the features to train the model on I normalize the data because normalized data leads to stable and fast training of the model. After that whole data has been split into two parts with a 90/10 ratio so, that we can evaluate the performance of our model on unseen data.

# Model Development and Evaluation

Now is the time to train some state-of-the-art machine learning models(Logistic Regression, Support Vector Machine, XGBClassifier), and then based on their performance on the training and validation data we will choose which ML model is serving the purpose at hand better.

For the evaluation metric, we will use the ROC-AUC curve but why this is because instead of predicting the hard probability that is 0 or 1 we would like it to predict soft probabilities that are continuous values between 0 to 1. And with soft probabilities, the ROC-AUC curve is generally used to measure the accuracy of the predictions.

```
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
    models[i].fit(X_train, Y_train)

    print(f'{models[i]} : ')
    print('Training Accuracy : ', metrics.roc_auc_score(
        Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy : ', metrics.roc_auc_score(
        Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()
```

* Output:

![Capture15](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/dcededf3-aef1-4470-ba12-782f844ff36a)

Among the three models, we have trained XGBClassifier has the highest performance but it is pruned to overfitting as the difference between the training and the validation accuracy is too high. But in the case of the Logistic Regression, this is not the case.

Now let’s plot a confusion matrix for the validation data.

```
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()
```

* Output:

![Capture16](https://github.com/Ajish777/Stock_Price_Prediction_Machine_Learning/assets/110074935/adef5038-adec-4ec9-a984-082689be3556)

# Conclusion:

We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.
