# Stock-Prediction-through-ML

**Idea**: 
- Fortune 500 stocks follow the same trend as each other. Can we benefit from the lag in between the stocks before their prices catch up?

**What**: 
- Achieved 74% accuracy from the model.

**How**:
- Used Beautiful Soup to scrape Fortune 500 tickers from Wikipedia and get their information from Yahoo finance.
- Verified strong correlation between Fortune 500 stocks through a heat map in Matplotlib
- Used Ensemble Method, Voting Classifier from Scikit-Learn, to build the machine learning model.

**Additional Details**
- Made addtional columns for percent change in price day_1, ... day_7.
- Used the additional columns to make the target attribute of -1, 0, 1 -> buy/sell/hold rating.
- If the percentage change would exceed a certain threshold, it would give a buy/sell rating. Else a hold rating.
- Used Linear Support Vector Classifier, K Neighbours Classifier, Random Forest Classifier in Voting Classifier.
  - Tried single classifiers. Didn't give an accuracy more than Ensemble Method. Even the proportion of buy/sell/hold would be wrong. 
  - Had the following options for classifiers:
        - Linear Classfier: Logistic Regression and Naive Bayes Classifier - We considered, features as dependent, hence not used.
        - Nearest Neighbor - Chosen.
        - Support Vector Machines - Chosen.
        - Decision Trees - Random forest accounts for decision trees.
        - Random Forest - Chosen. It corrects for decision trees’ habit of over fitting to their training set.
        - Neural Networks - Too much data manipulation needed like encoding, array dimensions etc.
    

**Difficulties**
- All fortune 500 tickers not available in Yahoo for the given starting and ending time. (Made a list of N/A tickers to solve this problem)
- Had to account for na's in the dataset.
- Making the visualization.
- Had to choose 7 days as a window against which correlation is considered useful. Else, the accuracy would fall.
- Had to increase the volatility threshold used to give a buy/sell rating to account for unreasoned deviation. Led to increase in accuracy.

*Correlation Heat Map*
> ![alt text](https://github.com/yousufafroze/Stock-Prediction-through-ML/blob/master/visualization_1.png)

*Zoomed in Heat Map*
> ![alt text](https://github.com/yousufafroze/Stock-Prediction-through-ML/blob/master/visualization_2.png)





