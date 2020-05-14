# Stock-Prediction-through-ML

**Idea**: 
Fortune 500 stocks follow the same trend as each other. Can we benefit from the lag in between the stocks before their prices catch up?

**What**: 
- Achieved 74% accuracy from the model.

**How**:
- Used Flask to scrape Fortune 500 tickers from Wikipedia and get their information from Yahoo finance.
- Verified strong correlation between Fortune 500 stocks through a heatmap in Matplotlib
- Used Voting Classifier from TensorFlow, to build the machine learning model.

**Additional Details**
- Made addtional columns for percent change in price day_1, ... day_7
- Used the additional columns to make the target attribute of -1, 0, 1 -> buy/sell/hold rating.
- If the percentage change would exceed a certain threshold, it would get a buy/sell rating. Else a hold rating.


**Difficulties**
- Making the visualization.
- Had to choose 7 days as a window against which correlation is considered useful. Else, the accuracy would fall.
- Had to increase the volatility threshold used to give a buy/sell rating to account for unreasoned deviation.





