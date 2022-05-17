# Stock-Portfolio-Diversification-Using-Clustering-and-Volatility-Prediction

## Problem Statement
How can we diversify an investor’s portfolio such that it can be profitable and has stocks with a security worth investing? How do we present information to an investor looking to diversify their stock portfolio, maximizing their profits such that they can study the company and make an informed decision before investing?
One of the answers to this question is forecasting the stock’s closing price. However, it is very difficult to predict something that depends on the future which is uncertain and unpredictable. Stock data is also very volatile. As you can see in the figure below, the stock closing price trends look like that of a random walk.

![image](https://user-images.githubusercontent.com/41315903/168646768-3764d894-694e-459a-82b0-7fe7ba0e55d8.png)


Keeping the above in mind and multiple ways the problem can be approached, this project intends to focus on volatility and how stock’s observed historical volatility and prediction can help diversify an investor’s portfolio. The level of volatility is an important factor for an investor as higher volatility creates greater risk and uncertainty of the portfolio’s performance at any given time. The increased uncertainty also reduces the ability to predict performance and use financial models. The idea of diversifying a portfolio is to have securities in one’s portfolio such that they cancel out each other’s volatility.

## Data Sources
The data (symbol, name, sector, and sub-sector) of companies in the S&P 500 index is scraped from the Wikipedia page at this [link](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies). The individual stock data for the company and for the S&P 500 index is gathered using the YFinance library in Python. The YFinance library offers to download market data from Yahoo Finance.

## Methodology
This step consists of two parts. The first part focuses on clustering stocks based on their weekly percentage returns. The second part focuses on volatility clustering and prediction.

### K-Means Clustering
K-Means Clustering is used to profile stocks with similar trends in weekly percentage returns. K-Means Clustering is an unsupervised machine learning technique in which the algorithm clusters the data in an iterative manner. It is used to cluster or profile unlabeled data to predict the class of observations without a target vector. The ‘K’ value here denotes the number of clusters.

In this project, percentage of weekly returns of each stock was calculated. Next, the elbow method was utilized to find the optimal value of number of clusters – ‘K’. The elbow method aims to find that value of ‘K’ such that there is not substantial decrease in the within cluster sum of squares (WCSS) value. The below figure is the graph generated through the yellowbricks python library which shows the optimal value of ‘K’ selected. **The optimal number of clusters is 5.**

![image](https://user-images.githubusercontent.com/41315903/168650632-43f209ff-c95d-4963-a32d-1e74545c7a0e.png)

The below figure shows the weekly percentage of returns for stocks grouped together in a cluster.

a. Cluster 1 – 305 stocks (e.g. Bank of America, AT&T).

b. Cluster 2 – 6 stocks (e.g. AMD, Tesla).

c. Cluster 3 – 40 stocks (e.g. Apple, Moderna).

d. Cluster 4 – 1 stock (NVIDIA).

e. Cluster 5 – 152 stocks (e.g. Google, Texas Instruments).

![image](https://user-images.githubusercontent.com/41315903/168651002-0713adea-ee85-4c74-b441-7bbe9228433a.png)

The Silhouette score of 0.4568 was achieved by this clustering algorithm.
One major inference from this is that Stocks that are volatile for a long time will remain volatile while stocks that are non-volatile for a long time, will remain non-volatile.

### Volatility Clustering and Prediction.
Modelling volatility is essentially modelling uncertainty. When forecasting volatility, we are forecasting realized volatility or return volatility. Realized volatility or return volatility is the square root of sum of squared returns. Below is the graph showing the realized volatility for the S&P 500 index.

![image](https://user-images.githubusercontent.com/41315903/168651257-c1b6b2c6-3dc0-4f62-9d73-ca9c8bc37590.png)

The result of this section can be viewed on the models deployed on Streamlit (Please use Firefox) - https://share.streamlit.io/kedarghule/stock-portfolio-diversification-using-clustering-and-volatility-prediction/main/main.py
