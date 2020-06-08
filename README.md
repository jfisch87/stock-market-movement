# Predicting Short Term Price Changes in the Stock Market
---

## Executive Summary

I am testing the efficient markets hypothesis by building a neural network model to predict price movement in the 30 companies that make up the Dow Jones Industrial Average.  Each company will be look at individually and resampled down to seconds and minutes from individual tick data.  I calculated percent changes for a number of different intervals and a sentiment score from relevant twitter posts.  The second data shows that most of the time the price did not change, however, it did about two-thirds of the time with the minute data.  

### Data

All of the market data was downloaded from [this]() website.  The data seems fine, but I don't 100% trust the source.  However, to get four months of tick data would be quite expensive for this project.  The twitter data was pulled using the [GetOldTweets3]() library that I had to modify so it would stop returning `549-too many requests` errors using python's `time` library and a [RateLimit]() library.  I had to add a few clauses in so allow the script to sleep before continuing on.

With all the data collected, I cleaned up each file and had everything uniformly layed out.  Using [vaderSentiment](https://github.com/cjhutto/vaderSentiment) in the [Natural Language Toolkit](http://www.nltk.org/index.html), I was able to find a sentiment score for each tweet.  After resampling all the data, a weighted mean for the tweet score was calculated.  Then, after removing non-trading hours and days (weekends, 4pm-9:30am, Thanksgiving, Christmas) I calculated a percent change for the period's high and mean prices over a number of periods (1, 2, 3, 4, 5, 10, 15, 30, 60).  Each period with no trades was given a volume of 0 and the price from the previous period.  

### Model

I built a recurrent neural network using Keras and Tensorflow.  I have 4 hidden layers with 1 output layer.  Using the time series generator with a lenth of 5 and a batch size of 512 to model this. 

| Type | Nodes |
| ----- | ---- |
| GRU | 32 |
| GRU | 16 |
| Dense | 8 |
| Dense | 4 |
| Dense (output) | 3 |

### Conclusions/Next Steps

There are so many different places to go from here to try to improve the models.  First is to use smaller companies that don't have as many people trading them.  There may be a better pattern to find in less liquid data.  Furthermore, working with tick data may be the best even though I didn't do it here.  Next, more market data could be useful, including bid/ask size and depth, the movement of the market as a whole, more company information or trying to correlate it with competitors.  Or, I can try pulling in commodity or foreign exchange data and using that for relevant companies or just trying to model those.  Adding additional weights and language processing for twitter posts and news headlines could work on building a model for something more long term.  Additionally, I would need to use a more powerful computer next time so it doesn't run for a few days.  