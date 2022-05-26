import pandas
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as smtsa

import matplotlib.pyplot as pyplot

import thinkplot
import thinkstats2

FORMATS = ['png']

def ReadData():
    """Reads data about avocado transactions.
    returns: DataFrame
    """
    transactions = pandas.read_csv('avocado.csv', parse_dates=[0])
    return transactions

transactions = ReadData()

def GroupByDay(transactions, func=np.mean):
    grouped = transactions[['date', 'average_price']].groupby('date')
    daily = grouped.aggregate(func)
 
    daily['date'] = daily.index
    start = daily.date[0]
    one_year = np.timedelta64(1, 'Y')
    daily['years'] = (daily.date - start) / one_year
 
    return daily
    
def GroupByTypeAndDay(transactions):
    groups = transactions.groupby('type')
    dailies = {}
    for name, group in groups:
        dailies[name] = GroupByDay(group)        
 
    return dailies
    
def RunLinearModel(daily):
    model = smf.ols('average_price ~ years', data=daily)
    results = model.fit()
    return model, results
  
dailies = GroupByTypeAndDay(transactions)
for name, daily in dailies.items():
    model, results = RunLinearModel(daily)
    print(results.summary())
    
def PlotFittedValues(model, results, label=''):
    years = model.exog[:,1]
    values = model.endog
    thinkplot.Scatter(years, values, s=15, label=label)
    thinkplot.Plot(years, results.fittedvalues, label='model' )

def SerialCorr(series, lag=1):
    xs = series[lag:]
    ys = series.shift(lag)[lag:]
    corr = thinkstats2.Corr(xs, ys)
    return corr

resid = (transactions[transactions.type=='conventional'].average_price).dropna()
corr = SerialCorr(resid, 1)

def PlotResiduals(model, results):
    """Plots the residuals of a model.
    model: StatsModel model object
    results: StatsModel results object    
    """
    years = model.exog[:, 1]
    thinkplot.Plot(years, results.resid, linewidth=0.5, alpha=0.5)


def PlotResidualPercentiles(model, results, index=1, num_bins=20):
    """Plots percentiles of the residuals.
    model: StatsModel model object
    results: StatsModel results object
    index: which exogenous variable to use
    num_bins: how many bins to divide the x-axis into
    """
    exog = model.exog[:, index]
    resid = results.resid.values
    df = pandas.DataFrame(dict(exog=exog, resid=resid))

    bins = np.linspace(np.min(exog), np.max(exog), num_bins)
    indices = np.digitize(exog, bins)
    groups = df.groupby(indices)

    means = [group.exog.mean() for _, group in groups][1:-1]
    cdfs = [thinkstats2.Cdf(group.resid) for _, group in groups][1:-1]

    thinkplot.PrePlot(3)
    for percent in [75, 50, 25]:
        percentiles = [cdf.Percentile(percent) for cdf in cdfs]
        label = '%dth' % percent
        thinkplot.Plot(means, percentiles, label=label)
        
def FillMissing(daily, span=30):
    """Fills missing values with an exponentially weighted moving average.
    Resulting DataFrame has new columns 'ewma' and 'resid'.
    daily: DataFrame of daily prices
    span: window size (sort of) passed to ewma
    returns: new DataFrame of daily prices
    """
    dates = pandas.date_range(daily.index.min(), daily.index.max())
    reindexed = daily.reindex(dates)
    reindexed=reindexed.dropna()
    ewm=reindexed['average_price'].ewm(span=30).mean()

    resid = reindexed.average_price - ewm
    fake_data = ewm + thinkstats2.Resample(resid, len(reindexed))
    reindexed.average_price.fillna(fake_data, inplace=True)

    reindexed['ewma'] = ewm
    reindexed['resid'] = reindexed.average_price - ewm
    return reindexed
def PrintSerialCorrelations(dailies):
    """Prints a table of correlations with different lags.
    dailies: map from category name to DataFrame of daily prices
    """
    filled_dailies = {}
    for name, daily in dailies.items():
        filled_dailies[name] = FillMissing(daily, span=30)

    # print serial correlations for raw price data
    for name, filled in filled_dailies.items():            
        corr = thinkstats2.SerialCorr(filled.average_price, lag=1)
        print(name, corr)

    rows = []
    for lag in [1, 7, 30, 300]:
        row = [str(lag)]
        for name, filled in filled_dailies.items():         
            corr = thinkstats2.SerialCorr(filled.resid, lag)
            row.append('%.2g' % corr)
        rows.append(row)

    print(r'\begin{tabular}{|c|c|c|c|}')
    print(r'\hline')
    print(r'lag & conventional & organic \\ \hline')
    for row in rows:
        print(' & '.join(row) + r' \\')
    print(r'\hline')
    print(r'\end{tabular}')
    filled = filled_dailies['conventional']
    acf = smtsa.acf(filled.resid, nlags=365)
    print('%0.3f, %0.3f, %0.3f, %0.3f, %0.3f' % 
          (acf[0], acf[1], acf[7], acf[30], acf[300]))
          
def SimulateAutocorrelation(daily, iters=1001, nlags=40):
    """Resample residuals, compute autocorrelation, and plot percentiles.
    daily: DataFrame
    iters: number of simulations to run
    nlags: maximum lags to compute autocorrelation
    """
    # run simulations
    t = []
    for _ in range(iters):
        filled = FillMissing(daily, span=30)
        resid = thinkstats2.Resample(filled.resid)
        acf = smtsa.acf(resid, nlags=nlags )[1:]
        t.append(np.abs(acf))

    high = thinkstats2.PercentileRows(t, [97.5])[0]
    low = -high
    lags = list(range(1, nlags+1))
    thinkplot.FillBetween(lags, low, high, alpha=0.2, color='gray')

def PlotAutoCorrelation(dailies, nlags=40, add_weekly=False):
    """Plots autocorrelation functions.
    dailies: map from type name to DataFrame of daily prices
    nlags: number of lags to compute
    add_weekly: boolean, whether to add a simulated weekly pattern
    """
    thinkplot.PrePlot(2)
    daily = dailies['organic']
    SimulateAutocorrelation(daily)

    for name, daily in dailies.items():

        if add_weekly:
            daily = AddWeeklySeasonality(daily)

        filled = FillMissing(daily, span=30)

        acf = smtsa.acf(filled.resid, nlags=nlags )
        lags = np.arange(len(acf))
        thinkplot.Plot(lags[1:], acf[1:], label=name)
   
def PlotDailies(dailies):
    """Makes a plot with daily prices for different qualities.
    dailies: map from name to DataFrame
    """
    thinkplot.PrePlot(rows=2)
    for i, (name, daily) in enumerate(dailies.items()):
        thinkplot.SubPlot(i+1)
        title = 'Average Price' if i == 0 else ''
        thinkplot.Config(ylim=[0, 3], title=title)
        thinkplot.Scatter(daily.average_price, s=10, label=name)
        if i == 1: 
            pyplot.xticks(rotation=30)
        else:
            thinkplot.Config(xticks=[])
    #thinkplot.show()
    thinkplot.Save(root='timeseries1',
                   formats=FORMATS)
    
def SimulateIntervals(daily, iters=101, func=RunLinearModel):
    """Run simulations based on different subsets of the data.
    daily: DataFrame of daily prices
    iters: number of simulations
    func: function that fits a model to the data
    returns: list of result objects
    """
    result_seq = []
    starts = np.linspace(0, len(daily), iters).astype(int)

    for start in starts[:-2]:
        subset = daily[start:]
        _, results = func(subset)
        fake = subset.copy()

        for _ in range(iters):
            fake.average_price = (results.fittedvalues + 
                        thinkstats2.Resample(results.resid))
            _, fake_results = func(fake)
            result_seq.append(fake_results)

    return result_seq

def PlotIntervals(daily, years, iters=101, percent=90, func=RunLinearModel):
    """Plots predictions based on different intervals.
    daily: DataFrame of daily prices
    years: sequence of times (in years) to make predictions for
    iters: number of simulations
    percent: what percentile range to show
    func: function that fits a model to the data
    """
    result_seq = SimulateIntervals(daily, iters=iters, func=func)
    p = (100 - percent) / 2
    percents = p, 100-p

    predict_seq = GeneratePredictions(result_seq, years, add_resid=True)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.2, color='gray')
    
def RunModels(dailies):
    """Runs linear regression for each group in dailies.
    dailies: map from group name to DataFrame
    """
    rows = []
    for daily in dailies.values():
        _, results = RunLinearModel(daily)
        intercept, slope = results.params
        p1, p2 = results.pvalues
        r2 = results.rsquared
        s = r'%0.3f (%0.2g) & %0.3f (%0.2g) & %0.3f \\'
        row = s % (intercept, p1, slope, p2, r2)
        rows.append(row)

    # print results in a LaTeX table
    print(r'\begin{tabular}{|c|c|c|}')
    print(r'\hline')
    print(r'intercept & slope & $R^2$ \\ \hline')
    for row in rows:
        print(row)
    print(r'\hline')
    print(r'\end{tabular}')
    
def MakeAcfPlot(dailies):
    """Makes a figure showing autocorrelation functions.
    dailies: map from type  name to DataFrame of daily prices    
    """
    axis = [0, 41, -0.5, 0.5]

    thinkplot.PrePlot(cols=2)
    PlotAutoCorrelation(dailies, add_weekly=False)
    thinkplot.Config(axis=axis, 
                     loc='upper right',
                     ylabel='correlation',
                     xlabel='lag (day)')

    thinkplot.SubPlot(2)
    
    PlotAutoCorrelation(dailies, add_weekly=True)
    
    thinkplot.Save(root='timeseries9',
                   axis=axis,
                   loc='upper right',
                   xlabel='lag (days)',
                   formats=FORMATS)
  
def AddWeeklySeasonality(daily):
    frisat = (daily.index.dayofweek==4) | (daily.index.dayofweek==5)
    fake = daily.copy()
    fake.average_price[frisat] += np.random.uniform(0, 2, frisat.sum())
    return fake
    
def GenerateSimplePrediction(results, years):
    n = len(years)
    inter = np.ones(n)
    d = dict(Intercept=inter, years=years)
    predict_df = pandas.DataFrame(d)
    predict = results.predict(predict_df)
    return predict
    
def SimulateResults(daily, iters=101):
    model, results = RunLinearModel(daily)
    fake = daily.copy()
    
    result_seq = []
    for i in range(iters):
        fake.average_price = results.fittedvalues + thinkstats2.Resample(results.resid)
        _, fake_results = RunLinearModel(fake)
        result_seq.append(fake_results)
 
    return result_seq
    
def GeneratePredictions(result_seq, years, add_resid=False):
    n = len(years)
    d = dict(Intercept=np.ones(n), years=years, years2=years**2)
    predict_df = pandas.DataFrame(d)
    
    predict_seq = []
    for fake_results in result_seq:
        predict = fake_results.predict(predict_df)
        if add_resid:
            predict += thinkstats2.Resample(fake_results.resid, n)
        predict_seq.append(predict)
 
    return predict_seq

def PlotPredictions(daily, years, iters=101, percent=90):
    result_seq = SimulateResults(daily, iters=iters)
    p = (100 - percent) / 2
    percents = p, 100-p
 
    predict_seq = GeneratePredictions(result_seq, years, True)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.3, color='gray')
 
    predict_seq = GeneratePredictions(result_seq, years, False)
    low, high = thinkstats2.PercentileRows(predict_seq, percents)
    thinkplot.FillBetween(years, low, high, alpha=0.5, color='gray')
    
def PlotRollingMean(daily, name):
    """Plots rolling mean and EWMA.
    daily: DataFrame of daily prices
    """
    dates = pandas.date_range(daily.index.min(), daily.index.max())
    reindexed = daily.reindex(dates)

    thinkplot.PrePlot(cols=2)
    thinkplot.Scatter(reindexed.average_price, s=15, alpha=0.1, label=name)
    reindexed=reindexed.dropna()
    #roll_mean = pandas.rolling_mean(reindexed.average_price, 30)
    roll_mean = reindexed.average_price.rolling(30).mean()
    thinkplot.Plot(roll_mean, label='rolling mean')
    pyplot.xticks(rotation=30)
    thinkplot.Config(ylabel='Average Price ($)')

    thinkplot.SubPlot(2)
    thinkplot.Scatter(reindexed.average_price, s=15, alpha=0.1, label=name)
    ewma=reindexed['average_price'].ewm(span=30).mean()
 
    thinkplot.Plot(ewma, label='EWMA')
    pyplot.xticks(rotation=30)
    thinkplot.Save(root='timeseries10',
                   formats=FORMATS)


def PlotFilled(daily, name):
    """Plots the EWMA and filled data.
    daily: DataFrame of daily prices
    """
    filled = FillMissing(daily, span=30)
    thinkplot.Scatter(filled.average_price, s=15, alpha=0.3, label=name)
    thinkplot.Plot(filled.ewma, label='EWMA', alpha=0.4)
    pyplot.xticks(rotation=30)
    thinkplot.Save(root='timeseries8',
                   ylabel='Average Price ($)',
                   formats=FORMATS)

def PlotLinearModel(daily, name):
    """Plots a linear fit to a sequence of prices, and the residuals.
    
    daily: DataFrame of daily prices
    name: string
    """
    model, results = RunLinearModel(daily)
    PlotFittedValues(model, results, label=name)
    thinkplot.Save(root='timeseries2',
                   title='fitted values',
                   xlabel='years',
                   xlim=[-0.1, 3.8],
                   ylabel='Average Price ($)',
                   formats=FORMATS)

    PlotResidualPercentiles(model, results)
    thinkplot.Save(root='timeseries3',
                   title='residuals',
                   xlabel='years',
                   ylabel='Average Price ($)',
                   formats=FORMATS)
    
    #years = np.linspace(0, 5, 101)
    #predict = GenerateSimplePrediction(results, years)
    
def main():
    thinkstats2.RandomSeed(18)
    transactions = ReadData()

    dailies = GroupByTypeAndDay(transactions)
    PlotDailies(dailies)
    RunModels(dailies)
    PrintSerialCorrelations(dailies)
    MakeAcfPlot(dailies)

    name = 'organic'
    daily = dailies[name]

    PlotLinearModel(daily, name)
    PlotRollingMean(daily, name)
    PlotFilled(daily, name)

    years = np.linspace(0, 5, 101)
    thinkplot.Scatter(daily.years, daily.average_price, alpha=0.1, label=name)
    PlotPredictions(daily, years)
    xlim = years[0]-0.1, years[-1]+0.1
    thinkplot.Save(root='timeseries4',
                   title='predictions',
                   xlabel='years',
                   xlim=xlim,
                   ylabel='Average Price ($)',
                   formats=FORMATS)

    name = 'conventional'
    daily = dailies[name]

    thinkplot.Scatter(daily.years, daily.average_price, alpha=0.1, label=name)
    PlotIntervals(daily, years)
    PlotPredictions(daily, years)
    xlim = years[0]-0.1, years[-1]+0.1
    thinkplot.Save(root='timeseries5',
                   title='predictions',
                   xlabel='years',
                   xlim=xlim,
                   ylabel='Average Price ($)',
                   formats=FORMATS)
    
main()
