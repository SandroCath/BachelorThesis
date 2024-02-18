from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import chi2
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error
import math
from statistics import mean
import os
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings('ignore')

folder_name = r'C:\Users\sandr\OneDrive\Bachelorarbeit\Data\Main Data\Clean Data' + '\\'
results_filename = r'C:\Users\sandr\OneDrive\Bachelorarbeit\Code\Output'

def create_joint_df(folder_name=str):
    # Read in the three CSV files
    commodities_df = pd.read_csv(folder_name + "CMO-Historical-Data-Monthly Indexes CHF Real.csv", sep=",")
    commodities_df['Date'] = pd.to_datetime(commodities_df['Date'])

    cpi_df = pd.read_csv(folder_name + "cpi_variables Switzerland.csv", sep=";")
    cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
    interestr_df = pd.read_csv(folder_name + "snb_key_interest_rate_20230421_0900.csv", sep=";")
    interestr_df['Date'] = pd.to_datetime(interestr_df['Date'])
    m2_df = pd.read_csv(folder_name + "snb_monetary_aggregate_20230421_0900.csv", sep=";")
    m2_df['Date'] = pd.to_datetime(m2_df['Date'])
    gdp_df = pd.read_csv(folder_name + "snb_GDP-20230421_0900.csv", sep=";")
    gdp_df['Date'] = pd.to_datetime(gdp_df['Date'])
    fxrates_df = pd.read_csv(folder_name + "snb_Effective_exchange_rate_indices-20230411_0900.csv", sep=";")
    fxrates_df['Date'] = pd.to_datetime(fxrates_df['Date'])

    mortgage_df=pd.read_csv(folder_name + "snb_mortgage_loans-20230421_0900.csv", sep=";")
    mortgage_df['Date'] = pd.to_datetime(mortgage_df['Date'])
    rents_df=pd.read_csv(folder_name + "snb_rents_index-20230421_0900.csv", sep=";")
    rents_df['Date'] = pd.to_datetime(rents_df['Date'])
    bondyield_df=pd.read_csv(folder_name + "snb_10y_bond_yield-20230403_1430.csv", sep=";")
    bondyield_df['Date'] = pd.to_datetime(bondyield_df['Date'])


    # Merge the three dataframes on the dates column
    controlvariables_df = pd.merge(cpi_df, interestr_df, on="Date", how="inner")
    controlvariables_df = pd.merge(controlvariables_df, m2_df, on="Date", how="inner")
    controlvariables_df = pd.merge(controlvariables_df, gdp_df, on="Date", how="inner")
    controlvariables_df = pd.merge(controlvariables_df, fxrates_df, on="Date", how="inner")
    controlvariables_df = pd.merge(controlvariables_df, mortgage_df, on="Date", how="inner")
    controlvariables_df = pd.merge(controlvariables_df, rents_df, on="Date", how="inner")
    controlvariables_df = pd.merge(controlvariables_df, bondyield_df, on="Date", how="inner")

    merged_df = pd.merge(commodities_df, controlvariables_df, on="Date", how="inner")
    merged_df = merged_df.set_index("Date")
    return merged_df


def adf_test(series):
    result = adfuller(series)
    print("ADF Test for", series.name)
    print("Test Statistic:", result[0])
    print("P-value:", result[1])
    print("Critical Values:", result[4])
    print("Is Stationary:", result[0] < result[4]["5%"])
    print()


def coint_test(series1, series2):
    result = coint(series1, series2)
    print("Cointegration Test for", series1.name, "and", series2.name)
    print("Test Statistic:", result[0])
    print("P-value:", result[1])
    print("Critical Values:", result[2])
    print("Is Cointegrated:", result[1] < 0.05)


initial_data = create_joint_df(folder_name)
print(initial_data.keys())



all_cols = ['Energy Index', 'Fertilizer Index',
       'CPI Total', '3M LIBOR', 'Monetary aggregate M2 in mio CHF',
       'Monetary aggregate M3 in mio CHF', 'real GDP Switzerland (in mio CHF)',
       'Effective Exchange Rate Index Real',
       'Credit Volume Mortgage Loans (in mio CHF)',
       'Real Estate Price Index Q1 2000=100',
       '10y Yield on CHF Swiss Confederation bond issues']

exogen_variables = ['Energy Index', 'Fertilizer Index',
        '3M LIBOR', 'Monetary aggregate M2 in mio CHF',
       'Monetary aggregate M3 in mio CHF', 'real GDP Switzerland (in mio CHF)',
       'Effective Exchange Rate Index Real',
       'Credit Volume Mortgage Loans (in mio CHF)',
       'Real Estate Price Index Q1 2000=100',
       '10y Yield on CHF Swiss Confederation bond issues']


endogene_variables = ['CPI Total','Nahrungsmittel und alkoholfreie Getränke','Nahrungsmittel','Brot, Mehl und Getreideprodukte','Fleisch, Fleischwaren','Milch, Käse, Eier','Früchte, Gemüse, Kartoffeln und Pilze']

pd.options.display.max_columns = None
pd.options.display.max_rows=None
# Remove outliers
print(initial_data.shape)



moving_average_lag=0
while moving_average_lag<3:
    output_filename=results_filename+'\\Moving Average Lag '+str(moving_average_lag)
    try:
        os.makedirs(output_filename)
    except:
        pass
    for temp_output in endogene_variables:
        print("\n \n \n \n "+"New variable: "+str(temp_output))
        temp_output_filename=str(temp_output).replace('/',' per ')
        temp_output_filename = str(temp_output_filename).replace('*', '')

        #First, we have to clean and ready the dataset:
        #Choose timeseries to be analyzed in this run
        current_dataset = [temp_output] + exogen_variables

        macro_data = initial_data.filter(current_dataset)


        print(macro_data.head())
        # make out the 80% and 20% barrier to create train- and test-set
        train_data = macro_data[:int(-((len(macro_data) - 1) / 5))]
        test_data = macro_data[int(-((len(macro_data) - 1) / 5)):]



        # Show amount of Null-values in current data set
        print("\nAmount of empty rows to be dropped: "+str(train_data.isnull().sum()))
        train_data = train_data.dropna()
        test_data=test_data.dropna()


        #Test raw data for non-stationarity:
        print("\nCheck ADF on non-adjusted data: \n")
        train_data.apply(adf_test)


        #Convert the data to percentage-changes as this makes it easier for the statistical model to process than absolute values
        # Log the values
        print("\n\n\nCheck ADF on logarithmic scales: \n")
        df1 = train_data.loc[:, ['Energy Index', 'Fertilizer Index',
                                temp_output, 'Monetary aggregate M2 in mio CHF',
                                   'Monetary aggregate M3 in mio CHF', 'real GDP Switzerland (in mio CHF)',
                                   'Effective Exchange Rate Index Real',
                                   'Credit Volume Mortgage Loans (in mio CHF)',
                                   'Real Estate Price Index Q1 2000=100']]
        df2 = train_data.loc[:, ['10y Yield on CHF Swiss Confederation bond issues', '3M LIBOR']]
        df1 = df1.apply(np.log)

        train_data = pd.merge(df1, df2, "inner", on="Date")
        print(initial_data.head())
        train_data = train_data.dropna()
        train_data.apply(adf_test)
        print(test_data)
        df1 = test_data.loc[:, ['Energy Index', 'Fertilizer Index',
                                temp_output, 'Monetary aggregate M2 in mio CHF',
                                   'Monetary aggregate M3 in mio CHF', 'real GDP Switzerland (in mio CHF)',
                                   'Effective Exchange Rate Index Real',
                                   'Credit Volume Mortgage Loans (in mio CHF)',
                                   'Real Estate Price Index Q1 2000=100']]
        df2 = test_data.loc[:, ['10y Yield on CHF Swiss Confederation bond issues', '3M LIBOR']]
        df1 = df1.apply(np.log)

        test_data = pd.merge(df1, df2, "inner", on="Date")
        test_data = test_data.dropna()

        # Diff the values
        print("\n\n\nCheck ADF on first level differences: \n")
        train_data = train_data.diff()
        train_data = train_data.dropna()
        test_data = test_data.diff()
        test_data = test_data.dropna()

        #ADF-Test to check for stationarity
        train_data.apply(adf_test)

        #Also Check for PhillipsPerron
        print("\n\n\nCheck PP on first level differences: \n")
        for col1 in train_data.columns:
            pp=PhillipsPerron(train_data[col1])
            print("P-Value for nonstationary for "+str(col1)+" is rejected at 5%? "+str(pp.pvalue))

        #Now check for Jarque-Bera (Normality-Test)
        for col1 in train_data.columns:
            jb=jarque_bera(train_data[col1])
            print("P-Value for normality for "+str(col1)+" is rejected at 5%? "+str(jb[1]))

        #Now check for Granger Causality
        for col1 in train_data.columns:
            if col1==temp_output:
                pass
            else:
                print("\nGranger causality of "+str(col1)+" and "+str(temp_output)+"\n")
                print(grangercausalitytests(train_data[[temp_output,col1]],maxlag=4))




        #Show summary of current data: Mean, SD, MIN/MAX Values
        print(train_data.describe())


        '''
        #Test for Cointegration
        for col1 in train_data.columns:
            print(col1)
            if col1 != temp_output:
                coint_test(train_data[col1], train_data[temp_output])
        '''

        #train_data = train_data[[temp_output] + exogen_variables]
        train_data.to_csv(output_filename+'current_dataframe.csv', encoding='utf-8')
        print("The specs of the dataset for this set of exogenous variables is: \n")
        print(train_data.shape)
        print(train_data.head())


        # model = VAR(train_data_norm[temp_output],exog=train_data_norm[temp_input])
        print("\n\nThe summary of the selected lags order is: \n")
        model = VAR(train_data)

        # look out the header for endogenous is missing, only period in
        # print(train_data_norm[temp_output])

        sorted_order = model.select_order(maxlags=4)
        number_of_lags = int(sorted_order.aic)

        print(sorted_order.summary())
        print("\n \n")
        # All the parameters should be as small as possible

        # To fit the model we use VARMAX Class, because it makes it very easy. Provide NON-STATIONARY (not .diff()) Data, the order=(4.0) means we're gonna use a simple VAR_Model, 0 stands for we're not using the moving average model. We enforce stationarity, so it will normalize our data automatically.

        #Varmax order(p,q) -> p = number of lags, q = information on Moving Average. If 0, means it's just a VAR Model
        train_data.to_csv(r"C:\Users\sandr\OneDrive\Bachelorarbeit\Code\current_dataframe.csv")
        var_model = VARMAX(train_data, order=(number_of_lags, moving_average_lag))
        print('The summary of the fitted model is: \n')
        fitted_model = var_model.fit(disp=False)

        try:
            os.makedirs(output_filename+'\VARMAX Summaries')
        except:
            pass
        with open(output_filename+'\VARMAX Summaries'+"\\"+str(temp_output_filename)+" Summary.txt",'w') as summaryfile:
            summaryfile.write(fitted_model.summary().as_text())
        print(fitted_model.summary())
        # Now we see the equations (with all coefficients etc.) for our prediction model

        n_forecast = len(test_data) #+1 since we dropped the first row due to diff() when preprocessing
        print("Number of predictions is "+str(n_forecast))
        print("\nThat means period "+str(len(train_data))+" to "+str(len(
            train_data) + n_forecast))
        # Start date of prediction is length of data set +1 since we dropped the first row due to diff() when preprocessing, end date is n_forecast steps into the future
        predict = fitted_model.predict(start=len(train_data)+1, end=len(
            train_data) + n_forecast)
        #predict = fitted_model.get_prediction(start=len(train_data_norm), end=len(train_data_norm) + n_forecast - 1)  # start="1989-07-01",end='1999-01-01')

        # calculate mean of all the predictions
        predictions = predict #.predicted_mean

        # Create the names of the headers of the predicted parameters
        if type(temp_output) == list:
            cols = [str(endogenous_var) + '_predicted' for endogenous_var in temp_output]
        else:
            cols = [temp_output + '_predicted']
        if type(exogen_variables) == list:
            cols = cols + [str(exogenous_var) + '_predicted' for exogenous_var in exogen_variables]
        else:
            cols.append(exogen_variables + '_predicted')




        #print("List of predictions for " + str(temp_output) + ":\n")
        predictions.columns=cols
        predictions.set_index(test_data.index, inplace=True)

        '''       
        print("Now test data")
        print(test_data)
        unscaled_data = pd.merge(test_data, unscaled_pred_df,how='outer',left_index=True, right_index=True)
        unscaled_data.to_csv(output_filename + '\\' + 'Unscaled Predictions ' + str(
            temp_output_filename) + ".csv")


        test_data.to_csv(output_filename + '\\' + 'Test Data ' + str(
            temp_output_filename) + ".csv")
        '''

        # Compare predictions vs. test-data set
        test_vs_pred = pd.merge(test_data, predictions, how="outer",left_index=True, right_index=True)
        #test_vs_pred.plot(figsize=(12, 5))
        test_vs_pred.to_csv(output_filename + '\\' + 'Predictions ' + str(temp_output_filename) + ".csv")

        try:
            os.makedirs(output_filename+'\Prediction Error Log')
        except:
            pass
        with open(output_filename+'\Prediction Error Log\Prediction Error Log '+str(temp_output_filename)+'.txt', 'w') as f:
            rmse_output=math.sqrt(mean_squared_error(predictions[temp_output+'_predicted'],test_data[temp_output]))
            print(var_model.nobs-len(var_model.param_names))
            degrees_of_freedom=(var_model.nobs-len(var_model.param_names))
            critical_value=chi2.ppf(0.95,degrees_of_freedom)
            print('Mean value of '+temp_output+' is : {}. Root Mean Squared Error is :{}'.format(mean(test_data[temp_output]),rmse_output))
            f.write('Mean value of '+temp_output+' is : {}. Root Mean Squared Error is :{}'.format(mean(test_data[temp_output]),rmse_output))
        f.close()

        moving_average_lag+=1
