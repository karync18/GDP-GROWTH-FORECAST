## Introduction and Research Background
In this project, I aim to determine whether firm-level accounting and market data can improve GDP growth prediction. Using WRDS datasets—including financial statements and stock market information—I test if these variables, which reflect real economic activity and investor sentiment, offer predictive signals beyond traditional macroeconomic indicators. I apply models like linear regression, LASSO, Ridge, Elastic Net, XGBoost, and ensemble methods to evaluate their effectiveness in forecasting GDP growth.

## Project Objective 
The objective of this project is to analyze the effectiveness of different forecasting models in predicting GDP growth based on economic variables, market variables, and accounting variables.

## Data Collected
The economic data that I collected is for the period of 1990 to 2020 from various sources, such as:
1. Survey of Professional Forecasters (SPF)
2. Bureau of Economic Analysis (BEA)
3. Federal Reserve Economic Data (FRED)

Stock price data is from:
1. WRDS Compustat (Fundamental Quarterly)
2. CRSP data (CRSP monthly stock market index)

Accounting data from:
1. WRDS Compustat (financial related data)

GDP growth data from:
1. Kaggle (training and testing data)

These data were processed and separated into 3 primary datasets:
1. Economic data
2. Economic data and market data
3. Economic data and accounting data

## Methodology Used
I started by building a base model using linear regression with only economic variables. Then, I expanded the dataset to include market variables (Set 2) and accounting variables (Set 3), comparing model performance across these sets using RMSE. Adding market and accounting data didn’t improve the linear model’s accuracy. To explore further, I applied machine learning algorithms—LASSO, Ridge, Elastic Net, XGBoost, and an ensemble of these methods—on all three data sets to assess whether more advanced models could better capture the predictive signals.

## Project Outcome
My best-performing model was a regularized LASSO using Set 3 data (economic + market variables), achieving an RMSE of 1.08628. To assess the value of accounting information in GDP forecasting, I found that it didn’t significantly enhance model performance. The best accounting-based model—a non-regularized linear regression—had an RMSE of 1.42852, which is notably higher than the base model. Based on these results, I recommend incorporating market data over accounting data, as models using market information consistently performed better. This suggests that market variables are more useful than accounting variables for accurate GDP growth prediction.

## Library
load all necessary libraries first
```R
library(dplyr)
library(tidyverse)
library(ggplot2)
library(lattice)
library(caret)
library(psych)
library(lubridate)
library(tidyr)
library(DescTools)
library(broom)
library(corrplot)
library(corrr)
library(reshape2)
library(ggcorrplot)
library(glmnet)
library(coefplot)
library(xgboost)
```
## Load all the necessary data
```R
### load all necessary data ###

# kaggle training data
setwd("/Users/karynchristabella/Downloads/FFA WIP")
gdp_data <- read.csv("GDP_Forecast_train.csv")
# 104 obs, 2 vars

# qrtly accounting data from WRDS compustat -> NorthAmerica -> fundamentals quarterly
accounting_data <- read.csv("compustat-qrtly-accdata.csv") 
# 1 218 003 obs, 68 vars

# economic data (analysts' forecast of GDP)
economic_data <- read.csv("economic_data.csv", sep = ";", header = TRUE) 
# 124 obs, 32 vars

# market and stock price data from WRDS CRPS and compustat daily stock index
stockprice_data <- read.csv("stockpricedata.csv") 
# 999999 obs, 15 vars
crsp_data <- read.csv("crsp.csv") 
# 2 820 818 obs, 13 vars
```
## Dependent Variable
### GDP Growth
Data source: Kaggle GDP Training Data
Variable: average_GDP_growth as our model's dependent variable

**Insights from the visualization:**

From the graph, we see that GDP growth has been highly volatile throughout the year from 1990 to 2015. It appears that the peak for GDP growth occurred in 1992 while the lowest drop in GDP growth was in 2008, due to the global financial crisis.

There have been several major downturns such as in 2001 and 2008-2009, but the post-crisis recoveries have been evident as seen in the sharp inclines after each period of decline. This recovery may be due to the effectiveness of economic policies, resources allocation, and other external factors that contribute to its economic restoration.

This evidence that external data such as the accounting and market data should be able to help improve forecasting model.

```R
# check structure of gdp data
str(gdp_data)

# make a new column for Year and Quarter from the YQ column
gdp_data$Year <- as.numeric(substr(gdp_data$YQ, 1, 4))
gdp_data$Quarter <- as.numeric(substr(gdp_data$YQ, 6, 7))

# gdp data for merging
final_gdp_data <- gdp_data
# 104 obs, 4 vars

# calculate the average GDP growth per year
gdp_growth_data <- final_gdp_data %>% group_by(Year) %>%
  summarize(average_GDP_growth= mean(NGDP, na.rm = TRUE)) 
# 26 obs 2 vars

# visualise the GDP growth over the years
gdp_plot <-ggplot(gdp_growth_data, aes(x = Year, y = average_GDP_growth)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  theme(axis.text.x = element_text(angle=50, size=8)) +
  labs(title = "GDP Growth from 1990 to 2015", x="Year", y="GDP growth") +
  theme_minimal()

library(plotly)
ggplotly(gdp_plot)
```
<img width="705" height="432" alt="Screenshot 2025-07-14 at 7 11 13 PM" src="https://github.com/user-attachments/assets/234cccdc-d5aa-4df8-867c-d2fc421dc95a" />

## Data Processing and Extracting Independent Variables
### Accounting Data
**Data Source & Period:** WRDS Compustat (North America), Quarterly data from 1990 to 2020

**Variables Used:**
**Financial metrics:** revenue, cost, equity, assets, cash, depreciation, earnings, shares, capital expenditure, sales, etc.

**Constructed Accounting Ratios:**
**Profitability: **Operating Profit Margin, Net Operating Profit After Tax (NOPAT), Return on Assets (ROA), Return on Equity (ROE)
**Leverage: **Debt Ratio

**Data Cleaning Steps:**
1. Removed entries with missing year/quarter data
2. Constructed financial ratios
3. Excluded financial sector firms (based on SIC codes)
4. Dropped variables with >50% missing values; filled others with median values
5. Calculated quarter-on-quarter growth and averaged it
6. Winsorized outliers at 1st and 99th percentiles
7. Created a Year-Quarter (YQ) column for merging
8. Merged accounting data with GDP and economic data

**Insights:**
After cleaning the data, I selected key accounting ratios—operating profit margin (OPM), profit margin (PM), net operating profit after taxes (NOPAT), return on assets (ROA), and debt ratio—to add to my base model. These variables represent core aspects of company performance, including assets, returns, and profitability.

When I compared these ratios to nominal GDP (NGDP) using line graphs, I noticed that profitability metrics showed sharper spikes and dips than GDP, indicating more volatility. Interestingly, the debt ratio closely followed the GDP trend, suggesting a potential link between leverage and economic growth.

Based on these observations, I aim to test whether accounting variables significantly contribute to predicting GDP growth rates.

```R
# check first few rows of acc data
head(accounting_data)

# check structure of acc data
str(accounting_data)

# summary stats of data to check if got NA
summary(accounting_data)

# drop default columns from WRDS that is unnecessary
accounting_data <- accounting_data %>%
  select(gvkey, datadate, datacqtr, sic, everything()) %>%
  select(-fyearq, -fqtr, -indfmt, -consol, -popsrc, -datafmt, -datafqtr, -costat) 
# 1218003 obs, 60 vars

# remove rows missing datacqtr 
accounting_data <- accounting_data %>% filter(datacqtr != "")
# originally, acc data has 1218003 observations
# after dropping, it has 1215819 observations
# therefore, 2184 observations were dropped

# then create year and quarter row from the variable 'datacqtr'
accounting_data <- accounting_data %>%
  mutate(year = as.numeric(substr(datacqtr, 1, 4)),
         quarter = as.numeric(substr(datacqtr, 6, 6)),
         datadate = ymd(datadate))  # converting 'datadate' to Date format
# 1215819 obs, 62 vars

# remove firms with missing quarter data
accounting_data <- accounting_data %>% 
  group_by(gvkey) %>% 
  distinct(year, quarter, .keep_all=TRUE) %>% 
  ungroup() 
# 1215815 observations so 4 observations were dropped

# create financial ratios
accounting_data <- accounting_data %>%
  mutate(roa = niq/atq,
         roe = revtq/teqq,
         pm = niq / revtq,
         opm = oiadpq /revtq,
         nopat = oiadpq*(1-0.35),
         debt_ratio = ltq / atq)
# 1215815 obs, 68 vars

# replace all infinite with NA
accounting_data <- accounting_data %>% mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

# classify the firms into their own industry using sic
# then, remove finance firms 
accounting_data <- accounting_data %>%
  mutate(sector = case_when(
    between(sic, 100, 999) ~ "Agriculture, Forestry, Fishing",
    between(sic, 1000, 1499) ~ "Mining",
    between(sic, 1500, 1799) ~ "Construction",
    between(sic, 2000, 3999) ~ "Manufacturing",
    between(sic, 4000, 4999) ~ "Utilities",
    between(sic, 5000, 5199) ~ "Wholesale Trade",
    between(sic, 5200, 5999) ~ "Resale Trade",
    between(sic, 6000, 6799) ~ "Finance",
    between(sic, 7000, 8999) ~ "Services",
    between(sic, 9100, 9999) ~ "Public Administration",
    TRUE ~ "Others")) %>% relocate(sector, .after = sic)

accounting_data <- accounting_data %>% filter(sector != "Finance") 
# 854155 obs 50 vars

# compute % missing for all variables
missing_values <- colSums(is.na(accounting_data)) / nrow(accounting_data) * 100
missing_values

# identify the variables that have more than 50% missing values
variables_50 <- names(missing_values[missing_values > 50])
variables_50

# remove those variables including roe 
accounting_data <- accounting_data %>%
  select(-mibnq, -ncoq, -pllq, -rllq, -teqq, -tieq, -roe)
# 854155 obs, 62 vars

# TO DEAL WITH MISSING VALUES
# select numeric columns
numeric_columns <- sapply(accounting_data, is.numeric)

# then apply preprocessing to replace missing values using median imputation
preProcValues <- preProcess(accounting_data[, numeric_columns], 
                            method = "medianImpute")

# apply transformations to replace missing values with median
accounting_data_processed <- accounting_data
accounting_data_processed[, numeric_columns] <- predict(preProcValues, accounting_data[, numeric_columns])

# check if there are still missing values after transformation
sum(is.na(accounting_data_processed))

# remove or impute missing values for gvkey, year and quarter just in case
accounting_data_processed <- accounting_data_processed %>%
  filter(!is.na(gvkey) & !is.na(year) & !is.na(quarter))
# 854155 obs 62 vars

# convert gvkey to character
accounting_data_processed <- accounting_data_processed %>%
  mutate(gvkey = as.character(gvkey))

# identify numeric columns (excluding gvkey, year, and quarter)
numeric_columns <- names(select(accounting_data_processed, where(is.numeric)))
# exclude 'year', 'quarter', 'gvkey'
numeric_columns <- setdiff(numeric_columns, c("year", "quarter", "gvkey", "fyearq", "fqtr"))

# apply QoQ growth calculation ONLY to numeric columns
accounting_data_processed <- accounting_data_processed %>%
  group_by(gvkey) %>%
  arrange(year, quarter, .by_group = TRUE) %>%
  mutate(across(all_of(numeric_columns), ~ (. / lag(.)) - 1, .names = "{.col}_qoq")) %>%
  ungroup()

# replace all infinite values with NA
accounting_data_processed <- accounting_data_processed %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

# compute mean qoq growth rate for each year-quarter
accounting_data_mean_qoq <- accounting_data_processed %>%
  group_by(year, quarter) %>%
  mutate(across(c(actq_qoq:debt_ratio_qoq), 
                .fns = list(mean = ~mean(., na.rm = TRUE)), 
                .names = paste0('{col}','_{.fn}'))) %>%
  dplyr::slice(1) %>%  # Keep only the first row for each year-quarter
  ungroup()  # 125 obs 171 vars

# replace all inf with NA
accounting_data_mean_qoq <- accounting_data_mean_qoq %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

summary(accounting_data_mean_qoq$year)

# save all relevant columns into acc dataset for merging later
# remove NAs for 1st quarter lag
accounting_data_mean_qoq <- accounting_data_mean_qoq %>%
  select(c("year", "quarter", 
           actq_qoq_mean:csh12q_qoq_mean, 
           cstkq_qoq_mean:epsf12_qoq_mean,
           ibcomq_qoq_mean:debt_ratio_qoq_mean)) %>%
  filter(year != 1989) # 124 obs 48 vars

# check structure and statistics of all selected variables
str(accounting_data_mean_qoq)
describe(accounting_data_mean_qoq)

# winsorize top and bottom 1% to mitigate influence of outliers
# define the winsor function
winsorize <- function(x, probs = c(0.01, 0.99)) {
  quantiles <- quantile(x, probs = probs, na.rm = TRUE)
  pmin(pmax(x, quantiles[1]), quantiles[2])
}

# apply the winsorization function to columns 3 through 48
accounting_winsor <- accounting_data_mean_qoq %>%
  mutate(across(3:48, winsorize)) # 124 obs 48 vars

# check the summary of the resulting data
describe(accounting_winsor)

final_accounting_data <- accounting_winsor 
# 124 obs, 48 vars

# MERGE GDP AND ACCOUNTING DATA FIRST
# create YQ variable by combining 'year' and 'quarter'
final_accounting_data <- final_accounting_data %>%
  mutate(YQ = paste(year, "Q", quarter, sep = ""))

final_data_merged <- left_join(final_accounting_data, final_gdp_data, by = 'YQ')
# 124 obs, 52 vars
```
<img width="481" height="431" alt="newplot 1" src="https://github.com/user-attachments/assets/5e62e12d-47c5-4614-aa64-8a284e3b9553" />


### Economic Data
**Timeframe:** 1990–2020 (Quarterly)
**Sources & Variables:**
**1. Surveys of Professional Forecasters (SPF):** Mean and median forecasts of nominal (NGDP) and real (RGDP) GDP growth for the current and upcoming quarters (up to four quarters ahead).
**2. Bureau of Economic Analysis (BEA):** First, second, and third (final) estimates of GDP growth.
**3. Federal Reserve Economic Data (FRED):** CPI, unemployment rate, PCE, government expenditures, net exports, industrial production, and personal income.

**Data Cleaning Process:**
1. Merged SPF, BEA, and FRED datasets into a single CSV (economic_data.csv).
2. Data already in quarterly growth format, ready for integration with accounting and GDP data.
3. Final merged dataset prepared for combining with stock price and CRSP data.

**Graph 1:**
Comparison of actual GDP growth with SPF’s nominal GDP forecasts (mean and median) from 1990–2015 shows strong alignment, indicating reliable predictive accuracy.
```R
# check data type
str(economic_data)

# clean supposed-to-be numeric columns by substituting commas with dots
# then convert it to numeric
economic_data <- economic_data %>%
  mutate(across(where(is.character), ~ as.numeric(gsub(",", ".", .)))) 
# 124 obs 32 vars

library(dplyr)
# create econ growth dataset to visualise diff data against gdp growth
econ_growth_selected <- economic_data %>%
  select(-QUARTER) %>%
  group_by(YEAR) %>%
  mutate(across(everything(), ~ mean(. , na.rm = TRUE))) %>%
  ungroup() # 124 obs 31 vars

# merge economic_data_selected and gdp_growth_data to visualise trend
economic_growth_trend <- left_join(gdp_growth_data, econ_growth_selected, by = c("Year" = "YEAR"))
# 124 obs , 32 vars 


# NOMINAL GDP VS avg GDP growth
# select relevant columns (incl average_GDP_growth) and reshape to long format
econ_ngrowth_long <- economic_growth_trend %>%
  select(Year, average_GDP_growth, mean_NGDP2, mean_NGDP3, mean_NGDP4, mean_NGDP5, mean_NGDP6) %>%
  gather(key = "Variable", value = "Value", -Year) 

# plot line graph to see growth trend of surveys & estimates with different horizons
econ_ngrowth_plot <- ggplot(econ_ngrowth_long, aes(x = Year, y = Value, color = Variable)) +
  geom_line() + 
  geom_line(data = subset(econ_ngrowth_long, Variable == "average_GDP_growth"), 
            aes(x = Year, y = Value), color = "red", size = 1) +  # Highlight average_GDP_growth 
  labs(title = "Trends in Nominal Economic Growth and Key Variable", 
       x = "Year", 
       y = "Value", 
       color = "Variables") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggplotly(econ_ngrowth_plot)
```
<img width="481" height="431" alt="newplot" src="https://github.com/user-attachments/assets/4683974f-f9e9-41e8-b86c-bf8457fe6299" />

**Graph 2:**
SPF’s real GDP forecasts also closely follow actual GDP trends, reinforcing the usefulness of these forecasts.
```R
# REAL GDP VS avg GDP growth
# select relevant columns (incl average_GDP_growth) and reshape to long format
econ_rgrowth_long <- economic_growth_trend %>%
  select(Year, average_GDP_growth, mean_RGDP2, mean_RGDP3, mean_RGDP4, mean_RGDP5, mean_RGDP6) %>%
  gather(key = "Variable", value = "Value", -Year) 

# plot line graph to see growth trend of surveys & estimates with different horizons
econ_rgrowth_plot <- ggplot(econ_rgrowth_long, aes(x = Year, y = Value, color = Variable)) +
  geom_line() + 
  geom_line(data = subset(econ_rgrowth_long, Variable == "average_GDP_growth"), 
            aes(x = Year, y = Value), color = "red", size = 1) +  # Highlight average_GDP_growth 
  labs(title = "Trends in Real Economic Growth and Key Variable", 
       x = "Year", 
       y = "Value", 
       color = "Variables") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggplotly(econ_rgrowth_plot)
```
<img width="481" height="431" alt="newplot 4" src="https://github.com/user-attachments/assets/dbfb36f8-d373-4d07-a9e1-ff96e40b0ab6" />


**Graph 3:**
BEA’s GDP estimates (first, second, and final) show increasing accuracy over time, with final estimates closely matching actual GDP growth.
```R
# REAL GDP VS avg GDP growth
# select relevant columns (incl average_GDP_growth) and reshape to long format
econ_rgrowth_long <- economic_growth_trend %>%
  select(Year, average_GDP_growth, mean_RGDP2, mean_RGDP3, mean_RGDP4, mean_RGDP5, mean_RGDP6) %>%
  gather(key = "Variable", value = "Value", -Year) 

# plot line graph to see growth trend of surveys & estimates with different horizons
econ_rgrowth_plot <- ggplot(econ_rgrowth_long, aes(x = Year, y = Value, color = Variable)) +
  geom_line() + 
  geom_line(data = subset(econ_rgrowth_long, Variable == "average_GDP_growth"), 
            aes(x = Year, y = Value), color = "red", size = 1) +  # Highlight average_GDP_growth 
  labs(title = "Trends in Real Economic Growth and Key Variable", 
       x = "Year", 
       y = "Value", 
       color = "Variables") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggplotly(econ_rgrowth_plot)
```
<img width="481" height="431" alt="newplot 2" src="https://github.com/user-attachments/assets/fc9b83de-3ebc-4349-9af2-e458833e8415" />


**Graph 4:**
Macroeconomic variables—especially government expenditure, industrial production, and personal income—exhibit trends similar to GDP growth. Deviations occur during major economic events (e.g., early 2000s, 2008 crisis), but overall, these variables correlate with GDP as they reflect key components of economic activity.
```R
# MACROECONIMC VARIABLES VS GDP GROWTH
# select columns related to macroeconomic variables and average_GDP_growth, and reshape to long format
econ_growth_long_macro <- economic_growth_trend %>%
  select(Year, average_GDP_growth, cpi, unemployment_rate, PCE, government_expenditure, net_export, Industrial_Production, Personal_income) %>%
  gather(key = "Variable", value = "Value", -Year)

# plot line graph for growth trend of macro variables
econ_macro_growth <-ggplot(econ_growth_long_macro, aes(x = Year, y = Value, color = Variable)) +
  geom_line() + 
  geom_line(data = subset(econ_growth_long_macro, Variable == "average_GDP_growth"), 
            aes(x = Year, y = Value), color = "red", size = 1) +
  labs(title = "Macroeconomic Variables and Average GDP Growth", 
       x = "Year", 
       y = "Value", 
       color = "Variables") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggplotly(econ_macro_growth)
```
<img width="481" height="431" alt="newplot 5" src="https://github.com/user-attachments/assets/9f338c64-efd4-4ce0-be83-fb640154ae42" />

## Update the merged data
```R
# final_economic_data
final_economic_data <- economic_data
# 124 obs, 32 vars

# prepare to merge economic data with the final data
# create YQ for economic_data (combining YEAR and Quarter)
final_economic_data <- final_economic_data %>%
  mutate(YQ = paste(YEAR, QUARTER, sep = "Q")) 

# merge final economic data with the final data by YQ
final_data_merged <- left_join(final_data_merged, final_economic_data, by = 'YQ')
# 124 obs, 84 vars
```

## Stock Price Data
**Data source:** WRDS Compustat - Fundemental Quarterly
**Data Period:** 1990 – 2020 (Quarterly)
**Variable:** Closing Stock Price Closing (prccq), High Stock Price (prchq), Low Stock Price (prclq)

**Cleaning Process:**
•	Removing variables with missing data year and quarter, as the period is crucial for us to arrange the data with the other data.
•	Removing duplicate data
•	Dealing with missing values by replacing the missing values with its mean (given how stock price fluctuates based on market conditions, replacing with mean makes more sense as it captures the variability in the data) - this is done by the preprocess function 
•	Deal with outliers using winsorization by capping extreme values at the 1st and 99th percentiles. That means any value below the 1st percentile is replaced with the 1st percentile value, and any value above the 99th percentile is replaced with the 99th percentile value.
•	Constructing the growth variable by calculating the quarter-on-quarter (QoQ) growth for the 3 variablest, then we compute the mean growth. At this point we have our data as their quarter growth from 1990 – 2020 (124 obs 48 vars)

**Data Insights:**
The line graph shows us how high stock prices (prchq), low stock prices (prclq), closing stock prices (prccq) followed closely to NGDP. Consequently, we concluded that stockprice also serve as reliable predictors for short-term GDP growth rates.

```R
### STOCKPRICE DATA ####
# first see structure of data
str(stockprice_data)
# 999999 obs, 15 vars

# summary of data
summary(stockprice_data)

# remove any missing datacqtr obs
stockprice_data <- stockprice_data %>% filter(datacqtr != "") 
# 998286 obs, 15 vars

# change datadate format from chr to Date
stockprice_data$datadate <- as.Date(stockprice_data$datadate)

# extract Year and Quarter
stockprice_data$year <- year(stockprice_data$datadate)
stockprice_data$quarter <- quarter(stockprice_data$datadate)

# select relevant vars
stockprice_data <- stockprice_data %>%
  mutate(gvkey = gvkey) %>%
  select(gvkey, datadate, year, quarter, prccq, prchq, prclq)
# 998286 obs, 7 vars

# remove duplicates based on the combination of year and qtr for each gvkey
stockprice_data <- stockprice_data %>% 
  group_by(gvkey) %>% 
  distinct(year, quarter, .keep_all = TRUE) %>% 
  ungroup() # 998083 obs, 7 vars

# check for NAs
summary(stockprice_data)

# replace NAs with mean values based on gvkey
stockprice_data <- stockprice_data %>% 
  group_by(gvkey) %>%
  mutate(prccq = ifelse(is.na(prccq), mean(prccq, na.rm = TRUE), prccq)) %>%
  mutate(prchq = ifelse(is.na(prchq), mean(prchq, na.rm = TRUE), prchq)) %>%
  mutate(prclq = ifelse(is.na(prclq), mean(prclq, na.rm = TRUE), prclq)) %>%
  ungroup() 

# check for NAs again
summary(stockprice_data)

# remove remaining NAs
stockprice_data <- filter(stockprice_data, !is.na(prccq) & !is.na(prchq) & !is.na(prclq))
# 944026 obs 7 vars

is.na(stockprice_data)

# winsorize top and bottom 1% to mitigate infleunce of outliers
# define winsor function
winsorization <- function(x) {
  winsor(x, trim = 0.01)
}

stockprice_data <- stockprice_data %>% mutate(prccq = winsor(prccq,trim=0.01),
                                              prchq = winsor(prchq,trim=0.01),
                                              prclq = winsor(prclq,trim=0.01))

# check summary of resulting data
describe(stockprice_data)

# create QoQ growth variables for prccq, prchq, prclq
stockprice_data_growth <- stockprice_data %>%
  # ensure data is ordered by gvkey, year, and quarter
  arrange(year, quarter) %>%
  # create new columns for the QoQ growth
  group_by(gvkey) %>%
  mutate(prccq_qoq_growth = (prccq - lag(prccq)) / lag(prccq),   
         prchq_qoq_growth = (prchq - lag(prchq)) / lag(prchq),   
         prclq_qoq_growth = (prclq - lag(prclq)) / lag(prclq)) %>%
  ungroup()   # 944026 obs 10 vars

# replace all infinite values with NA
stockprice_data_growth <- stockprice_data_growth %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

# compute mean of QoQ variables for each year and quarter
# replace NAs with mean values based on gvkey
stockprice_mean_growth <- stockprice_data_growth %>%
  group_by(year, quarter) %>%
  mutate(across(c(prccq_qoq_growth, prchq_qoq_growth, prclq_qoq_growth), 
                .fns = list(mean = ~mean(., na.rm = TRUE)), 
                .names = "{col}_{.fn}")) %>%
  dplyr::slice(1) %>%
  ungroup() # 124 obs 13 vars

# Replace all infinite values with NA
stockprice_mean_growth <- stockprice_mean_growth %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

summary(stockprice_mean_growth$year)
final_stockprice_data <- stockprice_mean_growth %>%
  select("year", "quarter", prccq_qoq_growth_mean, prchq_qoq_growth_mean, prclq_qoq_growth_mean)
# 124 obs, 5 vars
```
## CRSP Data
**Data source:** CRSP Monthly Stock Market Index
**Data Period:** 1990 – 2020 (Monthly)
**Variable:** Value weighted return including dividend (vwretd), Value weighted return excluding dividend (vwretx), Return on the S&P 500 Index (sprtrn)

**Variable Constructed :**
- market capitalisation (market_cap):  share outstanding (shr0ut) x price (MthPrc) (market_cap)
- cumulative return on the sprtrn over 3-months (sret_3_months)
- cumulative return on the sprtrn over 6-months (sret_6_months)
- cumulative return on the sprtrn over 12-months (sret_12_months)
- cumulative return on the sprtrn over 24-months (sret_24_months)
- cumulative return on the vwretd over 3-months (ret_3_months)
- cumulative return on the vwretd over 6-months (ret_6_months)
- cumulative return on the vwretd over 12-months (ret_12_months)
- cumulative return on the vwretd over 24-months (ret_24_months)

**Cleaning Process:**
- Adjust date format and get quarterly data
- Removing duplicate data
- Dealing with missing values by replacing the missing values with its mean (given how stock price fluctuates based on market conditions, replacing with mean makes more sense as it captures the variability in the data) - this is done by the preprocess function
- Visualize box plot too detect outliers in Value weighted return including dividend (vwretd), Value weighted return excluding dividend (vwretx), Return on the S&P 500 Index (sprtrn).
- Deal with outliers using winsorization by capping extreme values at the 1st and 99th percentiles. That means any value below the 1st percentile is replaced with the 1st percentile value, and any value above the 99th percentile is replaced with the 99th percentile value.
- Create market capitalization variable by multiplying share outstanding (shr0ut) and price (MthPrc) (market_cap). Construct the growth variable by calculating the quarter-on-quarter (QoQ) growth for market_cap, then we compute the mean growth. At this point we have our data as their quarter growth from 1990 – 2020 (124 obs ) final market capitalisation data is ready for merging
- Create cumulative return on the sprtrn and vwretd (We use vwretd and not vwretx because it reflects the total return to investors, including both capital gains and dividend income. This gives a more complete and realistic measure of investment performance)
- Construct the growth variable by calculating the quarter-on-quarter (QoQ) growth for cumulative return, then we compute the mean growth. At this point we have our data as their quarter growth from 1990 – 2020 (124 obs ) final cumulative return data is ready for merging
- Merge stockprice data and crsp data, then merge it with the merged data brought forward.

**Data Insights:**
The graph provides evidence on how stock market returns fluctuate more than the average GDP growth; this is because Stock market returns are more volatile than macroeconomic growth. Despite that, it seems to move in the same direction, indicating that market return movement can potentially forecast GDP growth.

```R
### CRSP DATASET ###
# first, see structure of data
str(crsp_data)
# 2820818 obs, 13 vars

# summary stats 
summary(crsp_data)

# convert datadate from chr to Date format
crsp_data$MthCalDt <- as.Date(crsp_data$MthCalDt)

# create new columns for year, quarter, and month
crsp_data <- crsp_data %>%
  mutate(year = year(MthCalDt),
         quarter = quarter(MthCalDt),
         month = month(MthCalDt),
         year_quarters = paste0(year, "Q", quarter),
         qtr = quarter) # 2820818 obs 18 vars

# VISUSALISE CRSP VARIABLE TREND AGAINST GDP GROWTH
# group the data by Year and calculate the mean for stock returns
crsp_data_yearly <- crsp_data %>%
  group_by(year) %>%
  summarize(avg_vwretx = mean(vwretx, na.rm = TRUE) * 100,  
            avg_ewretx = mean(ewretx, na.rm = TRUE) * 100,  
            avg_sprtrn = mean(sprtrn, na.rm = TRUE) * 100) 
# 31 obs 4 vars

# left join with gdp_growth_data to combine the data
crsp_gdp_growth_data <- left_join(crsp_data_yearly, gdp_growth_data, by = c("year" = "Year"))

# plot for growth trend of crsp market variables
crsp_gdp_plot <- ggplot(crsp_gdp_growth_data, aes(x = year)) +
  geom_line(aes(y = avg_vwretx, color = "vwretx"), size = 1) +
  geom_line(aes(y = avg_ewretx, color = "ewretx"), size = 1) +
  geom_line(aes(y = avg_sprtrn, color = "sprtrn"), size = 1) +
  geom_line(aes(y = average_GDP_growth, color = "average GDP growth"), size = 1) +
  labs(title = "Growth Trend of CRSP Market Variables",
       x = "Year", y = "Value") +
  theme_minimal() +
  theme(legend.title = element_blank()) +
  scale_color_manual(values = c("blue", "green", "red", "black"))

ggplotly(crsp_gdp_plot)
```
<img width="481" height="431" alt="newplot 6" src="https://github.com/user-attachments/assets/742c0988-e5c1-41c1-9a60-90a877d6ff8f" />

### Cleaning CRSP Data
```R
# extract quarterly data 
crsp_data_quarterly <- crsp_data %>%
  group_by(PERMNO, year) %>%
  filter(month %in% c(3, 6, 9, 12)) %>%
    ungroup() # 948227 obs 18 vars

# check the result
head(crsp_data_quarterly)

# check for missing data
summary(crsp_data_quarterly)

# replace NA for MthPRC and SHROUT with mean, grouped by PERMNO
crsp_data_quarterly <- crsp_data_quarterly %>%
  group_by(PERMNO) %>%
  mutate(MthPrc = ifelse(is.na(MthPrc), mean(MthPrc, na.rm = TRUE), MthPrc),
         ShrOut = ifelse(is.na(ShrOut), mean(ShrOut, na.rm = TRUE), ShrOut)) %>%
  ungroup() # 948227 obs 18 vars

# check for missing data
summary(crsp_data_quarterly)

# remove remaining NAs
crsp_data_quarterly <- crsp_data_quarterly %>% 
  filter(!is.na(MthPrc) & !is.na(ShrOut)) # 947855 obs

# check outliers for vwretd, vwretx, sprtrn
# gather them to do boxplot to spot outliers 
variables_to_plot <- crsp_data_quarterly %>% 
  select(vwretd, vwretx, sprtrn) %>%
  gather(key = "variable", value = "value") # 2843565 obs, 2 vars

# create boxplot with adjusted y-axis for better visualization
ggplot(variables_to_plot, aes(x = variable, y = value)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = "Boxplots for Outliers in vwretd, vwretx, sprtrn", x = "Variable", y = "Value") +
  scale_y_continuous(breaks = seq(min(variables_to_plot$value, na.rm = TRUE), 
                                  max(variables_to_plot$value, na.rm = TRUE), 
                                  by = 0.5))

# winsorize top and bottom 1% to eliminate outliers
# apply winsorization for vwretd
crsp_data_quarterly$vwretd <- pmin(pmax(crsp_data_quarterly$vwretd,
                                        quantile(crsp_data_quarterly$vwretd, 0.01, na.rm = TRUE)),
                                   quantile(crsp_data_quarterly$vwretd, 0.99, na.rm = TRUE))

# apply winsorization for vwretx
crsp_data_quarterly$vwretx <- pmin(pmax(crsp_data_quarterly$vwretx,
                                    quantile(crsp_data_quarterly$vwretx, 0.01, na.rm = TRUE)),
                               quantile(crsp_data_quarterly$vwretx, 0.99, na.rm = TRUE))

# apply winsorization for sprtrn
crsp_data_quarterly$sprtrn <- pmin(pmax(crsp_data_quarterly$sprtrn,
                                    quantile(crsp_data_quarterly$sprtrn, 0.01, na.rm = TRUE)),
                               quantile(crsp_data_quarterly$sprtrn, 0.99, na.rm = TRUE))

# for crsp value weighted market return which will be computed later
crsp_return_data <- crsp_data_quarterly # 947855 obs , 18 vars

# create market capitalisation variable 
crsp_data_quarterly <- crsp_data_quarterly %>%
  mutate(market_cap = abs(MthPrc) * ShrOut) # 947855 obs , 19 vars

# winsorise market capitalisation to remove outliers
crsp_data_quarterly <- crsp_data_quarterly %>%
  mutate(market_cap = winsor(market_cap,trim=0.01)) # 947855 obs, 19 vars
```
<img width="962" height="862" alt="Rplot" src="https://github.com/user-attachments/assets/25114a94-3ec0-4442-bfa9-b82244b86723" />

### Create CRSP Growth Variable
```R
# apply QoQ growth calculation 
crsp_data_growth <- crsp_data_quarterly %>%
  group_by(PERMNO) %>%
  arrange(year, quarter) %>%
  mutate(marketcap_qoq = market_cap/ lag(market_cap) - 1) %>%
  ungroup() # 20 vars

# replace infinite values with NA
crsp_data_growth <- crsp_data_growth %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

# compute mean qoq growth rate for each year-quarter
crsp_data_mean_growth <- crsp_data_growth %>%
  group_by(year, quarter) %>%
  mutate(marketcap_qoq_mean = mean(marketcap_qoq, na.rm = TRUE)) %>%
  dplyr::slice(1) %>% 
  ungroup() # 124 obs, 21 vars

# replace infinite values with NA
crsp_data_mean_growth <- crsp_data_mean_growth %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

# final market capitalisation dataset for merging
final_marketcap_data <- crsp_data_mean_growth %>%
  select("year", "quarter", marketcap_qoq_mean) # 124 obs, 3 vars

# construct stock market returns
# calculated relative to the 3rd month of each quarter

crsp_return_data <- crsp_return_data %>%
  group_by(PERMNO) %>%
  arrange(year, quarter) %>%
  mutate(ret = vwretd,
         ret_3_months = vwretd + lag(vwretd, 1),
         ret_6_months = vwretd + lag(vwretd, 1) + lag(vwretd, 2),
         ret_12_months = vwretd + lag(vwretd, 1) + lag(vwretd, 2) + lag(vwretd, 3) + lag(vwretd, 4),
         ret_24_months = vwretd + lag(vwretd, 1) + lag(vwretd, 2) + lag(vwretd, 3) + lag(vwretd, 4) +
           lag(vwretd, 5) + lag(vwretd, 6) + lag(vwretd, 7) + lag(vwretd, 8) + lag(vwretd, 9) +
           lag(vwretd, 10) + lag(vwretd, 11)) # 947855 obs, 23 vars

crsp_return_data <- crsp_return_data %>%
  group_by(PERMNO) %>%
  arrange(year, quarter) %>%
  mutate(sret = sprtrn,
         sret_3_months = sprtrn + lag(sprtrn, 1),
         sret_6_months = sprtrn + lag(sprtrn, 1) + lag(sprtrn, 2),
         sret_12_months = sprtrn + lag(sprtrn, 1) + lag(sprtrn, 2) + lag(sprtrn, 3) + lag(sprtrn, 4),
         sret_24_months = sprtrn + lag(sprtrn, 1) + lag(sprtrn, 2) + lag(sprtrn, 3) + lag(sprtrn, 4) +
           lag(sprtrn, 5) + lag(sprtrn, 6) + lag(sprtrn, 7) + lag(sprtrn, 8)) # 28 vars

# calculate mean for each quarter
crsp_return_data <- crsp_return_data %>%
  group_by(year, quarter) %>%
  mutate(across(ret:sret_24_months, ~ mean(., na.rm = TRUE), .names = "{col}_mean")) %>%
  dplyr::slice(1) %>% 
  ungroup()  # 124 obs, 38 vars

# replace all infinite values with NA
crsp_return_data <- crsp_return_data %>% 
  mutate_if(is.numeric, list(~replace(., !is.finite(.), NA)))

final_market_return <- crsp_return_data %>%
  select("year", "quarter", ret_mean:sret_24_months_mean) # 124 obs, 12 vars
```
### Merge CRSP Data and Stock Price Data
```R
# merging all market dataset
final_merged_market_data <- left_join(final_stockprice_data, final_marketcap_data, by=c("year", "quarter"))
final_merged_market_data <- left_join(final_merged_market_data, final_market_return, by=c("year", "quarter"))
# 124 obs, 16 vars

# prepare to merge with final data
# create the YQ variable by combining 'year' and 'quarter'
final_merged_market_data <- final_merged_market_data %>%
  mutate(YQ = paste(year, "Q", quarter, sep = "")) # 124 obs, 17 vars

final_data_merged <- left_join(final_data_merged, final_merged_market_data, by = 'YQ')
# 124 obs 100 vars
```

### Visualizing Market Data
When I analyzed the line graph, I observed that high (prchq), low (prcla), and closing (prcca) stock prices, along with market capitalization, moved in sync. Additionally, nominal GDP (NGDP) and average market return (ret mean) showed similar patterns. While NGDP followed a relatively stable trend, its movement aligned with major economic shifts reflected in stock prices and market cap changes.

These fluctuations in stock-related variables appear to be influenced by macroeconomic factors, including NGDP growth. Based on this, I concluded that stock prices can serve as reliable predictors for GDP growth rates.
```R
# MARKET RATIOS
mkt_plot <- training_data %>%
  mutate(across(c(prccq_qoq_growth_mean, prchq_qoq_growth_mean, 
                  prclq_qoq_growth_mean, marketcap_qoq_mean, ret_mean), 
                list(percent = ~ . * 100))) 

# compute the mean of market variables growth rate per year 
mkt_plot <- mkt_plot %>% 
  group_by(YEAR) %>% 
  mutate(prccq_qoq_growth_mean_percent = mean(prccq_qoq_growth_mean_percent)) %>% 
  mutate(prchq_qoq_growth_mean_percent = mean(prchq_qoq_growth_mean_percent)) %>%
  mutate(prclq_qoq_growth_mean_percent = mean(prclq_qoq_growth_mean_percent)) %>%
  mutate(marketcap_qoq_mean_percent = mean(marketcap_qoq_mean_percent)) %>%
  mutate(ret_mean_percent = mean(ret_mean_percent)) %>%
  dplyr::slice(1) %>% 
  ungroup() # 26 obs, 96 vars


# plot of qoq growth of market variables and GDP growth
mkt_growth_plot <-ggplot(mkt_plot, aes(x = YEAR)) + 
  geom_line(aes(y = prccq_qoq_growth_mean_percent, color = "Quarterly Mean Closing Stock Price"), size = 1) + 
  geom_line(aes(y = prchq_qoq_growth_mean_percent, color = "Quarterly Mean High Stock Price"), size = 1) + 
  geom_line(aes(y = prclq_qoq_growth_mean_percent, color = "Quarterly Mean Low Stock Price"), size = 1) + 
  geom_line(aes(y = marketcap_qoq_mean_percent, color = "Market Cap QoQ Mean"), size = 1) + 
  geom_line(aes(y = ret_mean_percent, color = "Return Mean")) +
  geom_line(aes(y = NGDP, color = "NGDP")) +
  labs(title = "Growth Trend of Multiple Variables",
       x = "Year",
       y = "Value",
       color = "Variable") +
  scale_color_manual(values = c("Quarterly Mean Closing Stock Price" = "green",
                                "Quarterly Mean High Stock Price" = "gray",
                                "Quarterly Mean Low Stock Price" = "red",
                                "Return Mean" = "orange",
                                "Market Cap QoQ Mean" = "blue",
                                "NGDP" = "purple")) +
  theme_minimal() +  
  theme(legend.position = "bottom")

ggplotly(mkt_growth_plot)
```
<img width="925" height="801" alt="newplot 7" src="https://github.com/user-attachments/assets/30d343a7-d6ab-486d-b223-4671911a9a5d" />

## Multicollinearity Prevention
Given the number of variables in my final dataset, it was essential to address multicollinearity, which occurs when independent variables are highly correlated and carry overlapping information. This can distort model estimates and lead to overfitting.

To mitigate this, I constructed a correlation matrix and visualized it using a correlation plot. This helped me assess the relationships between variables and identify those with high correlation values. I then excluded highly correlated variables and retained those that were more independent.

Based on the correlation plot, the following variables showed consistently low correlations and were selected for inclusion in both the base model and the base + accounting model, as they contribute unique information:

**SPF Forecasts:**
mean_RGDP2, mean_RGDP6, mean_NGDP2, mean_NGDP6
median_RGDP2, median_RGDP6, median_NGDP2, median_NGDP6

**BEA Estimate:**
final_est

**Macroeconomic Indicators:**
cpi, unemployment_rate, PCE, government_expenditure, net_export, Industrial_Production, Personal_income

**Accounting Ratios:**
pm_qoq_mean, nopat_qoq_mean, roa_qoq_mean, debt_ratio_qoq_mean

These variables were chosen to ensure the model remains robust and avoids overfitting by minimizing redundancy.

```R
### CORRELATION ###
colnames(training_data)
```

## Correlation Heat Map Plot
```R
# ACCOUNTING DATA
# calculate the correlation matrix directly with the selected variables
cor(training_data[, c("nopat_qoq_mean", "opm_qoq_mean", "pm_qoq_mean",
                      "debt_ratio_qoq_mean","roa_qoq_mean")],
    use = "complete.obs") # to exclude row that is missing

corrplot(cor(training_data[, c("nopat_qoq_mean", "opm_qoq_mean", "pm_qoq_mean",
                               "debt_ratio_qoq_mean","roa_qoq_mean")],
             use = "complete.obs"))
```
<img width="962" height="862" alt="Rplot01" src="https://github.com/user-attachments/assets/536a2cb4-1fa7-486c-97b3-1dae9e7404ff" />

```R
library(ggcorrplot)
# plot the correlation matrix with number
ggcorrplot(cor(training_data[, c("nopat_qoq_mean", "opm_qoq_mean", "pm_qoq_mean",
                                 "debt_ratio_qoq_mean","roa_qoq_mean")],
               use = "complete.obs"), 
           type = "lower", # Show only the lower triangle
           lab = TRUE,  # Add correlation coefficients as labels
           lab_size = 3, # Control size of labels
           colors = c("blue", "white", "red"), # Color palette for correlations
           title = "Correlation Matrix of Accounting Ratios")
```

<img width="962" height="862" alt="Rplot02" src="https://github.com/user-attachments/assets/3e445e57-af2d-4b68-818c-522895a77c35" />

```R
# plot for multicollinearity for acc variables
training_data %>% select(c(
  "mean_RGDP2", "mean_RGDP3", "mean_RGDP4", "mean_RGDP5", "mean_RGDP6", 
  "mean_NGDP2", "mean_NGDP3", "mean_NGDP4", "mean_NGDP5", "mean_NGDP6", 
  "median_RGDP2", "median_RGDP3", "median_RGDP4", "median_RGDP5", "median_RGDP6", 
  "median_NGDP2", "median_NGDP3", "median_NGDP4", "median_NGDP5", "median_NGDP6",
  "final_est", "cpi", "unemployment_rate", "PCE", "government_expenditure", 
  "net_export", "Industrial_Production", "Personal_income","nopat_qoq_mean", 
  "opm_qoq_mean", "pm_qoq_mean", "debt_ratio_qoq_mean", "roa_qoq_mean")) %>%
  correlate() %>% shave() %>% rplot(print_cor = TRUE) + theme(axis.text.x = element_text(angle = 90, hjust = 1))
```
<img width="962" height="862" alt="Rplot03" src="https://github.com/user-attachments/assets/32b02213-b4a8-4c4d-aae1-3936c1b6e5a8" />

## Market Data Correlation Plot
```R
# MARKET DATA
# compute the correlation matrix directly for the selected variables
cor(training_data[, c("prccq_qoq_growth_mean", 
                      "prchq_qoq_growth_mean", 
                      "prclq_qoq_growth_mean", 
                      "marketcap_qoq_mean", 
                      "ret_mean")],
    use = "complete.obs")

corrplot(cor(training_data[, c("prccq_qoq_growth_mean", 
                               "prchq_qoq_growth_mean", 
                               "prclq_qoq_growth_mean", 
                               "marketcap_qoq_mean", 
                               "ret_mean")],
             use = "complete.obs"))
```
<img width="962" height="862" alt="Rplot04" src="https://github.com/user-attachments/assets/b8b291d8-4e27-4a2f-9788-84b1a62ea5c9" />

```R

# Plot the correlation matrix
ggcorrplot(cor(training_data[, c("prccq_qoq_growth_mean", 
                                 "prchq_qoq_growth_mean", 
                                 "prclq_qoq_growth_mean", 
                                 "marketcap_qoq_mean", 
                                 "ret_mean")],
               use = "complete.obs"), 
           type = "lower",  # Show only the lower triangle for clarity
           lab = TRUE,  # Add correlation coefficients as labels
           lab_size = 3,  # Control label size
           colors = c("blue", "white", "red"),  # Color palette for correlations
           title = "Correlation Matrix of Selected Variables")
```
<img width="962" height="862" alt="Rplot05" src="https://github.com/user-attachments/assets/a756c9ca-c339-4d34-a97a-f133fccef06b" />

## Variables Skewness
Plotted density distributions for some of the variables to inspect their skewness and distribution of the data. 

```R
# DENSITY/SKEWNESS
describe(training_data)

plot(density(training_data$mean_RGDP2))
plot(density(training_data$median_RGDP2))

plot(density(training_data$final_est))
plot(density(training_data$unemployment_rate))
plot(density(training_data$PCE))
plot(density(training_data$government_expenditure))
plot(density(training_data$net_export))
plot(density(training_data$Industrial_Production))
plot(density(training_data$Personal_income))
plot(density(training_data$NGDP))

plot(density(na.omit(training_data$prccq_qoq_growth_mean)))
plot(density(na.omit(training_data$prchq_qoq_growth_mean)))
plot(density(na.omit(training_data$prclq_qoq_growth_mean)))

plot(density(training_data$marketcap_qoq_mean))
plot(density(training_data$ret_mean))
```
```R
plot(density(training_data$roa_qoq_mean))
```
<img width="602" height="421" alt="Screenshot 2025-07-14 at 8 22 44 PM" src="https://github.com/user-attachments/assets/5ebcdaa8-90ff-4313-aaf9-0a283eaa975d" />

```R
plot(density(training_data$debt_ratio_qoq_mean))
```
<img width="606" height="424" alt="Screenshot 2025-07-14 at 8 22 59 PM" src="https://github.com/user-attachments/assets/f7745f66-547e-4ae2-947f-643a935a6f00" />

## Prediction Models
### Linear Regression
Running multiple linear regressions allowed us to see the relationship between the dependent variable (GDP growth) and the independent variables. Firstly, we built the base model using economics variables only. Then, adding onto the base model, we built a second model by adding accounting variables to it. Lastly, we built the third model by adding market variables onto the base model. By comparing these models, we assessed whether the addition of accounting and market variables improves the predictive power and accuracy of the model.

The adjusted R square tells us the goodness of fit of the regression model, by adjusting for the number of independent variables that we include in the model. The higher the value, the better the independent variables explain the variation in the dependent variable.

The p-value is a statistical measure that helps to determine the significance of the results in hypothesis testing. In the regression model, it helps us to assess whether the independent variables significantly contribute to explaining the dependent variable. A good p-value is typically < = 0.05, a value under less than this significance level suggests strong evidence against the null hypothesis.

## Baseline model (Economic Variables Only)
```R
### BUILD MODEL ###
### BASE MODEL
base <- lm(NGDP ~ mean_RGDP2 + mean_RGDP6 + mean_NGDP2 + mean_NGDP6 +
             median_RGDP2 + median_RGDP6 + median_NGDP2 + median_NGDP6 +
             final_est + cpi + unemployment_rate + PCE + government_expenditure + 
             net_export + Industrial_Production + Personal_income, 
           data = training_data)

# get statistics of the base model
summary(base)
# Adj R Square: 0.7801
# RSE: 1.28

glance(base)

# using the model to predict the test data
pred <- predict(base, testing_data)
testing_data$YQ <- paste0(testing_data$YEAR, 'Q', testing_data$QUARTER)

# store in csv 
testing_result <- testing_data %>%
  select(YQ) %>%
  mutate(NGDP = pred)

# Export economic_data as a CSV file
write.csv(testing_result, "testing_result.csv", row.names = FALSE)
# Kaggle Out of Sample RMSE: 1.47384
```
**Explanation:**
The results show us the R square value is 0.8142, meaning that the model explains 81.42% of the variation in the dependent variable (GDP growth). But if we look at the adjusted R-Square which gives penalty if variable is added, the value reduces to 0.7801, meaning that 78.01% of the variability in GDP growth is explained by the independent variables in the model, the high value suggests that the model fits the data.

From the p-value, the variables that are statistically significant (p-value ≤ 0.05) include final estimation (final_est )(p = 1.69e-07 ), personal consumption expenditure PCE (p = 0.000436), net export  (net_export) (p = 0.001241), and Industrial Production (Industrial_Production) (p = 0.009602). These p-values indicate that these variables have a statistically significant relationship with the dependent variable, GDP growth. We also concluded that independent variables, median of survey of real GDP growth of 4th quarter forward (median_RGDP6), median of survey of nominal GDP growth of current quarter (median_NGPD2), cpi, and personal_income has significant negative correlation with gdp, while mean of survey of real GDP growth of current quarter (mean_RGDP2), mean of survey of nominal GDP growth of 4th quarter forward (mean_NGDP6), and especially personal consumption expenditure (PCE) are strongly and positively correlated with GDP.

The model F-statistic is 23.83 with 16 and 87 degrees of freedom, and a very small p-value (< 2.2e-16), indicating that the overall model is statistically significant. The Kaggle Out of Sample RMSE is 1.47384 suggests that on average, this model's predictions are off by about 1.47384 percentage points. Economic predictions, especially for GDP growth, usually faces considerable variation due to political events, global economic changes and therefore, we assumed that an RMSE of 1.47384 could indicate that this model is capturing patterns reasonably well but could still be improved if possible.

## Baseline and Accounting Model
``` R
# BASE + ACCOUNTING VARIABLES
baseaccounting <- lm(NGDP ~ mean_RGDP2 + mean_RGDP6 + mean_NGDP2 + mean_NGDP6 +
                       median_RGDP2 + median_RGDP6 + median_NGDP2 + median_NGDP6 +
                       final_est + cpi + unemployment_rate + PCE + government_expenditure + 
                       net_export + Industrial_Production + Personal_income + 
                       pm_qoq_mean + nopat_qoq_mean + roa_qoq_mean + debt_ratio_qoq_mean,
                     data = training_data)
summary(baseaccounting) 
# Adj R Square: 0.7781
# RSE: 1.285
glance(baseaccounting)

#anova test
anova(base, baseaccounting, test = "Chisq")
# p value of 0.5197 suggest that the improvement on the RSS is not statistically significant

# using the model to predict the test data
pred <- predict(baseaccounting, testing_data)
testing_data$YQ <- paste0(testing_data$YEAR, 'Q', testing_data$QUARTER)

# store in csv 
testing_result_accounting <- testing_data %>%
  select(YQ) %>%
  mutate(NGDP = pred)

# Export economic_data as a CSV file
write.csv(testing_result_accounting, "testing_result_accounting.csv", row.names = FALSE)
# Kaggle Out of Sample RMSE: 1.42852
```
**Explanation:**
The adjusted R-squared is 0.7781, meaning that 77.81% of the variability in GDP growth is explained by the independent variables in the model which is not an improvement from the base model.  
The p-value result is just like the base model with only final estimation (final_est), personal consumption expenditure (PCE), net export (net_export) and industrial production (Industrial_Production) being statistically significant as the p-value is below 0.05.
The model F-statistic is 19.06 on 20 and 83 degrees of freedom and the same p-value as our base model , indicating that the overall model is statistically significant. The out of sample RMSE result from Kaggle is 1.42852 and this means that on average, this model’s predictions are off by 1.42852 percentage points from the actual values. This is a slight improvement from the base model which has an RMSE of 1.47384 and therefore, the accounting variables slightly improve the model.

## Baseline and Market Model
```R
### BASE + MARKET
basemarket <- lm(NGDP ~ mean_RGDP2 + mean_RGDP6 + mean_NGDP2 + mean_NGDP6 +
                   median_RGDP2 + median_RGDP6 + median_NGDP2 + median_NGDP6 +
                   final_est + cpi + unemployment_rate + PCE + government_expenditure + 
                   net_export + Industrial_Production + Personal_income + 
                   marketcap_qoq_mean + ret_mean,
                 data = training_data)
summary(basemarket)
#Adj R Square: 0.7847
#RSE: 1.266
glance(basemarket)

#using the model to predict the test data
pred <- predict(basemarket, testing_data)
testing_data$YQ <- paste0(testing_data$YEAR, 'Q', testing_data$QUARTER)

#anova test
anova(base, basemarket, test = "Chisq")
# 0.145 p-value suggest that the improvement on the RSS is not statistically significant

# store in csv 
testing_result_market <- testing_data %>%
  select(YQ) %>%
  mutate(NGDP = pred)

# Export economic_data as a CSV file
write.csv(testing_result_market, "testing_result_market.csv", row.names = FALSE)
# Kaggle Out of Sample RMSE: 1.13327
```
**Explanation:**
The adjusted R-squared is 0.7847, meaning that 78.47% of the variability in GDP growth is explained by the independent variables in the model, it's a slight improvement from the base model.

Based on the p-value, BEA final estimation (final_est), personal consumption expenditure (PCE), government expenditure (government_expenditure), net export (net_export) and industrial production (Industrial_Production) are statistically significant as the p-value is below 0.05. Compared to the previous model, there is an additional variable (government_expenditure) that became statistically significant. This means this expanded model is more informative and likely a better fit to the data. 

Additionally, the model F-statistic is 21.85 with 18 and 85 degrees of freedom which is even lower than the base model’s F-statistic and also has a very small p-value (< 2.2e-16), indicating that the overall model is statistically significant and probably a better fit than the base model. An RMSE of 1.13327 suggests that this model is the best performing one amongst all three as it has the smallest RMSE. On average, this model’s predictions are off by only 1.252 percentage points from the actual values. This is a 0.34 percentage point increase from the base model’s result which means that the market variables improve the model.

## Regularized Models (LASSO, Ridge, Elastic Net)
```R
### REGULARISED MODELS (LASSO, RIDGE, ELASTIC NET) ###

# create YQ vector for prediction file
yq <- paste0(testing_data$YEAR, "Q", testing_data$QUARTER)

## BASE
training_base <- training_data[, c(47,50,54,55,59,60,64,65,69, 70:77)] # 104 obs, 17 vars
testing_base <- testing_data[, c(47,50,54,55,59,60,64,65,69, 70:77)] # 20 obs, 17 vars

# prepare training data
xb_train <- as.matrix(training_base[, setdiff(names(training_base), "NGDP")])

# extract IV (excluding NGDP)
yb_train <- training_base$NGDP

# replace NA in xb_train with 0
xb_train[is.na(xb_train)] <- 0

# prepare testing data 
xb_test <- as.matrix(testing_base[, setdiff(names(testing_base), "NGDP")])
```

**Explanation:**
To enhance the robustness of our model and minimize the risk of overfitting caused by the high dimensionality of our feature set. Due to the inclusion of multiple accounting variables, we are implementing three advanced regularization techniques to further build up our model. Firstly, LASSO (Least Absolute Shrinkage and Selection Operator) imposes an L1 penalty to shrink less relevant coefficients to zero for automatic feature selection; Then Ridge Regression applies an L2 penalty to reduce the magnitude of coefficients, effectively handling multicollinearity. The third technique is Elastic Net, which combines L1 and L2 penalties to balance feature selection and regularization. 

These techniques not only enhance model stability but also prevent misleading relationships from arising among economic, market, and accounting variables. Without proper regularization, not relevant correlations could distort the model’s interpretability and predictive accuracy. Our analysis has identified several key variables that exhibit significant positive or negative correlations with GDP growth. Below sections illustrate the model performance by visualizations.

### Data Preparation for Regularized Models
#### Baseline Model (Economic Variables Only)
```R
# create YQ vector for prediction file
yq <- paste0(testing_data$YEAR, "Q", testing_data$QUARTER)

## BASE
training_base <- training_data[, c(47,50,54,55,59,60,64,65,69, 70:77)] # 104 obs, 17 vars
testing_base <- testing_data[, c(47,50,54,55,59,60,64,65,69, 70:77)] # 20 obs, 17 vars

# prepare training data
xb_train <- as.matrix(training_base[, setdiff(names(training_base), "NGDP")])

# extract IV (excluding NGDP)
yb_train <- training_base$NGDP

# replace NA in xb_train with 0
xb_train[is.na(xb_train)] <- 0

# prepare testing data 
xb_test <- as.matrix(testing_base[, setdiff(names(testing_base), "NGDP")])
```

#### Baseline and Accounting Models
``` R
## BASE + ACCOUNTING VARIABLES
training_acc <- training_data[, c(42:43,45:47,50,54,55,59,60,64,65,69, 70:77)] # 104 obs 21 vars
testing_acc <- testing_data[, c(42:43,45:47,50,54,55,59,60,64,65,69, 70:77)] # 20 obs 21 vars

# prepare training data
xa_train <- as.matrix(training_acc[, setdiff(names(training_acc), "NGDP")])

# extract independent variable (excluding NGDP)
ya_train <- training_acc$NGDP 

# replace NA in xa_train with 0
xa_train[is.na(xa_train)] <- 0

# prepare testing data
xa_test <- as.matrix(testing_acc[, setdiff(names(testing_acc), "NGDP")])
```

#### Baseline and Market Models
``` R
## BASE + MARKET VARIABLES
training_market <- training_data[, c(47,50,54,55,59,60,64,65,69, 70:77, 81:82)] # 104 obs 19 vars
testing_market <- testing_data[, c(47,50,54,55,59,60,64,65,69, 70:77, 81:82)] # 20 obs 19 vars

# prepare training data
xm_train <- as.matrix(training_market[, setdiff(names(training_market), "NGDP")])

# extract independent variable (excluding NGDP)
ym_train <- training_market$NGDP

# replace NA in xm_train with 0
xm_train[is.na(xm_train)] <- 0

# prepare testing data 
xm_test <- as.matrix(testing_market[, setdiff(names(testing_market), "NGDP")])
```
## LASSO
Least Absolute Shrinkage and Selection Operator (LASSO) is a type of regularization technique used to improve the accuracy and interpretability of regression models. Unlike traditional linear regression, which includes all predictor variables. LASSO adds a penalty term to the regression equation and forces some less important coefficients to shrink to exactly zero. This shrinkage process serves two key purposes as feature selection and multicollinearity handling. By eliminating irrelevant variables, LASSO helps reduce the complexity of the model and improves interpretability. Moreover, only the most relevant features, LASSO helps mitigate issues where independent variables are highly correlated with each other (i.e., multicollinearity problem).

1. Before selecting the optimal lambda (λ), we first generate a LASSO coefficient profile plot. This plot illustrates how the regression coefficients change as the penalty term (lambda) varies. As lambda increases, more coefficients shrink toward zero, which allows us to observe the variables that remain influential even under strong regularization.
2. To identify the best penalty value, we perform 10-fold cross-validation to determine the optimal lambda (λ.min). The model is trained across a range of lambda values, and for each value, cross-validation calculates the Mean Squared Error (MSE). The optimal lambda (λ.min) is the value that results in the lowest average MSE across the validation folds. Meaning that the model achieves the best balance between accuracy and complexity since it minimizes overfitting by preventing the model from becoming too complex.
3. Once the optimal lambda is identified, we use it to train the final LASSO model and predict GDP growth on both the training and test sets. The performance of the model is evaluated using Root Mean Squared Error (RMSE), which measures how well the predicted GDP values align with actual GDP values. Finally, the RMSE for the test set is calculated and benchmarked on Kaggle to assess the model's predictive power. 

### Baseline Model (Economic Variables Only)
``` R
# BASE
set.seed(2020)
lasso_model <- glmnet(xb_train,yb_train, family = "gaussian", alpha = 1)

# before cross validation
plot(lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot06" src="https://github.com/user-attachments/assets/62ee0f93-d35a-425c-a656-60cda27e4caa" />

``` R
coefplot(lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot07" src="https://github.com/user-attachments/assets/b7eac2d1-848d-46d9-9b5f-b2ad065f07c2" />

``` R
# 10-fold cross validation
base_cv_lasso_model <- cv.glmnet(xb_train, yb_train, family = "gaussian", alpha = 1,
                                 type.measure = "mse")
# after cross validation
coefplot(base_cv_lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot08" src="https://github.com/user-attachments/assets/08b2c97f-b0a8-49cc-b100-9991b566eaa9" />

``` R
# with lambda.min (optimal lambda so best model for lasso)
base_lambda_min_lasso <- base_cv_lasso_model$lambda.min

# print the lambda.min value 
base_lambda_min_lasso
# 0.0336422

# predict using lambda.min
predict_lambda_min_lasso <- predict(base_cv_lasso_model, newx = xb_test, 
                                    s = base_lambda_min_lasso, type = "response")

# prepare data frame for for lasso
lasso_base_lambda_min <- data.frame(YQ = yq, NGDP = predict_lambda_min_lasso) # 20 obs 2 vars
colnames(lasso_base_lambda_min)[2] <- "NGDP" # rename column as NGDP

# Export lasso_base_lambdamin as CSV file
write.csv(lasso_base_lambda_min, "lasso_base_lambda_min.csv", row.names = FALSE)
# Kaggle Out of Sample RMSE: 1.6540
```
**Explanation:**
After performing cross-validation with the base model, LASSO identifies personal consumption expenditure (PCE), industrial production, and final estimated GDP (final_est) as key predictors of GDP growth. All of these variables show a positive correlation with the target variable. Meanwhile, the graph indicates that Consumer Price Index (cpi) and median forecast of real GDP growth for the next four quarters (median_RGDP6) are negatively correlated with GDP growth. 

The model using only the base variables achieves an out-of-sample Kaggle RMSE of 1.6540, meaning the average prediction error is approximately 1.6540 units in terms of GDP growth. Since a lower RMSE reflects higher predictive accuracy, the objective is to enhance model performance by incorporating additional variables that can further reduce prediction errors and improve alignment with actual GDP growth trends.

### Base and Accounting Model
``` R
# BASE + ACCOUNTING 
set.seed(2020)
lasso_model <- glmnet(xa_train, ya_train, family = "gaussian", alpha = 1)

# before cross validation
plot(lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot12" src="https://github.com/user-attachments/assets/90612601-be60-427a-868a-1f078d57e53e" />

``` R
coefplot(lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot13" src="https://github.com/user-attachments/assets/95035092-d18c-4a10-8525-fc0fd68a78e0" />


``` R
# 10-fold cross validation
acc_cv_lasso_model <- cv.glmnet(xa_train, ya_train, family = "gaussian", alpha = 1,
                                type.measure = "mse")
# after cross validation
coefplot(acc_cv_lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot14" src="https://github.com/user-attachments/assets/c0ecb899-0f2e-4554-b15f-84007d4725c7" />

``` R
# with lambda.min (optimal lambda so best model for lasso)
acc_lambda_min_lasso <- acc_cv_lasso_model$lambda.min

# print the lambda.min value 
acc_lambda_min_lasso
# 0.02793035

# predict using lambda.min
predict_lambda_min_lasso <- predict(acc_cv_lasso_model, newx = xa_test, 
                                    s = acc_lambda_min_lasso, type = "response")

# prepare data frame for for lasso
lasso_acc_lambda_min <- data.frame(YQ = yq, NGDP = predict_lambda_min_lasso) # 20 obs 2 vars
colnames(lasso_acc_lambda_min)[2] <- "NGDP" # rename column as NGDP

# Export lasso_acc_lambda_min as CSV file
write.csv(lasso_acc_lambda_min, "lasso_acc_lambda_min.csv", row.names = FALSE)
# Kaggle Out of Sample RMSE: 1.66836
```
**Explanation:**
After cross-validation, the coefficient plot visualizes how various predictors influence GDP growth, reflecting both the strength and direction of their relationships with the target variable. Each coefficient represents the estimated impact of a specific variable on GDP growth.

By selecting the optimal lambda, LASSO highlights personal consumption expenditure (PCE), industrial production, and final estimated GDP (final_est) as the most crucial predictors. A positive correlation between personal consumption expenditure (PCE) suggests that stronger stock market performance aligns with economic expansion. On the other hand, the negative correlation between Consumer Price Index (cpi) and median forecast of real GDP growth for the next four quarters (median_RGDP6) indicates that rising stock prices may coincide with slower economic growth, possibly due to speculative bubbles or market distortions.

Furthermore, the model’s Kaggle out-of-sample performance (RMSE = 1.66836) reveals room for improvement since it is lower than the base model performance. LASSO’s L1 penalty tends to select only one feature from a cluster of highly correlated variables while discarding the rest. This approach is useful for simplifying models; however, it may introduce selection bias and exclude meaningful predictors, potentially limiting the model’s effectiveness.

### Base and Market Variable
``` R
 BASE + MARKET 
set.seed(2020)
lasso_model <- glmnet(xm_train,ym_train, family = "gaussian", alpha = 1)

# before cross validation
plot(lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot15" src="https://github.com/user-attachments/assets/2c05b168-4835-4d03-afd4-eeea9306664e" />

``` R
coefplot(lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot16" src="https://github.com/user-attachments/assets/2b1119e1-dee0-4fa5-846b-98caa5bb7ea9" />


``` R
# 10-fold cross validation
market_cv_lasso_model <- cv.glmnet(xm_train, ym_train, family = "gaussian", alpha = 1,
                                   type.measure = "mse")
# after cross validation
coefplot(market_cv_lasso_model, xvar = "lambda", label = TRUE)
```
<img width="962" height="862" alt="Rplot17" src="https://github.com/user-attachments/assets/1b369acc-2e49-4460-ad13-6e5dfbe0dcf6" />

``` R
# with lambda.min (optimal lambda so best model for lasso)
market_lambda_min_lasso <- market_cv_lasso_model$lambda.min

# print the lambda.min value 
market_lambda_min_lasso
# 0.02793035

# predict using lambda.min
predict_lambda_min_lasso <- predict(market_cv_lasso_model, newx = xm_test, 
                                    s = market_lambda_min_lasso, type = "response")

# prepare data frame for for lasso
lasso_market_lambda_min <- data.frame(YQ = yq, NGDP = predict_lambda_min_lasso) # 20 obs 2 vars
colnames(lasso_market_lambda_min)[2] <- "NGDP" # rename column as NGDP

# Export lasso_market_lambda_min as CSV file
write.csv(lasso_market_lambda_min, "lasso_market_lambda_min.csv", row.names = FALSE)
# Kaggle Out of Sample RMSE: 1.08628
```
**Explanation:**
With the inclusion of additional marketing variables, LASSO selects several significant predictors of GDP growth, such as average return (ret_mean) and personal consumption expenditure (PCE) and Consumer Price Index (cpi). Among these variables, the Consumer Price Index (cpi) represents a negative correlation with GDP growth, whereas the remaining variables show a positive association.

The model incorporating accounting variables achieves an out-of-sample Kaggle score of RMSE 1.08628, which shows a noticeable improvement compared to the model based solely on market variables. As a result, this lower error rate suggests enhanced predictive accuracy, indicating that accounting data may offer a more precise and direct perspective on GDP fluctuations than market-based indicators.








