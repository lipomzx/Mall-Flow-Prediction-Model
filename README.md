# Mall-Flow-Prediction-Model
## Goal
- Predict the flow of the shopping malls to facilitate the clients who want to select site to open store; (No time series data)
- Features: ```'open_year', 'year_gap','business_area', 'store_num', 'mall_brand_level', 'developer_level','subway_num_500', 'subway_line_num_500', 'catering_num', 'data_time','flow_w_month', 'flow_daily_from_project', 'flow_Zhenliang','flow_GridArea', 'flow_circle_with_one_mall',```
- Label: ```'flow_daily'```
- Training data size: 700+
  
## Chosen Model - XGboost 
- Choose from regression models: log regression, random forest, elastic net, xgboost 
- Metrics: RMSE & R2 score
  
## Results of XGBoost model
- RMSE: 0.826254528945324
- Train R2_score: 0.6274700548608262
- Test R2_score:  0.5201399873873063

## XGboost Implementation from Scratch 
- Following the blog [What makes xgboost so extreme](https://medium.com/analytics-vidhya/what-makes-xgboost-so-extreme-e1544a4433bb) and implement the xgboost model while replicate sklearn api for binary classification and regression with only Python package
- For better understanding the mechanism and help finetuning parameters 

## Logs
- **Overfitting**
  - The r2 score between train and test data is over 0.2 which indicates the overfitting. I solve this problem by constraint the min_child_weight and make the model more conservative with regularization weights and gamma. 
- **Outliers**
  - Since the xgboost is based on the regression tree, it can be easily affected by outliers in our imbalanced dataset. We couldn't get the precise traffic flow from all malls officially. This problem is left for future work.  

