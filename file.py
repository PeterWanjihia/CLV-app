# CUSTOMER LIFETIME VALUE MODEL FOR FINANCIAL INSTITUTIONS

# LIBRARIES
import pandas as pd
import plydata.cat_tools as cat
import plotnine as pn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import joblib

pn.options.dpi = 300

# 1.0 DATA PREPARATION
cdnow_raw_df = pd.read_csv("data/CDNOW_master.txt", sep="\s+", names=["customer_id", "date", "transNumber", "transAmount"])

cdnow_raw_df.info()

cdnow_df = cdnow_raw_df \
    .assign(date=pd.to_datetime(cdnow_raw_df['date'], format="%Y%m%d")) \
    .dropna()

cdnow_df.info()

# 2.0 COHORT ANALYSIS
cdnow_first_transactions_tbl = cdnow_df \
    .sort_values(['customer_id', 'date']) \
    .groupby('customer_id') \
    .first()

cdnow_first_transactions_tbl
cdnow_first_transactions_tbl['date'].min()
cdnow_first_transactions_tbl['date'].max()

cdnow_df.reset_index() \
    .set_index('date') \
    [['transAmount']] \
    .resample(rule="MS") \
    .sum() \
    .plot()

ids = cdnow_df['customer_id'].unique()
ids_selected = ids[0:10]

cdnow_cust_id_subset_df = cdnow_df[cdnow_df['customer_id'].isin(ids_selected)] \
    .groupby(['customer_id', 'date']) \
    .sum() \
    .reset_index()

pn.ggplot(data=cdnow_cust_id_subset_df,
          mapping=pn.aes(x='date', y='transAmount', group='customer_id')) + \
    pn.geom_line() + \
    pn.geom_point() + \
    pn.facet_wrap('customer_id') + \
    pn.scale_x_date(date_breaks='1 Year', date_labels='%Y')

# 3.0 MACHINE LEARNING
# 3.1 TIME SPLITTING (STAGE 1)
n_days = 90
max_date = cdnow_df['date'].max()
cutoff = max_date - pd.to_timedelta(n_days, unit="d")

temporal_in_df = cdnow_df[cdnow_df['date'] <= cutoff]
temporal_out_df = cdnow_df[cdnow_df['date'] > cutoff]

# 3.2 FEATURE ENGINEERING (STAGE 2)
targets_df = temporal_out_df \
    .drop('transNumber', axis=1) \
    .groupby('customer_id') \
    .agg({'transAmount': 'sum', 'date': 'max'}) \
    .rename({'transAmount': 'trans_90_total'}, axis=1) \
    .assign(trans_90_flag=lambda x: (x['date'] >= cutoff).astype(int)) \
    .drop('date', axis=1)

max_date = temporal_in_df['date'].max()

recency_features_df = temporal_in_df \
    .groupby('customer_id') \
    .apply(lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, 'day')) \
    .to_frame() \
    .set_axis(['recency'], axis=1)

frequency_feature_df = temporal_in_df \
    .groupby('customer_id') \
    .size() \
    .to_frame() \
    .set_axis(['frequency'], axis=1)

transamount_feature_df = temporal_in_df \
    .groupby('customer_id') \
    .agg({'transAmount': ['sum', 'mean']}) \
    .droplevel(0, axis=1) \
    .rename(columns={'sum': 'transAmount_sum', 'mean': 'transAmount_mean'})

features_df = pd.concat([recency_features_df, frequency_feature_df, transamount_feature_df], axis=1) \
    .merge(targets_df, left_index=True, right_index=True, how='left') \
    .fillna(0)

# 3.3 MACHINE LEARNING
X = features_df[['recency', 'frequency', 'transAmount_sum', 'transAmount_mean']]

# 4.1 Next 90-day Transaction Amount
y_spend = features_df['trans_90_total']

xgb_reg_spec = XGBRegressor(objective="reg:squarederror", random_state=123)

xgb_reg_model = GridSearchCV(estimator=xgb_reg_spec, param_grid={'learning_rate': [0.01, 0.1, 0.3, 0.5]},
                             scoring='neg_mean_absolute_error', refit=True, cv=5)

xgb_reg_model.fit(X, y_spend)
xgb_reg_model.best_score_
xgb_reg_model.best_params_
xgb_reg_model.best_estimator_
predictions_reg = xgb_reg_model.predict(X)

# 4.2 Next 90 days probability (Predict whether or not they are going to transact with the bank)
y_prob = features_df['trans_90_flag']

xgb_clf_spec = XGBClassifier(objective='binary:logistic', random_state=123)

xgb_clf_model = GridSearchCV(estimator=xgb_clf_spec, param_grid={'learning_rate': [0.01, 0.1, 0.3, 0.5]},
                             scoring='roc_auc', refit=True, cv=5)

xgb_clf_model.fit(X, y_prob)
xgb_clf_model.best_score_
xgb_clf_model.best_params_
xgb_clf_model.best_estimator_
predictions_clf = xgb_clf_model.predict_proba(X)

# 4.3 Generating the Feature Importance in the global scope
imp_trans_amount_dict = xgb_reg_model.best_estimator_.get_booster().get_score(importance_type='gain')

imp_trans_amount_df = pd.DataFrame({'feature': list(imp_trans_amount_dict.keys()),
                                    'value': list(imp_trans_amount_dict.values())}) \
    .assign(feature=lambda x: cat.cat_reorder(x['feature'], x['value']))

pn.ggplot(pn.aes(x='feature', y='value'), data=imp_trans_amount_df) + \
    pn.geom_col() + \
    pn.coord_flip()

imp_trans_prob_dict = xgb_clf_model.best_estimator_.get_booster().get_score(importance_type='gain')

imp_trans_prob_df = pd.DataFrame({'feature': list(imp_trans_prob_dict.keys()),
                                  'value': list(imp_trans_prob_dict.values())}) \
    .assign(feature=lambda x: cat.cat_reorder(x['feature'], x['value']))

plot = pn.ggplot(pn.aes(x='feature', y='value'), data=imp_trans_prob_df) + \
    pn.geom_col() + \
    pn.coord_flip()

plot

# 5.0 SAVING THE MODEL TASKS
predictions_df = pd.concat([pd.DataFrame(predictions_reg, columns=['pred_trans']),
                           pd.DataFrame(predictions_clf[:, 1], columns=['pred_prob']),
                           features_df.reset_index()], axis=1)

predictions_df.to_pickle('artifacts/predictions_df.pkl')
pd.read_pickle('artifacts/predictions_df.pkl')

imp_trans_amount_df.to_pickle('artifacts/imp_trans_amount_df.pkl')
imp_trans_prob_df.to_pickle('artifacts/imp_trans_prob_df.pkl')

pd.read_pickle('artifacts/imp_trans_amount_df.pkl')

joblib.dump(xgb_reg_model, 'artifacts/xgb_reg_model.pkl')
joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')

model = joblib.load('artifacts/xgb_reg_model.pkl')
model.predict(X)

# 6.0 HOW TO USE THE INFORMATION ABOVE
# i) Customers with the highest probability of transacting with the bank in the next 90 days
predictions_df.sort_values(by='pred_trans', ascending=False)

# ii) Customers who have made transactions recently but are unlikely to continue
predictions_df[(predictions_df['recency'] > -90) & (predictions_df['pred_prob'] < 0.20)] \
    .sort_values(by='pred_prob', ascending=False)

# iii) Missed Opportunities
predictions_df[(predictions_df['trans_90_total'] == 0.0)].sort_values(by='pred_trans', ascending=False)


