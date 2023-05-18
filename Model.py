#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib 

import plydata.cat_tools as cat 
import plotnine as pn

from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import GridSearchCV

pn.options.dpi = 300



# In[2]:


cdnow_raw_df = pd.read_csv(
    "data/CDNOW_master.txt",
    sep = "\s+",
    names = ["customer_id","date","transNumber","transAmount"]
)


# In[3]:


cdnow_raw_df.info()


# In[4]:


cdnow_raw_df


# In[5]:


cdnow_df = cdnow_raw_df \
    .assign(
        date = lambda x: x['date'].astype(str)
    )\
    .assign(
        date = lambda x: pd.to_datetime(x['date'])        
    )\
    .dropna()


# In[6]:


cdnow_df.info()


# In[7]:


cdnow_df


# In[8]:


cdnow_first_transactions_tbl = cdnow_df\
    .sort_values(['customer_id','date'])\
    .groupby('customer_id')\
    .first()


# In[9]:


cdnow_first_transactions_tbl


# In[10]:


cdnow_first_transactions_tbl['date'].min()


# In[11]:


cdnow_first_transactions_tbl['date'].max()


# In[12]:


cdnow_df\
    .reset_index()\
    .set_index('date')\
    [['transAmount']]\
    .resample(
        rule = "MS"
    )\
    .sum()\
    .plot()


# In[13]:


ids = cdnow_df['customer_id'].unique()
ids_selected = ids[0:10]


# In[14]:


cdnow_cust_id_subset_df = cdnow_df\
    [cdnow_df['customer_id'].isin(ids_selected)]\
    .groupby(['customer_id','date'])\
    .sum()\
    .reset_index()


# In[15]:


pn.ggplot(
    data=cdnow_cust_id_subset_df,
    mapping=pn.aes(x='date', y='transAmount', group='customer_id')
) + \
pn.geom_line() + \
pn.geom_point() + \
pn.facet_wrap('customer_id') + \
pn.scale_x_date(
    date_breaks='1 Year',
    date_labels='%Y'
)

# With this code we can visualize how many times each selected customer made transactions with the bank 


# In[16]:


pn.ggplot(
    data=cdnow_cust_id_subset_df,
    mapping=pn.aes(x='date', y='transAmount', group='customer_id')
) + \
pn.geom_line() + \
pn.geom_point()
# This code helps us visualize all the purchases made by the ten selected customers all in one place


# In[17]:


n_days = 90
max_date = cdnow_df['date'].max()
cutoff = max_date - pd.to_timedelta(n_days, unit = "d")

temporal_in_df = cdnow_df \
    [cdnow_df['date']<=cutoff]
temporal_out_df = cdnow_df \
    [cdnow_df['date'] > cutoff]  


# In[18]:


cdnow_df


# In[19]:


max_date


# In[20]:


cutoff


# In[21]:


temporal_in_df


# In[22]:


temporal_out_df


# In[23]:


temporal_out_df\
    .drop('transNumber',axis=1)


# In[24]:


targets_df = temporal_out_df\
    .drop('transNumber', axis=1)\
    .groupby('customer_id')\
    .agg({'transAmount': 'sum', 'date': 'max'})\
    .rename({'transAmount': 'trans_90_total'}, axis=1)\
    .assign(trans_90_flag=lambda x: (x['date'] >= cutoff).astype(int))\
    .drop('date', axis=1)


# In[25]:


targets_df


# In[26]:


max_date = temporal_in_df['date'].max()


# In[27]:


temporal_in_df


# In[28]:


recency_features_df = temporal_in_df\
    [['customer_id', 'date']]\
    .groupby('customer_id')\
    .apply(
        lambda x: (x['date'].max() - max_date) / pd.to_timedelta(1, 'day')
    )\
    .to_frame()\
    .set_axis(['recency'], axis=1)


# In[29]:


recency_features_df


# In[30]:


frequency_feature_df = temporal_in_df\
    [['customer_id','date']]\
    .groupby('customer_id')\
    .count()\
    .set_axis(['frequency'],axis=1)            


# In[31]:


frequency_feature_df


# In[32]:


transamount_feature_df = temporal_in_df\
    .groupby('customer_id')\
    .aggregate(
        {'transAmount': ['sum','mean']}
    )\
    .set_axis(['transAmount_sum','transAmount_mean'], axis=1)


# In[33]:


transamount_feature_df


# In[34]:


features_df = pd.concat(
    [recency_features_df,frequency_feature_df,transamount_feature_df],axis=1
)\
    .merge(
        targets_df,
        left_index = True,
        right_index = True,
        how = 'left'
        
    )\
    .fillna(0)


# In[35]:


features_df


# In[36]:


from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import  GridSearchCV

X = features_df [['recency','frequency','transAmount_sum','transAmount_mean']]


# In[37]:


y_spend = features_df['trans_90_total']

xgb_reg_spec = XGBRegressor(
    objective = "reg:squarederror",
    random_state = 123
)


# In[38]:


y_spend = features_df['trans_90_total']

xgb_reg_spec = XGBRegressor(
    objective="reg:squarederror",
    random_state=123
)

xgb_reg_model = GridSearchCV(
    estimator=xgb_reg_spec,
    param_grid=dict(
        learning_rate=[0.01, 0.1, 0.3, 0.5]
    ),
    scoring='neg_mean_absolute_error',
    refit=True,
    cv=5
)

xgb_reg_model.fit(X, y_spend)
xgb_reg_model.best_score_
xgb_reg_model.best_params_
xgb_reg_model.best_estimator_
predictions_reg = xgb_reg_model.predict(X)


# In[41]:


xgb_reg_model.best_score_


# In[40]:


predictions_reg


# In[54]:


y_prob = features_df['trans_90_flag']

xgb_clf_spec = XGBClassifier(
    objective = 'binary:logistic',
    random_state = 123
)

xgb_clf_model = GridSearchCV(
    estimator = xgb_clf_spec,
    param_grid = dict(
        learning_rate = [0.01,0.1,0.3,0.5]
    ),
    scoring = 'roc_auc',
    refit = True,
    cv = 5   
    
)
    
xgb_clf_model.fit(X, y_prob)
xgb_clf_model.best_score_
xgb_clf_model.best_params_
xgb_clf_model.best_estimator_
predictions_clf = xgb_clf_model.predict_proba(X)


# In[55]:


predictions_clf


# In[56]:


xgb_clf_model.best_score_


# In[57]:


xgb_clf_model.best_estimator_


# In[58]:


imp_trans_amount_dict = xgb_reg_model\
    .best_estimator_\
    .get_booster()\
    .get_score(importance_type = 'gain')
    
imp_trans_amount_df = pd.DataFrame(
    data ={
        'feature': list (imp_trans_amount_dict.keys()),
        'value': list(imp_trans_amount_dict.values())
    }
)\
    .assign(
        feature = lambda x: cat.cat_reorder(x['feature'] ,x['value'])
    )
    


# In[61]:


imp_trans_amount_dict


# In[62]:


imp_trans_amount_df


# In[64]:


pn.ggplot(
    pn.aes('feature', 'value'),
    imp_trans_amount_df
) + pn.geom_col() + pn.coord_flip()


# In[65]:


xgb_clf_model


# In[67]:


imp_trans_prob_dict = xgb_clf_model\
    .best_estimator_\
    .get_booster()\
    .get_score(importance_type='gain')

imp_trans_prob_df = pd.DataFrame(
    data={
        'feature': list(imp_trans_prob_dict.keys()),
        'value': list(imp_trans_prob_dict.values())
    }
).assign(
    feature=lambda x: cat.cat_reorder(x['feature'], x['value'])
)

    


# In[71]:


plot = pn.ggplot(
    pn.aes('feature', 'value'),
    imp_trans_prob_df
) + pn.geom_col() + pn.coord_flip()



# In[73]:


plot


# In[74]:


predictions_df = pd.concat(
    [
        pd.DataFrame(predictions_reg).set_axis(['pred_trans'],axis=1),
        pd.DataFrame(predictions_clf)[[1]].set_axis(['pred_prob'],axis=1),
        features_df.reset_index()
    ],
    axis = 1
)


# In[75]:


predictions_df


# In[77]:


predictions_df.to_pickle('artifacts/predictions_df.pkl')
pd.read_pickle('artifacts/predictions_df.pkl')


# In[78]:


imp_trans_amount_df.to_pickle("artifacts/imp_trans_amount_df.pkl")
imp_trans_prob_df.to_pickle("artifacts/imp_trans_prob_df.pkl")

pd.read_pickle("artifacts/imp_trans_amount_df.pkl")


# In[81]:


pd.read_pickle("artifacts/imp_trans_amount_df.pkl")


# In[88]:


joblib.dump(xgb_reg_model,'artifacts/xgb_reg_model.pkl')
joblib.dump(xgb_clf_model, 'artifacts/xgb_clf_model.pkl')



# In[89]:


model = joblib.load('artifacts/xgb_reg_model.pkl')


# In[90]:


model.predict(X)


# In[92]:


predictions_df.sort_values(by='pred_trans', ascending=False)


# In[1]:


predictions_df.loc[(predictions_df['recency'] > -90) & (predictions_df['pred_prob'] < 0.20)].sort_values(by='pred_prob', ascending=False)


# In[97]:


predictions_df\
    [predictions_df['trans_90_total']==0.0]\
    .sort_values('pred_trans',ascending= False)

