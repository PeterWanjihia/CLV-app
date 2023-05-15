import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import joblib

# Load the trained models and feature importance
xgb_reg_model = joblib.load('artifacts/xgb_reg_model.pkl')
xgb_clf_model = joblib.load('artifacts/xgb_clf_model.pkl')
imp_trans_amount_df = pd.read_pickle('artifacts/imp_trans_amount_df.pkl')
imp_trans_prob_df = pd.read_pickle('artifacts/imp_trans_prob_df.pkl')

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(
    children=[
        html.H1("Customer Lifetime Value Model", style={"text-align": "center"}),
        html.Div(
            children=[
                html.H3("Enter Customer Features:"),
                dcc.Input(id="recency-input", type="number", placeholder="Recency (days)"),
                dcc.Input(id="frequency-input", type="number", placeholder="Frequency"),
                dcc.Input(id="trans-amount-sum-input", type="number", placeholder="Total Transaction Amount"),
                dcc.Input(id="trans-amount-mean-input", type="number", placeholder="Average Transaction Amount"),
                html.Button("Calculate", id="calculate-button", n_clicks=0),
            ],
            style={"text-align": "center", "margin-bottom": "20px"},
        ),
        html.Div(id="clv-output"),
        html.Div(
            children=[
                html.H3("Feature Importance - Transaction Amount"),
                dcc.Graph(
                    id="trans-amount-importance",
                    figure={
                        "data": [
                            {
                                "x": imp_trans_amount_df["value"],
                                "y": imp_trans_amount_df["feature"],
                                "type": "bar",
                                "orientation": "h",
                            }
                        ],
                        "layout": {
                            "title": "Feature Importance - Transaction Amount",
                            "xaxis_title": "Importance",
                            "yaxis_title": "Feature",
                        },
                    },
                ),
            ],
            style={"width": "50%", "display": "inline-block", "vertical-align": "top"},
        ),
        html.Div(
            children=[
                html.H3("Feature Importance - Transaction Probability"),
                dcc.Graph(
                    id="trans-prob-importance",
                    figure={
                        "data": [
                            {
                                "x": imp_trans_prob_df["value"],
                                "y": imp_trans_prob_df["feature"],
                                "type": "bar",
                                "orientation": "h",
                            }
                        ],
                        "layout": {
                            "title": "Feature Importance - Transaction Probability",
                            "xaxis_title": "Importance",
                            "yaxis_title": "Feature",
                        },
                    },
                ),
            ],
            style={"width": "50%", "display": "inline-block", "vertical-align": "top"},
        ),
    ]
)

# Callback function to calculate CLV and display the results
@app.callback(
    Output("clv-output", "children"),
    Input("calculate-button", "n_clicks"),
    [
        Input("recency-input", "value"),
        Input("frequency-input", "value"),
        Input("trans-amount-sum-input", "value"),
        Input("trans-amount-mean-input", "value"),
    ],
)
def calculate_clv(n_clicks, recency, frequency, trans_amount_sum, trans_amount_mean):
    if n_clicks > 0:
        # Create the input dataframe for prediction
        input_data = pd.DataFrame(
            {
                "recency": [recency],
                "frequency": [frequency],
                "transAmount_sum": [trans_amount_sum],
                "transAmount_mean": [trans_amount_mean],
            }
        )
        
        # Make predictions
        try:
            clv_prediction = xgb_reg_model.predict(input_data)
            prob_prediction = xgb_clf_model.predict_proba(input_data)
        except Exception as e:
            print(f"Error occurred during prediction: {e}")

        
        # Format the output
        output = html.Div(
            children=[
                html.H3("CLV Prediction"),
                html.P(f"Predicted Customer Lifetime Value: ${clv_prediction[0]:,.2f}"),
                html.H3("Transaction Probability"),
                html.P(f"Predicted Probability of Transaction: {prob_prediction[0][1]:.2%}"),
            ],
            style={"text-align": "center"},
        )
        return output
    else:
        return None

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)

