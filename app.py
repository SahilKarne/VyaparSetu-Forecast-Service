import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from prophet import Prophet
import pandas as pd
import holidays
from datetime import date
from dotenv import load_dotenv

load_dotenv()  # read .env

app = Flask(__name__)
CORS(app)

# ── MongoDB setup 
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME   = os.getenv("DB_NAME")
client    = MongoClient(MONGO_URI)
db        = client[DB_NAME]

# ── Holiday DataFrame 
def make_holidays_df():
    hols_in = holidays.CountryHoliday("IN")
    hols_us = holidays.CountryHoliday("US")
    df = pd.DataFrame(list((hols_in + hols_us).items()), columns=["ds","holiday"])
    df["ds"] = pd.to_datetime(df["ds"])
    return df

# ── Load & aggregate sales 
def load_timeseries(collection, user_field, user_id, product_id):
    cur = db[collection].find(
      {user_field: ObjectId(user_id), "product": ObjectId(product_id)},
      {"date":1, "quantity":1}
    )
    df = pd.DataFrame(list(cur))
    print(f"[load_timeseries] fetched {len(df)} documents from '{collection}' "f"for {user_field}={user_id}, product={product_id}:")
    print(df)
    if df.empty:
        return pd.DataFrame(columns=["ds","y"])
    df["ds"] = pd.to_datetime(df["date"]).dt.floor("d")
    df = df.groupby("ds").quantity.sum().reset_index().rename(columns={"quantity":"y"})
    return df

# ── Forecast function 
def run_prophet(df, periods):
    m = Prophet(
      holidays=make_holidays_df(),
      yearly_seasonality=True,
      weekly_seasonality=True,
      daily_seasonality=False
    )
    if df.empty or len(df)<2:
        today = pd.Timestamp.today().floor("d")
        df = pd.DataFrame({
          "ds": pd.date_range(end=today, periods=60),
          "y": [0]*60
        })
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    fc = m.predict(future)
    return fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods)

# ── Seller endpoint 
@app.route("/forecast/seller", methods=["GET"])
def forecast_seller():
    seller_id  = request.args.get("sellerId")
    product_id = request.args.get("productId")
    days       = int(request.args.get("days",30))
    if not seller_id or not product_id:
        return jsonify(error="sellerId & productId required"),400
    df = load_timeseries("sales","seller",seller_id,product_id)
    fc = run_prophet(df, days)
    return jsonify([
      {"ds": row.ds.strftime("%Y-%m-%d"),
       "yhat": row.yhat, "yhat_lower": row.yhat_lower, "yhat_upper": row.yhat_upper}
      for _,row in fc.iterrows()
    ])

# ── Buyer endpoint 
@app.route("/forecast/buyer", methods=["GET"])
def forecast_buyer():
    retailer_id= request.args.get("retailerId")
    product_id = request.args.get("productId")
    days       = int(request.args.get("days",30))
    if not retailer_id or not product_id:
        return jsonify(error="retailerId & productId required"),400
    df = load_timeseries("retailersales","retailer",retailer_id,product_id)
    fc = run_prophet(df, days)
    return jsonify([
      {"ds": row.ds.strftime("%Y-%m-%d"),
       "yhat": row.yhat, "yhat_lower": row.yhat_lower, "yhat_upper": row.yhat_upper}
      for _,row in fc.iterrows()
    ])

if __name__=="__main__":
    # app.run(host="0.0.0.0", port=5000)
    # default to 5000 locally, override on Render via $PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


# http://127.0.0.1:5000/forecast/buyer?retailerId=680776179e5465030ada5519&productId=68078a777c1ac7df22405220&days=30

# http://127.0.0.1:5000/forecast/seller?sellerId=680789ef7c1ac7df2240521d&productId=68078a777c1ac7df22405220&days=30
