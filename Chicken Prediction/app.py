from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

df = pd.read_csv('chicken.csv', parse_dates=['Date'])

# Prepare data for the model
X_train = df[['Feed_Cost']]
y_train = df['Chicken_Price']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        feed_cost = float(request.form['feed_cost'])
        feed_cost_scaled = scaler.transform([[feed_cost]])
        predicted_price = model.predict(feed_cost_scaled)[0]
        return render_template('index.html', feed_cost=feed_cost, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
