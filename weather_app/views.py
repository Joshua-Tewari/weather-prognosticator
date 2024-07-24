from django.shortcuts import render
import csv
import requests
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def index(request):
    api_key = '49cc8c821cd2aff9af04c9f98c36eb74'
    current_weather_url = 'https://api.openweathermap.org/data/2.5/weather?q={}&appid={}'
    forecast_url = 'https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&exclude=current,minutely,hourly,alerts&appid={}'

    if request.method == 'POST':
        city = request.POST['city']

        
        weather_data, daily_forecasts = fetch_weather_and_forecast(city, api_key, current_weather_url, forecast_url)

        
        model = train_model_from_file()

        
        next_five_days_forecast = predict_next_five_days(model)

        context = {
            'weather_data': weather_data,
            'daily_forecasts': daily_forecasts,
            'next_five_days_forecast': next_five_days_forecast
        }

        return render(request, 'weather_app/index.html', context)
    else:
        return render(request, 'weather_app/index.html')


def fetch_weather_and_forecast(city, api_key, current_weather_url, forecast_url):
    response = requests.get(current_weather_url.format(city, api_key)).json()
    lat, lon = response['coord']['lat'], response['coord']['lon']
    forecast_response = requests.get(forecast_url.format(lat, lon, api_key)).json()

    
    with open('weather_data.csv', 'a', newline='') as csv_file:  # Open in append mode
        writer = csv.writer(csv_file)
        writer.writerow(['city', 'temperature', 'description', 'icon'])
        writer.writerow([city,
                         round(response['main']['temp'] - 273.15, 2),
                         response['weather'][0]['description'],
                         response['weather'][0]['icon']])
        writer.writerow([])  # Add an empty row as a separator

        writer.writerow(['day', 'min_temp', 'max_temp', 'description', 'icon'])
        for daily_data in forecast_response['daily'][:5]:
            writer.writerow([datetime.datetime.fromtimestamp(daily_data['dt']).strftime('%A'),
                             round(daily_data['temp']['min'] - 273.15, 2),
                             round(daily_data['temp']['max'] - 273.15, 2),
                             daily_data['weather'][0]['description'],
                             daily_data['weather'][0]['icon']])

    
    weather_data = {
        'city': city,
        'temperature': round(response['main']['temp'] - 273.15, 2),
        'description': response['weather'][0]['description'],
        'icon': response['weather'][0]['icon'],
    }

    daily_forecasts = []
    for daily_data in forecast_response['daily'][:5]:
        daily_forecasts.append({
            'day': datetime.datetime.fromtimestamp(daily_data['dt']).strftime('%A'),
            'min_temp': round(daily_data['temp']['min'] - 273.15, 2),
            'max_temp': round(daily_data['temp']['max'] - 273.15, 2),
            'description': daily_data['weather'][0]['description'],
            'icon': daily_data['weather'][0]['icon'],
        })

    return weather_data, daily_forecasts


def preprocess_weather_data(weather_data):
    
    X = np.array([weather_data['temperature']]).reshape(-1, 1)
    y = np.array([weather_data['temperature']])  # Target variable (temperature)
    return X, y


def train_model_from_file():
    
    with open('weather_data.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)

    
    current_weather = {
        'city': data[1][0],
        'temperature': float(data[1][1]),
        'description': data[1][2],
        'icon': data[1][3],
    }

    
    forecast_data = data[4:]

    
    X_train, y_train = preprocess_weather_data(current_weather)
    model = GradientBoostingRegressor()  # Use Gradient Boosting Regressor
    model.fit(X_train, y_train)
    return model


def predict_next_five_days(model):
    next_five_days_forecast = []
    for i in range(1, 6):
        predicted_temp = model.predict(np.array([[i]])) 
        next_five_days_forecast.append({
            'day': (datetime.datetime.now() + datetime.timedelta(days=i)).strftime('%A'),
            'temperature': round(predicted_temp[0], 2),  
        })
    return next_five_days_forecast
