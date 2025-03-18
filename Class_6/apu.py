import requests
 # Define API endpoint
url = "https://api.openweathermap.org/data/2.5/weather"
params = {"q": "Singapore", "appid": "YOUR_API_KEY"}
# Make GET request
response = requests.get(url, params=params)
# Convert response to JSON
data = response.json()
# Print fetched data
print(data)