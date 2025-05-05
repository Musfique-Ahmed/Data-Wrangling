from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://coinmarketcap.com/?page=1"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

# print(soup)

# tags = soup.find_all("tr", class_="sc-902a96b4-0 kQBOjw")
tags = soup.find_all("tr", style="cursor:pointer")

data = {
    "Coin Name": [],
    "Price": [],
    "Market Cap": [],
    # "Volume(24h) in Dollar": [],
    # "Volume(24h)": [],
    "Cerculating Supply": []
}

for tag in tags:
    data["Coin Name"].append(tag.find("p", class_ = "sc-65e7f566-0 iPbTJf coin-item-name").text)
    data["Price"].append(tag.find("div", class_ = "sc-142c02c-0 lmjbLF").text)
    data["Market Cap"].append(tag.find("span", class_ = "sc-11478e5d-1 jfwGHx").text)
    # data["Volume(24h) in Dollar"].append(tag.find("p", class_ = "sc-71024e3e-0 fOLOxZ font_weight_500").text)
    # data["Volume(24h)"].append(tag.find("p", class_ = "sc-71024e3e-0 cuNNJt").text)
    data["Cerculating Supply"].append(tag.find("div", class_ = "circulating-supply-value").text)

#     print(tag)
#     print()
#     print("\n")

# print(tags[0].td.find())#.find("p", class_ = "sc-65e7f566-0 iPbTJf coin-item-name"))
# print(tags[0].find("p", class_ = "sc-65e7f566-0 iPbTJf coin-item-name").text)

print(data)