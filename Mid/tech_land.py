from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://www.startech.com.bd/laptop-notebook"
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

# print(soup)

cards = soup.find_all("div", class_="short-description")

data = {
    "Processor": [],
    "Ram": [],
    "Display": [],
    "Features": []
}

for i in range(1,11):
    print(f"Scrapping page no: {i}.....")
    url = f"https://www.startech.com.bd/laptop-notebook?page={i}"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

# print(soup)

    cards = soup.find_all("div", class_="short-description")
    for card in cards:
        # print(card)
        # print(card.li.text)
        info = card.find_all("li")
        # print(info)
        # for inf in info:
        #     print(inf.text)
        data["Processor"].append(info[0].text.strip("Processor: "))
        data["Ram"].append(info[1].text.strip("RAM: "))
        data["Display"].append(info[2].text.strip("Display: "))
        data["Features"].append(info[3].text.strip("Features: "))
        

df = pd.DataFrame(data)
# print(df)
df.to_csv("C:/Musfique's Folder/Python/Data-Wrangling/Mid/tech_land.csv", index=False)