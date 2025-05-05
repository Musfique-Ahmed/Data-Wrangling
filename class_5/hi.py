from bs4 import BeautifulSoup
import requests

html_text = requests.get("http://quotes.toscrape.com/").text

soup = BeautifulSoup(html_text, "lxml")
# print(soup.prettify())
quotes = soup.find_all("div", class_ = "quote")
# print(quotes)

for qoute in quotes:
    # print(qoute)
    demo = qoute.find_all("span", class_="text").get_text()
    print(demo)