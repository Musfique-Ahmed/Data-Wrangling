from bs4 import BeautifulSoup
import requests

html_text = requests.get("http://quotes.toscrape.com/").text

soup = BeautifulSoup(html_text, "lxml")
# print(soup.prettify())
# print(soup)

qoutes = soup.find_all("div", class_="quote")
# print(qoutes)
for qoute in qoutes:
    content = qoute.find_all("span")#, class_="text")
    # aouthor = qoute.find("span", class_="author")
    # print(content[0].text, end=" - ")
    # print(aouthor)
    # print("\n")
    # print(content[1].text.replace("(about)", ""))
    # print(content)
    a = content[1].find("a",href = True)
    # print(a["href"])
    html_2 = requests.get(f"https://quotes.toscrape.com{a["href"]}").text
    soup = BeautifulSoup(html_2, "lxml")
    # print(soup.prettify())
    # print(soup)
    details = soup.find("div", class_ = "author-details")
    dob = details.find("span", class_="author-born-date").text
    lob = details.find("span", class_="author-born-location").text
    print(dob)
    print(lob)
