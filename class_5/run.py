from bs4 import BeautifulSoup
import requests
import pandas as pd

html_text = requests.get('https://quotes.toscrape.com/').text # Get the HTML content of the page
soup = BeautifulSoup(html_text, 'lxml') # Parse the HTML content
quotes = soup.find_all('div', class_ = "quote")


quotes_list = []
author_list = []


for quote in quotes:
    demo = quote.find_all('span')
    quotes_list.append(demo[0].get_text()[1:-1])
    author_list.append(demo[1].small.get_text())
    # link = 

df = pd.DataFrame({'Quote' : quotes_list, 'Author' : author_list})

df.to_csv('Quotes.csv', index=False)
