from bs4 import BeautifulSoup

html_file=  open("home.html", "r")
content = html_file.read()
soup = BeautifulSoup(content, 'lxml')
print(soup.prettify())

tags = soup.find('h5') # find the first h5 tag
tags = soup.find_all('h5') # find all h5 tags
print(tags)