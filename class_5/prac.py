from bs4 import BeautifulSoup

html_file = open("home.html", "r")
content = html_file.read()
# print(content)

soup = BeautifulSoup(content, "lxml")
courses = soup.find_all("h5", class_ ="card-title")
prices = soup.find_all("a", class_="btn btn-primary")
# print(courses)
# print(prices)
for i in range(len(courses)):
    print(courses[i].text, end=" ")
    print(int(prices[i].text.replace("Start for ", "").strip("$")))

# soup = BeautifulSoup(content, "lxml")
# courses = soup.find("h5", class_ ="card-title")
# print(courses.text)