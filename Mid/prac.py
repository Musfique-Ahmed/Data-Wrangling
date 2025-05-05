from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

website = 'https://elms.uiu.ac.bd/'
path = "C:/Musfique's Folder/Python/Data-Wrangling/Mid/chromedriver-win64/chromedriver.exe"

service = Service(path)

driver = webdriver.Chrome(service=service)
driver.get(website)

login_button = driver.find_element(By.XPATH, '//*[@id="usernavigation"]/div[2]/div/span/a')
login_button.click()

time.sleep(5)

username = driver.find_element(By.ID, "username")
username.send_keys("015230101")
username.submit()

login_button_2 = driver.find_element(By.ID, "loginbtn")
login_button_2.click()

# inp = input("Enter 'q' to quit: ")
# while inp !="q":
#     driver.quit()


time.sleep(20)
driver.quit()