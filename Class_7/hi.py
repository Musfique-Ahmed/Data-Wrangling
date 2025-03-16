from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

website = 'https://ucam.uiu.ac.bd/Security/LogIn.aspx'
path = 'D:/0152330101/Driver/chromedriver.exe'


service = Service(path)

driver = webdriver.Chrome(service=service)
driver.get(website)

user_name = ""
password = ""

user_name_field = driver.find_element(By.ID, "logMain_UserName")
user_name_field.send_keys(user_name)
password_field = driver.find_element(By.ID, "logMain_Password")
password_field.send_keys(password)

login_button = driver.find_element(By.ID, "logMain_Button1")
login_button.click()

user = input("|Press Any key!")
driver.quit()