from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

website = 'https://ucam.uiu.ac.bd/'
path = "C:/Musfique's Folder/Python/Data-Wrangling/Class_6/Driver/chromedriver.exe"


service = Service(path)

driver = webdriver.Chrome(service=service)
driver.get(website)

login_button = driver.find_element(By.XPATH, '/html/body/form/div[3]/table/tbody/tr/td/div/div/div[2]/div/div[2]/input')
login_button.click()

# time.sleep(20)
input("Pressa any key to exit!")
driver.quit()
