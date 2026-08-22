from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException
import os

STREAMLIT_URL = os.environ.get("STREAMLIT_APP_URL", "본인_앱_URL")

def main():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(STREAMLIT_URL)
        wait = WebDriverWait(driver, 15)
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(),'Yes, get this app back up')]")))
            button.click()
            print("깨우기 버튼 클릭 완료 ✅")
        except TimeoutException:
            print("이미 깨어있음 ✅")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
