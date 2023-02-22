import sys
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def runJob(args):
    print(r"Daily check up {}".format(time.time()))

    if len(args) > 1 and args[1] == "--wait":
        # do not close chrome
        webdriver.ChromeOptions().add_experimental_option("detach", True)

    # Start chrome
    wd = webdriver.Chrome()
    wd.implicitly_wait(1)
    wd.get("https://www.macz.com/")

    # 等待20秒登录按钮
    # WebDriverWait(driver, 超时时长, 调用频率, 忽略异常).until(可执行方法, 超时时返回的信息)
    WebDriverWait(wd, 20).until(EC.element_to_be_clickable((By.ID, "to-login"))).click()
    WebDriverWait(wd, 20).until(EC.visibility_of_element_located((By.ID, "login-wrap")))
    print("Login window showed up...")

    WebDriverWait(wd, 20).until(
        EC.element_to_be_clickable((By.ID, "container"))
    ).click()


if __name__ == "__main__":
    runJob(sys.argv)
