from lxml import html
import requests
import ipdb
import time

from selenium import webdriver
from bs4 import BeautifulSoup as bs

import os
import urllib.request

driver = webdriver.Chrome('./chromedriver')

driver.get("https://www.data.gouv.fr/en/datasets/donnees-de-laboratoires-infra-departementales-durant-lepidemie-covid-19/")

# Maximize the window and let code stall
# for 10s to properly maximise the window.
driver.maximize_window()
time.sleep(10)

# # Obtain button by link text and click.
# button = driver.find_element_by_link_text("Sign In")
# button.click()

dynamic_button_xpath = "//a[starts-with(@href, 'u[update-additional]?')]"
# driver.find_element_by_xpath(dynamic_button_xpath).click()
driver.find_element_by_css_selector('.btn.btn-secondary.accordion-button.trigger-once.mt-md').click()

# html = driver.page_source
soup = bs(driver.page_source, 'lxml')
rows = soup.select('.card.resource-card')

# dates = ['2021-09','2021-08-01','2021-07-01','2021-06-01','2021-05-01','2021-04-01', '2021-03-01', '2021-02-01', '2021-01-01', '2020-12-01', '2020-11-01', '2020-10-01', '2020-09', '2020-08', '2020-07', '2020-06']

dates = ['2020-12', '2020-11', '2020-10', '2020-09', '2020-08', '2020-07', '2020-06', '2020-05']

iris_links = []
found = []

for row in rows:
    sel = row.select('dl.description-list > div > dd > a')
    links = [a['href'] for a in sel]
    # link = sel.select('a')
    link = links[0]
    if 'iris' in link:
        for date in dates:
            if date in link and date not in found:
                iris_links.append(link)
                path_file = link.split("/")[-1]
                if not os.path.isfile("../data/"+path_file):
                    with urllib.request.urlopen(link) as f:
                        html = f.read().decode('utf-8')
                        fw = open("../data/"+path_file, "w")
                        fw.write(html)
                        fw.close()

                    found.append(date)
