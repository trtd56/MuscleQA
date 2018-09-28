import re
import pandas as pd
from itertools import chain
from bs4 import BeautifulSoup
import requests

URL = 'http://weighttrainingfaq.org/wiki/index.php'
GET_INFO_1 = '%A4%E8%A4%AF%A4%A2%A4%EB%BC%C1%CC%E4'
GET_INFO_2 = '%A5%A6%A5%A7%A5%A4%A5%C8%A5%C8%A5%EC%A1%BC%A5%CB%A5%F3%A5%B0%CD%D1%B8%EC%BC%AD%C5%B5'
GET_INFO_3 = '%B8%BA%CE%CC%A1%A2%A5%C0%A5%A4%A5%A8%A5%C3%A5%C8'
GET_INFO_4 = '%B1%C9%CD%DC%A1%A2%A5%B5%A5%D7%A5%EA%A5%E1%A5%F3%A5%C8%2F%A5%D7%A5%ED%A5%C6%A5%A4%A5%F3'
GET_INFO_5 = '%A5%C8%A5%EC%A1%BC%A5%CB%A5%F3%A5%B0%CA%FD%CB%A1'
GET_INFO_6 = '%BD%C0%C6%F0%A1%A2%A5%B9%A5%C8%A5%EC%A5%C3%A5%C1'
GET_INFO_7 = '%A5%A4%A5%F3%A5%CA%A1%BC%A5%DE%A5%C3%A5%B9%A5%EB%A4%CE%A5%C8%A5%EC%A1%BC%A5%CB%A5%F3%A5%B0'
GET_INFO_8 = '%B6%BB%A4%CE%A5%C8%A5%EC%A1%BC%A5%CB%A5%F3%A5%B0'

STOP_PATTERN = '†|↑|A.|Q.'
MIN_LEN = 10


def get_text(get_info):
    headers = {"User-Agent": "trtd"}
    url = URL + '?' + get_info
    resp = requests.get(url, timeout=3, headers=headers, verify=False)
    soup = BeautifulSoup(resp.text, 'html5lib')
    elem = soup.find_all(id='body')[0]
    text = [line for line in elem.text.split('\n')]
    text = [re.sub(STOP_PATTERN, '', line) for line in text]
    text = list(chain.from_iterable([line.split('。') for line in text]))
    text = [line for line in text if len(line) > MIN_LEN]
    return text


texts = get_text(GET_INFO_1)
texts += get_text(GET_INFO_2)
texts += get_text(GET_INFO_3)
texts += get_text(GET_INFO_4)
texts += get_text(GET_INFO_5)
texts += get_text(GET_INFO_6)
texts += get_text(GET_INFO_7)
texts += get_text(GET_INFO_8)
df = pd.DataFrame({'text': texts})
df.to_csv('../data/muscle_text.csv', index=None)
