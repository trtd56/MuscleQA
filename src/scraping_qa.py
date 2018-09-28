import re
import pandas as pd
from bs4 import BeautifulSoup
import requests

URL = 'http://weighttrainingfaq.org/wiki/index.php'
GET_INFO = '%A4%E8%A4%AF%A4%A2%A4%EB%BC%C1%CC%E4'
URL += '?' + GET_INFO
A_TXT_1 = 'トレーニング用語の場合は、まずはウェイトトレーニング用語辞典を見てみてください。'
A_TXT_2 = '\
・現在9月の場合：来年の夏まで1年弱。頑張れば多少は見栄えがするようになるかもしれません。\n\
・現在12月の場合：来年の夏まで、約半年。筋量増加による体型改善は難しいので、減量に励みましょう。\n\
・現在3月の場合：夏まで数ヶ月。まだ減量は間に合うかもしれません。\n\
・現在6月の場合：夏直前。来年の夏なら……'


def get_q_text(elem):
    txt = elem.text
    if txt[:2] == 'Q.':
        return txt[2:-3]
    return None


def get_a_text(elem):
    txt = elem.text
    if txt[:2] == 'A.':
        return txt[2:]
    return None


headers = {"User-Agent": "trtd"}
resp = requests.get(URL, timeout=3, headers=headers, verify=False)

soup = BeautifulSoup(resp.text, 'html5lib')

q_elem = soup.find_all(id=re.compile('content_*'))
a_elem = soup.find_all('p', class_='quotation')

q_txts = [get_q_text(q) for q in q_elem]
a_txts = [get_a_text(a) for a in a_elem]
q_txts = [q for q in q_txts if q is not None]
a_txts = [a for a in a_txts if a is not None]
q_ids = ['Q{0:04d}'.format(i) for i in range(len(q_txts))]
a_txts.insert(0, A_TXT_1)
a_txts.insert(1, A_TXT_2)

qa_df = pd.DataFrame({'q_id': q_ids, 'q_txt': q_txts, 'a_txt': a_txts})
qa_df = qa_df.ix[:, ['q_id', 'q_txt', 'a_txt']]
qa_df.to_csv('../data/muscle_qa.csv', index=None)
