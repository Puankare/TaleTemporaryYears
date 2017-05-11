# -*- coding: utf-8 -*-

import os
from requests import post
from bs4 import BeautifulSoup
from re import sub, search

#reload(sys)
#sys.setdefaultencoding("utf-8")

PROJECT_URL = 'http://expositions.nlr.ru/LaurentianCodex/_Project/page_Show.php'

DEFAULT_STRING = u'Нет перевода'

PATH_DIR = 'src'
FILENAME = 'LavrentPaper'


def extract_text(list_curr):

    params = {'SelectListIzo': list_curr + 1, 'ListCurrent': list_curr,
              "n1": "on", "n2": "on", "n3": "on"}

    #params = {'ListCurrent': list_curr}

    page_data = post(PROJECT_URL, params=params)

    page_soup = BeautifulSoup(page_data.text, "lxml")

    try:
        page_text_struct = page_soup.select("#trlat")

        page_text = "\n".join([item.getText() for item in page_text_struct])
    except (TypeError, IndexError, AttributeError), e:
        print e.message
        return None

    page_text_no_delim = " ".join(sub(' +', ' ', page_text).splitlines())

    if search(DEFAULT_STRING, str(page_text_no_delim)):
        return None
    else:
        return page_text_no_delim


if __name__ == '__main__':

    cur_dir = os.path.abspath(os.path.curdir)

    file_path = os.path.join(cur_dir, PATH_DIR, FILENAME)

    with open(file_path, 'w') as logfile:

        for page in xrange(1, 200):

            if page % 10 == 0:
                print("Current Page -- {0}".format(page))

            page_data = extract_text(page)

            if page_data:
                logfile.write(page_data.decode('utf-8','ignore').encode("utf-8"))
