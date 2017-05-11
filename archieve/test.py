import scrapy
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://blog.scrapinghub.com']

    def parse(self, response):
        for title in response.css('h2.entry-title'):
            yield {'title': title.css('a ::text').extract_first()}

        next_page = response.css('div.prev-post > a ::attr(href)').extract_first()
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse)














import sys
from requests import post
from bs4 import BeautifulSoup
from re import sub

reload(sys)
sys.setdefaultencoding("utf-16")


PROJECT_URL = 'http://expositions.nlr.ru/LaurentianCodex/_Project/page_Show.php'

DEFAULT_NULL = unicode(u'Нет перевода')


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

    return page_text_no_delim

#!/opt/anaconda2/bin/python
#
# -*- coding: utf-8 -*-
print extract_text(121)