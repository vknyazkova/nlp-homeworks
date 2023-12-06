import re
import requests
from typing import Callable, Iterable

from bs4 import BeautifulSoup


class LexicographyCrawler:
    link = 'https://lexicography.online/'
    dictionaries = {
        'mas': 'explanatory/mas',
        'efremova': 'explanatory/efremova'
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/119.0.0.0 Safari/537.36"
    }

    def __init__(self,
                 dictionary='efremova',
                 defin_parser: Callable[[str], Iterable[str]] = None,
                 headers=None):
        """
        Initialize the LexicographyCrawler.

        Args:
            dictionary (str): Selected dictionary for crawling.
            defin_parser (Callable): Parser function for extracting definitions from HTML.
            headers (dict): HTTP headers for making requests.
        """

        self.dictionary = dictionary
        self.defin_parser = defin_parser
        if headers:
            self.headers = headers
        else:
            self.headers = LexicographyCrawler.headers

    @property
    def dictionary(self):
        return self._dict

    @dictionary.setter
    def dictionary(self, val):
        if val in LexicographyCrawler.dictionaries:
            self._dict = LexicographyCrawler.link + LexicographyCrawler.dictionaries[val]
        else:
            raise ValueError(
                f'Unknown dictionary. Available dictionaries: {list(LexicographyCrawler.dictionaries.keys())}'
            )

    @property
    def defin_parser(self):
        return self._defin_parser

    @defin_parser.setter
    def defin_parser(self, parser):
        if not parser:
            dict_name = self.dictionary.split('/')[-1]
            try:
                self._defin_parser = getattr(self, f'_{dict_name}_parser')
            except AttributeError:
                raise NotImplementedError('Yet there is no default parser for this dictionary.')
        elif callable(parser):
            self._defin_parser = parser
        else:
            raise TypeError

    def _efremova_parser(self, html: str) -> Iterable[str]:
        soup = BeautifulSoup(html, 'html.parser')
        definitions = []
        for defin in soup.find('article').find_all('li'):
            if defin.i:
                defin.i.extract()
            text_match = re.match(r'(\d. )(.*)', defin.text).group(2)
            text_match = text_match.replace('// ', "")
            definitions.append(text_match)
        return definitions

    def _mas_parser(self, html: str) -> Iterable[str]:
        raise NotImplementedError('Я еще не умею парсить малый академический словарь')

    def _create_search_link(self, word) -> str:
        first_letter = word[0]
        return f'{self.dictionary}/{first_letter}/{word}'

    def search_definitions(self, word: str) -> Iterable[str]:
        response = requests.get(self._create_search_link(word), headers=self.headers)
        if response.ok:
            definitions = self.defin_parser(response.text)
        elif response.status_code == 404:
            raise ValueError('This word does not have entry in selected dictionary')
        else:
            response.raise_for_status()
        return definitions

