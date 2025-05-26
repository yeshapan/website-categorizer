#to scrape website content

import requests #to send HTTP request
from bs4 import BeautifulSoup #to parse HTML and extract content

def scrape_website(url: str) -> str:
    #fetch + extract main text content from website
    try:
        response = requests.get(url, timeout=10) #try to connect to url (with 10 secs timeout)
        soup = BeautifulSoup(response.text, 'html.parser') #convert raw HTML to tree-like structure (for easy search/modification)

        #remove unwanted script/style
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
            #remove <script> , <style> , <noscript> tags and their contents coz they're code for browser instructions - no useful text in them

        #extract clean text
        text = ' '.join(soup.stripped_strings) #returns all visible text pieces stripped of extra spaces + combine into one long string
        return text

    except Exception as e: #failures such as site down, invalid url, timeout, etc
        print(f"[!] Failed to scrape {url}: {e}")
        return ""
