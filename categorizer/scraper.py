#to scrape website content

import requests #to send HTTP request
from bs4 import BeautifulSoup #to parse HTML and extract content

def scrape_website(url: str) -> str:
    #fetch + extract main text content from website
    try:
        #define a User-Agent header to mimic a web browser - as advised by Mitesh sir
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10) #try to connect to url (with 10 secs timeout)
        response.raise_for_status()  #raise an HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, 'html.parser') #convert raw HTML to tree-like structure (for easy search/modification)

        #remove unwanted script/style
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
            #remove <script> , <style> , <noscript> tags and their contents coz they're code for browser instructions - no useful text in them

        #extract clean text
        text = ' '.join(soup.stripped_strings) #returns all visible text pieces stripped of extra spaces + combine into one long string
        return text

    except requests.exceptions.HTTPError as e:
        print(f"[!] HTTP Error scraping {url}: {e}")
        return ""
    except requests.exceptions.ConnectionError as e:
        print(f"[!] Connection Error scraping {url}: {e}")
        return ""
    except requests.exceptions.Timeout as e:
        print(f"[!] Timeout Error scraping {url}: {e}")
        return ""
    except requests.exceptions.RequestException as e: # Catch all other request exceptions
        print(f"[!] An unexpected Request error occurred while scraping {url}: {e}")
        return ""
    except Exception as e: #failures such as site down, invalid url, timeout, etc
        print(f"[!] Failed to scrape {url}: {e}")
        return ""
