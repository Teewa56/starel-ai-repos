import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import Dict, Any
from serpapi import GoogleSearch


class FetchFromNet:    
    API_URL = "https://tokari-core.onrender.com/api/v1/ai/chat-completion"
    API_KEY = os.getenv("API_KEY")
    GOOGLE_URL = "https://www.google.com"

    def get_keyword(user_prompt, self):
        prompt = f"""
        Help me get the key word from this user request that i can put in
        to google to get relevant information on the topic, 
        respond with only what i asked for. this  is the promt: {user_prompt}
        """
        headers = {"x-api-key": self.API_KEY}
        payload = {"prompt": prompt}
        response = requests.post(self.API_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        keyword = response.json().get('response', '')
        return keyword
    
    def search_google(user_prompt, self):
        keyword = self.get_keyword(user_prompt)
        params = {
            "q": keyword,
            "api-key": os.getenv("serp-api-key")
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        sources = []
        for result in results:
            link_elem = {"Text": result['title'], "Link": result['link']}
            sources.append(link_elem)
        return sources
    def search_websites_from_google(links):
        for link in links:
            #this will loop through all the websites and
            #scrape all the needed information and return
            #this is best for up to date information(RAG)
            print(link)