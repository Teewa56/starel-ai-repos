import requests
import os
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import urljoin, urlparse
import re

class FetchFromNet:
    API_URL = "https://tokari-core.onrender.com/api/v1/ai/chat-completion"
    API_KEY = os.getenv("API_KEY")

    def get_keyword(self, user_prompt):
        """Extract keywords from user prompt for better search results"""
        prompt = f"""
        Help me get the key word from this user request that i can put in
        to search engines to get relevant information on the topic, 
        respond with only what i asked for. This is the prompt: {user_prompt}
        """
        headers = {"x-api-key": self.API_KEY}
        payload = {"prompt": prompt}
        
        try:
            response = requests.post(self.API_URL, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            keyword = response.json().get('response', '').strip()
            return keyword if keyword else user_prompt  # Fallback to original prompt
        except Exception as e:
            print(f"Error getting keyword: {e}")
            return user_prompt  # Fallback to original prompt

    def search_duckduckgo(self, user_prompt):
        """Search DuckDuckGo for relevant information"""
        keyword = self.get_keyword(user_prompt)
        url = "https://api.duckduckgo.com/"
        params = {
            'q': keyword,
            'format': 'json',
            'no_redirect': '1',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            sources = []
            
            # Get results from RelatedTopics
            for result in data.get('RelatedTopics', [])[:5]:
                if isinstance(result, dict) and 'Text' in result and 'FirstURL' in result:
                    sources.append({
                        "Text": result['Text'][:150] + '...' if len(result['Text']) > 150 else result['Text'],
                        "Link": result['FirstURL']
                    })
            
            # If no RelatedTopics, try Abstract
            if not sources and data.get('Abstract'):
                sources.append({
                    "Text": data['Abstract'][:150] + '...' if len(data['Abstract']) > 150 else data['Abstract'],
                    "Link": data.get('AbstractURL', '')
                })
            
            # If still no results, try Answer
            if not sources and data.get('Answer'):
                sources.append({
                    "Text": data['Answer'][:150] + '...' if len(data['Answer']) > 150 else data['Answer'],
                    "Link": data.get('AnswerURL', '')
                })
            
            return sources
            
        except Exception as e:
            print(f"Error searching DuckDuckGo: {e}")
            return []

    def scrape_website_content(self, url, max_chars=2000):
        """Scrape content from a single website"""
        try:
            # Add random delay to be respectful
            time.sleep(random.uniform(1, 2))
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'menu']):
                element.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'article', 'main', '.content', '#content', '.post', '.article',
                '.entry-content', '.post-content', '.article-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no specific content area found, get body text
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text(separator=' ', strip=True)
            
            # Clean up the text
            content_text = re.sub(r'\s+', ' ', content_text)  # Remove extra whitespace
            content_text = content_text.strip()
            
            # Truncate if too long
            if len(content_text) > max_chars:
                content_text = content_text[:max_chars] + "..."
            
            return content_text
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def search_and_scrape(self, user_prompt, max_sites=3):
        """Search DuckDuckGo and scrape content from the results"""
        # Get search results
        search_results = self.search_duckduckgo(user_prompt)
        
        if not search_results:
            return []
        
        enriched_results = []
        scraped_count = 0
        
        for result in search_results:
            if scraped_count >= max_sites:
                break
                
            link = result.get('Link', '')
            if not link or not self._is_valid_url(link):
                continue
            
            print(f"Scraping: {link}")
            scraped_content = self.scrape_website_content(link)
            
            if scraped_content:
                enriched_results.append({
                    "Title": result.get('Text', 'No title'),
                    "Link": link,
                    "Content": scraped_content,
                    "Summary": result.get('Text', '')
                })
                scraped_count += 1
            else:
                # Keep the original result even if scraping failed
                enriched_results.append({
                    "Title": result.get('Text', 'No title'),
                    "Link": link,
                    "Content": "",
                    "Summary": result.get('Text', '')
                })
        
        return enriched_results

    def _is_valid_url(self, url):
        """Check if URL is valid and accessible"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ['http', 'https']
        except:
            return False

    def get_search_summary(self, user_prompt):
        """Get a concise summary of search results for RAG integration"""
        results = self.search_and_scrape(user_prompt, max_sites=2)
        
        if not results:
            return "No additional information found online."
        
        summary_parts = []
        for i, result in enumerate(results[:2], 1):
            content = result.get('Content', '')
            if content:
                # Get first 200 characters of content
                snippet = content[:200] + "..." if len(content) > 200 else content
                summary_parts.append(f"Source {i}: {snippet}")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "Search results found but content extraction failed."

    # Legacy method name for backward compatibility
    def search_google(self, user_prompt):
        """Legacy method - now uses DuckDuckGo"""
        return self.search_duckduckgo(user_prompt)

    def search_websites_from_google(self, links):
        """Legacy method - scrape content from provided links"""
        enriched_results = []
        
        for link_data in links:
            if isinstance(link_data, dict):
                url = link_data.get('Link', '')
            else:
                url = str(link_data)
            
            if url and self._is_valid_url(url):
                content = self.scrape_website_content(url)
                enriched_results.append({
                    "Link": url,
                    "Content": content
                })
        
        return enriched_results