import trafilatura
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import json
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)

def get_bbc_headlines():
    """Get the latest headlines from BBC News"""
    try:
        url = "https://www.bbc.com/news"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch BBC News: Status code {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Find headline articles
        articles = soup.select('h3.gs-c-promo-heading__title')
        for article in articles[:10]:  # Get top 10 headlines
            headlines.append(article.text.strip())
            
        if not headlines:
            logger.warning("No BBC headlines found")
            
        return headlines
    except Exception as e:
        logger.error(f"Error fetching BBC headlines: {str(e)}")
        return None

def get_reuters_headlines():
    """Get the latest headlines from Reuters"""
    try:
        url = "https://www.reuters.com/world/"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch Reuters: Status code {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Find headline articles
        articles = soup.select('h3.text__heading')
        for article in articles[:10]:  # Get top 10 headlines
            headlines.append(article.text.strip())
            
        if not headlines:
            logger.warning("No Reuters headlines found")
            
        return headlines
    except Exception as e:
        logger.error(f"Error fetching Reuters headlines: {str(e)}")
        return None

def get_all_headlines():
    """Get headlines from multiple sources"""
    headlines = {}
    
    bbc_headlines = get_bbc_headlines()
    if bbc_headlines:
        headlines["BBC News"] = bbc_headlines
        
    reuters_headlines = get_reuters_headlines()
    if reuters_headlines:
        headlines["Reuters"] = reuters_headlines
    
    sports_headlines = get_espn_sports_news()
    if sports_headlines:
        headlines["ESPN Sports"] = sports_headlines
    
    return headlines

def format_headlines_for_context(headlines):
    """Format headlines as context for AI prompt"""
    if not headlines:
        return "Unable to fetch current headlines. Please try again later."
        
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    context = f"Today's date is {current_date}. Here are today's top news headlines:\n\n"
    
    for source, source_headlines in headlines.items():
        context += f"## {source}:\n"
        for i, headline in enumerate(source_headlines, 1):
            context += f"{i}. {headline}\n"
        context += "\n"
        
    return context

def get_espn_sports_news():
    """Get the latest sports headlines from ESPN"""
    try:
        url = "https://www.espn.com/"
        response = requests.get(url, timeout=10, 
                               headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch ESPN: Status code {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = []
        
        # Find headline articles
        articles = soup.select('.contentItem__title')
        for article in articles[:8]:  # Get top 8 headlines
            headlines.append(article.text.strip())
            
        if not headlines:
            logger.warning("No ESPN headlines found")
            
        return headlines
    except Exception as e:
        logger.error(f"Error fetching ESPN sports news: {str(e)}")
        return None

def search_web(query, num_results=5):
    """
    Search the web for results using a combination of sources
    """
    logger.info(f"Performing web search for: {query}")
    
    try:
        # URL encode the query
        encoded_query = quote(query)
        
        # Use a search API (here we'll use a public search endpoint)
        url = f"https://ddg-api.herokuapp.com/search?q={encoded_query}&limit={num_results}"
        
        response = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
            timeout=15
        )
        
        if response.status_code == 200:
            results = response.json()
            return results
        else:
            # Fallback to direct scraping
            return scrape_search_results(query, num_results)
    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        try:
            # Try backup scraping method if the API failed
            return scrape_search_results(query, num_results)
        except Exception as e2:
            logger.error(f"Backup search also failed: {str(e2)}")
            return None

def scrape_search_results(query, num_results=5):
    """
    Fallback method to scrape search results directly
    """
    try:
        encoded_query = quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        response = requests.get(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml'
            },
            timeout=15
        )
        
        results = []
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            result_elements = soup.select('.result')
            
            for i, result in enumerate(result_elements):
                if i >= num_results:
                    break
                    
                title_elem = result.select_one('.result__title')
                snippet_elem = result.select_one('.result__snippet')
                url_elem = result.select_one('.result__url')
                
                if title_elem and snippet_elem:
                    title = title_elem.text.strip()
                    snippet = snippet_elem.text.strip()
                    url = url_elem.text.strip() if url_elem else ""
                    
                    results.append({
                        "title": title,
                        "body": snippet,
                        "href": url
                    })
            
            return results
        else:
            logger.error(f"Search scraping failed with status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error scraping search results: {str(e)}")
        return None

def format_search_results_for_context(query, results):
    """Format search results as context for AI prompt"""
    if not results or len(results) == 0:
        return f"A web search for '{query}' was performed but no relevant results were found."
        
    context = f"Here are the top search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        snippet = result.get('body', result.get('snippet', 'No description available'))
        url = result.get('href', result.get('url', 'No URL available'))
        
        context += f"{i}. {title}\n"
        context += f"   {snippet}\n"
        context += f"   Source: {url}\n\n"
        
    return context

def get_mlb_scores(date=None):
    """
    Get the latest MLB scores or scores for a specific date
    
    Args:
        date (str, optional): Date in YYYY-MM-DD format. If None, gets today's scores.
        
    Returns:
        dict: Dictionary with game information and scores
    """
    try:
        # Use current date if none provided
        if not date:
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")
            
        url = f"https://www.mlb.com/scores/{date}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch MLB scores: Status code {response.status_code}")
            return None
            
        # Extract the text content using trafilatura for better parsing
        text_content = trafilatura.extract(response.text)
        
        if not text_content:
            # Fallback to BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for game information
            games_info = []
            game_elements = soup.select('.game-card')
            
            for game in game_elements:
                try:
                    teams = game.select('.team-name')
                    team_names = [team.text.strip() for team in teams]
                    
                    scores = game.select('.score')
                    score_values = [score.text.strip() for score in scores]
                    
                    status_elem = game.select_one('.game-status')
                    status = status_elem.text.strip() if status_elem else "Unknown"
                    
                    if len(team_names) >= 2 and len(score_values) >= 2:
                        games_info.append({
                            'away_team': team_names[0],
                            'home_team': team_names[1],
                            'away_score': score_values[0],
                            'home_score': score_values[1],
                            'status': status
                        })
                except Exception as e:
                    logger.error(f"Error parsing MLB game card: {str(e)}")
            
            return {
                'date': date,
                'source': url,
                'games': games_info
            }
        
        # If trafilatura extracted content, format it nicely
        return {
            'date': date,
            'source': url,
            'text_content': text_content
        }
    
    except Exception as e:
        logger.error(f"Error fetching MLB scores: {str(e)}")
        return None

def is_sports_score_query(prompt):
    """
    Check if the prompt is asking for sports scores, especially MLB
    
    Args:
        prompt (str): The user prompt
        
    Returns:
        bool: True if the prompt is asking for sports scores
    """
    # Patterns that indicate the user is asking for sports scores
    sports_patterns = [
        r'(mlb|baseball) scores?',
        r'what (were|was|are|is) the (mlb|baseball) scores?',
        r'show me (mlb|baseball) scores?',
        r'sports scores?',
        r'(baseball|mlb) (results|game|match|outcome)',
        r'(who won|score) (the|last night\'s) (baseball|mlb) game',
        r'how did the [\w\s]+ (play|do|perform)',
        r'what was the score of the [\w\s]+ game'
    ]
    
    # Case insensitive check for sports-related terms
    for pattern in sports_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return True
            
    return False

def format_mlb_scores_for_display(mlb_data):
    """
    Format MLB scores data for display in the AI response
    
    Args:
        mlb_data (dict): MLB scores data from get_mlb_scores()
        
    Returns:
        str: Formatted MLB scores text
    """
    if not mlb_data:
        return "Unable to fetch MLB scores. Please try again later."
    
    date = mlb_data.get('date', 'Today')
    games = mlb_data.get('games', [])
    text_content = mlb_data.get('text_content')
    
    # If we have the parsed text content from trafilatura, use it directly
    if text_content:
        # Clean up the text content to make it more readable
        text_content = re.sub(r'\n{3,}', '\n\n', text_content)
        return f"MLB Scores for {date}:\n\n{text_content}"
        
    # Otherwise format from the structured data
    if not games:
        return f"No MLB games found for {date}."
    
    formatted_text = f"# MLB Scores for {date}\n\n"
    
    for game in games:
        away_team = game.get('away_team', 'Unknown')
        home_team = game.get('home_team', 'Unknown')
        away_score = game.get('away_score', '?')
        home_score = game.get('home_score', '?')
        status = game.get('status', 'Unknown')
        
        formatted_text += f"## {away_team} @ {home_team}\n"
        formatted_text += f"**Score**: {away_team} {away_score} - {home_team} {home_score}\n"
        formatted_text += f"**Status**: {status}\n\n"
    
    formatted_text += f"Source: [MLB.com]({mlb_data.get('source', 'https://www.mlb.com/scores/')})"
    
    return formatted_text