import streamlit as st
import pandas as pd
from serpapi import GoogleSearch
from datetime import timedelta
from transformers import pipeline
import logging
import altair as alt

api_key = st.secrets["serpapi_api_key"]

topic_token_map = {
    "Business": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
    "Entertainment": "CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB",
    "Health": "CAAqIQgKIhtDQkFTRGdvSUwyMHZNR3QwTlRFU0FtVnVLQUFQAQ",
    "Science": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB",
    "Sports": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp1ZEdvU0FtVnVHZ0pWVXlnQVAB",
    "Technology": "CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB",
}

@st.cache_data(ttl=timedelta(days=1))
def fetch_news(country="us", language="en", category="Business"):
    params = {
        "engine": "google_news",
        "gl": country,
        "hl": language,
        "api_key": api_key,
        "topic_token": topic_token_map.get(category, topic_token_map["Business"]),
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    try:
        news_results = results["news_results"]
    except KeyError:
        logging.error("No news results found.")
        return []
    logging.info(f"Fetched {len(news_results)} news articles from Google News API.")
    return news_results 

def format_news_data(news_data):
    formatted_data = []
    for article in news_data:
        if "stories" not in article:
            formatted_article = {
                "title": article.get("title"),
                "source": article.get("source", {}).get("name"),
            }
            formatted_data.append(formatted_article)
        else:
            for story in article["stories"]:
                formatted_story = {
                    "title": story.get("title"),
                    "source": story.get("source", {}).get("name"),
                }
                formatted_data.append(formatted_story)
    df = pd.DataFrame(formatted_data)
    return df

@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_pipeline():
    """
    Load and cache the sentiment analysis pipeline.
    """
    logging.info("Initializing sentiment analysis model: distilbert-base-uncased-finetuned-sst-2-english")
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_data(ttl=timedelta(days=7))
def score_sentiment(df, _sentiment_pipeline):
    """
    Score the sentiment of the news articles using a sentiment analysis model from Hugging Face.
    """
    sentiments = _sentiment_pipeline(df["title"].tolist())

    # Process sentiments: 'POSITIVE' scores are positive, 'NEGATIVE' scores are negative.
    # The score from the model is a confidence value (0 to 1).
    processed_scores = []
    for s in sentiments:
        if s["label"] == "POSITIVE":
            processed_scores.append(s["score"])
        elif s["label"] == "NEGATIVE":
            processed_scores.append(-s["score"])
        else: # Should ideally not happen with this model for binary sentiment
            processed_scores.append(0) 
    df["positivity_score"] = processed_scores
    
    # standardize the positivity score
    #df["positivity_score"] = (df["positivity_score"] - df["positivity_score"].mean()) / df["positivity_score"].std()
    return df

def main():
    st.set_page_config(
        page_title="Some Good News", # center title
        page_icon=":newspaper:", # set icon
        initial_sidebar_state="expanded", # set sidebar to expanded
        layout="wide", # set layout to wide
    )
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the sentiment pipeline once
    sentiment_model = get_sentiment_pipeline()
    
    with st.sidebar:
        st.header("Filter News")
        country = st.selectbox("Country", ["us", "gb", "ca", "au"])
        language = st.selectbox("Language", ["en", "es", "fr", "de"])
        category = st.selectbox("Category", ["Business", "Entertainment", "Health", "Science", "Sports", "Technology"])
    
    st.title(":newspaper: Some Good News")

    if st.button("Fetch News"):
        with st.spinner("Fetching news articles..."):
            news_data = fetch_news(country=country, language=language, category=category)
        if not news_data:
            st.error("No news articles found.")
            return
        st.expander("View JSON Data").json(news_data, expanded=True)
        df = format_news_data(news_data)
        with st.spinner("Scoring sentiment in news articles..."):
            df = score_sentiment(df, sentiment_model)
        df = df.sort_values(by="positivity_score", ascending=False)

        st.write("### Top News Articles")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Bar chart for positivity score
        st.write("### Positivity Score by Source")
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("source", sort="-y"),
            y="mean(positivity_score)",
            tooltip=["source", "mean(positivity_score)"]
        ).properties(title="Average Positivity Score by Source"), use_container_width=True)
        # Plot number of articles by source
        st.write("### Number of Articles by Source")
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            x=alt.X("source", sort="-y"),
            y="count()",
            tooltip=["source", "count()"]
        ).properties(title="Number of Articles by Source"), use_container_width=True)
       
        
if __name__ == "__main__":
    main()
