# Fan Sentiment Analyzer

## Overview
This project analyzes fan sentiment for Arsenal FC by processing discussions from the [r/Gunners](https://www.reddit.com/r/Gunners/) subreddit. It correlates daily sentiment scores with actual match results to visualize how team performance impacts fan mood.

## Features
- **Reddit Scraping**: Fetches "Post Match", "Full Time", and "Daily Discussion" threads using the Pushshift/Photon API.
- **Sentiment Analysis**: Utilizes the `cardiffnlp/twitter-roberta-base-sentiment-latest` model from Hugging Face to score comments.
- **Weighted Aggregation**: Calculates daily sentiment scores weighted by comment upvotes.
- **Match Data Integration**: Scrapes match results (scores, opponents, competitions) from Soccerbase.
- **Visualization**: Generates time-series plots of sentiment overlaid with match results (Wins, Draws, Losses).

## Requirements
- Python 3.x
- Jupyter Notebook
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `requests`
- `beautifulsoup4`
- `transformers`
- `torch` (or `tensorflow` depending on the transformers backend)
- `scipy`


## Usage
1.  Clone the repository.
2.  Install the required dependencies.
3.  Run the `fan_sentiment_analyzer.ipynb` notebook.
    -   The notebook will fetch Reddit threads for the specified date range.
    -   It will download and cache the sentiment model.
    -   It will scrape match data.
    -   Finally, it will display a graph showing sentiment trends and match outcomes.

## Acknowledgements
-   Data provided by Reddit and Soccerbase.
-   Sentiment model by Cardiff NLP.

  <img width="4157" height="2068" alt="Arsenal_sentiment_analysis" src="https://github.com/user-attachments/assets/81f0d86d-5f8d-48b5-9ea5-8a0e6cdb0798" />
<img width="4160" height="2068" alt="Tottenham_sentiment_analysis" src="https://github.com/user-attachments/assets/a1d0fe21-b22c-444d-a6e3-f9eafca24729" />
