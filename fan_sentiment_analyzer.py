import os
import sys
import pickle
import time
import logging
import re
from transformers import AutoModelForSequenceClassification
# from transformers import TFAutoModelForSequenceClassification # Removed this line
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import datetime as dt
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup as bs
import torch


logging.basicConfig(
    filename= 'app.log',
    level=logging.INFO,
    force=True
)

def get_reddit_threads_url(subreddit,start_date,end_date,keyword):
    logging.info(f"getting data from {subreddit}")
    response = requests.get(f'https://arctic-shift.photon-reddit.com/api/posts/search?sort=asc&after={start_date}&before={end_date}&subreddit={subreddit}&title={keyword}&limit=25')
    sc = response.status_code
    res_json = response.json()
    res_data = res_json['data']
    url_list = []

    if res_data:
        for i in res_data:
            url_list.append(i["url"])

    return url_list,sc

def get_soccer_threads(start_date,subreddit,keywords):
    end_date = dt.datetime.now().date()
    results = []
    for k in keywords:
        print(k)
        win_start = start_date
        win_end = win_start + dt.timedelta(days=7)

        while win_start <= end_date:
            print(win_start, win_end)
            temp_results, status_code = get_reddit_threads_url(subreddit,win_start,win_end,k)
            for i in range(5):
                if status_code!= 200:
                    print(status_code)
                    time.sleep(30)
                    temp_results, status_code = get_reddit_threads_url(subreddit,win_start,win_end,k)
                else:
                    print(status_code)
                    break
            print(temp_results)
            results.extend(temp_results)
            win_start = win_end
            win_end = win_start + dt.timedelta(days=7)

    filter_results = []
    filter_phrases = "|".join(['post_match','postmatch','full_time','fulltime','ft','daily_discussion'])
    for r in results:
        if re.search(filter_phrases,r) and re.search("reddit",r) :
            pattern = re.escape("?") + r'.*'
            new_s =  re.sub(r'/'+re.escape("?") + r'.*' + '|' + '/$',r".json?sort=confidence&limit=500",r)
            filter_results.append(new_s)

    filter_results = list(set(filter_results))

    return filter_results

def get_reddit_comments(json_url):
    headers = {
        'User-Agent': 'pc:fan_sentiment_analyzer:v1.0 (by /u/alltakenistaken)'
    }
    response = requests.get(json_url,headers=headers)

    # Check if request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None

    data = response.json()

    # Reddit thread JSON: data[0] is the post, data[1] is the comment listing
    comment_listing = data[1]['data']['children']

    comments_list = []

    for comment in comment_listing:
        # 'kind' == 't1' indicates a comment object
        if comment['kind'] == 't1':
            c_data = comment['data']
            comments_list.append({
                'body': c_data.get('body'),
                'upvotes': c_data.get('ups'),
            })
            if comment["data"]["replies"]:
                child_comments = comment["data"]["replies"]["data"]['children']
                for child in child_comments:
                    if child['kind'] == 't1':
                            ch_data = child['data']
                            comments_list.append({
                                'body': ch_data.get('body'),
                                'upvotes': ch_data.get('ups'),
                            })

    metadata = data[0]['data']["children"]
    upvotes = metadata[0]['data']["score"]
    title =  metadata[0]['data']["title"]
    time_created = dt.datetime.fromtimestamp((metadata[0]['data']["created_utc"]))


    comments = pd.DataFrame(comments_list)
    if comments_list:
        comments = comments[comments["upvotes"]>0]


    return {"title": title, "comments": comments, "upvotes": upvotes, "time_created": time_created}

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
logging.info(f"CUDA is available?{torch.cuda.is_available()}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
if torch.cuda.is_available():
    model.to("cuda")

def get_sentiment_score(ars_posts_df):

    i = 0
    scores_list = []
    while i < len(ars_posts_df):
      text_list = ars_posts_df.body.to_list()[i:i+10]

      encoded_input = tokenizer(text_list, padding=True , truncation=True,  return_tensors='pt', max_length=512)
      output = model(**encoded_input)
      scores = output[0].detach().numpy()
      scores = softmax(scores,axis=1)
      scores[:,2] = scores[:,2] +  scores[:,1]/2
      scores_list.append(scores)
      i += 10
    full_array = np.vstack(scores_list)
    ars_posts_df["pos_score"] = full_array[:,2]

    return ars_posts_df

def get_post_sentiment_score(post_list,post_sentiment_list):
    for p in post_list:
        logging.info(f"getting sentiment score for {p}")
        post_dict = get_reddit_comments(p)
        for i in range(5):
            if not post_dict:
                time.sleep(60)
                post_dict = get_reddit_comments(p)
            else:
                break
        if not post_dict:
            continue
        if not post_dict["comments"].empty:
            post_sent = get_sentiment_score(post_dict["comments"])
            post_dict["comments"] = post_sent
            post_sentiment_list.append(post_dict)

    return post_sentiment_list

def get_daily_weighted_sentiment(sentiment_score,perc_threshold):
    sent_sc_copy = sentiment_score.copy()
    for sc in sent_sc_copy:
        sent_score = np.percentile(sc['comments']['pos_score'],perc_threshold, weights=sc['comments']['upvotes'], method = 'inverted_cdf')
        sc["sentiment"] = sent_score

    sentiment_df = pd.DataFrame([(d['title'],d['sentiment'], d['time_created'], d['upvotes']) for d in sent_sc_copy], columns=['title','sentiment', 'timestamp','upvotes'])
    # Resample every 1 day and calculate the weighted average of sentiment

    # Ensure timestamp is datetime and set as index
    if 'timestamp' in sentiment_df.columns:
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        # Ensure timestamp is timezone-naive to avoid comparison issues later
        if sentiment_df['timestamp'].dt.tz is not None:
            sentiment_df['timestamp'] = sentiment_df['timestamp'].dt.tz_localize(None)
        sentiment_df = sentiment_df.set_index('timestamp')

    def weighted_avg(x):
        if x['upvotes'].sum() == 0:
            return np.nan
        return np.average(x['sentiment'], weights=x['upvotes'])

    daily_weighted_sentiment = sentiment_df.resample('1D').apply(weighted_avg)
    daily_weighted_sentiment = daily_weighted_sentiment.dropna()

    return daily_weighted_sentiment

url = 'https://www.soccerbase.com/teams/home.sd'
r = requests.get(url)
soup = bs(r.content, 'html.parser')
teams = soup.find('div', {'class': 'headlineBlock'}, text='Team').next_sibling.find_all('li')

teams_dict = {}
for team in teams:
    link = 'https://www.soccerbase.com' + team.find('a')['href']
    team = team.text

    teams_dict[team] = link


team = []
comps = []
dates = []
h_teams = []
a_teams = []
h_scores = []
a_scores = []

consolidated = []
for k, v in teams_dict.items():
    print('Acquiring %s data...' % k)

    headers = ['Team', 'Competition', 'Home Team', 'Home Score', 'Away Team', 'Away Score', 'Date Keep']
    r = requests.get('%s&teamTabs=results' % v)
    soup = bs(r.content, 'html.parser')

    h_scores.extend([int(i.text) for i in soup.select('.score a em:first-child')])
    limit_scores = [int(i.text) for i in soup.select('.score a em + em')]
    a_scores.extend([int(i.text) for i in soup.select('.score a em + em')])

    limit = len(limit_scores)
    team.extend([k for i in soup.select('.tournament', limit=limit)])
    comps.extend([i.text for i in soup.select('.tournament a', limit=limit)])
    dates.extend([i.text for i in soup.select('.dateTime .hide', limit=limit)])
    h_teams.extend([i.text for i in soup.select('.homeTeam a', limit=limit)])
    a_teams.extend([i.text for i in soup.select('.awayTeam a', limit=limit)])



match_data = pd.DataFrame(list(zip(team, comps, h_teams, h_scores, a_teams, a_scores, dates)),
                      columns=headers)

def plot_sentiment(sentiment_score, perc_threshold, team, title_name, match_data_all):
    logging.info(f"plotting seniment for {team}")
    daily_weighted_sentiment = get_daily_weighted_sentiment(sentiment_score,perc_threshold)
    # 1. Prepare match data
    match_data = match_data_all[match_data_all.Team == team].copy()
    match_data['Date Keep'] = pd.to_datetime(match_data['Date Keep'])

    # Optional: Dictionary to map full team names to 3-letter short forms
    # Add or adjust these depending on the exact names in your dataset
    team_short_names = {
        'Man City': 'MCI', 'Man Utd': 'MUN',
        'Tottenham': 'TOT', 'Liverpool': 'LIV',
        'Chelsea': 'CHE', 'Newcastle United': 'NEW',
        'Aston Villa': 'AVL', 'Brighton & Hove Albion': 'BHA',
        'West Ham': 'WHU', 'Everton': 'EVE',
        'Brentford': 'BRE', 'Fulham': 'FUL', 'Crystal Palace': 'CRY',
        'Nottm Forest': 'NFO', 'Wolverhampton Wanderers': 'WOL',
        'Bournemouth': 'BOU', 'Luton Town': 'LUT', 'Burnley': 'BUR',
        'B Munich': 'BAY'
    }

    # 2. Filter matches to match sentiment date range
    min_date = daily_weighted_sentiment.index.min()
    max_date = daily_weighted_sentiment.index.max() + dt.timedelta(days=1)

    if hasattr(min_date, 'tzinfo') and min_date.tzinfo is not None:
        min_date = min_date.tz_localize(None)
    if hasattr(max_date, 'tzinfo') and max_date.tzinfo is not None:
        max_date = max_date.tz_localize(None)

    matches_in_range = match_data[(match_data['Date Keep'] >= min_date) & (match_data['Date Keep'] <= max_date)]

    # 3. Plotting preparation
    plt.figure(figsize=(14, 7))

    plot_data = daily_weighted_sentiment.copy()
    if plot_data.index.tz is not None:
        plot_data.index = plot_data.index.tz_localize(None)

    sns.lineplot(x=plot_data.index, y=plot_data.values, label='Sentiment Score', color='blue', linewidth=2)

    # Calculate a slight Y-offset so the text doesn't sit exactly on the line
    y_range = plt.ylim()[1] - plt.ylim()[0]
    text_y_offset = y_range * 0.03

    # 4. Annotate Match Results
    for index, row in matches_in_range.iterrows():
        print()
        date = row['Date Keep']
        home_team = row['Home Team']
        away_team = row['Away Team']
        home_score = row['Home Score']
        away_score = row['Away Score']

        # Determine match outcome and opponent
        is_home = home_team == team
        opponent_full = away_team if is_home else home_team

        # Get short name, fallback to first 3 letters if not in dictionary
        opponent_short = team_short_names.get(opponent_full, opponent_full[:3].upper())

        if is_home:
            if home_score > away_score: color, result = 'green', 'W'
            elif home_score < away_score: color, result = 'red', 'L'
            else: color, result = 'gray', 'D'
        else:
            if away_score > home_score: color, result = 'green', 'W'
            elif away_score < home_score: color, result = 'red', 'L'
            else: color, result = 'gray', 'D'

        # Dynamically find the Y-position (sentiment score) for the match date
        # .asof() fetches the value for that date, or the closest previous date if missing
        y_pos = plot_data.asof(date)

        # If plot_data is a DataFrame rather than a Series, extract the scalar value
        if isinstance(y_pos, pd.Series):
            y_pos = y_pos.iloc[0]

        # Add a subtle vertical line at the match date
        plt.axvline(x=date, color=color, linestyle=':', alpha=0.3)

        # Place the text at the dynamically calculated Y position
        plt.text(date, y_pos + text_y_offset,
                 f"{opponent_short}\n{home_score}-{away_score}\n({result})",
                 rotation=0, # Changed rotation to 0 for easier horizontal reading
                 verticalalignment='bottom',
                 horizontalalignment='center',
                 fontsize=9,
                 color=color,
                 fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor=color, alpha=0.8, boxstyle='round,pad=0.2'))

    # 5. Final Formatting

    plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.3)

    # Calculate a dynamic horizontal offset (e.g., 1.5% of the total date range)
    date_range = max_date - min_date
    x_offset = date_range * 0.04

    # Add the "Good", "Neutral", and "Bad" text sitting slightly right of the y-axis
    plt.text(min_date - x_offset, 1.0 + (y_range * 0.01), 'Good', color='green', va='bottom', ha='left', fontweight='bold')
    plt.text(min_date - x_offset, 0.5 - (y_range * 0.04), 'Sometimes maybe good sometimes maybe shit', color='gray', va='bottom', ha='left', fontweight='bold')
    plt.text(min_date - x_offset, 0.0 + (y_range * 0.01), 'Shit', color='red', va='bottom', ha='left', fontweight='bold')

    plt.title(f'{team} Fan Sentiment', fontsize=16, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)

    # Expand the y-axis slightly to ensure top labels aren't cut off
    plt.ylim(plt.ylim()[0], plt.ylim()[1] + (y_range * 0.1))

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(f'sentiment_graphs/{title_name}_sentiment_analysis.png',
                dpi=300,                # 'Dots per inch' - 300 is standard for high quality/print
                bbox_inches='tight',    # Ensures nothing gets cropped off the edges
                transparent=False)
    plt.show()

def get_team_sentiment(team_dict,start_date,perc_threshold,match_data):
    for k,v in team_dict.items():
        logging.info(f"get team sentiment for {k}")
        filtered_threads = get_soccer_threads(start_date,v['subreddit'],['post_match','postmatch','full_time','fulltime','ft','daily_discussion'])

        data_dir = './subreddit_threads'
        # Ensure the directory exists
        os.makedirs(data_dir, exist_ok=True)
        # Define the full filename for the pickled object
        pickle_filename = os.path.join(data_dir, k + '_threads_' + str(dt.date.today()) + '.pkl')

        with open(pickle_filename, 'wb') as f:
            pickle.dump(filtered_threads, f)

        post_sentiment_list = []
        sentiment_score = get_post_sentiment_score(filtered_threads,post_sentiment_list)

        try:
            with open(f'.\subreddit_sentiment\\{k}_sentiment_analysis.pkl', 'rb') as f:
                    archived_sentiment = pickle.load(f)
        except:
            archived_sentiment = []

        archived_sentiment.extend(sentiment_score)

        data_dir = './subreddit_sentiment'
        data_checkpoint_dir = './subreddit_checkpoint_sentiment'

        # Ensure the directory exists
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(data_checkpoint_dir, exist_ok=True)

        # Define the full filename for the pickled object
        pickle_filename = os.path.join(data_dir, k + '_sentiment_analysis.pkl')
        pickle_checkpoint_filename = os.path.join(data_checkpoint_dir, k + '_sentiment_analysis_' + str(dt.date.today()) + '.pkl' )

        # Open the file in binary write mode and pickle the ars_thread object
        with open(pickle_filename, 'wb') as f:
            pickle.dump(archived_sentiment, f)

        with open(pickle_checkpoint_filename, 'wb') as f:
            pickle.dump(sentiment_score, f)


        plot_sentiment(archived_sentiment,perc_threshold,v["title"],v["match_data_key"],match_data)

team_dict = {"man_utd":{"subreddit":"reddevils","title": "Manchester United", "match_data_key":"Man Utd"}, "man_city":{"subreddit":"MCFC","title": "Manchester City","match_data_key":"Man City"}}
get_team_sentiment(team_dict,dt.date(2025,8,1),66,match_data)