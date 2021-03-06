# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: dm_env
#     language: python
#     name: dm_env
# ---

# ```
#  ____   _               _      __  __  _                          
# | __ ) | |  __ _   ___ | | __ |  \/  |(_) _ __  _ __   ___   _ __ 
# |  _ \ | | / _` | / __|| |/ / | |\/| || || '__|| '__| / _ \ | '__|
# | |_) || || (_| || (__ |   <  | |  | || || |   | |   | (_) || |   
# |____/ |_| \__,_| \___||_|\_\ |_|  |_||_||_|   |_|    \___/ |_|   
#                                                                   
# ```                                                                  
#

# # Introduction

# - This project approaches different Data Mining techniques applied onto subtitle files of the Black Mirror TV Show, plus some full scripts from a few episodes. 
# - Term Frequency, Sentiment Analysis, Topic Mining and Entity Recognition are the main points covered by it. 

# # Dataset description

# The dataset is split into two distinct parts:
# 1. the subtitle file dataframe, including all TV Show episodes (22) + Bandersnatch (interactive movie)
# 2. the script file dataframe, including only the first season (3 episodes) and 2 more episodes (S3E4, S4E1)
#
# > The main dataset used is the first one, with a raw word count of 120k, and we were left off with 40k afterwards.
#
# Some preprocessing steps included:
# * Finding and collecting srt files
# * Making use of a srt parsing library
# * Getting rid of html tags and non-alphanumeric characters
#
# There were distinct cleaning procedures done for separate techniques, namely using unnest tokens and stopwords for Term Frequency and Sentiment Analysis, but relying on lemmatization for Topic Mining or just basic preprocessing for Entity Recognition.
#
# > The script dataset had an approximate raw word count of ~70k, and only 30k post-cleaning.
#
# Preprocessing steps included:
# * Finding and collecting pdf script files
# * Making use of a pdf parsing library (tika - java project with python interface)
# * Getting rid non-alphanumeric characters

# # Problem statement

# Out of the datasets described above, we attempted to:
# - extract basic term frequencies per season
# - perform sentiment analysis both with Afinn and NRCLex, and then compare data and also draw conclusions on a per-season basis.
# - perform sentiment analysis on the script dataset and display a radar chart 
# - mine topics on the subtitle dataset, however we expected to encounter difficulties here, since there are no clear or direct connections between episodes and the data is limited.
# - mine topics on the script dataset, which should have been easier because there was more data per episode.
# - find all references to (real) places, then attempt to search them using a geolocation library (geopy), and plot them episode-by-episode on a world map.

# # Data processing and cleaning

# +
import os
import re
import srt
import nltk
import pandas as pd

from afinn import Afinn
from nrclex import NRCLex
from tidytext import unnest_tokens
import spacy
nlp = spacy.load('en_core_web_lg')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('omw-1.4')
# -


# cell used for constants
SUBTITLE_DIR_PATH = 'bm_subtitles/'
SCRIPT_DIR_PATH = "bm_scripts/"

# +
episode_name_map = {
    (0, 1): "Bandersnatch",
    (1, 1): "The National Anthem",
    (1, 2): "Fifteen Million Merits",
    (1, 3): "The Entire History of You",
    (2, 1): "Be Right Back",
    (2, 2): "White Bear",
    (2, 3): "The Waldo Moment",
    (2, 4): "White Christmas",
    (3, 1): "Nosedive",
    (3, 2): "Playtest",
    (3, 3): "Shut Up and Dance",
    (3, 4): "San Junipero",
    (3, 5): "Men Against Fire",
    (3, 6): "Hated in the Nation",
    (4, 1): "USS Callister",
    (4, 2): "Arkangel",
    (4, 3): "Crocodile",
    (4, 4): "Hang the DJ",
    (4, 5): "Metalhead",
    (4, 6): "Black Museum",
    (5, 1): "Striking Vipers",
    (5, 2): "Smithereens",
    (5, 3): "Rachel, Jack and Ashley Too"
}

season_color_code = {
    0: "black",
    1: "orange",
    2: "red",
    3: "blue",
    4: "green",
    5: "purple"
}



# +
def create_df(with_bandersnatch=True):
    df_list = []
    
    for f_name in os.listdir(SUBTITLE_DIR_PATH):
        
        f_path = os.path.join(SUBTITLE_DIR_PATH, f_name)
        groups = re.match(r'Black\.Mirror\.S(\d+)E(\d+)\.srt', f_name)
        
        if groups is None:
            raise ValueError('file format does not match')
        
        season, episode = groups.groups()
        if (not with_bandersnatch) and int(season) == 0:
            continue
        
        print('Processing ' + f_name)
        
        try:
            with open(f_path, 'r', encoding='utf8') as fin:
                script = srt.parse(fin.read())
        except:
            with open(f_path, 'r', encoding='utf16') as fin:
                script = srt.parse(fin.read())
            
        lines = [x.content for x in script]

        content = '\n'.join(lines)

        # Parse content and eliminate text processing directives
        content = re.sub(r'<[^<>]+>', '', content) 

        bm_df = pd.DataFrame.from_dict({
            'content' : [content],
            'season'  : [int(season)],
            'episode' : [int(episode)]
            })
        
        df_list.append(bm_df)
    
    return pd.concat(df_list, ignore_index=True)
        
bm_df_merged = create_df(True)

bm_df = (unnest_tokens(bm_df_merged, "word", "content"))
bm_df.reset_index(inplace=True, drop=True)

# +
import tika.parser
print(tika)

def create_df_pdf(with_bandersnatch=True):
    df_list = []
    
    for f_name in os.listdir(SCRIPT_DIR_PATH):
        
        f_path = os.path.join(SCRIPT_DIR_PATH, f_name)
        groups = re.match(r'Black\.Mirror\.S(\d+)E(\d+)\.pdf', f_name)
        
        if groups is None:
            raise ValueError('file format does not match')
        
        season, episode = groups.groups()
        if (not with_bandersnatch) and int(season) == 0:
            continue
        
        print('Processing ' + f_name)
        
#         with open(f_path, 'r', encoding='utf8') as fin:
#             script = srt.parse(fin.read())
        
        parsed = tika.parser.from_file(f_path)

        content = parsed['content']

        bm_df = pd.DataFrame.from_dict({
            'content' : [content],
            'season'  : [int(season)],
            'episode' : [int(episode)]
            })
        
        df_list.append(bm_df)
    
    return pd.concat(df_list, ignore_index=True)
        
pdf_df_merged = create_df_pdf(True)

pdf_df = (unnest_tokens(pdf_df_merged, "word", "content"))
pdf_df.reset_index(inplace=True, drop=True)
# -

# check how many non alphanumeric words ar in the dataframe
non_alnum = bm_df[list(map(lambda x: not x.isalnum(), bm_df['word']))].copy()
non_alnum.reset_index()
non_alnum['word'].value_counts()

# +
# the actual data cleaning is done here 
basic_stop_words = nltk.corpus.stopwords.words('english')
custom_stop_words = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
extended_stop_words = basic_stop_words + custom_stop_words

bm_df_clean = bm_df[~bm_df['word'].isin(extended_stop_words)].copy()
bm_df_clean = bm_df_clean[list(map(lambda x: x.isalnum(), bm_df_clean['word']))].copy()
bm_df_clean.reset_index(inplace=True, drop=True)

bm_df_clean
# -

bm_df_clean['word'].value_counts()

bm_df_clean[bm_df_clean['word'] == 'na'].value_counts()

# testing the 'na' anomaly
df_mockup = pd.DataFrame.from_dict({'line': ['I\'m gonna do it']})
unnest_tokens(df_mockup, 'word', 'line')

# # Analazying word and document frequency

# +
from siuba import *

# Get the words for each season in Black Mirror
bm_df_words = count(bm_df_clean, _.season, _.word, sort=True)
bm_df_words.head(15)

# +
# Check how many times the word 'mirror' appeared
display(bm_df_words[bm_df_words['word'] == 'mirror'])

# Check how many times the word 'black' appeared
bm_df_words[bm_df_words['word'] == 'black']

# +
from tidytext import bind_tf_idf

# extend the dataframe with infromation regarding the term frequency, the inverse document frequency
# and the tf-idf statistic
bm_tf_idf = bind_tf_idf(bm_df_words, 'word', 'season', 'n')

arrange(bm_tf_idf, -_.tf_idf)

# +
# get the 10 most relevant terms for each season
bm_term_df = ungroup(group_by(arrange(bm_tf_idf, -_.tf_idf), 'season').head(10))

# sort it based on the season 
ordered_bm_term_df = arrange(bm_term_df, _.season, -_.tf_idf)
ordered_bm_term_df

# +
from siuba.dply.forcats import fct_reorder
from plotnine import *

ggplot(ordered_bm_term_df) +\
    aes(x=fct_reorder(ordered_bm_term_df['word'], x=ordered_bm_term_df['tf_idf']), y='tf_idf', fill='season') +\
    coord_flip() + geom_col(show_legend = False, alpha=0.8, width=0.8) +\
    facet_wrap('~season', ncol = 2, scales = "free") +\
    labs(x = "tf-idf", y = None) +\
    theme(subplots_adjust={'wspace': 0.1, 'hspace': 0.1}) +\
    theme(subplots_adjust={'wspace': 0.5, 'hspace': 0.4}) +\
    scale_x_discrete()

# +
# See the overall sentiment of each season vs the sentiment for the top 10 words per season
afinn = Afinn()
afinn_part_dict = dict()

# compute the partial sentiment
partial_grouped_df = ordered_bm_term_df.groupby(by='season')

# the partial sentiment did not change even if more words were added for each season
for season, group_df in partial_grouped_df:
    partial_sentiment = sum((afinn.score(row['word']) * row['n']) for _, row in group_df.iterrows())
    
    if season not in afinn_part_dict:
        afinn_part_dict[season] = partial_sentiment
    
partial_summarized_bm_df = pd.DataFrame.from_dict({
    'season' : afinn_part_dict.keys(),
    'season_score': afinn_part_dict.values()}) 

print('Sentiment with partial info:')
display(partial_summarized_bm_df)
print('\n')

# compute the overall sentiment
# add another column with the afinn score for each word
overall_bm_term_df = bm_df_clean.copy()
overall_bm_term_df = overall_bm_term_df.assign(afinn_score = [afinn.score(word) for word in overall_bm_term_df['word']])

overall_grouped_df = group_by(overall_bm_term_df, 'season')
summarized_bm_df = summarize(overall_grouped_df, season_score = _.afinn_score.sum())

print('Sentiment with overall info:')
summarized_bm_df

# The information is not so interesting, will not plot it
# -
# # Sentiment analysis

# sentiment analysis for each episode 
afn_bm_term_df = bm_df_clean.copy()
afn_bm_term_df = afn_bm_term_df.assign(afinn_score = [afinn.score(word) for word in afn_bm_term_df['word']])
# +
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

ep_grouped_df = group_by(afn_bm_term_df, 'season', 'episode')
ep_summarized_bm_df = summarize(ep_grouped_df, episode_score = _.afinn_score.sum())

ep_summarized_bm_df

# +
# plot the sentiment for each episode vs the sentiment for each season
idx = range(len(ep_summarized_bm_df))
var_width = ep_summarized_bm_df['season'].value_counts(dropna=False, sort=False).array
season_x = [1.0, 3.0, 6.5, 11.5, 17.5, 22]

ep_summarized_bm_df['ep_name'] = ep_summarized_bm_df.apply(lambda row: episode_name_map[(row.season, row.episode)], axis=1)

(ggplot() 
 + geom_bar(aes(x='ep_name', y='episode_score'), data=ep_summarized_bm_df, stat='identity', width=0.9, size=1) 
 + geom_bar(aes(x=season_x, y='season_score', fill=list(season_color_code.values())), 
                data=summarized_bm_df, 
                stat='identity', alpha=0.3, 
                width=var_width)
 + labs(x='Episode', y='Episode score')
 + ggtitle("Afinn sentiment")
 + theme(axis_text_x  = element_text(angle = 90))
 + scale_x_discrete(limits=ep_summarized_bm_df['ep_name'])
 + scale_fill_identity(limits=list(season_color_code.values()))
)

# +
# sentiment analysis with NRCLex
list_sentiments = [NRCLex(elem).top_emotions for elem in bm_df_clean['word']]
list_first_sent = [elem[0] for elem in list_sentiments]

nrc_bm_term_df = bm_df_clean.copy()
nrc_bm_term_df = nrc_bm_term_df.assign(sentiment=list_first_sent)

nrc_bm_term_df = nrc_bm_term_df[nrc_bm_term_df.apply(lambda x: x['sentiment'][1] > 0, axis=1)] 
nrc_bm_term_df

# +
nrc_bm_term_df['sentiment'].str[0].value_counts()

# As an observation, there are mostly negative sentiments

# +
# get the text per episode 

text_per_episode = bm_df_clean.groupby(['season', 'episode']).agg({'word': ', '.join}).reset_index()

text_per_episode

# +
from operator import itemgetter

sent_list = [dict(sorted(NRCLex(elem).raw_emotion_scores.items(), key=itemgetter(1), reverse=True)[:2]) for elem in text_per_episode['word']]

text_per_episode['ep_name'] = text_per_episode.apply(lambda row: episode_name_map[(row.season, row.episode)], axis=1)

x_axis = text_per_episode['ep_name']
y_sent1 = [list(sents.items())[0][1] for sents in sent_list]
y_sent2 = [list(sents.items())[1][1] for sents in sent_list]

y_sent1_label = [list(sents.items())[0][0] for sents in sent_list]
y_sent2_label = [list(sents.items())[1][0] for sents in sent_list]

(ggplot() 
 + geom_bar(aes(x=x_axis, y=y_sent1, fill=y_sent1_label), stat='identity', position='dodge', width=0.5) 
 + geom_bar(aes(x=x_axis, y=y_sent2, fill=y_sent2_label), stat='identity', position='dodge', width=0.8)
 + labs(x='Episode', y='Sentiment score')
 + ggtitle("NRCLex top 2 sentiments per episode")
 + theme(axis_text_x  = element_text(angle = 90))
 + scale_x_discrete(limits=ep_summarized_bm_df['ep_name'])
)

# +
# nltk
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

list_nltk_sentiments = ['positive' if sia.polarity_scores(word)['compound'] > 0 else 'negative' if sia.polarity_scores(word)['compound'] < 0 else 'neutral' for word in bm_df_clean['word']]

# +
nltk_bm_df = bm_df_clean.copy()
nltk_bm_df = nltk_bm_df.assign(nltk_sentiment=list_nltk_sentiments)

nltk_bm_df.head(10)

# +
from siuba import *

# count the positive and negative values per episode
ep_nltk_test_bm_df = count(filter(nltk_bm_df, _.nltk_sentiment != 'neutral'), 'season', 'episode', 'nltk_sentiment')
ep_nltk_test_bm_df.loc[ep_nltk_test_bm_df['nltk_sentiment'] == 'negative', 'n'] *= -1
display(ep_nltk_test_bm_df)

s_nltk_test_bm_df = count(filter(nltk_bm_df, _.nltk_sentiment != 'neutral'), 'season', 'nltk_sentiment')
s_nltk_test_bm_df.loc[s_nltk_test_bm_df['nltk_sentiment'] == 'negative', 'n'] *= -1
s_nltk_test_bm_df

# +
# get the sum of negative and positive sentiment per each episode
ep_nltk_sent_bm_df = ep_nltk_test_bm_df.groupby([ep_nltk_test_bm_df.season, ep_nltk_test_bm_df.episode]).sum().reset_index()
s_nltk_sent_bm_df = s_nltk_test_bm_df.groupby(s_nltk_test_bm_df.season).sum().reset_index()

# ep_nltk_sent_bm_df
s_nltk_sent_bm_df

# +
# plot the sentiment for each episode vs the sentiment for each season
idx = range(len(ep_nltk_sent_bm_df))
var_width = ep_nltk_sent_bm_df['season'].value_counts(dropna=False, sort=False).array
season_x = [1.0, 3.0, 6.5, 11.5, 17.5, 22]

ep_summarized_bm_df['ep_name'] = ep_summarized_bm_df.apply(lambda row: episode_name_map[(row.season, row.episode)], axis=1)

(ggplot() 
 + geom_bar(aes(x=ep_summarized_bm_df['ep_name'], y='n'), data=ep_nltk_sent_bm_df, stat='identity', width=0.9, size=1) 
 + geom_bar(aes(x=season_x, y='n', fill=list(season_color_code.values())), 
                data=s_nltk_sent_bm_df, 
                stat='identity', alpha=0.3, 
                width=var_width,
                show_legend=False)
 + labs(x='Episode', y='Episode sentiment')
 + ggtitle("NLTK Vader sentiment")
 + theme(axis_text_x  = element_text(angle = 90))
 + scale_x_discrete(limits=ep_summarized_bm_df['ep_name'])
 + scale_fill_identity(limits=list(season_color_code.values())))

# +
sent_pdf_df = pdf_df.copy()

# clean the scripts dataframe
sent_pdf_clean_df = sent_pdf_df[~sent_pdf_df['word'].isin(extended_stop_words)].copy()
sent_pdf_clean_df = sent_pdf_clean_df[list(map(lambda x: x.isalnum(), sent_pdf_clean_df['word']))].copy()
sent_pdf_clean_df.reset_index(inplace=True, drop=True)

sent_pdf_clean_df

# +
# group the text 
pdf_text_per_episode = sent_pdf_clean_df.groupby(['season', 'episode']).agg({'word': ', '.join}).reset_index()

pdf_text_per_episode

# +
from pprint import pprint

pdf_sent_list = [dict(sorted(NRCLex(elem).raw_emotion_scores.items(), key=itemgetter(1), reverse=True)) for elem in pdf_text_per_episode['word']]

pdf_text_per_episode['ep_name'] = pdf_text_per_episode.apply(lambda row: episode_name_map[(row.season, row.episode)], axis=1)
sent_sum_dict = dict()

for ep in pdf_sent_list:
    for k, v in ep.items():
        if k not in sent_sum_dict:
            sent_sum_dict[k] = v 
        else:
            sent_sum_dict[k] += v

sent_sum_list = [(k, v) for k, v in sent_sum_dict.items()]
sent_sum_list.sort(key=lambda x: x[1], reverse=True) 
allowed_keywords = {k for k, v in sent_sum_list[:5]}

x_axis = text_per_episode['ep_name']

new_cols = {k:[] for k in allowed_keywords} 

for ep in pdf_sent_list:
    for k, v in ep.items():
        if k in new_cols:
            new_cols[k].append(v)

pdf_sent_df = pd.concat([pdf_text_per_episode, pd.DataFrame.from_dict(new_cols)], axis=1)

pdf_sent_df 

# +
from math import pi
import matplotlib.pyplot as plt  

df = pdf_sent_df.copy()

def make_spider(row, title, color):
    categories=list(df)[4:] # number of variables to be added to the plot
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # init the spider plot + set the number of rows and cols 
    # basically the number of subplots to be shown
    ax = plt.subplot(2, 3, row + 1, polar=True, ) 

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([200,400,600], ["200","400","600"], color="grey", size=7)
    plt.ylim(0,800)

    # Ind1
    values=df.loc[row].drop(['season', 'episode', 'word', 'ep_name']).values.flatten().tolist()

    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=11, color=color, y=1.1)

    
# ------- PART 2: Apply the function to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(1500/my_dpi, 1500/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))
 
# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title='ep '+df['ep_name'][row], color=my_palette(row))
# -

# # Topic Modeling

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = set(nltk.corpus.stopwords.words('english'))
custom_stop = set(['fucking'])
extended_stop = en_stop.union(custom_stop)

# Create p_stemmer of class PorterStemmer
stemmer = nltk.stem.WordNetLemmatizer()

def generate_tokens(text):
        # Remove all the special characters
    text = re.sub(r'\W', ' ', str(text))

    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)

    # Converting to Lowercase
    text = text.lower()

    # Lemmatization
    tokens = tokenizer.tokenize(text)
    
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in extended_stop]
    tokens = [word for word in tokens if len(word)  > 5]
    
    # stem tokens
    return tokens

tokenized_bm_df = bm_df_merged.copy()
tokenized_bm_df['tokens'] = tokenized_bm_df['content'].apply(generate_tokens)    

corp_dict = corpora.Dictionary(tokenized_bm_df['tokens'].tolist())
tokenized_bm_df['bow'] = tokenized_bm_df['tokens'].apply(corp_dict.doc2bow)
tokenized_bm_df['len'] = tokenized_bm_df['tokens'].apply(len)  
tokenized_bm_df.head(5)

ldamodel = gensim.models.ldamodel.LdaModel(tokenized_bm_df['bow'].tolist(),
                                           num_topics=2, id2word = corp_dict, passes=30)
ldamodel.print_topics(num_words=15)

# +

for ind, data in tokenized_bm_df[['season', 'episode', 'bow']].iterrows():
    print(f'S{data["season"]}.E{data["episode"]} - {episode_name_map[(data["season"], data["episode"])]}')
    print(ldamodel.get_document_topics(data["bow"]))

# +
from gensim.models import CoherenceModel

coherence_score_lda = CoherenceModel(model=ldamodel, texts=tokenized_bm_df['tokens'].tolist(), dictionary=corp_dict, coherence='c_v')
coherence_score = coherence_score_lda.get_coherence_per_topic()

print('Coherence Score:', coherence_score)

# +

import pyLDAvis.gensim_models

lda_visualization = pyLDAvis.gensim_models.prepare(ldamodel, tokenized_bm_df['bow'].tolist(), corp_dict, sort_topics=False)
pyLDAvis.display(lda_visualization)

# -

lsimodel = gensim.models.lsimodel.LsiModel(tokenized_bm_df['bow'].tolist(),
                                           num_topics=3, id2word = corp_dict)
lsimodel.print_topics(num_words=10)

# +
from gensim.models import CoherenceModel

coherence_score_lsi = CoherenceModel(model=lsimodel, texts=tokenized_bm_df['tokens'].tolist(), dictionary=corp_dict, coherence='c_v')
coherence_score = coherence_score_lsi.get_coherence_per_topic()

print('Coherence Score:', coherence_score)
# -

tokenized_pdf_df = pdf_df_merged.copy()
tokenized_pdf_df['tokens'] = tokenized_pdf_df['content'].apply(generate_tokens)    

corp_dict = corpora.Dictionary(tokenized_bm_df['tokens'].tolist())
tokenized_pdf_df['bow'] = tokenized_pdf_df['tokens'].apply(corp_dict.doc2bow)
tokenized_pdf_df['len'] = tokenized_pdf_df['tokens'].apply(len)  
tokenized_pdf_df.head(5)

ldamodel = gensim.models.ldamodel.LdaModel(tokenized_pdf_df['bow'].tolist(),
                                           num_topics=2, id2word = corp_dict, passes=30)
ldamodel.print_topics(num_words=15)

for ind, data in tokenized_pdf_df[['season', 'episode', 'bow']].iterrows():
    print(f'S{data["season"]}.E{data["episode"]} - {episode_name_map[(data["season"], data["episode"])]}')
    print(ldamodel.get_document_topics(data["bow"]))

lda_visualization = pyLDAvis.gensim_models.prepare(ldamodel, tokenized_bm_df['bow'].tolist(), corp_dict, sort_topics=False)
pyLDAvis.display(lda_visualization)

# +
coherence_score_pdf = CoherenceModel(model=ldamodel, texts=tokenized_pdf_df['tokens'].tolist(), dictionary=corp_dict, coherence='c_v')
coherence_score = coherence_score_pdf.get_coherence_per_topic()

print('Coherence Score:', coherence_score)


# -

# # Entity Recognition

# +
def scrub(text):
    final_list = []
    doc = nlp(text)
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(ent)
    for ent in doc:
        if ent.ent_type_ == "GPE":
            final_list.append(ent)
    return final_list

bm_loc_df = bm_df_merged.copy()
bm_loc_df['places'] = bm_loc_df['content'].apply(scrub)
bm_loc_df


# +
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

app = Nominatim(user_agent="stefan200013@yahoo.com")
geocode = RateLimiter(app.geocode, min_delay_seconds=1)

def try_geolocation(loc_txt):
    print(loc_txt)
    loc = geocode(loc_txt, featuretype="country")
    if loc is not None:
        return loc.raw

    loc = geocode(loc_txt, featuretype="state")
    if loc is not None and loc.raw['importance'] > 0.7:
        return loc.raw
    
    loc = geocode(loc_txt, featuretype="city")
    if loc is not None and loc.raw['importance'] > 0.6:
        return loc.raw
    
    return None


# -

all_locs_dict = {}
for place_class in [e for l in bm_loc_df['places'].tolist() for e in l]:
    place = place_class.text
    if place not in all_locs_dict:
        all_locs_dict[place] = try_geolocation(place)
all_locs_dict


def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map

# +
import folium
from folium import DivIcon, Marker
world_map = folium.Map(location=[0, 0], tiles='openstreetmap', zoom_start=1)

def number_DivIcon(color, number):

    icon = DivIcon(
            icon_size=(150,36),
            icon_anchor=(14,40),
            html="""<span class="fa-stack " style="font-size: 12pt" >>
                    <!-- The icon that will wrap the number -->
                    <span class="fa fa-circle-o fa-stack-2x" style="color : {:s}"></span>
                    <!-- a strong element with the custom content, in this case a number -->
                    <strong class="fa-stack-1x">
                         {:d}  
                    </strong>
                </span>""".format(color,number)
        )
    return icon

for idx, data in bm_loc_df.iterrows():
    for place_class in data['places']:
        place = place_class.text
        if all_locs_dict[place] == None:
            continue
        ttip=episode_name_map[(data['season'], data['episode'])] + f",\n{place}"
        pos = [all_locs_dict[place]['lat'], all_locs_dict[place]['lon']]
        folium.Marker(
            location=pos,
            tooltip=ttip,
            icon=folium.Icon(color='white',icon_color='white'),
            markerColor='white',
        ).add_to(world_map)
        folium.Marker(
            location=pos,
            tooltip=ttip,
            icon= number_DivIcon(season_color_code[data['season']], data['episode'])
        ).add_to(world_map)
world_map = add_categorical_legend(world_map, 'Season Color Legend',
                             colors = season_color_code.values(),
                           labels = [(f'S{s}' if s != 0 else "Bandersnatch") for s in season_color_code.keys() ])
world_map
# -

# # Conclusion

# ### Data cleaning

# - the _unnest_tokens_ function extracts from words such as 'wanna', 'gonna' two components (we observed many appearances of the 'na' word)
# - curse words were filtered out where necessary
# - we found a lot of ??? character in subtitle files corresponding to song lyrics
# - we removed html blocks which were often found in subtitle files

# ### Data gathering

# - downloaded collection of .srt files and used the `srt` library
# - downloaded collection of .pdf files (only for a few episodes because they were hard to find) and used `tika` library for parsing

# ### tf-idf

# - the majority of the top 10 terms that had big tf-idf scores for each season seemed to be mostly main 
# characters names (Waldo, Stefan, Walton etc.) from episodes and thus not give a conclusive picture as to what the season was about   

# ### Sentiment analysis

# - the sentiment analysis was done using three different lexicons to label a word as having a certain
# sentiment, which yielded quite different results
# - for the more naive lexicon, Afinn, the score for each episode was mainly negative, which at a first
# glance seems to be appropriate of such a TV series, but the other two lexicons showed scores that 
# were associated with mainly positive sentiments for each episode
# - There was not much sentiment associated with the _Metalhead_ episode as the story was told mostly 
# through visuals, without much dialogue during the episode
# - We plotted 5 dominant sentiments for each episodes taken from the script dataset and observed that the _USS Callister_ episode has an overwhelmingly positive atmosphere as the plot follows a simulation in which the main character created his own perfect world 

# ### Topic Mining

# - We had a hard time processing topics since the content of episodes were mostly unrelated.
# - In retrospective, it seems really difficult to find elements of interest from Data Mining by processing text that is merely linked through abstract or indirect topics, as opposed to something that has continuity
# - Used both lda and lsi models and compared coherence, but the lsi model couldn't be used for visual representation.
# - Two or three topics were chosen, episodes seemed to fit fairly well, however, after lemmatization, the data remaining from one single episode was fairly small, so completely distinct topics kind of got merged into bigger piles that had a small coherence (.20). Most keywords of a topic were character names or actions/events unique to an episode.
# - For the scripts dataset, even if there were a smaller numeber of episodes, it still was better, presumably because there was more context associated with all events. The coherence increased to around .35, but the topic keywords were still mostly episode-specific.  

# ### Entity Recognition 

# - We thought it would be interesting to see the places most referenced by the series, and also plot them accordingly.
# - It was quite challenging to filter the irrelevant (or fictional) locations out, through the use of popularity. To this end, we processed the text GPE into coordinates and other data using the `geopy` module and the OpenStreetMap Nomiatim geo API. 
# - The actual representation on the world map was done using `folium`
# - Interesting observations: _Hated in the Nation_ and _Playtest_ have many locations mentioned world-wide, and they are both more "international" in nature from a plot point of view  
