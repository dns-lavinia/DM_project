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

# # Introduction

# # Dataset description

# # Problem statement

# # Data cleaning

# +
import os
import re
import srt
import nltk
import pandas as pd

from afinn import Afinn
from nrclex import NRCLex
from tidytext import unnest_tokens

nltk.download('punkt')
nltk.download('stopwords')
# -


# cell used for constants
SUBTITLE_DIR_PATH = 'bm_subtitles/'


# +
def create_df():
    df_list = []
    
    for f_name in os.listdir(SUBTITLE_DIR_PATH):
        print('Processing ' + f_name)

        f_path = os.path.join(SUBTITLE_DIR_PATH, f_name)
        groups = re.match(r'Black\.Mirror\.S(\d+)E(\d+)\.srt', f_name)
        
        if groups is None:
            raise ValueError('file format does not match')
        
        season, episode = groups.groups()
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
        
bm_df_merged = create_df()

bm_df = (unnest_tokens(bm_df_merged, "word", "content"))
bm_df.reset_index(inplace=True, drop=True)
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
df_mockup = pd.DataFrame.from_dict({'line': ['I am finna do it']})
unnest_tokens(df_mockup, 'word', 'line')

# # Analazying word and document frequency

# +
from siuba import *

# Get the words for each season in Black Mirror
bm_df_words = count(bm_df_clean, _.season, _.word, sort=True)
bm_df_words.head(15)

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
ordered_bm_term_df = ordered_bm_term_df.assign(afinn_score = [afinn.score(word) for word in ordered_bm_term_df['word']])

grouped_bm_term_df = group_by(ordered_bm_term_df, 'season')
summarized_bm_df = summarize(grouped_bm_term_df, season_score = _.afinn_score.sum())

print('Sentiment with overall info:')
summarized_bm_df

# The information is not so interesting, will not plot it
# -



# # Conclusion

# "na" + "ta" -> particle from "gonna", "wanna", "gotta" if using unnest_tokens
