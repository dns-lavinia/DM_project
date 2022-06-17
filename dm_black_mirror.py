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
import srt
import re
import pandas as pd

from tidytext import unnest_tokens

import nltk
nltk.download('punkt')
nltk.download('stopwords')
# -


SUBTITLE_DIR_PATH = 'bm_subtitles/'


# +
def create_df():
    df_list = []
    
    for f_name in os.listdir(SUBTITLE_DIR_PATH):
        print(f_name)
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
            
        bm_df = pd.DataFrame.from_dict({'lines' : list(script)})

        bm_df['lines'] = bm_df['lines'].apply(lambda x: x.content)
        
        bm_df['season'] = int(season)
        bm_df['episode'] = int(episode)

        bm_df = (unnest_tokens(bm_df, "word", "lines"))

        bm_df.reset_index(drop=True, inplace=True)
        bm_df = bm_df[bm_df.word.notnull()].reset_index(drop=True)
        
        df_list.append(bm_df)
    
    return pd.concat(df_list, ignore_index=True)
        
bm_df = create_df()
# -

bm_df

# # Conclusion
