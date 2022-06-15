# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: dm_env
#     language: python
#     name: dm_env
# ---

# %% [markdown]
# # Introduction

# %% [markdown]
# # Dataset description

# %% [markdown]
# # Problem statement

# %% [markdown]
# # Data cleaning

# %%
import os
import srt
import pandas as pd

from tidytext import unnest_tokens

import nltk
nltk.download('punkt')

# %%

# %%

# %%
with open('bm_s01e1.srt') as fin:
    script = srt.parse(fin.read())

bm_df = pd.DataFrame.from_dict({'lines' : list(script)})

bm_df['lines'] = bm_df['lines'].apply(lambda x: x.content)

bm_df['episode'] = 1

bm_df = (unnest_tokens(bm_df, "word", "lines"))

bm_df.reset_index(drop=True, inplace=True)
bm_df = bm_df[bm_df.word.notnull()].reset_index(drop=True)

bm_df

# %% [markdown]
# # Conclusion
