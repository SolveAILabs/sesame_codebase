import pandas as pd
import numpy as np
import os
import math
import re
from collections import defaultdict, Counter
from tqdm import tqdm
import plotly.graph_objects as go
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class Tokenize():
    
    def __init__(self, text_corpus, remove_stop_words = False):
        
        self.remove = remove_stop_words
        self.special_chars = {
            '!': "EXL",
            '#': "TAG",
            "$": "CUR",
            "?": "QUE",
            "%": "PER",
            "*": "CND",
            "&": "AND"
        }
        self.unk_chars = {
            '!': "UNKEXL",
            '#': "UNKTAG",
            "$": "UNKCUR",
            "?": "UNKQUE",
            "%": "UNKPER",
            "*": "UNKCND",
            "&": "UNKAND"
        }
        
        self.special_c = ['!', '#', '$', '?', '%', '*', '&']
        self.can_flt = lambda char: re.match('[A-Za-z]', char)
        self.text_corpus = text_corpus
        self.is_txt = lambda txt: re.search('([_a-zA-Z0-9_])', txt.strip())
#         self.vocab_set = []
        
        if self.remove:
            self.stopwords = list(set(stopwords.words('english')))
        
        self.build_vocab(self.text_corpus)
            
        
            
    def clean_data(self, text):
        
        return ' '.join([''.join(re.findall('([a-zA-Z0-9!#$?!%*&])', txt)) for txt in text.strip().split(' ')])
    
    def remove_stopwords(self, text):
        
        self.text = self.clean_data(text)
        
        return ' '.join([txt for txt in text.split(' ') if txt not in self.stopwords])
    
    def build_vocab(self, text_corpus):
        
        self.vocab_set = [] # for str_to_int
        self.vocab_dict = dict()
        
        if self.remove:
            text_corpus = self.remove_stopwords(self.clean_data(text_corpus))
        else:
            text_corpus = self.clean_data(text_corpus)
        
        self.special_ch = lambda char: re.search('[#$?!%*&]', char)
        
        for text in text_corpus.split(' '):
            if text not in self.vocab_set:
                if self.special_ch(text):
                    lst = list(text)
                    spc_char = [ch for ch in lst if ch in self.special_c]
                    if lst[0] == '$': self.vocab_set.extend([self.special_chars['$'] if self.special_chars['$'] not in self.vocab_set else '-1233', re.split('[$]', text)[-1] if re.split('[$]', text)[-1] not in self.vocab_set else '-1233']) 
                    if lst[0] == '?': self.vocab_set.extend([self.special_chars['?'] if self.special_chars['?'] not in self.vocab_set else '-1233', re.split('[?]', text)[-1] if re.split('[?]', text)[-1] not in self.vocab_set else '-1233']) 
                    if lst[0] == '!': self.vocab_set.extend([self.special_chars['!'] if self.special_chars['!'] not in self.vocab_set else '-1233', re.split('[!]', text)[-1] if re.split('[!]', text)[-1] not in self.vocab_set else '-1233']) 
                    if lst[0] == '%': self.vocab_set.extend([self.special_chars['%'] if self.special_chars['%'] not in self.vocab_set else '-1233', re.split('[%]', text)[-1] if re.split('[%]', text)[-1] not in self.vocab_set else '-1233']) 
                    if text == '&' and 'AND' not in self.vocab_set: self.vocab_set.append(self.special_chars['&']) 
                    if lst[0] == '*': self.vocab_set.extend([self.special_chars['*'] if self.special_chars['*'] not in self.vocab_set else '-1233', re.split('[*]', text)[-1] if re.split('[*]', text)[-1] not in self.vocab_set else '-1233']) 
                    if lst[0] == '#': self.vocab_set.extend([self.special_chars['#'] if self.special_chars['#'] not in self.vocab_set else '-1233', re.split('[#]', text)[-1] if re.split('[#]', text)[-1] not in self.vocab_set else '-1233']) 
                else:
                    self.vocab_set.append(text)
            else:
                continue
                
        
        if '-1233' in self.vocab_set:
            self.vocab_set = [vocab for vocab in self.vocab_set if not vocab == '-1233']
        
        unk_lst = []
        
        for knw in list(self.special_chars.values()):
            if knw not in self.vocab_set:
                ky = [key for key, value in self.special_chars.items() if value == knw][0]
                unk_lst.append(self.unk_chars[ky])
        
        self.vocab_set.extend(unk_lst)
        self.vocab_set.extend(['UNKNUM', 'UNKCAPSTR', 'UNKSTR'])
            
            
         
        self.vocab_dict = defaultdict(list)
        
        for index, data in enumerate(self.vocab_set):
            data = data.lower() if isinstance(data, str) else data
            self.vocab_dict[data].append(index)
            
        return self.vocab_set, self.vocab_dict
    
    def get_vocab_len(self):
        if len(self.vocab_set) > 0:
            return len(self.vocab_set)
        else:
            return "Build a vocab first"
    
    def get_tagged_sentence(self, text):
        
        if self.remove:
            text = self.remove_stopwords(self.clean_data(text))
        else:
            text = self.clean_data(text)
            
        text_lst = text.split(' ')
        if len(self.vocab_set) > 0:
            join_lst = []
            for txt in text_lst:
                if txt in self.vocab_set:
                    join_lst.append(txt)     
                else:
                    # not in vocab set
                    if self.special_ch(txt):
                        spc_chr = self.special_chars[re.findall('[#$?!%*&]', txt)[0]]
                        if spc_chr in self.vocab_set:
                            # special char in vocab
                            join_lst.append(spc_chr)
                            if self.is_txt(txt) and re.findall('[#$?!%*&]', txt)[0] in ['#', '$', '*']:
                                
                                t = re.split(['!#$?%*&'], txt)[-1]
                                if t in self.vocab_set:
                                    join_lst.append(t)
                                else:
                                    if not self.can_flt(t):
                                        join_lst.append('UNKNUM')
                                    elif t.isupper():
                                        join_lst.append('UNKCAPSTR')
                                    else:
                                        join_lst.append('UNKSTR')
                            else:
#                                 print(txt)
                                if self.is_txt(txt):
                                    t = re.split('[!#$?%*&]', txt)[0]
                                    if t in self.vocab_set:
                                        join_lst.append(t)
                                    else:
                                        if not self.can_flt(t):
                                            join_lst.append('UNKNUM')
                                        elif t.isupper():
                                            join_lst.append('UNKCAPSTR')
                                        else:
                                            join_lst.append('UNKSTR')
                        else:
                            # special char not in voacb
                            chr_ = re.findall('[#$?!%*&]', txt)[0]
                            join_lst.append(self.unk_chars[chr_])
                            if self.is_txt(txt) and chr_ in ['#', '$', '*']:
                                t = re.split('[!#$?%*&]', txt)[-1]
                                if t in self.vocab_set:
                                    join_lst.append(t)
                                else:
                                    if not self.can_flt(t):
                                        join_lst.append('UNKNUM')
                                    elif t.isupper():
                                        join_lst.append('UNKCAPSTR')
                                    else:
                                        join_lst.append('UNKSTR')
                            else:
                                if self.is_txt(txt):
                                    t = re.split('[!#$?%*&]', txt)[0]
                                    if t in self.vocab_set:
                                        join_lst.append(t)
                                    else:
                                        if not self.can_flt(t):
                                            join_lst.append('UNKNUM')
                                        elif t.isupper():
                                            join_lst.append('UNKCAPSTR')
                                        else:
                                            join_lst.append('UNKSTR')                         
                    else:
                        # not special char
                        if not self.can_flt(txt):
                            join_lst.append('UNKNUM')
                        elif txt.isupper():
                            join_lst.append('UNKCAPSTR')
                        else:
                            join_lst.append('UNKSTR')
            tagged_text = ' '.join(join_lst)
            return tagged_text
                            
        else:
            return "Build a vocab first"
        
        
        
    def get_indices(self, text):
        if self.remove:
            text = self.remove_stopwords(self.clean_data(text))
        else:
            text = self.clean_data(text)
            
            
        tagged_text = self.get_tagged_sentence(text)
        lst = []
        for txt in tagged_text.strip().split(' '):
            lst.append(self.vocab_set.index(txt))
        
        return np.array(lst)
        
        
            
            
    def from_df(self, dataframe, columname):
        
        join_lst = dataframe[columname].values.tolist()
        
        string = ' '.join(join_lst)
        
        self.build_vocab(string)
        
   