#%%
# importing the requests library 
import requests 
import json
import brotli  
import tqdm
import time
import sys
import pandas as pd
import argparse
import winsound
import bs4
import re
import os
import threading
#%%
def parse(id,url):
    
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
        ,'accept-encoding': 'gzip, deflate, br'
        ,'accept-language': 'en-US,en;q=0.9,es-MX;q=0.8,es;q=0.7,nb;q=0.6'
        ,'cache-control': 'max-age=0'
        ,'cookie': 'SessionGA=91503dbe97dd6e0686f7b138a72e9580; auth=no; __auc=1bab11651700d617da5d89d6650; _fbp=fb.1.1580772458015.1782158031; _hjid=d7277911-6d41-4c33-9877-6b53732ed871; _ga=GA1.2.175842662.1580772459; __cfduid=d3e5040d014b69768dd5094e808c4b7441586034113; _gid=GA1.2.1319508640.1586034116; SessionGA=aa7f912a17e5504fe1f02d859ecfedb9; acceptPrivacyPolicyClosed=true; acceptPrivacyPolicy=true; __cflb=0H28uvsdtvUjyvb8gkHue7ggx2Vity68jPDKqSPV7bq; _cb_ls=1; _cb=Cq5QpI8l9TGrcgwU; _cb_svref=null; __asc=a8974ebe17148284d351e20d5a8; AMP_TOKEN=%24NOT_FOUND; cointelegraph_com_session=eyJpdiI6Im5SNndcL1hCNU53S2hGaFV0U0F2N3pnPT0iLCJ2YWx1ZSI6ImdsSHNyT2NEZVwvRTNVUURIZUY4SERXTzB3K1o0YkhjXC8rWE9TdWdoazlSdGdEZmZFbWhUM1pERllOQ3NGS1hScSIsIm1hYyI6ImQ1NDJlNzk4NGY0MzAxOTQ3ZWM3MWZiNGY2ODFmZGM2NGFhYzM0ZWU5Y2FiNjI0MjEzNGMyOTk2N2M5N2ZiYmYifQ%3D%3D; _chartbeat2=.1586053532843.1586053563606.1.BqX6vHCLOePFCqZW3S10T_nDvgyDc.3'
        ,'sec-fetch-dest': 'document'
        ,'sec-fetch-mode': 'navigate'
        ,'sec-fetch-site': 'none'
        ,'sec-fetch-user': '?1'
        ,'upgrade-insecure-requests': '1'
        ,'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36 OPR/67.0.3575.115'
            
    } 

    r = requests.get(url = url,headers=headers) 
    bs = bs4.BeautifulSoup(r.content,'html.parser') 
    row_post = bs.find('div',{'row row_post'})
    header = None
    date = None
    total_views = None
    total_shares = None
    content = None

    if row_post != None:
        post_header = bs.find('div',{'post-header'})
        if post_header != None:
            header = post_header.find('h1',{'header'})
            if header != None:
                header = header.text
            date = post_header.find('div',{'date'})
            if date != None:
                date = date.attrs['datetime']
    
        referral_widget = row_post.find('div',{'referral_widget'})
        if referral_widget != None:
            referral_stats_views = referral_widget.find('div',{'referral_stats total-views'})
            if referral_stats_views != None:
                total_views = referral_stats_views.find('span',{'total-qty'})
                total_views = total_views.text

            referral_stats_shares = referral_widget.find('div',{'referral_stats total-shares'})
            if referral_stats_shares != None:
                total_shares = referral_stats_shares.find('span',{'total-qty'})
                if total_shares != None:
                    total_shares = total_shares.text

        post_content = row_post.find('div',{'post-content'})
        if post_content != None:
            content = post_content.find('div',attrs={'itemprop':'articleBody'})
            if content != None:
                content= content.text
                content = re.sub('\\xa0','',content).strip()

    return  [id,header,date,total_views,total_shares,content]


def run_item(f, val):
    result_info = [threading.Event(), None]
    def runit():
        id = val[0]
        url = val[1]
        result_info[1] = f(id,url)
        result_info[0].set()
    threading.Thread(target=runit).start()
    return result_info

def gather_results(result_infos):
    results = [] 
    for i in range(len(result_infos)):
        result_infos[i][0].wait()
        results.append(result_infos[i][1])
    results.sort(key=lambda x: x[0],reverse=True)
    return results

def get_arg(idx,default,val_type):
    if len(sys.argv) > idx:
        if val_type == 'int':
            val = int(sys.argv[idx])
            if val > 0:
                return val
            else:
                return default
        if val_type == 'str':   
            val = sys.argv[idx]
            if val != None and val != '' :
                return val
            else: 
                return default

#%%
if __name__ == "__main__":
    
    file_name = 'cointelegraph_news1-1000'
    df = pd.read_csv (f"{file_name}.csv",index_col=[0])
    start_index = get_arg(1,0,'int')
    end_index =  get_arg(2, len(df),'int')
    step = get_arg(3,10,'int')
    file_mode = get_arg(4, 'w','str')
    print('Getting data:')
    cols = ['id','header','date','total_views','total_shares','content']
    pbar = tqdm.tqdm(range(start_index,end_index,step))
    f_name = f'{file_name}_content.csv'
    if file_mode == 'w':
        content_df = pd.DataFrame(columns=cols)
        content_df.to_csv(f_name, mode=file_mode, header=True,index=False)
        file_mode = 'a'
    for row_idx in pbar:
            winsound.Beep(500, 100)
            idxs =list( range(row_idx,row_idx + step))
            vals = [(df.loc[x]['id'],df.loc[x]['url']) for x in idxs]
            results = gather_results([run_item(parse,val) for val in vals])
            #     #content_df.loc[len(content_df)] = result
            content_df = pd.DataFrame(results, columns=cols)
            content_df.to_csv(f_name, mode=file_mode, header=False,index=False)
            #     pbar.set_description(f"Fetching data for item {row_idx}-{id}")
    winsound.Beep(2000, 500)