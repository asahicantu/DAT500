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


# defining the api-endpoint  
API_ENDPOINT = "https://cointelegraph.com/api/v1/content/json/_mp"
  
# your API key here 
API_KEY = "zmkzA359FhLh38dnoBfmDBMDqpMmYDYsvUQ6rAIT"
#%%
def fetch_data(page,api_key):
    data = {
        "page":page
        ,"lang":"en"
        ,"_token":API_KEY
    } 

    data = json.dumps(data)
    
    # data to be sent to api 
    headers = {
    "accept":"application/json, text/plain, */*"
    ,"accept-encoding":"gzip, deflate, br"
    ,"accept-language":"en-US,en;q=0.9,es-MX;q=0.8,es;q=0.7,nb;q=0.6"
    ,"content-type":"application/json;charset=UTF-8"
    ,"cookie":"auth=no; __auc=1bab11651700d617da5d89d6650; _fbp=fb.1.1580772458015.1782158031; _hjid=d7277911-6d41-4c33-9877-6b53732ed871; _ga=GA1.2.175842662.1580772459; __cfduid=d3e5040d014b69768dd5094e808c4b7441586034113; __cflb=0H28uvsdtvUjyvb8gkHue7ggx2Vity68qteoM9dpLps; __asc=0dc6a92017146fffd43bf9978ac; AMP_TOKEN=%24NOT_FOUND; _gid=GA1.2.1319508640.1586034116; cointelegraph_com_session=eyJpdiI6IlhGSzBXVTN2akhBYWU0MkJGbXphbGc9PSIsInZhbHVlIjoiNm1nTlFMMzhwOVRpZlpaS3oyUkpaMHkzSWd4ZnJkY1lEU3Y2WXBNWTNLbTVOWW5TQ1V6c05QeGhPRGM4UG1EUiIsIm1hYyI6IjA1ZTUwMzA5MjRjMGViMDhmZDE5MGZhYzgyMmVhMDg1NzhlM2JhYTgzOGQ2ZTZiNDYyOWMyNmNhMmUwOWVmZGUifQ%3D%3D; SessionGA=aa7f912a17e5504fe1f02d859ecfedb9; acceptPrivacyPolicyClosed=true; acceptPrivacyPolicy=true"
    ,"origin":"https://cointelegraph.com"
    ,"referer":"https://cointelegraph.com/"
    ,"sec-fetch-dest":"empty"
    ,"content-length":str(len(data))
    ,"sec-fetch-mode":"cors"
    ,"sec-fetch-site":"same-origin"
    ,"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36 OPR/67.0.3575.115"
            
    } 
  
    # sending post request and saving response as response object 
    r = requests.post(url = API_ENDPOINT,headers=headers, data =data) 
    return r
df = None
#%%
if __name__ == "__main__":
    start_index = int(sys.argv[1])
    total_pages = int(sys.argv[2])
    file_name = sys.argv[3]
    print('Getting data:')
    
    pbar = tqdm.tqdm(range(start_index,start_index  + total_pages))
    total_items = 0
    for page in pbar:
        #time.sleep(1)
        # content = None
        # try:
        #     content = brotli.decompress(r.content)
        # except:
        #     print(sys.exc_info()[0])
        try:
            winsound.Beep(500, 100)
            pbar.set_description(f"Fetching data for page {page} ({total_items} items)")
            r = fetch_data(page,API_KEY)
            data = json.loads(r.text)
            if not 'posts' in data.keys():
                break
            total_items += len(data['posts'])
            pbar.set_description(f"Fetching data for page {page} ({total_items} items)")
            for row in data['posts']:
                for key in list(row.keys()):
                    val = row[key]
                    if type(val) == dict:
                        for s_key in val.keys():
                            s_val = val[s_key]
                            s_key = key + '_' + s_key
                            row.update({s_key:s_val})
                        row.pop(key)
                if df is None:
                    df = pd.DataFrame(row,index = [0])
                else :
                    df = df.append(row,ignore_index = True)
        except:
            info = sys.exc_info()[0]
            print(info[0],info[1],info[2])
            winsound.Beep(1500, 100)
        finally:
            df.to_csv(f'{file_name}{start_index}-{total_pages}.csv')
    winsound.Beep(2000, 500)
        
        
        
        


    # %%
