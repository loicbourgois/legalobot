import logging
logging.basicConfig(level=logging.INFO, format="")
logging.info("start")
import xmltodict
import re
import yaml
import json
import os
import openai
import pandas
import time
from random import random
import asyncio
from functools import wraps, partial
import numpy as np


data = []
g = {}


def read(path):
    with open(path, "r") as file:
        return file.read()


def write_force(path, content):
    folder = path.replace(path.split("/")[-1], '')
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(path, 'w', encoding='utf8') as f:
        f.write(content)


def process_article_txt(txt):
    if isinstance(txt, str):
        return txt
    elif isinstance(txt, list):
        txt = "\n".join( [ str(aa) for aa in txt ]  )
    elif isinstance(txt, dict):
        txt = str(txt)
    return process_article_txt(txt)


def process_article(article, niveau):
    if isinstance(article, dict):
        txt = ""
        if '#text' in article.keys():
            txt = article['#text']
        else:
            txt = article['p']
        txt = process_article_txt(txt)
        data.append({
            'code-name': g['code_name'],
            'num': article['@num'],
            'txt': txt,
        })
        # logging.info(f"{''.join(['  ' for _ in range(niveau+1)])}{article['@num']}")
    elif isinstance(article, list):
        for aa in article:
            list_articles(aa, article)
    else:
        logging.warning("Could no process #2")
        logging.warning(article)


def list_articles(x, parent):
    if isinstance(x, dict):
        niveau = 0
        try:
            niveau = int(x['@niveau'])
            str_ = "".join([ "  " for _ in range(niveau) ])
            str_ += x['@title'].replace('\n','')
            # logging.info(str_)
        except:
            # logging.warning("Could not process group")
            # logging.warning(x.keys())
            pass
        if 'article' in x.keys():
            if isinstance(x['article'], dict):
                process_article(x['article'], niveau)
            else:
                for article in x['article']:
                    process_article(article, niveau)
        if 't' in x.keys():
            if isinstance(x['t'], list):
                for t in x['t']:
                    list_articles(t, x)
            elif 'article' in x['t']:
                process_article(x['t']['article'], niveau)
            elif 't' in x['t']:
                list_articles(x['t']['t'], x['t'])
            else:
                logging.warning("Could not process #1")
                logging.warning(x['t'])
    else:
        # logging.info(type(parent))
        # logging.info(json.dumps(parent, indent=2))
        pass


def wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run


@wrap
def oai_get_embedding_async(text, model):
    return oai_get_embedding(text, model)


def oai_get_embedding(text, model, max_retry = 3, backoff = 2):
    try:
        text = text.replace("\n", " ")
        e = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        return e
    except Exception as e:
        logging.error(e)
        time.sleep(random()*backoff)
        return oai_get_embedding(text, model, max_retry-1, backoff*2)


async def get_embedding_inner(x):
    return await oai_get_embedding_async(x, 'text-embedding-ada-002')


async def get_embedding(data):
    for i,x in enumerate(data):
        logging.info(f"[start] get_embedding {i+1} / {len(data)}")
        x['task'] = asyncio.create_task( get_embedding_inner(x['txt']) )
        await asyncio.sleep(0.025)
    for i,x in enumerate(data):
        x['embedding'] = await x['task']
        x.pop("task")
        logging.info(f"[ end ] get_embedding {i+1} / {len(data)}")
    return data


@wrap
def query_openai_chat_async(x):
    return query_openai_chat(x)


def query_openai_chat(x, max_retry = 3, backoff = 2):
    model = "gpt-3.5-turbo"
    if max_retry < 0:
        return "", 0
    if not len(x['prompt']):
        return "", 0
    try:
        r = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": x['prompt']},
            ],
        )
        m = r['choices'][0]['message']['content']
        # global_['total_tokens'] += r['usage']['total_tokens']
        txt =  "\n".join([ "  [gpt]  "+ l for l in (x['prompt'] + '\n' + m).split("\n") ])
        logging.info(txt + "\n")
        return m, r['usage']['total_tokens'], model
    except Exception as e:
        logging.error(e)
        time.sleep(random()*backoff)
        return query_openai_chat(x, max_retry-1, backoff*2)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search(df, question, n):
    df['similarities'] = df.embedding.apply(lambda x: cosine_similarity(x, question.embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res


def answer_question(raw_question):
    raw_question = raw_question.replace("\n", " ").strip()
    questions = [
        {
            'txt': raw_question,
        }
    ]
    questions = pandas.DataFrame(asyncio.run(get_embedding(questions)))
    question = questions.iloc[0]
    logging.info("loading")
    df_1 = pandas.DataFrame(json.loads(read("/root/github.com/loicbourgois/legalobot/code-civil.json")))
    df_2 = pandas.DataFrame(json.loads(read("/root/github.com/loicbourgois/legalobot/code-penal.json")))
    df_3 = pandas.DataFrame(json.loads(read("/root/github.com/loicbourgois/legalobot/code-du-patrimoine.json")))
    df = pandas.concat([df_1, df_2, df_3], ignore_index=True)
    logging.info("get res")
    res = search(df, question, 10)
    lines = [f"Voici {len(res)} articles de lois:", ""]
    for x in res.iloc:
        code_title = {
            'code-civil': "Code civil",
            'code-penal': "Code pénal",
            'code-du-patrimoine': "Code du patrimoine",
        }[x['code-name']]
        lines.append(f"{code_title} - article {x.num} :")
        lines.append(f"{x.txt}")
        lines.append(f"")
    lines.append(f"")
    lines.append(f"En tenant compte de ces {len(res)} articles, répondre au problème suivant:")
    lines.append(question.txt)
    prompt = "\n".join(lines)
    logging.info("Asking")
    r, a, b = asyncio.run(query_openai_chat_async({
        'prompt': prompt,
        'max_tokens': 1000,
    }))
    # write_force("/root/github.com/loicbourgois/legalobot/_01_raw_question.txt", raw_question)
    # write_force("/root/github.com/loicbourgois/legalobot/_02_prompt.txt", prompt)
    # write_force("/root/github.com/loicbourgois/legalobot/_03_prompt_response.txt", prompt + "\n\nResponse:" + r)


def setup_code(code_name):
    g['code_name'] = code_name
    str_ = read(f"/root/github.com/loicbourgois/legalobot/{code_name}.xml")
    str_ = str_.replace("<br/>", "\n")
    str_ = str_.replace('<div align="left">', "")
    str_ = str_.replace("</div>", "")

    regex_lits = re.findall('(<!--[^-]*-->)', str_)
    for x in regex_lits:
        str_ = str_.replace(x, "")

    regex_lits = re.findall('(<a class=\"alpha\" type=\"article-internal\" [^>]*>([^<]*)</a>)', str_)
    for x in regex_lits:
        str_ = str_.replace(x[0], x[1])    

    regex_lits = re.findall('(<a href=\"/affichCodeArticle\.do[^>]*>([^<]*)</a>)', str_)
    for x in regex_lits:
        str_ = str_.replace(x[0], x[1])
    
    regex_lits = re.findall('(<a href=\"/affichTexteArticle\.do[^>]*>([^<]*)</a>)', str_)
    for x in regex_lits:
        str_ = str_.replace(x[0], x[1])    
    
    regex_lits = re.findall('(<a href=\"/affichCode\.do[^>]*>([^<]*)</a>)', str_)
    for x in regex_lits:
        str_ = str_.replace(x[0], x[1])
    
    regex_lits = re.findall('(<a href=\"/affichTexte\.do[^>]*>([^<]*)</a>)', str_)
    for x in regex_lits:
        str_ = str_.replace(x[0], x[1])    
    
    code = xmltodict.parse(str_ )
    list_articles(code['code'], code)

    logging.info(len(data))

    data_ = asyncio.run(get_embedding(data))
    df = pandas.DataFrame(data_)
    write_force(
        f"/root/github.com/loicbourgois/legalobot/{code_name}.json", 
        json.dumps(df.to_dict('records'), indent=2, ensure_ascii=False),
    )


def format_code(code_name):
    g['code_name'] = code_name
    str_ = read(f"/root/github.com/loicbourgois/legalobot/{code_name}.xml")
    for k, v in {
        "<br/>": "\n",
        '<div align="left">': "",
        "</div>": ""
    }.items():
        str_ = str_.replace(k,v)
    regex_lits = re.findall('(<!--[^-]*-->)', str_)
    for x in regex_lits:
        str_ = str_.replace(x, "")
    for regex in [
        '(<a class=\"alpha\" type=\"article-internal\" [^>]*>([^<]*)</a>)',
        '(<a href=\"/affichCodeArticle\.do[^>]*>([^<]*)</a>)',
        '(<a href=\"/affichTexteArticle\.do[^>]*>([^<]*)</a>)',
        '(<a href=\"/affichCode\.do[^>]*>([^<]*)</a>)',
        '(<a href=\"/affichTexte\.do[^>]*>([^<]*)</a>)',
    ]:
        for x in re.findall(regex, str_):
            str_ = str_.replace(x[0], x[1])    
    code = xmltodict.parse(str_ )
    list_articles(code['code'], code)
    write_force(
        f"/root/github.com/loicbourgois/legalobot/codes/{code_name}.json", 
        json.dumps(
            data,
            indent=2, 
            ensure_ascii=False
        ),
    )
    data_2 = {}
    data_3 = {}
    for x in data:
        k = x['code-name'] + "/" + x['num'] 
        data_3[k] = x
        for i, y in enumerate(x['txt'].replace("\n", ". ").split('. ')):
            kk = x['code-name'] + "/" + x['num'] + "/" + str(i)
            if y.strip().endswith("."):
                data_2[kk] = y.strip()
            else:
                data_2[kk] = y.strip() + "."
    write_force(
        f"/root/github.com/loicbourgois/legalobot/codes/{code_name}-2.json", 
        json.dumps(
            # pandas.DataFrame(data).to_dict('records'), 
            data_2,
            indent=2, 
            ensure_ascii=False
        ),
    )
    write_force(
        f"/root/github.com/loicbourgois/legalobot/codes/{code_name}-3.json", 
        json.dumps(
            data_3,
            indent=2, 
            ensure_ascii=False
        ),
    )


def embed_code(name):
    logging.info("[start] load model")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/gtr-t5-xl')
    logging.info("[ end ] load model")
    data = json.loads(read(f"/root/github.com/loicbourgois/legalobot/codes/{name}.json"))
    sentences = []
    for i, x in enumerate(data):
        logging.info(f"{i}/{len(data)}")
        data[i]['embeddings'] = model.encode( [x['txt'].replace("\n", " ")] )
    df = pandas.DataFrame(data)
    df.to_parquet(f"/root/github.com/loicbourgois/legalobot/codes/{name}-4.parquet")


def load_embeds(name):
    return pandas.read_parquet(f"/root/github.com/loicbourgois/legalobot/codes/{name}-4.parquet")


# format_code('code-civil')
# embed_code('code-civil')

data = load_embeds('code-civil')
embeddings = [ x['embeddings'] for x in data ][0..10]
query = "Je suis l'unique locataire d'un 27m² et j'envisage de faire habiter une connaissance avec moi. Dois-je le déclarer au propriétaire ? Cela implique-t-il de modifier le contrat de location ? Selon que ce soit le cas ou non, quel type de papiers devrais-je faire ? Que devrais-je prévoir selon vous ?"
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/gtr-t5-xl')
query_emb = model.encode(query)

scores = util.dot_score(query_emb, embeddings)[0].cpu().tolist()



# setup_code("code-du-patrimoine")


# answer_question("""
# Je suis l'unique locataire d'un 27m² et j'envisage de faire habiter une connaissance avec moi. Dois-je le déclarer au propriétaire ? Cela implique-t-il de modifier le contrat de location ? Selon que ce soit le cas ou non, quel type de papiers devrais-je faire ? Que devrais-je prévoir selon vous ?
# """)


# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# query_embedding = model.encode('How big is London')
# passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
#                                   'London is known for its finacial district'])

# print("Similarity:", util.dot_score(query_embedding, passage_embedding))
# https://sbert.net/ 
# from sentence_transformers import SentenceTransformer, util
# query = "Je suis l'unique locataire d'un 27m² et j'envisage de faire habiter une connaissance avec moi. Dois-je le déclarer au propriétaire ? Cela implique-t-il de modifier le contrat de location ? Selon que ce soit le cas ou non, quel type de papiers devrais-je faire ? Que devrais-je prévoir selon vous ?"
# # data = json.loads(read("/root/github.com/loicbourgois/legalobot/code-civil.json"))
# # data = json.loads(read("/root/github.com/loicbourgois/legalobot/code-du-patrimoine.json"))
# # for x in data:
# #     print(x['num'])
# docs = [ x['txt'] for x in data ]
# #Load the model
# model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
# # model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# # model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
# #Encode query and documents
# query_emb = model.encode(query)
# doc_emb = model.encode(docs)
# #Compute dot score between query and all document embeddings
# scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
# #Combine docs & scores
# doc_score_pairs = list(zip(docs, scores, data))
# #Sort by decreasing score
# doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
# #Output passages & scores
# for i, (doc, score, data_point) in enumerate(doc_score_pairs):
#     if i > 20:
#         break
#     print(score, data_point['num'])
#     print(doc)
