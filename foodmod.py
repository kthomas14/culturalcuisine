import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from gensim import corpora, models
from sklearn.manifold import TSNE
import folium
from folium.plugins import HeatMap

def get_menu_data(url, uid, cuisine):
    '''
    Takes allmenus.com url and a restaurant unique id number
    (so that restaurants with the same name are uniquely identifiable)
    Returns Dictionary: {UID:{Cuisine, Restaurant Name,
                              Coordinates, Menu:{Dish Names, Dish Descriptions}
                        }
    '''
    # Get HTML, parse it, and identify JSON containing Restaurant/Menu Data
    html = requests.get(url)
    soup = BeautifulSoup(html.text, "html.parser")
    restaurant_data = soup.find('script', type='application/ld+json')
    try:
        # set strict to False, so that control characters like \n are allowed within the text:
        restaurant_data_json = json.loads(restaurant_data.string, strict=False)
    except:
        return {uid: {'Cuisine': cuisine,
                       'Restaurant': None,
                       'Coordinates': None,
                       'Menu Items': None,
                       'Item Descriptions': None,
                       'URL': url
                }}
    try:
        # Store each menu item (a dictionary) into nested_items variable for retrieving item names and descriptions below:
        nested_items = [section['hasMenuItem'] for section in restaurant_data_json['hasMenu'][0]['hasMenuSection']]

        # Store Data for individual menu/restaurant in a dictionary to enable easy aggregation into a Pandas DataFrame:
        # Item Descriptions are cleaned, so no $, stand-alone numbers, or measurements remain
        menu_data = {uid: {'Cuisine': cuisine,
                       'Restaurant': restaurant_data_json['name'],
                       'Coordinates': (restaurant_data_json['geo']['latitude'], restaurant_data_json['geo']['longitude']),
                       'Menu Items': [j['name'] for i in nested_items for j in i],
                       'Item Descriptions': [clean_description(j['description']) for i in nested_items for j in i],
                       'URL': url
                }}
    except:
        menu_data = {uid: {'Cuisine': cuisine,
                       'Restaurant': restaurant_data_json['name'],
                       'Coordinates': (restaurant_data_json['geo']['latitude'], restaurant_data_json['geo']['longitude']),
                       'Menu Items': None,
                       'Item Descriptions': None,
                       'URL': url
                }}

    return menu_data

def clean_description(text):
    '''
    Removes prices, numbers, and measurements (in oz, liters) from input string, as well as other non-word characters
    '''
    clean = re.sub(r'(\$*\d+\.*\d*)|(\d*oz)|(liters*)|(&*amp;)|(comma;)', ' ', text)
    clean = re.sub(r'&#39;', "'", clean)
    return clean

def scrape_top_menus(cuisine_list, url, n=10):
    '''
    Scrapes the top `n` restaurant menus on `allmenus.com` from each cuisine in `cuisine_list`.
    
    Drops menus that are missing data.
    '''
    # Create a Dictionary for holding every cuisine's restaurant/menu data:
    cuisine_dict = {}
    # Incremental count for assigning unique identification numbers to each restaurant
    count = 0

    # Scrape top n (parameter n) restaurant links from allmenus.com for each cuisine:
    for cuisine in cuisine_list:
        # Get cuisine's popularity-sorted list of restaurants
        html = requests.get(url + cuisine + '/')
        soup = BeautifulSoup(html.text, "html.parser")
        
        if not soup.find('ul', class_="restaurant-list"):
            continue
        # Find links where class != grubhub (these take you to Grubhub's website, rather than the allmenus page we want)
        restaurant_anchors = soup.find('ul', class_="restaurant-list").findAll("a", class_=None)
        restaurant_links = ['https://www.allmenus.com' + i.get('href') for i in restaurant_anchors[:n]]

        # Loop through each Restaurant's allmenus.com webpage and get its menu data:
        menu_dict = {}
        for restaurant in restaurant_links:
            dict_temp = {}
            dict_temp = get_menu_data(url = restaurant,
                                      uid = count,
                                      cuisine = cuisine)
            menu_dict.update(dict_temp)
            count += 1
        cuisine_dict.update(menu_dict)

    # bring data into DataFrame and drop rows with missing data
    df = pd.DataFrame(cuisine_dict).T
    df.dropna(inplace = True)
    return df

def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': nltk.corpus.wordnet.ADJ,
                'N': nltk.corpus.wordnet.NOUN,
                'V': nltk.corpus.wordnet.VERB,
                'R': nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation, as well as a set of study-specific stop-words
    '''
    # Define stop words
    stop = nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['comma', 'amp', 'w/', 'w.', 'z']
    
    # Combine list elements together into a single string to use NLTK's tokenizer
    text = ' '.join(text)
    
    # tokenize + lemmatize words in text
    tokens = [i for i in nltk.word_tokenize(text.lower()) if i not in stop]
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t)) for t in tokens]
    return lemmas

def plot_top_tfidf(series, data_description, n = 10):
    '''
    Plots the top `n` TF-IDF words in a Pandas series of strings.
    '''
    # Get lemmas for each row in the input Series
    lemmas = series.apply(get_lemmas)

    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary([i for i in lemmas])

    # Convert dictionary into bag of words format: list of (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]

    # Calculate TFIDF based on bag of words counts for each token and return weights:
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_weights = {}
    for doc in tfidf[bow_corpus]:
        for ID, freq in doc:
            tfidf_weights[dictionary[ID]] = np.around(freq, decimals = 2)

    # highest TF-IDF values:
    top_n = pd.Series(tfidf_weights).nlargest(n)

    # Plot the top n weighted words:
    plt.plot(top_n.index, top_n.values, label=data_description)
    plt.xticks(rotation='vertical')
    plt.title('Top {} Lemmas (TFIDF) for '.format(n) + data_description);
    
    return

def get_top_tfidf(series, n = 10):
    '''
    Plots the top `n` TF-IDF words in a Pandas series of strings.
    '''
    # Get lemmas for each row in the input Series
    lemmas = series.apply(get_lemmas)

    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary([i for i in lemmas])

    # Convert dictionary into bag of words format: list of (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]

    # Calculate TFIDF based on bag of words counts for each token and return weights:
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_weights = {}
    for doc in tfidf[bow_corpus]:
        for ID, freq in doc:
            tfidf_weights[dictionary[ID]] = np.around(freq, decimals = 2)

    # highest TF-IDF values:
    top_n = pd.Series(tfidf_weights).nlargest(n)

    
    return top_n

def make_folium_pt_map(menu_df, location):
    '''
    Plots a folium point map that displays the location, name, and cuisine of each restaurant that we have scraped.
    
    Note that cuisines/color-codes are hard-coded into this function, so we cannot display restaurants from different sets
    of cuisines than the demo set of cuisines without adjusting the hard-coding.
    '''
    
    m = folium.Map(zoom_start=12, location=location, tiles='CartoDB positron')

    for point in menu_df.index:
        popup_content = '<b>Restaurant: </b>' + menu_df['Restaurant'][point] + '\n' + '<b>Cuisine: </b>' + menu_df['Cuisine'][point].capitalize() + '\n' + '<b>Plant Based Alternatives: </b>' + str(menu_df['Plant Based Alternatives'][point])
        
        # Colors are baked into the function, so we won't be able to reassign colors/cuisines without reorganizing:
        color_cuisine = {'italian': 'red','soul-food': 'blue', 'cajun-creole': 'green', 'american': 'purple',
                         'african': 'orange', 'chinese': 'darkred', 'thai': 'black', 'belgian': 'pink',
                         'mexican': 'darkblue', 'brazilian':'lightgreen', 'british-traditional': 'lightred', 
                         'caribbean': 'beige', 'colombian': 'darkgreen', 'cuban': 'darkblue', 'dutch': 'darkpurple',
                         'greek': 'white', 'filipino': 'lightblue', 'haitian': 'gray', 'japanese': 'black',
                         'russian': 'lightgray', 'puerto-rican': 'darkblue', 'ethiopian': 'orange', 'indian': 'lightblue', 
                         'persian': 'black'
                        }
               
        color_icon_body = color_cuisine[menu_df['Cuisine'][point]]

        folium.Marker(menu_df['Coordinates'][point],
                      popup=popup_content,
                      icon=folium.Icon(color=color_icon_body,icon_color='white', icon='cutlery')
                     ).add_to(m)

    # Legend Inspired by: https://medium.com/@bobhaffner/creating-a-legend-for-a-folium-map-c1e0ffc34373
    legend_html = '''
             <div style="position: fixed;
             top: 50px; right: 50px; width: 150px; height: 325px;
             border:2px solid grey; z-index:9999; font-size:14px;
             ">&nbsp; Cuisine Color Code<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:red"></i>&nbsp; Italian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:blue"></i>&nbsp; Soul Food<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:green"></i>&nbsp; Cajun-Creole<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:purple"></i>&nbsp; American<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:orange"></i>&nbsp; African/Ethiopian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:darkred"></i>&nbsp; Chinese<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:black"></i>&nbsp; Thai/Japanese<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:pink"></i>&nbsp; Belgian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:darkblue"></i>&nbsp; Mexican<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:lightgreen"></i>&nbsp; Brazilian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:lightred"></i>&nbsp; British-Traditional<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:beige"></i>&nbsp; Caribbean<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:darkgreen"></i>&nbsp; Colombian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:darkblue"></i>&nbsp; Cuban/PuertoRican<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:darkpurple"></i>&nbsp; Dutch<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:white"></i>&nbsp; Greek<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:lightblue"></i>&nbsp; Filipino/Indian/Persian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:gray"></i>&nbsp; Haitian<br>
             &nbsp;<i class="fa fa-map-marker fa-2x" style="color:lightgray"></i>&nbsp; Russian
             </div>
            '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def word2vec_tsne_plot(model, perplexity=40, n_iter=2500):
    '''
    Creates and TSNE model based on a Gensim word2vec model and plots it, 
    given parameter inputs of perplexity and number of iterations.
    '''
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)

    # Reduce 100 dimensional vectors down into 2-dimensional space so that we can see them
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    
    plt.show()
    return

def doc2vec_tsne_plot(doc_model, labels, perplexity=40, n_iter=2500):
    '''
    Creates and TSNE model based on a Gensim doc2vec model and plots it, 
    given parameter inputs of perplexity and number of iterations.
    '''
    tokens = []
    for i in range(len(doc_model.docvecs.vectors_docs)):
        tokens.append(doc_model.docvecs.vectors_docs[i])

    # Reduce 100 dimensional vectors down into 2-dimensional space so that we can see them
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    X = [doc[0] for doc in new_values]
    y = [doc[1] for doc in new_values]

    # Combine data into DataFrame, so that we plot it easily using Seaborn
    df = pd.DataFrame({'X':X, 'y':y, 'Cuisine':labels})
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x="X", y="y", hue="Cuisine", data=df)
    return