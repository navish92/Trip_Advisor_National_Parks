# Trip Advisor National Parks

## Motivation
Whether you're a travel aficionado or a once-in-a-blue-moon vacationer, one of the most vital aspects for a successful trip is figuring out how you should spend your time at your destination - which is always a laborious undertaking!  It's very easy to get a huge list of "Must-see" for any place you are planning to visit, but many additional hours of research is required to understand which must-see points fits in with your preferences.

## Objective
Focusing on a single tourism destination - **Yosemite National Park**, I use **reviews of attractions** listed under "Top Things To Do" on its **Trip Advisor** page, to create a **travel preferences based recommendation system**, using NLP based **Topic Modeling** techniques.

## Approach

The following steps were undertaken to deliver on the project's objective.

1. [Data Collection](#data-collection) - Reviews scrapped from Trip Advisor 
1. [NLP Preprocessing](#nlp-preprocessing) - Series of text cleaning & manipulation to prepare corpus for topic modeling
1. [Topic Modelling](#topic-modeling) - Dimensionality reduction using Corex algorithm & capture the recurrent themes across the documents (reviews)
1. [Topic Interpretation and Recommendation System](#topic-interpretation-and-recommendation-system) - Exploring the impact of topics on sentiment outcomes along with other trends & creating a recommendation system.

### [Data Collection](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/1-Data_Acquisition.ipynb)

10,000+ reviews were scrapped from [Trip Advisor](https://www.tripadvisor.in).   
Specifically, Trip Advisor's top attractions for [Yosemite National Park](https://www.tripadvisor.in/Attractions-g61000-Activities-Yosemite_National_Park_California.html) was used to obtain a list of attractions and then reviews from each individual [attraction's page](https://www.tripadvisor.in/Attraction_Review-g61000-d139187-Reviews-Glacier_Point-Yosemite_National_Park_California.html) was used to obtain the reviews itself.  

![Data Source Collection Process Screenshots](./Visuals/data_source_screenshots.png)

### [NLP Preprocessing](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/2-NLP_Preprocessing.ipynb)

The corpus was preprocessed for noise removal, lemmatization & stop word removal (via regex, string functions, NLTK & Spacy) using the following steps:
1. Noise Removal using **Regex** and **String functions** to convert to lower case, remove emails/website links, separate out words joined punctuations, and remove everything except all alphabets & whitespace.
2. Word Lemmatization using **Spacy**
3. Stop Word Removal using a starter list from **NLTK** but also curated a custom list of additional words

Create a new column called 'Outlook Sentiment' was made using the ratings on each review - a 4 or 5 rating corresponded to a positive outcome, and the rest negative.

The word cloud below provides a glimpse of the words present in the cleaned corpus.
<p align="center"> <img src="/Visuals/wordcloud.png" alt="Reviews Word Cloud" width="500"/> </p>

### [Topic Modeling](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/3-Topic_Modeling_Corex.ipynb)

Various topic modeling approaches were tried, including NMF, LDA & Corex. Ultimately, Corex was found to yield the best results, especially due to its semi-supervised nature enabled by the use of anchor words. 18 topics were finally settled on, with a Total Correlation (TC) score of 24.75.  

The chart below shows the various topics & the number of documents falling under each of them. (Note: A document can & often belongs under more than one topic; but differ in how strongly they score under each topic)
<p align="center"> <img src="/Visuals/cumulative_topic_frequency.png" alt="Cumulative Topic Frequency" width="600"/> </p>


### [Topic Interpretation and Recommendation System](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/4-Topic_Interpretation_and_Recommender.ipynb)  

The obtained topics were further analyzed using Logistic Regression. They were used as features with review outlook sentiment serving as the target variable. The coeffecients for each topic were informative in providing insight on which topics usually led to a more positive outcome.  
Additional analysis on the trend for topics over time & dimensionality reduction/clustering was also performed to gain a deeper understanding of the topics.  
  
Most importantly, a recommender system was built to finally wrap everything in a user-centric manner. 

## [Results: Attractions Recommendation App](https://share.streamlit.io/navish92/personalized_trip_advisor/main/streamlit_attractions_recommender.py)

From the 18 topics that were found, 12 were chosen to be used for the front end aspect of the recommendation system. These are:
<p align="center"> <img src="/Visuals/corex_topics.png" alt="Recommenders Topics" width="800"/> </p>

The choice & naming for the above topics were done based on domain knowledge, as I am an avid traveler and a huge fan of the U.S. National Parks system.  
    
The user can enter their top 3 priorities for their trip to Yosemite National Park. Using cosine similarity, they will be provided 3 attractions to prioritize on accordingly. **The interactive version of this app was deployed onto a web interface using Streamlit, and can be viewed [here](https://share.streamlit.io/navish92/personalized_trip_advisor/main/streamlit_attractions_recommender.py)**.   

Below are screenshots of various parts of the app, for easy reference.
<p align="center"> <img src="/Visuals/streamlit_app.png" alt="Recommenders App" width="800"/> </p>

## Results: Other Findings

Based on simple aggregate measures, **Shuttle Bus, Organized Tours,** and **Hiking Advice** topics had the highest negative reviews contribution, at 5% of their respective total occurances under that topic. This isn't surprising, as all of these would be areas where people may face higher grievances and potentially express them. 

Using a logistic Regression model, where the topics served as features and the sentiment outlook as target, the following feature coefficients were obtained to gain further insight. 
<p align="center"> <img src="/Visuals/Logreg_Features.png" alt="Logreg Features" width="600"/> </p>

**Hiking Advice** contributing the most towards a higher negative outcome makes sense, since people who faced certain issues would be more expressive of their problems & share advice to help future travelers. Correspondingly, **Climbing Advice** and **Beautiful Views** corresponding to highest postive outcomes intuitively also makes sense, as Yosemite does have exceedingly good views and people who tend to undertake climbing activities are known to very openly & freely share advice to help the next climber (missteps in climbing can lead to serious injuries/death).   

Overall, since the number of positive reviews far outnumbered the negative reviews (25:1), the above findings should be considered with a grain of salt.


## Future Work

The work done thus far looks incredibly promising. As part of next steps, the following can be undertaken:  
1. Expand to include all attractions that contain 10+ reviews on Trip Advisor for Yosemite National Park
1. Change recommender to provide 'n' destinations, instead of just 3, based on user preference
1. Repeat process for 10+ National Parks and extend the above recommender facility.
1. Add option to be able to state likes/dislikes in one park's attractions to recommend attractions' in another park.

## Tools Used

- NLTK
- SpaCy
- CorEx
- Pandas
- Scikit-learn (Feature Extraction, Decomposition, Linear Models, Pairwise Metrics)
- Matplotlib / Seaborn / Word Cloud / Scattertext
- Selenium / BeautifulSoup / requests
- Streamlit

## Skills Demonstrated

- Natural Language Processing (Pre-processing, Topic Modeling)
- Unsupervised Learning
- Dimensionality Reduction 
- Visualization
- Web scraping
