# Trip Advisor National Parks

### Motivation
Whether you're a travel aficionado or a once-in-a-blue-moon vacationer, one of the most vital aspects for a successful trip is figuring out how you should spend your time at your destination - which is always a laborious undertaking!  It's very easy to get a huge list of "Must-see" for any place you are planning to visit, but many additional hours of research is required to understand which must-see points fits in with your preferences.

### Objective
Focusing on a single tourism destination - Yosemite National Park, I will be using reviews of attractions listed under "Top Things To Do" on its **Trip Advisor** page, to create a travel preferences recommendation system, using NLP based Topic Modeling techniques.

### Data Sources

10,000+ reviews were scrapped from [Trip Advisor](www.tripadvisor.com).   
Specifically, their top attractions for [Yosemite National Park](https://www.tripadvisor.in/Attractions-g61000-Activities-Yosemite_National_Park_California.html) was used to obtain a list of attractions and then reviews from each individual [attraction's page](https://www.tripadvisor.in/Attraction_Review-g61000-d139187-Reviews-Glacier_Point-Yosemite_National_Park_California.html) was used to obtain the reviews itself.  

### Approach

The scraped reviews underwent three steps of further processing.

#### 1. [Pre-processing (Cleaning)](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/2-NLP_Preprocessing.ipynb)

The corpus of documents were cleaned using NLTK & Spacy libraries using the following steps:
- All characters were converted to lower case
- All website links were removed by referencing any text that started with "http"
- All emails were removed by identifying any text with a '@' in between its characters
- Text with certain punctuations were substituted with whitespace (a lot of reviews contain text such as difficult/strenous - this helps clean that up).
- Everything except English alphabets & whitespace were dropped
- Using Spacy, all words were lemmatized
- Using the default 'Stop Words' list from sklearn, a majority of the stop words were removed (words such as "not", "no")
- A list of custom additional stop words were made. Majorly, this comprised of the names of attractions.  
Finally, a new column for 'Outlook Sentiment' was made using the ratings on each review - a 4 or 5 rating corresponded to a positive outcome, and the rest negative.

The word cloud below provides a glimpse of the words present in the cleaned corpus.

**IMAGE**

#### 1. [Topic Modeling](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/3-Topic_Modeling_Corex.ipynb)

Various topic modeling approaches were tried, including NMF, LSA, LDA & Corex. Ultimately, Corex was found to yield the best results, especially due to its semi-supervised nature enabled by the use of anchor words. 18 topics were finally settled on.

The chart below shows the various topics & the number of documents falling under each of them. (Note: A document can & often belongs under more than one topic; but differ in how strongly they score under each topic)

**IMAGE**

#### 1. [Topic Interpretation and Recommendion System](https://github.com/navish92/Trip_Advisor_National_Parks/blob/main/Notebooks_Python_Files/4-Topic_Interpretation_and_Recommender.ipynb)  

The obtained topics were further analyzed using Logistic Regression. They were used as features with the outlook sentiment serving as the target variable. The coeffecients for each topic were informative in providing insight on which topics usually led to a more positive outcome.  
Additional analysis on the trend for topics over time & dimensionality reduction/clustering was also performed to gain a deeper understanding of the topics.  
Most importantly, a recommender system was built to finally wrap everything in a user-centric manner. 

### Findings

Goals: Following are the comprehensive list of goals. Definitely won't be able to achieve them all, but its to help drive the overall direction:
1. Use one national park as proof of concept. Scrape all "Things To Do" & all reviews for each "Thing". Create a summary segregated by sentiment analysis.
1. Present an overall summarized itinerary that can be filtered based on top level preferences and time available.
1. Make a customer facing app that will allow them to interact with above processed National Park.
1. Repeat process for 10+ National Parks and store to be readily accessible.
1. Add an option to be able enter any destination from Trip Advisor, for which the above process can be repeated on the fly.

## Learning Goals

1. Support Tools
    * Cloud Services (GCP)
    * VS Code
1. NLP 
