"""
This python module is for scraping information from Trip Advisor. 
"""

from bs4 import BeautifulSoup
from selenium import webdriver
import pickle
import time
import numpy as np
import pandas as pd
import requests
import re

def ta_userreviews_review_parser(review_url):

    """
    Take in the url for a specific user review in Trip Advisor and returns just the text portion of the review as a tring
    'ta' in the function name stands for Trip Advisor
    Args:
        review_url (str): review url
    Output:
        review_text (str): paragraph containing the review; will return None if the review could not be retrieved. 
    """

    url = "https://www.tripadvisor.in" + review_url

    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html5lib')
            review_text = soup.find('span', class_ = 'fullText').text
            return review_text
        else:
            return None
    except:
        return None



def ta_attraction_reviews_parser(ta_reviews_page_soup):

    """
    Takes in a Trip Advisor html page source containing reviews for a particular attraction/destination &
    parses the reviews in that page.
    'ta' in the above function name stands for 'Trip Advisor'

    Input: HTML Page Source, Location ID
    Output: reviews_list - List containing all the reviews in the page.
            Each list element is a dictionary with the following keys: attraction_id, user_id, user_name, review_date, rating,                     review_title,  review, expr_date
    """


    reviews_list = []
    reviews_soup = ta_reviews_page_soup.find_all('div', class_='Dq9MAugU T870kzTX LnVzGwUB')

    for review in reviews_soup:
        reviews_dict = {}
    # If the main review text cannot be retrieved, all other information is skipped & this review instance is ignored
        try:
            reviews_trimmed_text = review.find('q', class_='IRsGHoPm').text

            if reviews_trimmed_text:

                try:
                    reviews_dict['user_name'] = review.find('a' , class_="_1r_My98y").text
                    reviews_dict['user_profile_link'] = review.find('a' , class_="_1r_My98y").get('href')
                except:
                    reviews_dict['user_profile_link'] = None
                    reviews_dict['user_name'] = None

                try:
                    reviews_dict['review_date'] = ' '.join(review.find('div', class_= '_2fxQ4TOx').text.split()[-2:])
                except:
                    reviews_dict['review_date'] = None

                try:
                    reviews_dict['helpful_votes'] = int(review.find('span', class_='_1fk70GUn').text)
                except:
                    reviews_dict['helpful_votes'] = np.nan

                try:
                    rating_text = review.find('span', class_='ui_bubble_rating').get('class')[1]
                    reviews_dict['rating'] = int(re.findall('[0-9]', rating_text)[0])

                except:
                    reviews_dict['rating'] = np.nan
                
                try:
                    reviews_dict['review_link'] = review.find('a', class_='ocfR3SKN').get('href')
                    review_text = ta_userreviews_review_parser(reviews_dict['review_link'])

                    if review_text:
                        reviews_dict['review_text'] = review_text
                    else:
                        continue
                except:
                    reviews_dict['review_link'] = None                
                
                try:
                    reviews_dict['review_title'] = review.find('a', class_='ocfR3SKN').text
                except:
                    reviews_dict['review_title'] = None
                
                try:
                    reviews_dict['experience_date'] = ' '.join(review.find('span', class_='_34Xs-BQm').text.split()[-2:])
                except:
                    reviews_dict['experience_date'] = None

        except:
            continue

        reviews_list.append(reviews_dict)



    return reviews_list