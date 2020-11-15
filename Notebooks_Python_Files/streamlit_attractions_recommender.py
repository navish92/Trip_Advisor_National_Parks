"""
Python Script to render 'Yosemite NP Attractions Recommender' on a webpage using Streamlit & a bit of custom HTML.

To run streamlit:
1. install streamlit: pip install streamlit
2. Check if its installed successfully by typing (in anaconda terminal): streamlit hello
3. To run a specifc file, cd into the correct directory.
4. Enter command: streamlit run your_script.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import numpy as np
from collections import defaultdict
from recommender_and_other_functions import yosemite_attraction_reco
from sklearn.metrics.pairwise import cosine_similarity

# Loading Dataframe & Storing list of options to show to user from dataframe's column names
df = pd.read_csv("../Data/Attractions_Topics_Summary_Scores.csv", index_col = 0)
options = df.columns.tolist()[0:12]

# Creating a sidebar for aesthetic appeal
st.sidebar.markdown(" **Yosemite NP Trip Advisor** ")
# page = st.sidebar.radio("Please select a section", options=pages)
st.sidebar.markdown('---')
st.sidebar.write('Created by Navish Agarwal')

# Color gradient for sidebar
sidebar_gradient = """
<style> .sidebar .sidebar-content {
    background-image: linear-gradient(#6fdc6f,#1e7b1e);
    color: black; }
</style>
"""
st.sidebar.markdown(sidebar_gradient,  unsafe_allow_html=True)

# Code for main page area
st.title("Yosemite National Park Trip Advisor")

st.markdown('''
Welcome!  
  
**Traveling is one of the most relaxing endevours in life**.  
But if you get weary planning for every trip like I do, FRET NOT!  
My Trip Advisor app is here to **save you**! ''')

st.markdown("")
st.markdown('''
Please select the top 3 things that are most important to you from your upcoming trip to Yosemite National Park
''')

# User inputs
priority_1 = st.radio("Highest Priority:", options, index = 1, key="priority1_options")
priority_2 = st.radio("Second Highest Priority:", options, index = 4, key="priority2_options")
priority_3 = st.radio("Third Highest Priority:", options, index = 8, key="priority3_options")

#Alternate display option --- disregarding for now
# priorities = st.multiselect("",options)
# priority_1, priority_2, priority_3 = priorities
# priority_1 = 'Must Visits'
# priority_2 = 'Easy Trails'
# priority_3 = 'Wildlife'

user_inputs = defaultdict(int)
user_inputs[priority_1] = 1
user_inputs[priority_2] = 1
user_inputs[priority_3] = 1

button = st.button("Show me the suggestions!")


if button:
    attractions = yosemite_attraction_reco(df, user_inputs)
    st.write(''' *Based off your preferences, following are the top 3 places recommended places:* ''')
    for i in range(len(attractions)):
        st.write(f'{i+1}.   {attractions[i]}')
    st.markdown('''
    ###
    **Keep changing your inputs to see other suggestions!**
    ''')



st.markdown(
'''
#
#
Further details on how this recommender was built:
[Github](https://github.com/navish92/Trip_Advisor_National_Parks)
[LinkedIn](https://www.linkedin.com/in/navishofficial/)
#'''
)



