import pandas as pd
import boto3
import cohere
import pickle
import re
import json
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import streamlit as st
import textwrap
import os

nltk.download('stopwords')

session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

s3 = session.resource('s3')

file_keys = {
    'Courses/new_data_courses.pkl': r'C:\Users\Ritesh\Desktop\SIH_FINAL_JOB_COURSE\Sih_data\new_data_courses.pkl',
    'Courses/TFIDF_MATRIX_COURSES.pkl': r'C:\Users\Ritesh\Desktop\SIH_FINAL_JOB_COURSE\Sih_data\TFIDF_MATRIX_COURSES.pkl',
    'Courses/vectorizer_courses.pkl': r'C:\Users\Ritesh\Desktop\SIH_FINAL_JOB_COURSE\Sih_data\vectorizer_courses.pkl',
    'One_Seventy/one_seventy.pkl': r'C:\Users\Ritesh\Desktop\SIH_FINAL_JOB_COURSE\Sih_data\one_seventy.pkl',
    'One_Seventy/TFIDF_MATRIX.pkl': r'C:\Users\Ritesh\Desktop\SIH_FINAL_JOB_COURSE\Sih_data\TFIDF_MATRIX.pkl',
    'One_Seventy/VEC.pkl': r'C:\Users\Ritesh\Desktop\SIH_FINAL_JOB_COURSE\Sih_data\VEC.pkl'
}

bucket_name = 'jobrecommendationbucket'
def download_file_from_s3(bucket_name, file_key, local_file_path):
    # Check if the file already exists locally
    if not os.path.exists(local_file_path):
        # Initialize a session using Amazon S3

        # Download the file from S3
        s3.meta.client.download_file(bucket_name, file_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path}.")
    else:
        print(f"File already exists at {local_file_path}. Skipping download.")


co = cohere.Client("xqcnCdnVrePjAuW0mCdlTlxCaiZgvOIkLF1TI1AW")
# Loading the courses dataframe
# with open(r'C:\Users\Ritesh\Desktop\SIH_Job_Course\Courses\new_data_courses.pkl', 'rb') as file:
#     cdf = pickle.load(file)
#
# courses = pd.DataFrame(cdf)
#
# with open(r'C:\Users\Ritesh\Desktop\SIH_Job_Course\Courses\TFIDF_MATRIX_COURSES.pkl', 'rb') as file:
#     tfidf_matrices_courses = pickle.load(file)
#
# with open(r'C:\Users\Ritesh\Desktop\SIH_Job_Course\Courses\vectorizer_courses.pkl', 'rb') as file:
#     vectorizer_courses = pickle.load(file)

for file_key, local_file_name in file_keys.items():
    download_file_from_s3(bucket_name, file_key, local_file_name)

# Load the pickle files after downloading or verifying existence
with open('./Sih_data/new_data_courses.pkl', 'rb') as file:
    cdf = pickle.load(file)
courses = pd.DataFrame(cdf)

with open('./Sih_data/TFIDF_MATRIX_COURSES.pkl', 'rb') as file:
    tfidf_matrices_courses = pickle.load(file)

with open('./Sih_data/vectorizer_courses.pkl', 'rb') as file:
    vectorizer_courses = pickle.load(file)

with open('./Sih_data/one_seventy.pkl', 'rb') as file:
    ndf = pickle.load(file)

with open('./Sih_data/TFIDF_MATRIX.pkl', 'rb') as file:
    tfidf_matrices = pickle.load(file)

with open('./Sih_data/VEC.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

recommendation = []


def recommend_course(missing):
    processed = preprocessing(missing)
    missing_values = vectorizer_courses.transform([processed])

    similar = cosine_similarity(missing_values, tfidf_matrices_courses)

    courses['similarity'] = similar.flatten()

    recommended_courses = courses.sort_values(by='similarity', ascending=False)
    item = recommended_courses['info'].iloc[0]
    recommendation.append(item)


def to_markdown(text):
    text = text.replace('\n', ' ').strip()
    text = text.replace('•', ' *')
    text = text.replace('>', ' ')
    formatted_text = textwrap.fill(text, width=80)
    return formatted_text


# with open(r'C:\Users\Ritesh\Desktop\SIH_Job_Course\One_Seventy\one_seventy.pkl', 'rb') as file:
#     ndf = pickle.load(file)
#
# with open(r'C:\Users\Ritesh\Desktop\SIH_Job_Course\One_Seventy\TFIDF_MATRIX.pkl', 'rb') as file:
#     tfidf_matrices = pickle.load(file)
#
# with open(r'C:\Users\Ritesh\Desktop\SIH_Job_Course\One_Seventy\VEC.pkl', 'rb') as file:
#     vectorizer = pickle.load(file)

newsdf = pd.DataFrame(ndf)

# Ensure 'tags' column is treated as strings
newsdf['tagz'] = newsdf['tagz'].astype(str)

port_stem = PorterStemmer()

# genai.configure(api_key="AIzaSyAz25OadDsrsh6ymaObJTiPG6gsVKdiCZ8")

prompt = ("""
            Tell me in keyword format as short as possible
          what are the keywords that are there in company skills
          and absent in user skills. Match every single word in company
          skills with user skills if even one word is same do not print
          it in missing skills. If all keywords of company skills are
          present in user skills then print 0 in the missing skills output.
          Please ensure you stick to the company's requirements and give output
          in array format where every missing skill is a string. 
          Respond with the missing skills in the form of plain text keywords and no mardown is needed
          also not in the form of an array it should just be plain text like paragraph.
          The text generated should be a plain text.
          """)

model = genai.GenerativeModel('gemini-1.5-flash')


def preprocessing(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)


def format_json_string(json_like_str):
    json_like_str = re.sub(r"(?<=:)\s*'(.*?)'", r'"\1"', json_like_str)
    json_like_str = re.sub(r"(?<!\w)(\w+):", r'"\1":', json_like_str)
    json_like_str = re.sub(r'(\w+):', r'"\1":', json_like_str)
    return json_like_str


st.title("Job Recommendation System")

user_input = st.text_area("Enter your skills or keywords here:")

if st.button("Recommend Companies"):
    if user_input:
        processed_input = preprocessing(user_input)

        user_vector = vectorizer.transform([processed_input])

        similarities = cosine_similarity(user_vector, tfidf_matrices)

        newsdf['similarity'] = similarities.flatten()

        recommended_companies = newsdf.sort_values(by='similarity', ascending=False)

        # Get the top 10 companies
        recommend = recommended_companies.iloc[:10]

        job_json = []
        missing_tags = []
        for i in range(10):
            item = recommend['inform'].iloc[i]
            element = recommend['tagz'].iloc[i]
            try:
                formatted_item = format_json_string(item)
                input_text = f"{prompt} \n Here's the user skillset: {user_input} \n and company skillset : {element}"
                # response = model.generate_content([input_text])
                # response.resolve()
                response = co.chat(
                    message=input_text,
                )
                # recommend_course(response.text)
                recommend_course(response.text)
                missing_tags.append(response.text)
                job_info_json = json.loads(formatted_item)
                job_json.append(job_info_json)
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")

        # Assign missing_tags to the top 10 companies only
        recommend['missing'] = missing_tags
        recommend['course'] = recommendation

        st.write("Top 10 Recommended Companies:")
        for job_info in job_json:
            st.write(f"Name: {job_info.get('name')}")
            st.write(f"Sector: {job_info.get('Sector')}")
            st.write(f"Role: {job_info.get('role')}")
            st.write(f"Description: {job_info.get('description')}")
            st.write(f"Pay: {job_info.get('pay')}")
            st.write(f"Posted: {job_info.get('posted')}")
            st.write(f"Type: {job_info.get('type')}")
            st.write("------")
        # st.dataframe(recommend[['inform', 'tagz', 'similarity', 'Job Title', 'Role', 'missing','course']])
        st.dataframe(recommend[['inform', 'similarity', 'Job Title', 'Role', 'missing', 'course']])
    else:
        st.write("Please enter some skills or keywords.")