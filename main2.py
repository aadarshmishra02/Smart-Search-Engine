import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from preprocess import preprocess

# File paths
csv_file_path = 'courses_with_embeddings.csv'
embeddings_file_path = 'course_embeddings.npy'

# Function to check if data files exist and contain data
def check_files_exist():
    if os.path.exists(csv_file_path) and os.path.exists(embeddings_file_path):
        # Check if CSV and Numpy files have data
        if os.path.getsize(csv_file_path) > 0 and os.path.getsize(embeddings_file_path) > 0:
            return True
    return False

# Run preprocess if files don't exist or are empty
if not check_files_exist():
    preprocess()

# Load data and embeddings
df = pd.read_csv(csv_file_path)
embeddings = np.load(embeddings_file_path)

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Search function
def search_courses(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = similarities.topk(k=top_k)
    results = [df.iloc[idx.item()] for idx in top_results.indices]
    return results

def main():
    # Streamlit interface
    st.title("🌟 Smart Course Search 🔎")
    st.markdown(
        """
        ### Find the Most Relevant Free Courses on Analytics Vidhya
        Welcome to **Smart Course Search**! Simply type in your area of interest, and we'll show you the best courses available.
        """
    )

    # User input for the search query
    query = st.text_input("Enter your search query:")

    if query:
        st.markdown(f"### Showing results for: *'{query}'* 📜")
        results = search_courses(query)
        for result in results:
            st.markdown("---")
            st.markdown(f"## {result['title']}")
            st.markdown(f"**📗 Description:**")
            st.markdown(result['description'])
            
            # Course Curriculum Section
            if 'Course curriculum' in result:
                st.markdown("### 📙 Course Curriculum:")
                st.markdown(result['Course curriculum'])
            
            # About the Instructor Section
            if 'About the Instructor' in result:
                st.markdown("### 👨‍🏫 About the Instructor:")
                st.write(result['About the Instructor'])

            # Adding a button to enroll or learn more about the course
            if 'url' in result:
                st.markdown("**[Learn More and Enroll Here](%s)**" % result['url'])

if __name__ == "__main__":
    main()
