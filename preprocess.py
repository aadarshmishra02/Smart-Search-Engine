import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

def preprocess():
    # Base URL for navigation
    base_url = 'https://courses.analyticsvidhya.com/collections/courses?page='
    course_list_url = "https://courses.analyticsvidhya.com/"

    # List to hold course data
    courses = []

    page_number = 1  # Start with the first page
    while True:
        # Construct URL for the current page
        current_page_url = base_url + str(page_number)
        print(f"Processing page {page_number}...")

        # Get the current page content
        response = requests.get(current_page_url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page_number}. Status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all course cards
        course_cards = soup.find_all('li', class_='products__list-item')
        if not course_cards:
            print("No more courses found. Ending extraction.")
            break
        
        # Extract course data from each card
        for course_card in course_cards:
            title_tag = course_card.find('h3')
            link_tag = course_card.find('a')

            if title_tag and link_tag:  # Check if both title and link exist
                title = title_tag.text.strip()
                course_link = link_tag['href']
                
                # Construct full course URL (assume relative links)
                course_url = course_list_url.rstrip('/') + course_link

                # Visit each course link to get the description
                course_response = requests.get(course_url)
                if course_response.status_code == 200:
                    course_soup = BeautifulSoup(course_response.content, 'html.parser')
                    description_tag = course_soup.find('div', class_='fr-view')  # Adjust based on actual class or tag
                    description = description_tag.text.strip() if description_tag else 'No description available'
                    
                    curriculum_tag = course_soup.find('ul', class_='course-curriculum__chapter-content')  # Adjust based on actual class or tag
                    curriculum = curriculum_tag.text.strip() if curriculum_tag else 'No curriculum available'
                    
                    #enroll_tag = course_soup.find('article', class_='section__content section__content___ae733')  # Adjust based on actual class or tag
                    #enroll = enroll_tag.text.strip() if enroll_tag else 'No enroll available'
                    
                    instructor_tag = course_soup.find('section', class_='text-image section-height__medium section__content-alignment--left text-image___07200')  # Adjust based on actual class or tag
                    instructor = instructor_tag.text.strip() if instructor_tag else 'No instructor available'
                    
                    # Append the data to the list
                    courses.append({'title': title, 'description': description, 'Course curriculum': curriculum, 'About the Instructor': instructor})
                else:
                    print(f"Failed to fetch course page: {course_url}")
                
                # Sleep to avoid overwhelming the server (optional)
                time.sleep(1)
            else:
                print("Skipped a course card due to missing title or link.")
        
        # Move to the next page
        page_number += 1
    #     break

    # Save the collected data to a CSV file
    df = pd.DataFrame(courses)
    df.to_csv('courses.csv', index=False)
    print("Data collection complete. Saved to courses.csv.")


    # Load the data
    df = pd.read_csv('courses.csv')

    # Combine relevant text fields for embedding (e.g., title, description, curriculum)
    df['combined_text'] = df['title'] + ' ' + df['description'] + ' ' + df['Course curriculum'] + ' ' + df['About the Instructor']

    # Load a pre-trained model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for each course
    embeddings = model.encode(df['combined_text'].tolist(), convert_to_tensor=True)

    # Save embeddings and DataFrame for later use
    np.save('course_embeddings.npy', embeddings)
    df.to_csv('courses_with_embeddings.csv', index=False)

    # Load embeddings and DataFrame
    embeddings = np.load('course_embeddings.npy')
    df = pd.read_csv('courses_with_embeddings.csv')
    
  


