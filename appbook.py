import numpy as np
import streamlit as st
import pickle

st.title('Book Recommendation System')


model = pickle.load(open('model.pkl','rb'));
book_name = pickle.load(open('book_name.pkl','rb'));
final_rating = pickle.load(open('final_rating.pkl','rb'));
book_pivot = pickle.load(open('book_pivot.pkl','rb'));


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]:
        ids = np.where(final_rating['Book-Title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['Image-URL-S']
        poster_url.append(url)

    return  poster_url



def recommend_book(book_name):
    book_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    return book_list, poster_url



st.write(f"total books available: {len(book_pivot)}")
selected_books = st.selectbox(
    "Select Book",book_pivot.index)

if st.button('Show Recommendations'):
    recommendad_books, poster_url = recommend_book(selected_books)

    st.subheader('Recommended Books')


    cols = st.columns(5)

    for i, col in enumerate(cols):
        with col:
            google_search_url = f"https://www.google.com/search?q={recommendad_books[i+1].replace(' ', '+')} + book"

            st.markdown(
                f"""
                <div style="text-align: center; padding: 10px; border-radius: 10px; background-color: #f0f9f9; ">
                      <a href="{google_search_url}" target="_blank">
                           <img src="{poster_url[i+1]}" width="120" style="border-radius: 8px; cursor: pointer;">
                      </a><br>
                     <p style="font-weight: bold; color: #333;">{recommendad_books[i+1]}</p>
                </div>
                
                """,

                unsafe_allow_html=True

            )
