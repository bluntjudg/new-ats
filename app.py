import streamlit as st
import pandas as pd
import pickle
from io import StringIO

# --- Load the Trained Model and Tools ---
@st.cache_resource  # Use st.cache_resource for global resources like models
def load_model():
    """Loads the trained model, vectorizer, and label encoder from the pickle file."""
    pickle_filepath = 'resume_matcher_model.pkl'
    try:
        with open(pickle_filepath, 'rb') as file:
            loaded_data = pickle.load(file)
        model = loaded_data['model']
        vectorizer = loaded_data['vectorizer']
        label_encoder = loaded_data['label_encoder']
        print("Model loaded successfully.")  # Keep this for debugging in Streamlit
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        st.error(f"Error: '{pickle_filepath}' not found.  Make sure you have run the training part and the file is in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# --- Get Alignment Score ---
def get_alignment_score(resume_text, target_category, model, vectorizer, label_encoder):
    """Calculates the alignment score."""
    if model is None or vectorizer is None or label_encoder is None:
        return None  # Model wasn't loaded

    processed_text = vectorizer.transform([resume_text])
    try:
        category_index = label_encoder.transform([target_category])[0]
        probabilities = model.predict_proba(processed_text)[0]
        alignment_score = probabilities[category_index]
        return alignment_score
    except ValueError:
        st.error(f"Error: Category '{target_category}' not found in the trained categories.")
        return None
    except Exception as e:
        st.error(f"Error calculating alignment score: {e}")
        return None

# --- Main Streamlit App ---
def main():
    st.title("Resume-to-Category Matching")

    # Load the model
    model, vectorizer, label_encoder = load_model()
    if model is None:
        st.stop()  # Stop if the model didn't load

    # Input: Resume Text (File Upload or Text Area)
    input_option = st.radio("Choose input method:", ["Upload Resume File", "Enter Resume Text"])
    resume_text = ""

    if input_option == "Upload Resume File":
        uploaded_file = st.file_uploader("Upload a resume file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".pdf"):
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    resume_text = ""
                    for page in pdf_reader.pages:
                        resume_text += page.extract_text() or "" # handle None
                elif uploaded_file.name.endswith(".txt"):
                    resume_text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.name.endswith(".docx"):
                    import docx
                    doc = docx.Document(uploaded_file)
                    resume_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                else:  #  Generic fallback, may work for plain text in other formats
                    resume_text = uploaded_file.read().decode("utf-8", errors='ignore')

            except Exception as e:
                st.error(f"Error reading file: {e}.  Please ensure it is a valid PDF, TXT, or DOCX file.")
                resume_text = ""  # Clear resume_text to prevent processing with bad data

    else:
        resume_text = st.text_area("Enter Resume Text", height=300)

    # Input: Target Category
    target_category = st.text_input("Enter Target Job Category", "Data Science")

    # Calculate and Display Alignment Score
    if st.button("Calculate Alignment Score") and resume_text and target_category: # prevent error
        alignment_score = get_alignment_score(resume_text, target_category, model, vectorizer, label_encoder)
        if alignment_score is not None:
            st.write(f"Alignment Score with '{target_category}': {alignment_score:.4f}")

if __name__ == "__main__":
    # --- Train and Save the Model (ONLY RUN ONCE or when training data changes) ---
    # This part should ideally be a separate script or run once before deploying the Streamlit app.
    # For simplicity, we'll include it here, but comment it out.  You would uncomment
    # this, run the script once, and then comment it out again.
    #
    # try:
    #     data = pd.read_csv('resume_data.csv')
    #     if 'Resume' not in data.columns or 'Category' not in data.columns:
    #         raise ValueError("CSV file must contain 'Resume' and 'Category' columns.")
    # except FileNotFoundError:
    #     print("Error: 'resume_data.csv' not found. Please make sure the file exists.")
    #     exit()
    # except ValueError as e:
    #     print(f"Error loading data: {e}")
    #     exit()
    #
    # X = data['Resume']
    # y = data['Category']
    #
    # label_encoder = LabelEncoder()
    # y_encoded = label_encoder.fit_transform(y)
    #
    # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # X_tfidf = tfidf_vectorizer.fit_transform(X)
    #
    # model = MultinomialNB()
    # model.fit(X_tfidf, y_encoded)
    #
    # pickle_filepath = 'resume_matcher_model.pkl'
    # with open(pickle_filepath, 'wb') as file:
    #     pickle.dump({'model': model, 'vectorizer': tfidf_vectorizer, 'label_encoder': label_encoder}, file)
    #
    # print(f"Model trained and saved to '{pickle_filepath}'")
    # print("Please comment out the training section after running it once!")

    main()  # Run the Streamlit app
