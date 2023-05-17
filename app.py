import streamlit as st 
import pickle
import functions
from PIL import Image

with open("model.pkl","rb") as f:
        model = pickle.load(f)   

st.set_page_config(layout="wide")

def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Image Recognition App
        </h1>
    """, unsafe_allow_html=True
                )

    st.caption("""
        <p style='text-align: center'>
        by team 18
        </p>
    """, unsafe_allow_html=True
               )

def main():
    image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"]) 
    
    col1, col2 = st.columns(2)
    
    if image:
        with col1:
            st.image(image)
        pred = functions.predictor(f"test/{image.name}",functions.mean_face, functions.eigvecs, model)
        with col2:
            st.success(pred)
            image = Image.open("images\output.png")
            st.image(image, caption="ROC curve")
        
if __name__ == '__main__':
    head()
    main()
        