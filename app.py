import streamlit as st
import numpy as np
import pickle
import PIL

FLOWER_CAT = ['setosa', 'versicolor', 'virginica']

def main():

	html_title = """
	<div>

	<h2 style="text-align: center; font-family: 'Helvetica Neue', sans-serif; font-size:50px;color:#7c795d">Iris Flower Classifier</h2>
	<br>
	<br>
	</div>

	"""

	st.markdown(html_title, unsafe_allow_html=True)

	
	sl = st.slider('Sepal Length (cm)', 4.0, 8.0, 6.0)
	sw = st.slider('Sepal Width (cm)', 2.0, 5.0, 3.5)
	pl = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
	pw = st.slider('Petal Width (cm)', 0.1, 2.0, 1.05)

	setosa = PIL.Image.open('images/setosa.jpg')
	versicolor = PIL.Image.open('images/versicolor.jpg')
	virginica = PIL.Image.open('images/virginica.jpg')


	with open('model/rfc_model.pkl', 'rb') as file:
		pickle_model = pickle.load(file)

	flower_details = np.array([[sl, sw, pl, pw]])
	prediction =pickle_model.predict(flower_details)

	if st.button("CLASSIFY"):
		flower_details = np.array([[sl, sw, pl, pw]])
		prediction =pickle_model.predict(flower_details)
		st.subheader("Prediction:")
		st.success(FLOWER_CAT[prediction[0]])

		if(prediction[0] == 0):
			st.image(setosa, width=300)
		elif prediction[0] == 1:
			st.image(versicolor, width=300)
		else:
			st.image(virginica, width=300)



if __name__ == '__main__':
	main()