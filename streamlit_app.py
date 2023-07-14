import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Définition des options de délimiteurs
delimiters = {'Virgule (,)': ',', 'Point-virgule (;)': ';', 'Tabulation (\\t)': '\t', 'Espace ( )': ' '}

# Titre de l'application
st.title("Mon application de visualisation ")

# Sidebar
st.sidebar.header("Paramètres")
parametre_1 = st.sidebar.slider("Paramètre 1", 0, 10, 5)
parametre_2 = st.sidebar.selectbox("Paramètre 2", ["Option 1", "Option 2", "Option 3"])

# Affichage des paramètres sélectionnés
st.write("Paramètre 1 sélectionné :", parametre_1)
st.write("Paramètre 2 sélectionné :", parametre_2)

# Section 1 de l'application
st.header("Section 1")
st.subheader("Graphique interactif")
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
chart_type = st.radio("Type de graphique", ["Ligne", "Barres"])
if chart_type == "Ligne":
    st.line_chart(data)
else:
    st.bar_chart(data)

# Section 2 de l'application
st.header("Section 2")
st.subheader("Analyse de texte")
text = st.text_area("Entrez du texte", "Saisissez votre texte ici...")
processed_text = text.upper()
st.write("Texte en majuscules :", processed_text)

# Section 3 de l'application
st.header("Section 3")
st.subheader("Téléchargement de fichiers")
uploaded_files = st.file_uploader("Télécharger plusieurs fichiers", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    delimiter_option = st.selectbox("Choisissez le délimiteur", list(delimiters.keys()))
    dataframes = [pd.read_csv(file, delimiter=delimiters[delimiter_option]) for file in uploaded_files]
    data = pd.concat(dataframes)
    st.write(data)
    if st.button("Tracer le graphique"):
        # Choix des colonnes pour les axes x et y
        x_axis = st.selectbox("Choisissez la colonne pour l'axe x", data.columns)
        y_axis = st.selectbox("Choisissez la colonne pour l'axe y", data.columns)

        # Choix du type de graphique
        plot_type = st.selectbox("Choisissez le type de graphique",
                                 ("Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Area Plot", "Pie Chart"))

        # Choix de la couleur
        color = st.color_picker("Choisissez une couleur")

        plt.figure(figsize=(12, 6))

        if plot_type == "Scatter Plot":
            plt.scatter(data[x_axis], data[y_axis], color=color)
        elif plot_type == "Line Plot":
            plt.plot(data[x_axis], data[y_axis], color=color)
        elif plot_type == "Bar Plot":
            plt.bar(data[x_axis], data[y_axis], color=color)

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(plot_type)
        st.pyplot()

# Section 4 de l'application
st.header("Section 4")
st.subheader("Affichage de vidéos YouTube")
video_url = st.text_input("URL de la vidéo YouTube")
if video_url:
    st.video(video_url)

# Footer
st.markdown("---")
st.write("Merci d'utiliser notre application !")
