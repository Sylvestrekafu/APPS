import streamlit as st

# Titre de l'application
st.title("Mon application complexe")

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
uploaded_file = st.file_uploader("Télécharger un fichier", type=["csv", "txt"])
if uploaded_file is not None:
    data = uploaded_file.read()
    st.write("Contenu du fichier :")
    st.write(data)

# Section 4 de l'application
st.header("Section 4")
st.subheader("Affichage de vidéos YouTube")
video_url = st.text_input("URL de la vidéo YouTube")
if video_url:
    st.video(video_url)

# Footer
st.markdown("---")
st.write("Merci d'utiliser notre application  !")
