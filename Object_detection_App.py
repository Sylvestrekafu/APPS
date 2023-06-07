import streamlit as st
import torch
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
    return model

def run_detection(model, img):
    results = model(img)
    return results

def count_classes(results, class_names):
    class_counts = {}
    for det in results.xyxy[0]:
        class_id = int(det[5])
        class_name = class_names[class_id]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    return class_counts

def main():
    st.title("Détection des objets avec YOLOv5 et Streamlit")
    model = load_model()

    files = st.file_uploader("Téléchargez une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if files:
        class_names = model.module.names if hasattr(model, 'module') else model.names  # Get class names

        for file in files:
            image = Image.open(file)
            st.image(image, caption=f"Image originale: {file.name}", use_column_width=True)

            if st.button(f"Détection pour {file.name}"):
                results = run_detection(model, image)
                image_with_boxes = results.render()
                st.image(image_with_boxes, caption=f"Image avec détections: {file.name}", use_column_width=True)

                # Compter le nombre d'objets détectés par classe
                class_counts = count_classes(results, class_names)
                st.write("Nombre d'objets détectés par classe:")
                for class_name, count in class_counts.items():
                    st.write(f"{class_name}: {count}")

if __name__ == "__main__":
    main()