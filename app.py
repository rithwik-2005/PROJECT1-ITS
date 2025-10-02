import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO
import easyocr
import os

st.title("Vehicle Number Plate Detection & Recognition")

# Upload video
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Load your trained YOLO model
    model = YOLO("best.pt")  # your best weights
    reader = easyocr.Reader(['en'])

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video
    output_video_path = "output_video.mp4"
    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    plates_set = set()

    stframe = st.empty()
    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]

                ocr_result = reader.readtext(crop)
                plate_text = ""
                if ocr_result:
                    plate_text = ocr_result[0][1]
                    plates_set.add(plate_text)

                # Draw detection box and recognized text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out_video.write(frame)

        # Show frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb)
        processed_frames += 1
        progress_bar.progress(min(processed_frames / frame_count, 1.0))

    cap.release()
    out_video.release()

    # Save Excel
    df = pd.DataFrame({"Detected_Plates": list(plates_set)})
    excel_path = "detected_plates.xlsx"
    df.to_excel(excel_path, index=False)

    st.success("Processing complete!")
    st.video(output_video_path)
    st.download_button("Download Excel", excel_path)
