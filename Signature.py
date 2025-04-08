import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import Image
import warnings
import os
from PyPDF2 import PdfReader, PdfWriter
import io
from reportlab.pdfgen import canvas

warnings.filterwarnings("ignore", message="missing ScriptRunContext")

# --- Configuration ---
LOGO_URL = "https://via.placeholder.com/150"
SIGNATURE_FILENAME = "signature.png"
OPTIMIZED_SIGNATURE_FILENAME = "signature_final_optimized.png"
TEMP_PDF_FILENAME = "temp_uploaded_document.pdf"
SIGNED_PDF_FILENAME = "signed_document.pdf"
SIGNATURE_WIDTH = 200
SIGNATURE_HEIGHT = 90
SIGNATURE_MARGIN = (20, 20)
DRAWING_THUMB_FINGER_DISTANCE_THRESHOLD = 60  # Adjust as needed
FIST_GESTURE_THRESHOLD = 0.1  # Adjust based on landmark visibility

# --- Helper Functions ---
def optimize_signature(signature_path):
    try:
        signature = cv2.imread(signature_path, cv2.IMREAD_COLOR)
        gray_signature = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
        _, binary_signature = cv2.threshold(gray_signature, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped_signature = binary_signature[y : y + h, x : x + w]
            blurred_signature = cv2.GaussianBlur(cropped_signature, (3, 3), 0)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            final_signature = cv2.dilate(blurred_signature, kernel, iterations=1)
            final_signature_pil = Image.fromarray(final_signature)
            return final_signature_pil
        else:
            raise ValueError("No contours found. Please check the input signature image.")
    except Exception as e:
        raise RuntimeError(f"Error during optimization: {e}")

def add_signature_to_pdf(input_pdf_path, output_pdf_path, signature_image_path, margin=(10, 10)):
    try:
        pdf_reader = PdfReader(input_pdf_path)
        pdf_writer = PdfWriter()
        signature = Image.open(signature_image_path)
        signature_width, signature_height = SIGNATURE_WIDTH, SIGNATURE_HEIGHT

        for page in pdf_reader.pages:
            page_width = float(page.mediabox[2])
            page_height = float(page.mediabox[3])
            x_position = page_width - signature_width - margin[0]
            y_position = margin[1]

            signature_pdf = io.BytesIO()
            c = canvas.Canvas(signature_pdf, pagesize=(page_width, page_height))
            c.drawImage(signature_image_path, x_position, y_position, width=signature_width, height=signature_height)
            c.save()

            signature_pdf.seek(0)
            signature_overlay = PdfReader(signature_pdf)
            page.merge_page(signature_overlay.pages[0])
            pdf_writer.add_page(page)

        with open(output_pdf_path, "wb") as output_file:
            pdf_writer.write(output_file)
    except Exception as e:
        raise RuntimeError(f"Error adding signature to PDF: {e}")

def is_fist(hand_landmarks):
    if hand_landmarks:
        # Landmark indices for finger tips (based on typical MediaPipe order)
        thumb_tip = hand_landmarks.landmark[4].y
        index_finger_tip = hand_landmarks.landmark[8].y
        middle_finger_tip = hand_landmarks.landmark[12].y
        ring_finger_tip = hand_landmarks.landmark[16].y
        pinky_finger_tip = hand_landmarks.landmark[20].y

        # Landmark indices for PIP joints
        index_finger_pip = hand_landmarks.landmark[6].y

        # Check if all fingers (except thumb) are below the PIP joint of the index finger
        if middle_finger_tip > index_finger_pip or \
           ring_finger_tip > index_finger_pip or \
           pinky_finger_tip > index_finger_pip:
            return False
        return True
    return False

# --- Streamlit UI ---
st.title("Automated Signature System")

st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        position: fixed;
        top: 10px;  /* Adjust the vertical placement */
        left: 10px; /* Adjust the horizontal placement */
        z-index: 1000; /* Ensures the logo stays on top */
    }
    .logo-container img {
        width: 150px; /* Adjust the width of the logo */
        height: auto; /* Maintain aspect ratio */
    }
    </style>
    <div class="logo-container">
        <img src="https://via.placeholder.com/150" alt="logo.png">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Air Canvas for Signature Capture ---
if st.sidebar.button("Start Air Canvas"):
    st.write("Press 'S' to save, 'C' to clear, 'B' for blue, 'G' for green, 'R' for red, 'Y' for yellow, and 'Q' to close.")

    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    paintWindow = np.zeros((800, 1200, 3), dtype=np.uint8) + 255

    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)
        hand_landmarks = None

        if result.multi_hand_landmarks:
            for handslms in result.multi_hand_landmarks:
                hand_landmarks = handslms
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                break # Process only the first detected hand

        can_draw = False
        if hand_landmarks:
            index_finger_tip = (int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * width),
                                int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * height))
            thumb_tip = (int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x * width),
                         int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y * height))

            distance = ((index_finger_tip[0] - thumb_tip[0]) ** 2 + (index_finger_tip[1] - thumb_tip[1]) ** 2) ** 0.5
            if distance > DRAWING_THUMB_FINGER_DISTANCE_THRESHOLD and not is_fist(hand_landmarks):
                can_draw = True
                center = index_finger_tip
                cv2.circle(frame, center, 5, colors[colorIndex], -1)
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
            else:
                bpoints.append(deque(maxlen=1024))
                blue_index += 1
                gpoints.append(deque(maxlen=1024))
                green_index += 1
                rpoints.append(deque(maxlen=1024))
                red_index += 1
                ypoints.append(deque(maxlen=1024))
                yellow_index += 1

        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cropped_paint_window = paintWindow.copy()
            filename = SIGNATURE_FILENAME
            cv2.imwrite(filename, cropped_paint_window)
            print(f"Signature saved as {filename}.")
        elif key == ord('q'):
            break
        elif key == ord('c'):
            paintWindow[:] = 255
            bpoints = [deque(maxlen=1024)]
            gpoints = [deque(maxlen=1024)]
            rpoints = [deque(maxlen=1024)]
            ypoints = [deque(maxlen=1024)]
            blue_index = 0
            green_index = 0
            red_index = 0
            yellow_index = 0
        elif key == ord('b'):
            colorIndex = 0  # Blue
        elif key == ord('g'):
            colorIndex = 1  # Green
        elif key == ord('r'):
            colorIndex = 2  # Red
        elif key == ord('y'):
            colorIndex = 3  # Yellow

    cap.release()
    cv2.destroyAllWindows()

# --- Signature Optimization ---
if st.sidebar.button("Optimize Signature"):
    try:
        if os.path.exists(SIGNATURE_FILENAME):
            optimized_img = optimize_signature(SIGNATURE_FILENAME)
            st.image(optimized_img, caption="Optimized Signature", use_column_width=True)
            optimized_img.save(OPTIMIZED_SIGNATURE_FILENAME, optimize=True)
            st.success("Signature optimized successfully!")
        else:
            st.error("Please capture a signature using 'Start Air Canvas' first.")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Upload and Sign PDF ---
st.sidebar.title("Upload and Sign PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if st.sidebar.button("Add Signature to PDF"):
    if uploaded_pdf:
        with open(TEMP_PDF_FILENAME, "wb") as temp_pdf:
            temp_pdf.write(uploaded_pdf.read())

        if not os.path.exists(OPTIMIZED_SIGNATURE_FILENAME):
            st.error("Optimized signature image not found! Please capture and optimize your signature first.")
        else:
            try:
                add_signature_to_pdf(TEMP_PDF_FILENAME, SIGNED_PDF_FILENAME, OPTIMIZED_SIGNATURE_FILENAME, SIGNATURE_MARGIN)
                st.success("Signature added to the PDF successfully!")

                with open(SIGNED_PDF_FILENAME, "rb") as signed_pdf:
                    st.download_button(
                        label="Download Signed PDF",
                        data=signed_pdf,
                        file_name=SIGNED_PDF_FILENAME,
                        mime="application/pdf",
                    )
            except RuntimeError as e:   
                st.error(f"Error while adding signature: {e}")
    else:
        st.error("Please upload a PDF file first.")
