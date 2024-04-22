import cv2
import easyocr
import os
import csv
from datetime import datetime

# Load the pre-trained Haar cascade classifier for detecting license plates
plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Path to the folder containing the images
folder_path = r'D:\numberplate\Car-Number-Plate-Recognition-Sysytem-master\Dataset'

# Create a CSV file to store the results
csv_file = 'detected_plates.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Serial Number', 'Time', 'Date', 'Day', 'Number Plate'])

# Get a list of all files in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize serial number
serial_number = 1

# Loop over each image file
for image_file in image_files:
    # Load the image
    img = cv2.imread(image_file)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the image
    plates = plat_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))

    # Initialize list to store detected license plate texts
    detected_texts = []

    # Loop over each detected license plate
    for (x, y, w, h) in plates:
        # Crop the detected license plate region
        plate_img = gray[y:y+h, x:x+w]

        # Use EasyOCR to extract text from the cropped license plate region
        result = reader.readtext(plate_img)

        # Extract text from EasyOCR result and append to detected_texts list
        detected_texts.append(' '.join([text[1] for text in result]))

        # Draw a bounding box around the detected license plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save the detected license plate text and other information to the CSV file
    for text in detected_texts:
        # Get current date and time
        now = datetime.now()
        date = now.strftime('%Y-%m-%d')
        time = now.strftime('%H:%M:%S')
        day = now.strftime('%A')

        # Append the results to the CSV file
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([serial_number, time, date, day, text])

        serial_number += 1

# Print a message indicating completion
print("License plate detection completed. Results saved in:", csv_file)

