import os
import cv2
import time

# ISL Hand Signs (updated list)
# To collect the words, you have to change the class names to required names of the words.
ISL_SIGNS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'W', 
    'X', 'Y', 'Z'
]

DATA_DIR = './isl_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of images to collect per sign
DATASET_SIZE = 1000

def collect_sign_images():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create directories for each sign
    for sign in ISL_SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

    # Collect images for each sign
    for sign in ISL_SIGNS:
        print(f'Preparing to collect data for sign: {sign}')

        # Wait for user to be ready
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display instruction for current sign
            cv2.putText(frame, f'Prepare to show ISL sign: {sign}', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Press "Q" when ready', (100, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('ISL Sign Collection', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # Collect 1000 images for the current sign
        counter = 0
        start_time = time.time()
        while counter < DATASET_SIZE:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Save the frame every 0.1 seconds
            current_time = time.time()
            if current_time - start_time >= 0.1:
                filename = os.path.join(DATA_DIR, sign, f'{counter}.jpg')
                cv2.imwrite(filename, frame)
                counter += 1
                start_time = current_time

            # Ask the user to switch hands after 500 images
            if counter == 500:
                print(f"Halfway there! Please switch hands for sign: {sign}.")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break

                    cv2.putText(frame, f"Switch hands for sign: {sign}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, "Press 'Q' to continue", (100, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('ISL Sign Collection', frame)
                    if cv2.waitKey(25) == ord('q'):
                        break

            # Display current progress
            cv2.putText(frame, f'Capturing {sign}: {counter}/{DATASET_SIZE}', 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('ISL Sign Collection', frame)

            # Allow user to interrupt
            if cv2.waitKey(25) == ord('q'):
                break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_sign_images()