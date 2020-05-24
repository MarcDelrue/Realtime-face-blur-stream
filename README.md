This project is the combination of two projects of Adrian Rosebrock from https://www.pyimagesearch.com/
which are "Blur and anonymize faces with OpenCV and Python" and "OpenCV – Stream video to web browser/HTML page"

# Realtime-face-blur-stream
uses OpenCV, Flask and Raspberry Pi 4 and its deidcated camera v2 in order to create a realtime face blur on web browser

# blur_face.py example
python blur_face.py --image examples/adrian.jpg --face face_detector --method pixelated

# blur_face_video.py example
python blur_face_video.py --face face_detector --method simple

# webstreaming.py example 
python webstreaming.py --ip 0.0.0.0 --port 8000 --face face_detector --method simple

