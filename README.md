Face Recognition Testing Instructions

1. Open 'face_recognition.py' file and save the following variables to the correct values:
    - imagesFolder (path to the folder containing the frames from the clip)
    - scriptSrc (path to the subtitles script file)
    - numCast (int referring to the number of speakers to be identified)
    - clipData (dict containing info on the clip being used)

2. Run 'face_recognition.py' which will save a JSON file 'segments.json' containing all data gathered and a folder 'faceshots' containing all faceshots created.

3. Open 'testing.py' file and save the following variables to the correct values:
    - dataFilePath (path to the JSON file 'segments.json')
    - frameFilePath (path to the folder containing the frames from the clip)
    - faceshotsFilePath (path to the folder containing the faceshots generated from running 'face_recognition.py')

4. Run 'testing.py' which will save an Excel spreadsheet representing all data generated from 'face_recognition.py'
