from deepface import DeepFace
from pysubparser import parser
from google.colab import drive
import face_recognition as fr
import numpy as np
import json

drive.mount('/content/drive')

castFolder = "./drive/MyDrive/Projects/CSC1028/cast"
imagesFolder = "./drive/MyDrive/Projects/CSC1028/YoutubeFiles/images"
scriptSrc = "./drive/MyDrive/Projects/CSC1028/YoutubeFiles/script.srt"
dbFolder = "./drive/MyDrive/Projects/CSC1028/cast"

castMetadata = [
    {
        "name": "Katniss Everdeen",
        "images": [
            "./drive/MyDrive/Projects/CSC1028/cast/0.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/5.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/9.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/10.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/15.jpg"
        ]
    },
    {
        "name": "Peeta Mellark",
        "images": ["./drive/MyDrive/Projects/CSC1028/cast/1.jpg"]
    },
    {
        "name": "Primrose Everdeen",
        "images": [
            "./drive/MyDrive/Projects/CSC1028/cast/3.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/6.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/8.jpg"
        ]
    },
    {
        "name": "Effy",
        "images": [
            "./drive/MyDrive/Projects/CSC1028/cast/4.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/7.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/14.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/16.jpg",
            "./drive/MyDrive/Projects/CSC1028/cast/17.jpg"
        ]
    }
]


def getEmbedding(img):
  # Method to retrieve embedding of face found in image
  
  try:
    embedding = DeepFace.represent(img_path = img, enforce_detection = False)
    return embedding[0]["embedding"]
  except:
    print("Image does not exist")

def embeddingsForSegment(subtitles):
  # Method to get embeddings for every frame in a subtitle segment

  embeddings = []
  for index, sub in enumerate(subtitles):

    print("Starting subtitles segment index: " + str(index))

    embeddings_segment = []
    start = sub.start
    end = sub.end

    # Starting image
    startSecond = (start.hour * 3600) + (start.minute * 60) + start.second + float("0." + str(start.microsecond))
    startCode = int(startSecond * 24)

    # End image
    endSecond = (end.hour * 3600) + (end.minute * 60) + end.second + float("0." + str(end.microsecond))
    endCode = int(endSecond * 24)

    # Loop through each image from start of segment to end
    code = startCode
    while code <= endCode:
      imgCode = formatImageCount(code)
      filePath = imagesFolder + "/" + imgCode + ".png"

      current = getEmbedding(filePath)

      embeddings_segment.append({
          "image": filePath,
          "embedding": current
      })
      code += 1
    
    embeddings.append({
        "id": sub.index,
        "text": sub.text,
        "embeddings": embeddings_segment,
        "start": start,
        "end": end
    })

  return embeddings


def getCastEmbeddings():
  # Method to get embeddings for each cast member

  for index, cast in enumerate(castMetadata):

    eb = getEmbedding(cast["images"][len(cast["images"]) - 1])
    # Save to castMetadata
    castMetadata[index]["embedding"] = eb


def mostPresentInSegment(sub):
  # Method to find who is seen most in each segment
    
  current = {}
  for image in sub["embeddings"]:

    for cast in castMetadata:

      try:
        # Compare embedding from frame with each cast member's embedding
        results = fr.compare_faces([np.array(image["embedding"])], np.array(cast["embedding"]), tolerance=0.4)

        cast_name = cast['name']

        if results[0] == True:
          # If match, increment count for cast by 1

          print(image["image"], cast_name)
          
          if cast_name in current.keys():
            current[cast_name]['count'] += 1
          else:
            current[cast_name] = {}
            current[cast_name]["name"] = cast_name
            current[cast_name]['count'] = 1
      
      except:
        print("Error 1")
        

  # Find who has highest count and return name and count
  highest_count = 0
  most_present_cast = {}
  for cast_name in current.keys():
    print(current[cast_name])

    if current[cast_name]["count"] > highest_count:
      highest_count = current[cast_name]["count"]
      most_present_cast = current[cast_name]
  
  return most_present_cast


def formatImageCount(count):
    # Method to format number into 4 digits

    newCount = count
    if count < 10:
      newCount = "000" + str(count)
    elif count < 100:
      newCount = "00" + str(count)
    elif count < 1000:
      newCount = "0" + str(count)
    
    return str(newCount)


def saveToJSON(data, fileName):
  # Method to save data to JSON file

  json_object = json.dumps(data, indent=4, default="str")
  
  with open(fileName + ".json", "w") as outfile:
    outfile.write(json_object)


if __name__ == '__main__':
    # Parse script
    script = parser.parse(scriptSrc)

    # Get embeddings
    eb_segments = embeddingsForSegment(script)
    getCastEmbeddings()

    data = []
    for index, seg in enumerate(eb_segments):

    print("Starting segment: " + str(index))

    most_present = mostPresentInSegment(seg)

    data.append({
        "cast": most_present,
        "id": index,
        "text": seg["text"],
        "start": seg["start"],
        "end": seg["end"]
    })

    saveToJSON(data, "segments")