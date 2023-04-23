from deepface import DeepFace
from deepface.commons import functions
import cv2
from pysubparser import parser
import numpy as np
import json
from sklearn.cluster import KMeans

imagesFolder = "<PATH_TO_FRAMES_FOLDER>"
scriptSrc = "<PATH_TO_SUBTITLES_SCRIPT>"
numCast = 5 # No. of speakers in clip to identify

# Info on the downloaded clip
clipData = {
    'title': "<CLIP_TITLE>",
    'yt_channel': "<YOUTUBE_CHANNEL_NAME>",
    'link': "<LINK_TO_VIDEO>"
}


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


def compareEmbeddings(embedding, embedding_cluster, threshold):
    # Calculate cosine similarity between the embedding and each embedding in the cluster
    similarities = np.dot(embedding, embedding_cluster.T) / (np.linalg.norm(embedding) * np.linalg.norm(embedding_cluster, axis=1))

    # Find the index of the embedding with the highest similarity
    max_index = np.argmax(similarities)

    # Return True if the similarity is above a threshold, False otherwise
    if similarities[max_index] > threshold:
        bio_score = similarities[max_index] - threshold
        return True, max_index, bio_score
    else:
        return False, None, -1


def getEmbedding(img):
  # Method to retrieve embedding of face found in image
  
  try:
    embedding = DeepFace.represent(img_path = img, enforce_detection = False)
    return embedding[0]["embedding"]
  except:
    print("Image does not exist")
    return None
  

def embeddingsForSegment(subtitles):
  # Method to get embeddings for every frame in a subtitle segment

  embeddings = []
  total_embeddings = [] ## Used for getting cast identities
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

        if current != None:

          embeddings_segment.append({
              "image": filePath,
              "img_code": imgCode,
              "embedding": current
          })
          total_embeddings.append(current)

        code += 1
      
      embeddings.append({
          "id": sub.index,
          "text": sub.text,
          "embeddings": embeddings_segment,
          "start": start,
          "end": end
      })

  return {
      "embeddings": embeddings,
      "total": total_embeddings
  }


def getCastEmbeddings(embeddings):
  # Method to cluster embeddings for each cast member
  
  # Initialize KMeans with the desired number of clusters
  kmeans = KMeans(n_clusters=numCast, random_state=42)

  # Convert the embeddings list to a numpy array
  embeddings_array = np.array(embeddings)

  # Fit KMeans to the embeddings array
  kmeans.fit(embeddings_array)

  # Get the cluster labels and centers
  cluster_labels = kmeans.labels_
  cluster_centers = kmeans.cluster_centers_

  # Print the cluster labels for each embedding
  print("Cluster labels:\n", cluster_labels)

  # Print the cluster centers
  print("Cluster centers:\n", cluster_centers)

  return kmeans


def mostPresentInSegment(sub, cast_cluster):
  # Method to find who is seen most in each segment
    
  current = {}
  for image in sub["embeddings"]:

    try:
      # Compare embedding from frame with each cast member's embedding
      results, identity_index, bio_score = compareEmbeddings(image["embedding"], cast_cluster.cluster_centers_, 0.4)

      if results == True:

        # If match, increment count for cast by 1
        if str(identity_index) in current.keys():
          current[str(identity_index)]['count'] += 1
          
          # If closer match, replace image
          if bio_score > current[str(identity_index)]["bio_score"]:
            current[str(identity_index)]["bio_score"] = bio_score
            current[str(identity_index)]["image"] = image["image"]
            current[str(identity_index)]["img_code"] = image["img_code"]
        else:
          current[str(identity_index)] = {}
          current[str(identity_index)]["id"] = identity_index
          current[str(identity_index)]["count"] = 1
          current[str(identity_index)]["image"] = image["image"]
          current[str(identity_index)]["img_code"] = image["img_code"]
          current[str(identity_index)]["bio_score"] = bio_score
    
    except Exception as e:
      print("ERROR: " + str(e))
        

  # Find who has highest count and return info
  highest_count = 0
  most_present_cast = {}
  for identity in current.keys():

    if current[identity]["count"] > highest_count:
      highest_count = current[identity]["count"]
      most_present_cast = current[identity]
  
  return most_present_cast


def save_matching_face_from_image(img_path, output_file_name, given_embedding):
  try:
    # Load the image
    img = cv2.imread(img_path)

    # Detect faces using the DeepFace library
    detected_faces = DeepFace.extract_faces(img_path=img)

    face_embeddings = []
    cropped_faces = []

    # Calculate embeddings for each detected face
    for face in detected_faces:
        x = face['facial_area']['x']
        y = face['facial_area']['y']
        w = face['facial_area']['w']
        h = face['facial_area']['h']

        # Add padding to the cropping coordinates
        x_padding = 50
        y_padding = 50
        x_start = max(0, x - x_padding)
        y_start = max(0, y - y_padding)
        x_end = min(img.shape[1], x + w + x_padding)
        y_end = min(img.shape[0], y + h + y_padding)

        cropped_face = img[y_start:y_end, x_start:x_end]
        img_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(img_rgb, model_name='VGG-Face', enforce_detection=False)
        face_embeddings.append(embedding)
        cropped_faces.append(cropped_face)

    # Compare the calculated embeddings with the given face embedding
    min_distance = float('inf')
    best_match_index = None

    for i, embedding in enumerate(face_embeddings):
        distance = np.linalg.norm(given_embedding - embedding[0]['embedding'])
        if distance < min_distance:
            min_distance = distance
            best_match_index = i

    # Save the matched face locally
    if best_match_index is not None:
        cv2.imwrite(output_file_name, cropped_faces[best_match_index])
  
  except Exception as e:
    print(e)


def saveToJSON(data, fileName):
  # Method to save data to JSON file

  # Save clip data and segment data
  output = {
      'clip_data': clipData,
      'scenes': data
  }

  json_object = json.dumps(output, indent=4, default=str)
  
  with open(fileName + ".json", "w") as outfile:
    outfile.write(json_object)


if __name__ == '__main__':
    # Parse script
    script = parser.parse(scriptSrc)

    # Get embeddings
    eb_segments = embeddingsForSegment(script)
    cast_cluster = getCastEmbeddings(eb_segments["total"])
    eb_segments = eb_segments["embeddings"]

    data = []
    for index, seg in enumerate(eb_segments):

        print("Starting segment: " + str(index))

        most_present = mostPresentInSegment(seg, cast_cluster)

        if 'image' in most_present.keys():
            save_matching_face_from_image(most_present["image"], "faceshots/" + str(index) + ".png", cast_cluster.cluster_centers_[int(most_present["id"])])

        data.append({
            "cast": most_present,
            "id": index,
            "text": seg["text"],
            "start": seg["start"],
            "end": seg["end"]
        })

    saveToJSON(data, "segments")