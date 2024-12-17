from deepface import DeepFace

img_path =r".\img2.jpg"
analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'])
print(analysis)
