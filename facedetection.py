from deepface import DeepFace

img_path = r"C:\Users\leela\OneDrive\Desktop\PROJECT\img1.jpg"

faces = DeepFace.extract_faces(img_path)
print(faces)

analysis = DeepFace.analyze(img_path, actions=['age', 'gender', 'emotion'])
print("Analysis of img1.jpg:")
print(analysis)

img1_path = r"C:\Users\leela\OneDrive\Desktop\PROJECT\img1.jpg"
img2_path = r"C:\Users\leela\OneDrive\Desktop\PROJECT\img2.jpg"

result = DeepFace.verify(img1_path, img2_path)
print("Verification result between img1.jpg and img2.jpg:")
print(result)

imgs = [
    r"C:\Users\leela\OneDrive\Desktop\PROJECT\img1.jpg",
    r"C:\Users\leela\OneDrive\Desktop\PROJECT\img2.jpg",
]
df = DeepFace.find(imgs, db_path=r"C:\Users\leela\OneDrive\Desktop\Database")
print("Similar faces found:")
print(df)