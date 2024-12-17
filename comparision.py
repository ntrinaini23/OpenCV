from deepface import DeepFace

img1_path = r"C:\Users\leela\OneDrive\Desktop\PROJECT\img1.jpg"
img2_path = r"C:\Users\leela\OneDrive\Desktop\PROJECT\img2.jpg"

result = DeepFace.verify(img1_path, img2_path)
print(result)
