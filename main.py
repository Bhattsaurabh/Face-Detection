import face_recognition as fr
import cv2
import matplotlib.pyplot as plt

img1=fr.load_image_file("D:\\face-detection\\img1.jpg")
img2=fr.load_image_file("D:\face-detection\img2.JPG")
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(img1[:, :, ::-1])
plt.show()
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2[:, :, ::-1])
plt.show()

enc1 = fr.face_encodings(img1)[0]
print(enc1)
enc2 = fr.face_encodings(img2)[0]
print(enc2)

result = fr.compare_faces([enc1], enc2)
print(result)
