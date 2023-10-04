import face_recognition as fr
import cv2
import matplotlib.pyplot as plt

img1=fr.load_image_file("C:\\Users\\hp\\Documents\\my doc\\passport.jpg")
img2=fr.load_image_file("D:\\mussorie\\IMG_7654.JPG")
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

# detect faces in the image

face_locations = fr.face_locations(img1)

# loop through the face locations and draw rectangles around the faces

for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(img1, (left, top), (right, bottom), (0, 0, 255), 2)
plt.imshow(img1[:, :, ::-1])
plt.show()

img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


face_locations = fr.face_locations(img2)

for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(img2, (left, top), (right, bottom), (0, 0, 255), 2)
plt.imshow(img2[:, :, ::-1])
plt.show()

enc1 = fr.face_encodings(img1)[0]
print(enc1)
enc2 = fr.face_encodings(img2)[0]
print(enc2)

result = fr.compare_faces([enc1], enc2)
print(result)
