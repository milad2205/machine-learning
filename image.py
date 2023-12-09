from matplotlib import pyplot as plt
import cv2

image = cv2.imread("Me.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
model = cv2.CascadeClassifier("model.xml")

face = model.detectMultiScale(image)
print(face)
print(image.shape)

# image = cv2.rectangle(image, (100,100), (500,500) , (0,0,255) , 4 )
x = face[0][0]
y = face[0][1]
a = face[0][2]
b = face[0][3]

image = cv2.rectangle(image, (x,y), (x+a,y+b) , (0,0,255) , 4 )

plt.imshow(image)
plt.show()