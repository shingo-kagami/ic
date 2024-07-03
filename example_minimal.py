import cv2

img = cv2.imread("lena.jpg")
cv2.imshow("test_win", img)
cv2.waitKey(5000)
cv2.destroyAllWindows()
