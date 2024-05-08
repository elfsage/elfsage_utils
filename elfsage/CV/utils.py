from elfsage.images import to_grey
import cv2


def mask_to_rect(mask, eps=0.05):
    mask = to_grey(mask)
    thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = eps*cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    rect = cv2.boundingRect(approx)

    return rect
