import cv2
import numpy as np

# def find_largest_contour(mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         return max(contours, key=cv2.contourArea)
#     else:
#         return None
    

def find_largest_contours(mask, n=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        return sorted_contours[:len(sorted_contours)]
    else:
        return None
def main():
    # Read the image
    image = cv2.imread('C:\\Users\\quest\\Pictures\\Screenshots\\318_adris.png')

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find the largest contour (assumed to be the largest grass area)
    top_contours = find_largest_contours(mask, n=3)

    for cont in top_contours:
        x, y, w, h = cv2.boundingRect(cont)

        # Calculate the center point of the bounding rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw the bounding rectangle and center point on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Display the result
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
    largest_contour = top_contours[0]
    second_largest = top_contours[2]
    



    if largest_contour is not None:
        # Calculate the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Calculate the center point of the bounding rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw the bounding rectangle and center point on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Display the result
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if second_largest is not None:
        # Calculate the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(second_largest)

        # Calculate the center point of the bounding rectangle
        center_x = x + w // 2
        center_y = y + h // 2

        # Draw the bounding rectangle and center point on the original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Display the result
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No grass found in the image.")

if __name__ == "__main__":
    main()