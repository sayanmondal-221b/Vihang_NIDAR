import cv2
import numpy as np

def detect_and_mark_yellow_spots(image_path, output_path="yellow_spots_detected.jpg", min_area=10):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_spots = []
    output = image.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
                
            detected_spots.append({
                "contour": contour,
                "area": area,
                "centroid": (cX, cY)
            })
            
            
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)
            
            cv2.circle(output, (cX, cY), 4, (0, 0, 255), -1)
            
            cv2.putText(output, f"{area}", (cX-20, cY-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    
    cv2.imwrite(output_path, output)
    
    cv2.imshow("Detected Yellow Spots", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return detected_spots, output

if __name__ == "__main__":
    input_image = "input_image.png"  
    output_image = "detected_spots.png"
    
    spots, marked_image = detect_and_mark_yellow_spots(
        image_path=input_image,
        output_path=output_image,
        min_area=20
    )
    
    print(f"Detection complete. Output saved to {output_image}")
    print(f"Detected {len(spots)} yellow spots:")
    for i, spot in enumerate(spots, 1):
        print(f"Spot {i}: Area = {spot['area']} px, Center = {spot['centroid']}")
