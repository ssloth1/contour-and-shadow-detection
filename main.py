# Author: James Bebarski
# Date: 6/28/2024

import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Default values:
        # These can be changed, but I got the best results with these values

        # Kernel size, for Gaussian blur
        self.kernel_size = 11

        # Hue, Saturation, and Value (HSV) threshold values
        self.hue_lower_thresh = 0
        self.hue_upper_thresh = 180
        self.saturation_lower_thresh = 150
        self.saturation_upper_thresh = 255
        self.v_lower_thresh = 110
        self.v_upper_thresh = 255

        # Dilation and Erosion iterations
        self.dilation_iterations = 5
        self.erosion_iterations = 5

        # methods to calculate the window size, scale, and process the image
        self.screen_res = 1536, 1152
        self.scale_width = self.screen_res[0] / self.image.shape[1]
        self.scale_height = self.screen_res[1] / self.image.shape[0]
        self.scale = min(self.scale_width, self.scale_height)
        self.window_width = int(self.scale * self.image.shape[1])
        self.window_height = int(self.scale * self.image.shape[0])


    # In my implementation, I used the HSV color space to threshold the image.
    # I actually got the idea to explore other color spaces other than grayscale from class.
    # Initially I was getting pretty bad results since the shadow of the yellow block was nearly
    # the same color as the block to it's left.
    # 
    # Although, using the HSV space yielded better results, there were still some issues 
    # with the shadow of the yellow block and the block to it's left. 
    def process_image(self):
        output_image = self.image.copy()

        # thresholding the HSV image to get the ideal hue, saturation, and brightness/lightness values
        image = cv2.inRange(self.hsv_image, 
                           np.array([self.hue_lower_thresh, self.saturation_lower_thresh, self.v_lower_thresh]), 
                           np.array([self.hue_upper_thresh, self.saturation_upper_thresh, self.v_upper_thresh]))

        # apply Gaussian blur to the image
        image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

        # apply dilation and erosion to the image
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        image = cv2.dilate(image, kernel, iterations=self.dilation_iterations)
        image = cv2.erode(image, kernel, iterations=self.erosion_iterations)

        # get the contours of the image
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_list = []
        
        # draw the contours and centers of the contours
        for contour in contours:

            # only draw contours with an area greater than 50 pixels
            if cv2.contourArea(contour) > 50:
                contours_list.append(contour)
                moment = cv2.moments(contour)

                # if the contour has a moment, calculate the center of the contour
                if moment["m00"] != 0:
                    centerX = int(moment["m10"] / moment["m00"])
                    centerY = int(moment["m01"] / moment["m00"])

                # if the contour does not have a moment, set the center to 0, 0
                else:
                    centerX, centerY = 0, 0

                # draw the contour and center, green for the contour and purple for the center
                cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(output_image, (centerX, centerY), 5, (200, 0, 200), -1)

        resized_output_image = cv2.resize(output_image, (self.window_width, self.window_height))
        cv2.imshow('Contours & Centers', resized_output_image)
        return output_image
    
    # to string method to print the values of the trackbars
    def __str__(self):
        return (f"Parameters:\n"
                f"  Kernel Size: {self.kernel_size}\n"
                f"  Hue Range: [{self.hue_lower_thresh}, {self.hue_upper_thresh}]\n"
                f"  Saturation Range: [{self.saturation_lower_thresh}, {self.saturation_upper_thresh}]\n"
                f"  Value Range: [{self.v_lower_thresh}, {self.v_upper_thresh}]\n"
                f"  Dilation Iterations: {self.dilation_iterations}\n"
                f"  Erosion Iterations: {self.erosion_iterations}")
    
    # All these methods help update the image via the values selected from the trackbars
    def update_kernel_size(self, value):
        self.kernel_size = value if value % 2 == 1 else value + 1
        self.process_image()

    def update_hue_lower(self, value):
        self.hue_lower_thresh = value
        self.process_image()

    def update_hue_upper(self, value):
        self.hue_upper_thresh = value
        self.process_image()

    def update_saturation_lower(self, value):
        self.saturation_lower_thresh = value
        self.process_image()

    def update_saturation_upper(self, value):
        self.saturation_upper_thresh = value
        self.process_image()

    def update_v_lower(self, value):
        self.v_lower_thresh = value
        self.process_image()

    def update_v_upper(self, value):
        self.v_upper_thresh = value
        self.process_image()

    def update_dilation_iterations(self, value):
        self.dilation_iterations = value
        self.process_image()

    def update_erosion_iterations(self, value):
        self.erosion_iterations = value
        self.process_image()

def main():
    
    # read in blocks image and create a window for the image and the trackbars
    blocks = ImageProcessor('images/blocks.jpg')

    cv2.namedWindow('Contours & Centers', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Contours & Centers', blocks.window_width, blocks.window_height)

    # Sliders for the kernel size, hue, saturation, and value thresholds, and dilation and erosion iterations
    cv2.createTrackbar('Kernel Size', 'Contours & Centers', blocks.kernel_size, 20, blocks.update_kernel_size)
    cv2.createTrackbar('Hue Lower', 'Contours & Centers', blocks.hue_lower_thresh, 180, blocks.update_hue_lower)
    cv2.createTrackbar('Hue Upper', 'Contours & Centers', blocks.hue_upper_thresh, 180, blocks.update_hue_upper)
    cv2.createTrackbar('Saturation Lower', 'Contours & Centers', blocks.saturation_lower_thresh,180, blocks.update_saturation_lower)
    cv2.createTrackbar('Saturation Upper', 'Contours & Centers', blocks.saturation_upper_thresh, 255, blocks.update_saturation_upper)
    cv2.createTrackbar('V Lower', 'Contours & Centers', blocks.v_lower_thresh, 255, blocks.update_v_lower)
    cv2.createTrackbar('V Upper', 'Contours & Centers', blocks.v_upper_thresh, 255, blocks.update_v_upper)
    cv2.createTrackbar('Dilation Iterations', 'Contours & Centers', blocks.dilation_iterations, 10, blocks.update_dilation_iterations)
    cv2.createTrackbar('Erosion Iterations', 'Contours & Centers', blocks.erosion_iterations, 10, blocks.update_erosion_iterations)

    blocks.process_image()

    print('Press any key to save the image and exit')

    # wait for a key press to save the image and exit
    while True:
        key = cv2.waitKey(1)
        if key != -1:
            output_image = blocks.process_image()
            cv2.imwrite('images/output.jpg', output_image)
            print(blocks)
            print('Output image saved as output.jpg')
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
