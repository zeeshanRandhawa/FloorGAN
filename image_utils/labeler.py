from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
'''


'''

class Labeler:

    def __init__(self):
        self.future = None

        # Define the colors and corresponding room names
        self.room_colors = {
                (255,210,116,255): 'Bedroom',     # yellow with alpha channel
                (238,77,77,255): 'Living Room',   # Red with alpha channel
                (191,227,232,255): 'Balcony', # blue with alpha channel
                (190,190,190,255): 'Bathroom', #Gray Bathroom
                (198,124,123,255): 'Kitchen'
            }

    def _find_regions_of_color(self, image, target_color):
        np_image = np.array(image)
        mask = np.all(np_image[:, :, :4] == target_color, axis=-1).astype(np.uint8) * 255

        # Use connected component labeling
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = []
        for i in range(1, num_labels):  # Skip the background
            x, y, width, height, area = stats[i] #plot width and height
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            regions.append(((cx, cy), (width, height)))
        
        return regions

    def label_rooms(self, image_path, output_path):
        '''
        Label each room
        '''
        print(f"Opening image at: {image_path}")
        image = Image.open(image_path).convert('RGBA')
        draw = ImageDraw.Draw(image)

        for color, room_name in self.room_colors.items():
            regions = self._find_regions_of_color(image, color)
            for center, (width, height) in regions:
                x, y = center
                factor = 0.80
                # Find appropriate font size
                if(room_name != "Living Room"):
                    factor = 1
                font_size = 10  # Start with a small font size
                font = ImageFont.truetype("arial.ttf", font_size)
                bbox = draw.textbbox((0, 0), room_name, font=font)
                text_width = bbox[2] - bbox[0]

                # Increase font size until text fits within the room's width
                while text_width < (width*factor) and font_size < height:
                    font_size += 1
                    font = ImageFont.truetype("arial.ttf", font_size)
                    bbox = draw.textbbox((0, 0), room_name, font=font)
                    text_width = bbox[2] - bbox[0]

                # Decrease font size if it exceeds the room's width
                while text_width > (width*factor):
                    font_size -= 1
                    font = ImageFont.truetype("arial.ttf", font_size)
                    bbox = draw.textbbox((0, 0), room_name, font=font)
                    text_width = bbox[2] - bbox[0]

                # Draw the text
                draw.text((x - text_width // 2, y - bbox[3] // 2), room_name, fill=(0, 0, 0, 255), font=font)

        print(f"Saving labeled image to: {output_path}")
        image.save(output_path)


    





# Testing
labeler = Labeler()
labeler.label_rooms(r'D:\Floor generator\houseganpp\dump\fp_final_2.png', r'D:\Floor generator\houseganpp\dump\labeled_floorplan.png')
