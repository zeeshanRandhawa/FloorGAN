import requests
from PIL import Image
from io import BytesIO

def get_satellite_image(address, api_key):
    # Step 1: Get the coordinates of the address using Geocoding API
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(geocode_url)
    geocode_data = response.json()
    
    if geocode_data['status'] != 'OK':
        print("Error in geocoding address.")
        return None
    
    location = geocode_data['results'][0]['geometry']['location']
    lat, lng = location['lat'], location['lng']
    
    # Step 2: Get the satellite image using Maps Static API
    static_map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom=20&size=640x640&maptype=satellite&key={api_key}"
    response = requests.get(static_map_url)
    
    if response.status_code != 200:
        print("Error in getting satellite image.", response.text)
        return None
    
    # Open the image
    image = Image.open(BytesIO(response.content))
    return image

# Example usage
address = "351 Fenley Ave, San Jose, CA, USA"
address = "391, adris ave, san jose, CA, USA"
api_key = "AIzaSyCxc8iUZTUQ8q-uu2snjE5xNmT7WBZ3Fxw"

satellite_image = get_satellite_image(address, api_key)
if satellite_image:
    satellite_image.show()