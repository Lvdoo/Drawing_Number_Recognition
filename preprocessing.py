from torchvision import transforms
from PIL import Image


# Traite l'image récupérer du canvas pour ensuite appliquer le CNN et prédire le nombre
def preprocess(image) : 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Convertit l'image en grayscale 
    image = image.convert("L")
    width, height = image.size

    # Récupère les coordonnées des pixels blanc
    pixel_coords = get_white_pixels(image, width, height)

    if len(pixel_coords) > 0 :

        # Récupère les coords max et min
        x_min, y_min, x_max, y_max = get_bounding_box(pixel_coords)

        # Ajoute une marge
        x_min, y_min, x_max, y_max = add_margin(x_min, y_min, x_max, y_max, width, height)

        # Crop l'image dans la zone du chiffre
        image = crop_image(image, x_min, y_min, x_max, y_max)

        # Rend l'image carré
        image = pad_to_square(image)

        # Met l'image sous un format 28x28
        image = resize_image(image)
        image.save("preprocessed_image.png")

        image = transform(image)
            
        # rajoute une dimension qui vaut 1 au tensor et qui correspond au batch
        image = image.unsqueeze(0)
        return image
    else : 
        return None
        
    
def get_white_pixels(image, width, height) : 
    pixels = image.load()
    coords = []
    for i in range(width) : 
        for j in range(height) :
            if pixels[i,j] > 25 :
                coords.append((i,j))
    return coords

def get_bounding_box(coords):
    x_list = [coord[0] for coord in coords]
    y_list = [coord[1] for coord in coords]

    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    return x_min, y_min, x_max, y_max

def add_margin(x_min, y_min, x_max, y_max, width, height) : 
    margin = 10
    x_min_margin = max(x_min - margin, 0)
    y_min_margin = max(y_min - margin, 0)
    x_max_margin = min(x_max + margin, width)
    y_max_margin = min(y_max + margin, height)

    return x_min_margin, y_min_margin, x_max_margin, y_max_margin

def crop_image(image, x_min, y_min, x_max, y_max) : 
    cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped_image

def pad_to_square(image) :
    width, height = image.size
    position_x = int((max(width, height) - width) / 2)
    position_y = int((max(width, height) - height) / 2)
    new_image = Image.new(size = (max(width, height), max(width, height)), color = 0, mode = "L")
    Image.Image.paste(new_image, image, (position_x, position_y))
    return new_image

def resize_image(image) : 
    resized_image = image.resize((20,20))
    position_x = int((28 - 20) / 2)
    position_y = int((28 - 20) / 2)
    last_image = Image.new(size = ((28,28)), color = 0, mode = "L")
    Image.Image.paste(last_image, resized_image, (position_x, position_y))
    return last_image