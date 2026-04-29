from torchvision import transforms

# Traite l'image récupérer du canvas pour ensuite appliquer le CNN et prédire le nombre
def preprocess(image) : 
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Convertit l'image en grayscale 
    image = image.convert("L")

    # Récupère les coordonnées des pixels blanc
    pixel_coords = get_white_pixels(image)

    if len(pixel_coords) > 0 : 
        # Récupère les coords max et min
        min_coords, max_coords = get_max_min_coords(pixel_coords)
        image = transform(image)
            
            # rajoute une dimension qui vaut 1 au tensor et qui correspond au batch
        image = image.unsqueeze(0)
        return image
    
    else :
        text = "You didn't draw anything"
        return text
    
def get_white_pixels(image) : 
    pixels = image.load()
    width, height = image.size()
    coords = []
    for i in range(width) : 
        for j in range(height) :
            if pixels[i,j] > 25 :
                coords.append((i,j))
    return coords

def get_max_min_coords(coords) : 
    min_coords = []
    max_coords = []
    
