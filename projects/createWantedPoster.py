import os
from wantedposter.wantedposter import WantedPoster
from PIL import Image


def create_wanted_poster(first_name,last_name,bounty_amount,portrait_poster_path = None,show_image = True):
    # Create WantedPoster object
    wanted_poster = WantedPoster(portrait_poster_path, first_name, last_name, bounty_amount)

    # Generate poster
    path = wanted_poster.generate()
    
    # Show image
    if show_image==True:
        Image.open(path).show()
    return path

def create_wanted_posters(first_names,last_names,bounty_amounts,portrait_poster_paths):
    dataset_paths = [
        create_wanted_poster(first_name,last_name,bounty_amount,portrait_poster_path,False) 
        for first_name,last_name,bounty_amount,portrait_poster_path in 
        zip(
                first_names,
                last_names,
                bounty_amounts,
                portrait_poster_paths
        )
    ]
    return dataset_paths

if __name__ == '__main__':
    first_name = "Prakhar"
    last_name = "Gandhi K."
    bounty_amount = pow(2,31)
    portrait_poster_path = r"C:\Users\gprak\Downloads\DiscUdemy Courses\resumephoto.jpg"
    create_wanted_poster(first_name,last_name,bounty_amount,portrait_poster_path)