def remove_small_values(image):
    """
    Input : image (2d array)
    Output: Masked image where all values below the mean of the image are set to 0
    """
    image_mean = np.mean(image)
    image = np.ma.masked_array(np.ma.masked_less(image, image_mean))
    return image
