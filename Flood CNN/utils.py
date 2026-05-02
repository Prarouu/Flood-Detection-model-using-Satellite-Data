import matplotlib.pyplot as plt

def show_prediction(image, mask, pred):
    plt.figure(figsize=(10,5))

    plt.subplot(1,3,1)
    plt.title("VV")
    plt.imshow(image[0], cmap='gray')

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(mask[0], cmap='gray')

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred, cmap='gray')

    plt.show()