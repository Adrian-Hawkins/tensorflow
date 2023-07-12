import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

# def plot(feature, x=None, y=None):
#     plt.figure(figsize=(10,8))
#     plt.scatter(train_features[feature], train_labels, label='Data')
#     if x is not None and y is not None:
#         plt.plot(x, y, color='k', label='Predictions')
#     plt.xlabel(feature)
#     plt.ylabel('MPG')
#     plt.show()