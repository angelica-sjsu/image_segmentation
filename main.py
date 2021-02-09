import data_prep
import neural_net
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from IPython.display import clear_output
from tensorflow.keras.optimizers import Adam


def train_model(X, y):
    '''
    calls all model related operations
    @X: dictionary containing images
    @y: dictionary containing groundtruth
    '''
    x_train = X['train']
    y_train = y['train']
    x_test = X['test']
    y_test = y['test']

    weight_path = "{}_weights.best.hdf5".format('cxr_reg')

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=3,
                                       verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=15)  # probably needs to be more patient, but kaggle time is limited
    callbacks_list = [checkpoint, early, reduceLROnPlat]

    unet_model = neural_net.unet((512, 512, 3))

    unet_model.compile(optimizer=Adam(),
                       # loss=tf.keras.losses.CategoricalCrossentropy(),
                       loss = tf.keras.losses.BinaryCrossentropy(),
                       metrics=['accuracy'])

    # real-time data augmentation
    data_aug = data_prep.data_augmentation()
    data_aug.fit(x_train)

    history = unet_model.fit(x_train, y_train,
                             batch_size=4,
                             epochs=50,
                             validation_data=(x_test, y_test),
                             callbacks=callbacks_list)

    #print(history)
    neural_net.history_plotter(history)


if __name__ == '__main__':
    train_images_path = ['ss_train_voc/angelica/JPEGImages/',
                         'ss_train_voc/olivia/JPEGImages']
    train_masks_path = ['ss_train_voc/angelica/SegmentationClassPNG/',
                        'ss_train_voc/olivia/SegmentationClassPNG/']
    test_images_path= ['ss_test_voc/angelica/JPEGImages/',
                       'ss_test_voc/olivia/JPEGImages/']
    test_masks_path = ['ss_test_voc/angelica/SegmentationClassPNG/',
                       'ss_test_voc/olivia/SegmentationClassPNG/']

    train_imgs, train_msks = data_prep.get_data(train_images_path, train_masks_path)
    data_prep.display_images([train_imgs[117], train_msks[117]])
    test_imgs, test_msks = data_prep.get_data(test_images_path, test_masks_path)
    data_prep.display_images([test_imgs[0], test_msks[0]])
    
    # train_model(X={'train':train_imgs, 'test':test_imgs},
    #             y={'train':train_msks, 'test':test_msks})




