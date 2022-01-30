from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from dext.model.marine_debris_utils import create_multibox_head
from dext.model.marine_debris_utils import create_prior_boxes


def SSD_ResNet20(num_classes=21, input_shape=(96, 96, 1),
                 num_priors=[4, 6, 6, 6, 4, 4], l2_loss=0.0005,
                 return_base=False, weight_folder=None):
    image = Input(shape=input_shape, name='image')
    resnet20 = load_model(weight_folder + 'fls-turntable-objects-pretrained-resnet20-platform-96x96.hdf5')
    resnet20.trainable = True
    resnet_out = resnet20(image)

    conv4_3_norm = resnet20.get_layer('conv2d_98').output
    fc7 = resnet20.get_layer('conv2d_105').output

    # EXTRA layers in SSD -----------------------------------------------------
    # Block 6 -----------------------------------------------------------------
    conv6_1 = Conv2D(256, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss))(fc7)
    conv6_1z = ZeroPadding2D()(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='valid',
                     activation='relu', name='branch_3',
                     kernel_regularizer=l2(l2_loss))(conv6_1z)

    # Block 7 -----------------------------------------------------------------
    conv7_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss))(conv6_2)
    conv7_1z = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), padding='valid', strides=(2, 2),
                     activation='relu', name='branch_4',
                     kernel_regularizer=l2(l2_loss))(conv7_1z)

    # Block 8 -----------------------------------------------------------------
    conv8_1 = Conv2D(128, (1, 1), padding='same', activation='relu',
                     kernel_regularizer=l2(l2_loss))(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu', name='branch_5',
                     kernel_regularizer=l2(l2_loss))(conv8_1)

    # Block 9 -----------------------------------------------------------------
    conv9_1 = Conv2D(128, (2, 2), padding='valid', activation='relu',
                     kernel_regularizer=l2(l2_loss))(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), padding='valid', strides=(1, 1),
                     activation='relu', name='branch_6',
                     kernel_regularizer=l2(l2_loss))(conv9_1)

    branch_tensors = [conv4_3_norm, fc7, conv6_2, conv7_2, conv8_2, conv9_2]

    if return_base:
        outputs = branch_tensors
    else:
        outputs = create_multibox_head(
            branch_tensors, num_classes, num_priors, l2_loss)

    model = Model(inputs=resnet20.inputs, outputs=outputs, name='SSD-ResNet20')
    model.prior_boxes = create_prior_boxes('SSD-ResNet20')
    return model
