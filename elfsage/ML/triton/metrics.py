from keras.losses import binary_crossentropy
import keras.backend as k


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def my_dice_coeff(y_true, y_pred):
    score = 1.0 - my_dice_loss( y_true, y_pred)
    return score


def my_dice_loss(y_true, y_pred):
    y_true_flat = k.flatten( y_true )
    y_pred_flat = k.flatten( y_pred )
    intersection = k.sum( y_true_flat * y_pred_flat )
    absdiff = k.sum( k.abs( y_true_flat - y_pred_flat ) )
    loss = 1.0 / (1.0 + (DICE_MULT * intersection + DICE_SMOOTH) / (DICE_MULT * absdiff + DICE_SMOOTH))
    return loss
