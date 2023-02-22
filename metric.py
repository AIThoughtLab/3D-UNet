

# This class is to see the performance of the segmented masks

import tensorflow as tf

class CustomMetrics:
    def __init__(self, num_classes=4, epsilon=1e-6):
        self.num_classes = num_classes
        self.epsilon = epsilon

    def dice_coeff(self, y_true, y_pred):
        # Flatten the tensors to 2D and calculate the intersection and union
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        # Calculate the dice coefficient and return it
        dc = (2.0 * intersection + self.epsilon) / (union + self.epsilon)
        return dc

    def dice_loss(self, y_true, y_pred):
        # Calculate the dice coefficient and return the complement as loss
        dc = self.dice_coeff(y_true, y_pred)
        loss = 1.0 - dc
        return loss

    def iou(self, y_true, y_pred):
        # Flatten the tensors to 2D and calculate the intersection and union
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))

        # Calculate the IoU and return it
        iou_ = (intersection + self.epsilon) / (union + self.epsilon)
        return iou_

    def mean_iou(self, y_true, y_pred):
        # Create an array of the class indices and one-hot encode y_true and y_pred
        class_indices = tf.range(self.num_classes)
        y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), self.num_classes)
        y_pred = tf.one_hot(tf.cast(y_pred, dtype=tf.int32), self.num_classes)

        # Calculate the IoU for each class and return the mean
        iou_values = []
        for i in range(self.num_classes):
            class_iou = self.iou(y_true[..., i], y_pred[..., i])
            iou_values.append(class_iou)
        mean_iou_ = tf.reduce_mean(iou_values)
        return mean_iou_
