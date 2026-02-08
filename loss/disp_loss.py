import torch


# def gradient_x(img):
#     gx = img[:, :, :-1, :] - img[:, :, 1:, :]
#     return gx
#
#
# def gradient_y(img):
#     gy = img[:, :-1, :, :] - img[:, 1:, :, :]
#     return gy

def gradient_x(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx


def gradient_y(img):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def get_disparity_smoothness_list(disp, pyramid):
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [torch.exp(-torch.mean(torch.abs(g), 3, keepdim=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-torch.mean(torch.abs(g), 3, keepdim=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y


def get_disparity_smoothness(depth_map, ref_img):
    disp_gradients_x = gradient_x(depth_map)
    disp_gradients_y = gradient_y(depth_map)

    image_gradients_x = gradient_x(ref_img)
    image_gradients_y = gradient_y(ref_img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    disparity = torch.mean(torch.abs(smoothness_x)) / 2.0 + torch.mean(torch.abs(smoothness_y)) / 2.0
    # return [smoothness_x, smoothness_y]
    return disparity


def get_disparity_loss(depth_map, ref_img):
    # self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))
    grad_depth = get_disparity_smoothness(depth_map, ref_img)
    disparity_loss = torch.mean(torch.abs(grad_depth))

    return disparity_loss
