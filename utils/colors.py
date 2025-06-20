import torch
import numpy as np

def pt_rgb2ycbcr(image, only_y=True):
    """
    Converts an RGB image (PyTorch Tensor) to YCbCr using the BT.601 standard.
    
    The function expects an input tensor of shape (B, 3, H, W) with values in [0,1].
    It scales the image to [0,255], applies the conversion, and finally scales back to [0,1].
    
    For a float input:
      - The Y channel will lie roughly in [16/255, 235/255].
      - The Cb and Cr channels will be in ranges centered around 128/255.
      
    Args:
        image (torch.Tensor): Input RGB image of shape (B, 3, H, W), values in [0,1].
        only_y (bool): If True, return only the Y (luminance) channel (shape (B, 1, H, W)).
                       Otherwise, return the full YCbCr image (shape (B, 3, H, W)).
    
    Returns:
        torch.Tensor: The converted image (values in [0,1]). 
                      If only_y is True, shape is (B, 1, H, W); otherwise (B, 3, H, W).
    """
    if image.size(1) != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    
    # Scale from [0,1] to [0,255]
    image_255 = image * 255.0

    # Define BT.601 conversion matrix (for [0,255] RGB)
    # Note: Dividing the matrix by 255.0 lets us combine scaling into one step later.
    matrix = torch.tensor([
        [ 65.481, 128.553,  24.966],   # Y
        [-37.797, -74.203, 112.000],    # Cb
        [112.000, -93.786, -18.214]     # Cr
    ], device=image.device, dtype=image.dtype) / 255.0

    # Define offset values; note these remain in the [0,255] domain until the final division.
    offset = torch.tensor([16, 128, 128], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

    # Apply conversion using einsum:
    # For each pixel, this computes: ycbcr = matrix @ [R, G, B] + offset.
    ycbcr = torch.einsum('bchw,rc->brhw', image_255, matrix) + offset

    # Scale back to [0,1]
    ycbcr = ycbcr / 255.0

    return ycbcr[:, 0, :, :] if only_y else ycbcr

def pt_rgb2ycbcr_norm(image, only_y=True):
    """
    Converts an RGB image (PyTorch Tensor) to YCbCr using the BT.601 standard.
    
    The function expects an input tensor of shape (B, 3, H, W) with values in [0,1].
    
    For a float input:
      - The Y channel will lie roughly in [16/255, 235/255].
      - The Cb and Cr channels will be in ranges centered around 128/255.
      
    Args:
        image (torch.Tensor): Input RGB image of shape (B, 3, H, W), values in [0,1].
        only_y (bool): If True, return only the Y (luminance) channel (shape (B, 1, H, W)).
                       Otherwise, return the full YCbCr image (shape (B, 3, H, W)).
    
    Returns:
        torch.Tensor: The converted image (values in [0,1]). 
                      If only_y is True, shape is (B, 1, H, W); otherwise (B, 3, H, W).
    """
    if image.size(1) != 3:
        raise ValueError("Input image must have 3 channels (RGB).")
    
    # Matriz de conversión para rango [0,1] en lugar de [0,255]
    matrix = torch.tensor([
        [ 0.257,  0.504,  0.098],  # Y
        [-0.148, -0.291,  0.439],  # Cb
        [ 0.439, -0.368, -0.071]   # Cr
    ], device=image.device, dtype=image.dtype)

    # Offset ajustado para valores en [0,1] (dividiendo los valores tradicionales por 255)
    offset = torch.tensor([16/255, 128/255, 128/255], device=image.device, dtype=image.dtype).view(1, 3, 1, 1)

    # Conversión con einsum: YCbCr = matrix @ [R, G, B] + offset
    ycbcr = torch.einsum('bchw,rc->brhw', image, matrix) + offset

    return ycbcr[:, 0, :, :] if only_y else ycbcr

def rescale_and_ycbcr2rgb(ycbcr):
    """
    Asegura que la imagen YCbCr está en el rango [0,1] y la convierte a RGB.

    Args:
        ycbcr (torch.Tensor): Tensor (B, 3, H, W) en YCbCr (posiblemente desescalado).
    
    Returns:
        torch.Tensor: Imagen en RGB con valores en [0,1].
    """
    # Asegurar que los valores están en [0,1] (clampeando)
    ycbcr = torch.clamp(ycbcr, 0, 1)

    # Separar canales
    Y = ycbcr[:, 0:1, :, :]
    Cb = ycbcr[:, 1:2, :, :]
    Cr = ycbcr[:, 2:3, :, :]

    # Escalar Cb y Cr de [0,1] → [-0.5, 0.5] (si es necesario)
    Cb = Cb - 0.5
    Cr = Cr - 0.5

    # Matriz inversa de transformación (BT.601 estándar)
    matrix_inv = torch.tensor([
        [1.000,  0.000,  1.402],    # R
        [1.000, -0.344, -0.714],    # G
        [1.000,  1.772,  0.000]     # B
    ], device=ycbcr.device, dtype=ycbcr.dtype)

    # Aplicar la conversión (usando einsum)
    rgb = torch.einsum('bchw,rc->brhw', torch.cat([Y, Cb, Cr], dim=1), matrix_inv)

    # Clampeamos nuevamente para asegurarnos de que los valores están entre 0 y 1
    rgb = torch.clamp(rgb, 0, 1)

    return rgb

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def rgb_to_yuv(image, y_only=True):
    """
    Converts an RGB image (Tensor) to YUV using BT.601 standard.
    
    Args:
        image (torch.Tensor): RGB image tensor of shape (B, 3, H, W), values in [0, 1] or [0, 255].
    
    Returns:
        torch.Tensor: Y channel of shape (B, 1, H, W)
    """
    # YUV transformation matrix (BT.601)
    matrix = torch.tensor([
        [0.299, 0.587, 0.114],  # Y
        [-0.14713, -0.28886, 0.436],  # U
        [0.615, -0.51499, -0.10001]   # V
    ], device=image.device, dtype=image.dtype)

    # Convert RGB to YUV
    yuv = torch.tensordot(image, matrix, dims=([1], [1]))  # Apply transformation
    yuv = yuv.permute(0, 3, 1, 2)  # Rearrange to (B, C, H, W)

    if y_only:
        return yuv[:, :1, :, :]  # Extract Y channel (Luminance)
    else:
        return yuv