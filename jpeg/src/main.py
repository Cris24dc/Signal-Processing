import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets
from scipy.fft import dctn, idctn
import cv2

Q = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def rgb_ycbcr(img):
    matrix = np.array([[.299, .587, .114], 
                       [-.1687, -.3313, .5], 
                       [.5, -.4187, -.0813]])
    
    ycbcr = img.dot(matrix.T)
    ycbcr[:,:,[1,2]] += 128
    
    return ycbcr


def ycbcr_rgb(img):
    matrix = np.array([[1, 0, 1.402], 
                       [1, -0.34414, -.71414], 
                       [1, 1.772, 0]])
    
    rgb = img.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(matrix.T)

    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    
    return np.round(rgb).astype(np.uint8)


def process_channel(channel, Q):
    h, w = channel.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    chan_pad = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
    h_pad, w_pad = chan_pad.shape
    final_chan = np.zeros((h_pad, w_pad))
    
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            block = chan_pad[i:i+8, j:j+8]
            block = block - 128
            
            freq_block = dctn(block, norm='ortho')
            quant_block = Q * np.round(freq_block / Q)
            block_rec = idctn(quant_block, norm='ortho')

            block_rec = block_rec + 128
            final_chan[i:i+8, j:j+8] = block_rec

    final_chan = final_chan[:h, :w]
    return final_chan


def pipeline_and_mse(img, scale):
    Q_scaled = Q * scale
    
    Y_rec = process_channel(img[:,:,0], Q_scaled)
    Cb_rec = process_channel(img[:,:,1], Q_scaled)
    Cr_rec = process_channel(img[:,:,2], Q_scaled)
    
    img_rec_ycbcr = np.dstack((Y_rec, Cb_rec, Cr_rec))
    img_rec_rgb = ycbcr_rgb(img_rec_ycbcr)
    
    mse = np.mean((ycbcr_rgb(img).astype(float) - img_rec_rgb.astype(float)) ** 2)
    return mse, img_rec_rgb


TARGET_MSE = int(input('Target MSE: '))

img_original = datasets.face()
img_ycbcr = rgb_ycbcr(img_original)

low = 0.01
high = 100.0
iterations = 20
best_scale = 1.0

for k in range(iterations):
    mid = (low + high) / 2
    
    current_mse, _ = pipeline_and_mse(img_ycbcr, mid)
    print(f"MSE: {current_mse:.2f} | Scale: {mid:.2f}")
    
    if current_mse < TARGET_MSE:
        low = mid
        best_scale = mid
    else:
        high = mid


Q_final = Q * best_scale

Y_rec = process_channel(img_ycbcr[:,:,0], Q_final)
Cb_rec = process_channel(img_ycbcr[:,:,1], Q_final)
Cr_rec = process_channel(img_ycbcr[:,:,2], Q_final)

img_rec_ycbcr = np.dstack((Y_rec, Cb_rec, Cr_rec))
img_final_rgb = ycbcr_rgb(img_rec_ycbcr)

final_mse = np.mean((img_original.astype(float) - img_final_rgb.astype(float)) ** 2)
print(f"MSE final: {final_mse:.2f}")

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(img_original)
plt.title("Original")

plt.subplot(122)
plt.imshow(img_final_rgb)
plt.title(f"Reconstruit (MSE={final_mse:.2f})")

plt.show()
plt.imsave("./img/image.jpg", img_final_rgb)


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    video.release()
    return frames, fps


def save_video(frames, fps, output_path):
    if not frames:
        return
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(new_frame)
        
    out.release()


def compress_frame(img_rgb):
    img_ycbcr = rgb_ycbcr(img_rgb)
    
    Y_rec = process_channel(img_ycbcr[:,:,0], Q)
    Cb_rec = process_channel(img_ycbcr[:,:,1], Q)
    Cr_rec = process_channel(img_ycbcr[:,:,2], Q)
    
    img_rec_ycbcr = np.dstack((Y_rec, Cb_rec, Cr_rec))
    img_rec_rgb = ycbcr_rgb(img_rec_ycbcr)
    
    return img_rec_rgb


frames, fps = extract_frames('./img/input_video.mp4')
processed_frames = []

for i, frame in enumerate(frames):
    new_frame = compress_frame(frame)
    processed_frames.append(new_frame)

save_video(processed_frames, fps, './img/output_video.mp4')
