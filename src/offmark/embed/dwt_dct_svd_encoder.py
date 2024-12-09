import numpy as np
import pywt
import cv2

class DwtDctSvdEncoder:
    def __init__(self, key=None, scales=[0,15,0], blk=4):
        self.key = key
        self.scales = scales
        self.blk = blk

    def read_wm(self, wm):
        self.wm = wm[0]

    def wm_capacity(self, frame_shape):
        row, col, channels = frame_shape
        block_num = row * col // 64
        return (1, block_num)

    def encode(self, yuv):
        (row, col, channels) = yuv.shape
        for channel in range(3):
            if self.scales[channel] <= 0:
                continue
            ca, hvd = pywt.dwt2(yuv[:row // 4 * 4,:col // 4 * 4, channel].astype(np.float32), 'haar')
            self.__encode_frame(ca, self.scales[channel])
            yuv[:row // 4 * 4, :col // 4 * 4, channel] = pywt.idwt2((ca, hvd), 'haar')
        return yuv

    def __encode_frame(self, frame, scale):
        # (row, col) = frame.shape
        # c = 0
        # for i in range(row // self.blk):
        #    for j in range(col // self.blk):
        #        blk = frame[i * self.blk : i * self.blk + self.blk,
        #                      j * self.blk : j * self.blk + self.blk]
        #        wm_bit = self.wm[c]
        #        embedded_blk = self.__blk_embed_wm(blk, wm_bit, scale)
        #        frame[i * self.blk : i * self.blk + self.blk,
        #              j * self.blk : j * self.blk + self.blk] = embedded_blk
        #        c += 1
        (row, col) = frame.shape
        blk_size = self.blk
        n_blocks_row = row // blk_size
        n_blocks_col = col // blk_size

        frames_reshaped = frame[:n_blocks_row * blk_size, :n_blocks_col * blk_size]
        frames_reshaped = frames_reshaped.reshape(n_blocks_row, blk_size, n_blocks_col, blk_size)
        frames_reshaped = frames_reshaped.transpose(0, 2, 1, 3)
        blocks = frames_reshaped.reshape(-1, blk_size, blk_size)

        wm_bits = self.wm[:blocks.shape[0]]

        for idx in range(blocks.shape[0]):
            blk = blocks[idx]
            wm_bit = wm_bits[idx]
            embedded_blk = self.__blk_embed_wm(blk, wm_bit, scale)
            blocks[idx] = embedded_blk

        frames_reshaped = blocks.reshape(n_blocks_row, n_blocks_col, blk_size, blk_size)
        frames_reshaped = frames_reshaped.transpose(0, 2, 1, 3)
        frame[:n_blocks_row * blk_size, :n_blocks_col * blk_size] = frames_reshaped.reshape(row, col)
            

    def __blk_embed_wm(self, blk, wm_bit, scale):
        # u, s, v = np.linalg.svd(cv2.dct(blk))
        # s[0] = (s[0] // scale + 0.25 + 0.5 * wm_bit) * scale
        # return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))
        
        # V2:
        D = cv2.dct(blk.astype(np.float32))
        s, u, vt = cv2.SVDecomp(D)
        s = s.flatten()
        s[0] = (s[0] // scale + 0.25 + 0.5 * wm_bit) * scale
        D_modified = u @ np.diag(s) @ vt
        return cv2.idct(D_modified)
        
        # V3:
        # D = cv2.dct(blk.astype(np.float32))
        # coeff_idx = (1, 1)
        # D_coeff = D[coeff_idx]
        # D_coeff_modified = (D_coeff // scale + 0.25 + 0.5 * wm_bit) * scale
        # D[coeff_idx] = D_coeff_modified
        # return cv2.idct(D)