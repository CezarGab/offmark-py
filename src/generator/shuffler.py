import numpy as np
import math

class Shuffler:
	
	def __init__(self, key=None):
		self.key = key

	def wm_type(self):
		return "bits"

	def generate_wm(self, payload, capacity):
		payload = np.copy(payload)
		wm_len = np.array(payload.shape).prod()
		c = int(math.ceil(capacity / wm_len))
		np.random.RandomState(self.key).shuffle(payload)
		wm = np.stack([payload for _ in range(c)], axis=0).flatten()[:capacity]
		return wm
