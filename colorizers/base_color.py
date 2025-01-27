
import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm # Normalize the L channel by subtracting the mean and dividing by the standard deviation (Z-score normalization)

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent # Unnormalize the L channel by multiplying by the standard deviation and adding the mean

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm # Normalize the ab channels by dividing by the standard deviation

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm # Unnormalize the ab channels by multiplying by the standard deviation

