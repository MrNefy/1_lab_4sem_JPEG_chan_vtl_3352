import os
import math
import numpy as np
from typing import List, Tuple, Dict, Any
import struct
from collections import defaultdict
from itertools import chain
import matplotlib.pyplot as plt
from PIL import Image

# Constants
N = 8
PI = 3.14159265358979323846
SQRT2 = 1.41421356237309504880
C0 = 1.0 / SQRT2
C1 = 1.0

# Quantization tables
Luminance_quantization_table = [
	[16, 11, 10, 16, 24, 40, 51, 61],
	[12, 12, 14, 19, 26, 58, 60, 55],
	[14, 13, 16, 24, 40, 57, 69, 56],
	[14, 17, 22, 29, 51, 87, 80, 62],
	[18, 22, 37, 56, 68, 109, 103, 77],
	[24, 35, 55, 64, 81, 104, 113, 92],
	[49, 64, 78, 87, 103, 121, 120, 101],
	[72, 92, 95, 98, 112, 100, 103, 99]
]

Chrominance_quantization_table = [
	[17, 18, 24, 47, 99, 99, 99, 99],
	[18, 21, 26, 66, 99, 99, 99, 99],
	[24, 26, 56, 99, 99, 99, 99, 99],
	[47, 66, 99, 99, 99, 99, 99, 99],
	[99, 99, 99, 99, 99, 99, 99, 99],
	[99, 99, 99, 99, 99, 99, 99, 99],
	[99, 99, 99, 99, 99, 99, 99, 99],
	[99, 99, 99, 99, 99, 99, 99, 99]
]

# Huffman tables
Luminance_DC_differences = [
	"00", "010", "011", "100", "101", "110", 
	"1110", "11110", "111110", "1111110", 
	"11111110", "111111110"
]

Chrominance_DC_differences = [
	"00", "01", "10", "110", "1110", "11110", 
	"111110", "1111110", "11111110", "111111110", 
	"1111111110", "11111111110"
]

Luminance_AC = [
	["1010", "00", "01", "100", "1011", "11010", "1111000", "11111000", "1111110110", "1111111110000010", "1111111110000011"],
	["", "1100", "11011", "1111001", "111110110", "11111110110", "1111111110000100", "1111111110000101", "1111111110000110", "1111111110000111", "1111111110001000"],
	["", "11100", "11111001", "1111110111", "111111110100", "1111111110001001", "1111111110001010", "1111111110001011", "1111111110001100", "1111111110001101", "1111111110001110"],
	["", "111010", "111110111", "111111110101", "1111111110001111", "1111111110010000", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101"],
	["", "111011", "1111111000", "1111111110010110", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101"],
	["", "1111010", "11111110111", "1111111110011110", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101"],
	["", "1111011", "111111110110", "1111111110100110", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101"],
	["", "11111010", "111111110111", "1111111110101110", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101"],
	["", "111111000", "111111111000000", "1111111110110110", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101"],
	["", "111111001", "1111111110111110", "1111111110111111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110"],
	["", "111111010", "1111111111000111", "1111111111001000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111"],
	["", "1111111001", "1111111111010000", "1111111111010001", "1111111111010010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000"],
	["", "1111111010", "1111111111011001", "1111111111011010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001"],
	["", "11111111000", "1111111111100010", "1111111111100011", "1111111111100100", "1111111111100101", "1111111111100110", "1111111111100111", "1111111111101000", "1111111111101001", "1111111111101010"],
	["", "1111111111101011", "1111111111101100", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100"],
	["11111111001", "1111111111110101", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111011", "1111111111111100", "1111111111111101", "1111111111111110"]
]

Chrominance_AC = [
	["00", "01", "100", "1010", "11000", "11001", "111000", "1111000", "111110100", "1111110110", "111111110100"],
	["", "1011", "111001", "11110110", "111110101", "11111110110", "111111110101", "1111111110001000", "1111111110001001", "1111111110001010", "1111111110001011"],
	["", "11010", "11110111", "1111110111", "111111110110", "111111111000010", "1111111110001100", "1111111110001101", "1111111110001110", "1111111110001111", "1111111110010000"],
	["", "11011", "11111000", "1111111000", "111111110111", "1111111110010001", "1111111110010010", "1111111110010011", "1111111110010100", "1111111110010101", "1111111110010110"],
	["", "111010", "111110110", "1111111110010111", "1111111110011000", "1111111110011001", "1111111110011010", "1111111110011011", "1111111110011100", "1111111110011101", "1111111110011110"],
	["", "111011", "1111111001", "1111111110011111", "1111111110100000", "1111111110100001", "1111111110100010", "1111111110100011", "1111111110100100", "1111111110100101", "1111111110100110"],
	["", "1111001", "11111110111", "1111111110100111", "1111111110101000", "1111111110101001", "1111111110101010", "1111111110101011", "1111111110101100", "1111111110101101", "1111111110101110"],
	["", "1111010", "11111111000", "1111111110101111", "1111111110110000", "1111111110110001", "1111111110110010", "1111111110110011", "1111111110110100", "1111111110110101", "1111111110110110"],
	["", "11111001", "1111111110110111", "1111111110111000", "1111111110111001", "1111111110111010", "1111111110111011", "1111111110111100", "1111111110111101", "1111111110111110", "1111111110111111"],
	["", "111110111", "1111111111000000", "1111111111000001", "1111111111000010", "1111111111000011", "1111111111000100", "1111111111000101", "1111111111000110", "1111111111000111", "1111111111001000"],
	["", "111111000", "1111111111001001", "1111111111001010", "1111111111001011", "1111111111001100", "1111111111001101", "1111111111001110", "1111111111001111", "1111111111010000", "1111111111010001"],
	["", "111111001", "1111111111010010", "1111111111010011", "1111111111010100", "1111111111010101", "1111111111010110", "1111111111010111", "1111111111011000", "1111111111011001", "1111111111011010"],
	["", "111111010", "1111111111011011", "1111111111011100", "1111111111011101", "1111111111011110", "1111111111011111", "1111111111100000", "1111111111100001", "1111111111100010", "1111111111100011"],
	["", "11111111001", "1111111111100100", "1111111111100101", "1111111111100110", "1111111111100111", "1111111111101000", "1111111111101001", "1111111111101010", "1111111111101011", "1111111111101100"],
	["", "11111111100000", "1111111111101101", "1111111111101110", "1111111111101111", "1111111111110000", "1111111111110001", "1111111111110010", "1111111111110011", "1111111111110100", "1111111111110101"],
	["1111111010", "111111111000011", "1111111111110110", "1111111111110111", "1111111111111000", "1111111111111001", "1111111111111010", "1111111111111011", "1111111111111100", "1111111111111101", "1111111111111110"]
]

zigzag_sequence = [
	0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 
	12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 
	28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 
	44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 
	55, 62, 63
]

def div_up(x: int, y: int) -> int:
	return (x - 1) // y + 1

def rgb_to_ycbcr(height: int, width: int, R: np.ndarray, G: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	Y = np.zeros((height, width), dtype=np.int16)
	Cb = np.zeros((height, width), dtype=np.int16)
	Cr = np.zeros((height, width), dtype=np.int16)
	
	for y in range(height):
		for x in range(width):
			r = R[y][x]
			g = G[y][x]
			b = B[y][x]
			
			Y[y][x] = int(0.299 * r + 0.587 * g + 0.114 * b)
			Cb[y][x] = int(128 - 0.168736 * r - 0.331264 * g + 0.5 * b)
			Cr[y][x] = int(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)

	return Y, Cb, Cr

def ycbcr_to_rgb(height: int, width: int, Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	R = np.zeros((height, width), dtype=np.int16)
	G = np.zeros((height, width), dtype=np.int16)
	B = np.zeros((height, width), dtype=np.int16)
	
	for y in range(height):
		for x in range(width):
			y_val = Y[y][x]
			cb_val = Cb[y][x] - 128
			cr_val = Cr[y][x] - 128
			
			r = y_val + 1.402 * cr_val
			g = y_val - 0.344136 * cb_val - 0.714136 * cr_val
			b = y_val + 1.772 * cb_val
			
			R[y][x] = int(max(0.0, min(255.0, r)))
			G[y][x] = int(max(0.0, min(255.0, g)))
			B[y][x] = int(max(0.0, min(255.0, b)))
	
	return R, G, B

def downsample(height: int, width: int, c: np.ndarray) -> np.ndarray:
	odd_h = height % 2 != 0
	odd_w = width % 2 != 0
	
	new_height = div_up(height, 2)
	new_width = div_up(width, 2)
	downsampled = np.zeros((new_height, new_width), dtype=np.int16)
	
	for y in range(0, height - odd_h, 2):
		for x in range(0, width - odd_w, 2):
			arithmetic_mean = (c[y][x] + c[y][x+1] + c[y+1][x] + c[y+1][x+1]) // 4
			downsampled[y//2][x//2] = arithmetic_mean
	
	if odd_w:
		w = width - 1
		x = width // 2
		for y in range(0, height - odd_h, 2):
			arithmetic_mean = (c[y][w] + c[y+1][w]) // 2
			downsampled[y//2][x] = arithmetic_mean
	
	if odd_h:
		h = height - 1
		y = height // 2
		for x in range(0, width - odd_w, 2):
			arithmetic_mean = (c[h][x] + c[h][x+1]) // 2
			downsampled[y][x//2] = arithmetic_mean
	
	if odd_h and odd_w:
		h = height - 1
		w = width - 1
		y = height // 2
		x = width // 2
		downsampled[y][x] = c[h][w]
	
	return downsampled

def up_scale(height: int, width: int, c: np.ndarray) -> np.ndarray:
	up = np.zeros((height, width), dtype=np.int16)
	odd_h = height % 2 != 0
	odd_w = width % 2 != 0
	
	for y in range(height // 2):
		for x in range(width // 2):
			pixel = c[y][x]
			i = y * 2
			j = x * 2
			up[i][j] = pixel
			up[i][j+1] = pixel
			up[i+1][j] = pixel
			up[i+1][j+1] = pixel
	
	if odd_w:
		w = width - 1
		x = width // 2
		for y in range(height // 2):
			pixel = c[y][x]
			i = y * 2
			up[i][w] = pixel
			up[i+1][w] = pixel
	
	if odd_h:
		h = height - 1
		y = height // 2
		for x in range(width // 2):
			pixel = c[y][x]
			j = x * 2
			up[h][j] = pixel
			up[h][j+1] = pixel
	
	if odd_h and odd_w:
		h = height - 1
		w = width - 1
		y = height // 2
		x = width // 2
		up[h][w] = c[y][x]
	
	return up

def precompute_cos_table() -> np.ndarray:
	cos_table = np.zeros((N, N))
	for u in range(N):
		for x in range(N):
			cos_table[u][x] = math.cos((2 * x + 1) * u * PI / (2 * N))
	return cos_table

COS_TABLE = precompute_cos_table()

def dct_1d(input_arr: np.ndarray) -> np.ndarray:
	output = np.zeros(N)
	for u in range(N):
		sum_val = 0.0
		cu = C0 if u == 0 else C1
		for x in range(N):
			sum_val += input_arr[x] * COS_TABLE[u][x]
		output[u] = sum_val * cu * 0.5
	return output

def dct_2d_8x8(block: np.ndarray) -> np.ndarray:
	temp = np.zeros((N, N))
	coeffs = np.zeros((N, N))
	
	# Apply 1D DCT to each row
	for y in range(N):
		row = block[y].copy()
		dct_row = dct_1d(row)
		temp[y] = dct_row
	
	# Apply 1D DCT to each column
	for u in range(N):
		column = temp[:, u].copy()
		dct_column = dct_1d(column)
		coeffs[:, u] = dct_column
	
	return coeffs

def idct_1d(input_arr: np.ndarray) -> np.ndarray:
	output = np.zeros(N)
	for x in range(N):
		sum_val = 0.0
		for u in range(N):
			cu = C0 if u == 0 else C1
			sum_val += cu * input_arr[u] * COS_TABLE[u][x]
		output[x] = sum_val * 0.5
	return output

def classic_round(x):
    return int(x + 0.5) if x >= 0 else int(x - 0.5)

def idct_2d_8x8(coeffs: np.ndarray) -> np.ndarray:
	temp = np.zeros((N, N))
	block = np.zeros((N, N), dtype=np.int16)
	
	# Apply 1D IDCT to each column
	for u in range(N):
		column = coeffs[:, u].copy()
		idct_column = idct_1d(column)
		temp[:, u] = idct_column
	
	# Apply 1D IDCT to each row
	for y in range(N):
		row = temp[y].copy()
		idct_row = idct_1d(row)
		for x in range(N):
			block[y][x] = int(classic_round(idct_row[x]))
	
	return block

def generate_quantization_matrix(quality: int, quant_table: List[List[int]]) -> np.ndarray:
	quality = max(1, min(100, quality))
	if quality < 50:
		scale_factor = 200.0 / quality
	else:
		scale_factor = 8 * (1.0 - 0.01 * quality)
	
	q_matrix = np.zeros((8, 8), dtype=np.int16)
	for y in range(8):
		for x in range(8):
			q = quant_table[y][x] * scale_factor
			q_matrix[y][x] = max(1, min(255, q))
	
	return q_matrix

def quantize(dct_coeffs: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
	quantized = np.zeros((N, N), dtype=np.int16)
	for y in range(N):
		for x in range(N):
			a = dct_coeffs[y][x] / np.float32(q_matrix[y][x])
			b = classic_round(a)
			quantized[y][x] = int(b)
	return quantized

def dequantize(quantized: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
	dct_coeffs = np.zeros((N, N))
	for y in range(N):
		for x in range(N):
			dct_coeffs[y][x] = quantized[y][x] * q_matrix[y][x]
	return dct_coeffs

def zigzag_scan(block: np.ndarray) -> List[int]:
	scanned = [0] * 64
	for i in range(64):
		idx = zigzag_sequence[i]
		scanned[i] = block[idx // 8][idx % 8]
	return scanned

def inverse_zigzag_scan(scanned: List[int]) -> np.ndarray:
	block = np.zeros((8, 8), dtype=np.int16)
	for i in range(64):
		idx = zigzag_sequence[i]
		block[idx // 8][idx % 8] = scanned[i]
	return block

def dc_difference(data: List[int]) -> None:
	size = len(data)
	temp = []
	for i in range(0, size, 64):
		temp.append(data[i])
	
	for i in range(64, size, 64):
		data[i] -= temp[i//64 - 1]

def reverse_dc_difference(data: List[List[int]]) -> None:
	size = len(data)
	for i in range(1, size):
		data[i][0] += data[i-1][0]

def int_to_binary(num: int, positive: int = 1) -> List[int]:
	if num == 0:
		return [0]
	
	if positive == 0:
		num *= -1
	
	bits = []
	while num > 0:
		bits.append(1 if num % 2 == positive else 0)
		num = num // 2
	
	bits.reverse()
	return bits

def rle_encode_ac(cur: int, out_rle: List[int], zero_count: int, EOB: bool, ZRL: bool, tempZRL: List[int]) -> bool:
	if cur == 0:
		zero_count[0] += 1
		EOB[0] = True
		if zero_count[0] == 15:
			tempZRL.append(15)
			tempZRL.append(0)
			zero_count[0] = 0
			ZRL[0] = True
		return True
	else:
		if ZRL[0]:
			out_rle.extend(tempZRL)
			tempZRL.clear()
			ZRL[0] = False
		
		out_rle.append(zero_count[0])
		zero_count[0] = 0
		EOB[0] = False
		return False

def preparing_for_coding_dc_and_ac(data: List[int]) -> List[int]:
	output = []
	dc_difference(data)
	
	size = len(data)
	for i in range(0, size, 64):
		# DC coefficient
		if data[i] == 0:
			output.append(0)
		else:
			if data[i] > 0:
				temp = int_to_binary(data[i], 1)
			else:
				temp = int_to_binary(data[i], 0)
			output.append(len(temp))
			output.extend(temp)
		
		# AC coefficients
		zero_count = [0]
		EOB = [False]
		ZRL = [False]
		tempZRL = []
		
		for j in range(1, 64):
			if i + j >= size:
				break
				
			if rle_encode_ac(data[i + j], output, zero_count, EOB, ZRL, tempZRL):
				continue
			
			if data[i + j] >= 0:
				temp = int_to_binary(data[i + j], 1)
			else:
				temp = int_to_binary(data[i + j], 0)
			
			output.append(len(temp))
			output.extend(temp)
		
		if EOB[0]:
			output.append(0)
			output.append(0)
	
	return output

def ha_encode(data: List[int], dc_table: List[str], ac_table: List[List[str]]) -> str:
	encoded = ""
	size = len(data)
	i = 0
	
	while i < size:
		# DC coefficient
		encoded += dc_table[data[i]]
		k_size = data[i]
		
		for k in range(k_size):
			i += 1
			encoded += chr(data[i] + ord('0'))
		
		# AC coefficients
		count = 1
		while count < 64 and i < size - 1:
			i += 1
			
			if data[i] == 0 and data[i + 1] == 0:
				encoded += ac_table[0][0]
				i += 1
				break
			
			if data[i] == 15 and data[i + 1] == 0:
				encoded += ac_table[15][0]
				count += 15
				i += 1
				continue
			
			count += 1 + data[i]
			encoded += ac_table[data[i]][data[i + 1]]
			
			i += 1
			k_size = data[i]
			for k in range(k_size):
				i += 1
				encoded += chr(data[i] + ord('0'))
		i += 1
	
	return encoded

def pack_bits_to_bytes(bit_str: str) -> bytes:
	output = bytearray()
	length = len(bit_str)
	zero_bits = 0
	
	for i in range(0, length, 8):
		byte_str = bit_str[i:i+8]
		
		if len(byte_str) < 8:
			zero_bits = 8 - len(byte_str)
			byte_str += '0' * zero_bits
		
		byte = int(byte_str, 2)
		output.append(byte)
	
	return bytes(output)

def split_into_8x8_blocks(height: int, width: int, data: np.ndarray) -> List[np.ndarray]:
	blocks = []
	
	for i in range(0, height, 8):
		block_height = 8
		if height - i < 8:
			block_height = height - i
		
		for j in range(0, width, 8):
			block_width = 8
			if width - j < 8:
				block_width = width - j
			
			block = np.zeros((8, 8), dtype=np.int16)
			for bi in range(block_height):
				for bj in range(block_width):
					block[bi][bj] = data[i + bi][j + bj]
			
			blocks.append(block)
	
	return blocks

def merge_8x8_blocks(height: int, width: int, blocks: List[np.ndarray], is_chroma: bool) -> np.ndarray:
	if is_chroma:
		height = div_up(height, 2)
		width = div_up(width, 2)
	
	output = np.zeros((height, width), dtype=np.int16)
	h_blocks = div_up(height, 8)
	w_blocks = div_up(width, 8)
	position = 0
	
	for i in range(h_blocks):
		for u in range(w_blocks):
			for bi in range(8):

				if i * 8 + bi >= height:
					continue

				for bu in range(8):

					if u * 8 + bu >= width:
						continue

					output[i * 8 + bi][u * 8 + bu] = blocks[position][bi][bu]
			position += 1
	
	return output

def jpeg_decode_ha_rle(pos: int, bit_str: str, num_blocks: int, dc_table: List[str], ac_table: List[List[str]]) -> Tuple[List[List[int]], int]:
	with open('C:/Users/lin/source/repos/jpeg/Decoder JPEG/console.txt', 'w', encoding='utf-8') as f:
		out = [[0] * 64 for _ in range(num_blocks)]
	
		for num_block in range(num_blocks):
			# DC coefficient
			if bit_str[pos:pos+2] == "00":
				pos += 2
				print(pos, file = f)
			else:
				search = True
				length = 2
			
				while search:
					if length > 11:
						print(f"DC Difference length ERROR: not found. tmp = {pos}")
						break
				
					code = bit_str[pos:pos+length]
				
					for d in range(1, 12):
						if len(dc_table[d]) == length and code == dc_table[d]:
							search = False
							pos += length
							print(pos, file = f)

							bits = bit_str[pos:pos+d]
							minus = 1
						
							if bits[0] == '0':
								inverted = ''.join(['1' if c == '0' else '0' for c in bits])
								minus = -1
								bits = inverted
						
							pos += d
							print(pos, file = f)
							out[num_block][0] = minus * int(bits, 2)
							break
				
					length += 1
		
			# AC coefficients
			EOB = False
			count = 0
		
			while count < 63 and pos < len(bit_str):
				search = True
				length = 2
			
				while search and pos + length <= len(bit_str):
					code = bit_str[pos:pos+length]
				
					if length > 16:
						print(f"AC length ERROR: not found. tmp = {pos}")
						break
				
					if code == ac_table[0][0]:
						EOB = True
						pos += len(ac_table[0][0])
						print(pos, file = f)
						break
				
					for a in range(16):
						for c in range(11):
							if len(ac_table[a][c]) == length and code == ac_table[a][c]:
								search = False
								pos += length
								print(pos, file = f)
								count += a
							
								if a == 15 and c == 0:
									break
							
								bits = bit_str[pos:pos+c]
								minus = 1
							
								if bits and bits[0] == '0':
									inverted = ''.join(['1' if c == '0' else '0' for c in bits])
									minus = -1
									bits = inverted
							
								pos += c
								print(pos, file = f)
								count += 1
								out[num_block][count] = minus * int(bits, 2) if bits else 0
								break
					
						if not search:
							break
				
					length += 1
			
				if EOB:
					break
			if num_block - num_blocks == -1:
				print("num block: ", pos)

	
	return out, pos

def write_bmp(filename: str, R: np.ndarray, G: np.ndarray, B: np.ndarray, width: int, height: int) -> bool:
	try:
		with open(filename, 'wb') as file:
			# BMP header (14 bytes)
			file_size = 54 + 3 * width * height
			bmp_header = struct.pack(
				'<2sIHHI', 
				b'BM', file_size, 0, 0, 54
			)
			
			# DIB header (40 bytes)
			dib_header = struct.pack(
				'<IIIHHIIIIII', 
				40, width, height, 1, 24, 
				0, 0, 0, 0, 0, 0
			)
			
			file.write(bmp_header)
			file.write(dib_header)
			
			# Pixel data (BGR order, bottom-up)
			padding = (4 - (width * 3) % 4) % 4
			pad_bytes = b'\x00' * padding
			
			for y in range(height-1, -1, -1):
				for x in range(width):
					blue = min(255, max(0, B[y][x]))
					green = min(255, max(0, G[y][x]))
					red = min(255, max(0, R[y][x]))
					
					file.write(bytes([blue, green, red]))
				
				if padding > 0:
					file.write(pad_bytes)
		
		return True
	
	except Exception as e:
		print(f"Error writing BMP file: {e}")
		return False

def main():
	# Configuration
	file = "Lenna.raw"
	compress = False
	original = False
	width = 512
	height = 512
	min_quality = 0
	max_quality = 100
	step = 100
	
	# Extract filename without extension
	dot_pos = file.rfind('.')
	name = file[:dot_pos] if dot_pos != -1 else file
	
	# Read raw image data
	try:
		with open(file, 'rb') as f:
			data = f.read()
		
		if len(data) != 3 * height * width:
			print(f"Error: File size {len(data)} doesn't match expected size {3 * height * width}")
			return
		
		print(f"Image parameters:\n1) Size: {len(data)} bytes")
		print(f"2) Pixel count: {len(data)//3}")
		print(f"3) Resolution: {width}x{height} bytes")
		
		# Convert raw data to RGB arrays
		R = np.zeros((height, width), dtype=np.uint8)
		G = np.zeros((height, width), dtype=np.uint8)
		B = np.zeros((height, width), dtype=np.uint8)
		
		index = 0
		for y in range(height):
			for x in range(width):
				R[y][x] = data[index]
				G[y][x] = data[index+1]
				B[y][x] = data[index+2]
				index += 3
		
		for quality in range(max_quality, min_quality, -step):
			print(f"\nQuality: {quality}")
			
			# Convert RGB to YCbCr
			Y, Cb, Cr = rgb_to_ycbcr(height, width, R, G, B)
			
			# Downsample chroma components (4:2:0)
			Cb_down = downsample(height, width, Cb)
			Cr_down = downsample(height, width, Cr)

			# Split into 8x8 blocks
			Y_blocks = split_into_8x8_blocks(height, width, Y)
			Cb_blocks = split_into_8x8_blocks(div_up(height, 2), div_up(width, 2), Cb_down)
			Cr_blocks = split_into_8x8_blocks(div_up(height, 2), div_up(width, 2), Cr_down)

			# # # Generate quantization matrices
			q0_matrix = generate_quantization_matrix(quality, Luminance_quantization_table)
			q1_matrix = generate_quantization_matrix(quality, Chrominance_quantization_table)
			
			# Process each block (DCT + Quantization)
			for i in range(len(Y_blocks)):
				dct_coeffs = dct_2d_8x8(Y_blocks[i])
				Y_blocks[i] = quantize(dct_coeffs, q0_matrix)
			
			for i in range(len(Cb_blocks)):
				dct_coeffs = dct_2d_8x8(Cb_blocks[i])
				Cb_blocks[i] = quantize(dct_coeffs, q1_matrix)
			
			for i in range(len(Cr_blocks)):
				dct_coeffs = dct_2d_8x8(Cr_blocks[i])
				Cr_blocks[i] = quantize(dct_coeffs, q1_matrix)

			#Zigzag scan and prepare for Huffman encoding
			Y_scanned = []
			Cb_scanned = []
			Cr_scanned = []
			
			for block in Y_blocks:
				Y_scanned.extend(zigzag_scan(block))
			
			for block in Cb_blocks:
				Cb_scanned.extend(zigzag_scan(block))
			
			for block in Cr_blocks:
				Cr_scanned.extend(zigzag_scan(block))

			# Prepare DC and AC coefficients for encoding
			Y_encoded = preparing_for_coding_dc_and_ac(Y_scanned.copy())
			Cb_encoded = preparing_for_coding_dc_and_ac(Cb_scanned.copy())
			Cr_encoded = preparing_for_coding_dc_and_ac(Cr_scanned.copy())
			
			# Huffman encoding
			encoded_bits = ""
			encoded_bits += ha_encode(Y_encoded, Luminance_DC_differences, Luminance_AC)
			encoded_bits += ha_encode(Cb_encoded, Chrominance_DC_differences, Chrominance_AC)
			encoded_bits += ha_encode(Cr_encoded, Chrominance_DC_differences, Chrominance_AC)

			# Pack bits into bytes
			compressed_data = pack_bits_to_bytes(encoded_bits)
			
			print(f"\nCompressed data size: {len(compressed_data)} bytes")
			print(f"or {len(compressed_data)/1024:.2f} KB")
			
			# Save compressed data if needed
			
			
			# Decoding process
			# Convert bytes back to bit string
			bit_str = ""
			for byte in compressed_data:
				bit_str += f"{byte:08b}"
			
			# Calculate number of blocks
			num_y_blocks = div_up(width, 8) * div_up(height, 8)
			num_c_blocks = div_up(width//2, 8) * div_up(height//2, 8)
			
			# Decode Huffman and RLE
			pos = 0
			Y_decoded, pos = jpeg_decode_ha_rle(pos, bit_str, num_y_blocks, 
											Luminance_DC_differences, Luminance_AC)
			Cb_decoded, pos = jpeg_decode_ha_rle(pos, bit_str, num_c_blocks, 
											   Chrominance_DC_differences, Chrominance_AC)
			Cr_decoded, _ = jpeg_decode_ha_rle(pos, bit_str, num_c_blocks, 
											 Chrominance_DC_differences, Chrominance_AC)
			
			# Reverse DC difference coding
			reverse_dc_difference(Y_decoded)
			reverse_dc_difference(Cb_decoded)
			reverse_dc_difference(Cr_decoded)
			
			# Inverse zigzag and create blocks
			Y_blocks_decoded = []
			for block in Y_decoded:
				Y_blocks_decoded.append(inverse_zigzag_scan(block))
			
			Cb_blocks_decoded = []
			for block in Cb_decoded:
				Cb_blocks_decoded.append(inverse_zigzag_scan(block))
			
			Cr_blocks_decoded = []
			for block in Cr_decoded:
				Cr_blocks_decoded.append(inverse_zigzag_scan(block))
			
			# Dequantize and IDCT
			Y_blocks_reconstructed = []
			for i in range(len(Y_blocks_decoded)):
				dequantized = dequantize(Y_blocks_decoded[i], q0_matrix)
				Y_blocks_reconstructed.append(idct_2d_8x8(dequantized))
			
			Cb_blocks_reconstructed = []
			for i in range(len(Cb_blocks_decoded)):
				dequantized = dequantize(Cb_blocks_decoded[i], q1_matrix)
				Cb_blocks_reconstructed.append(idct_2d_8x8(dequantized))
			
			Cr_blocks_reconstructed = []
			for i in range(len(Cr_blocks_decoded)):
				dequantized = dequantize(Cr_blocks_decoded[i], q1_matrix)
				Cr_blocks_reconstructed.append(idct_2d_8x8(dequantized))
			
			# Merge blocks back into full images
			Y_reconstructed = merge_8x8_blocks(height, width, Y_blocks_reconstructed, False)
			Cb_reconstructed = merge_8x8_blocks(height, width, Cb_blocks_reconstructed, True)
			Cr_reconstructed = merge_8x8_blocks(height, width, Cr_blocks_reconstructed, True)
			
			# Upscale chroma components
			Cb_upscaled = up_scale(height, width, Cb_reconstructed)
			Cr_upscaled = up_scale(height, width, Cr_reconstructed)
			
			# Convert back to RGB
			R_reconstructed, G_reconstructed, B_reconstructed = ycbcr_to_rgb(
				height, width, Y_reconstructed, Cb_upscaled, Cr_upscaled)
			
			# Save reconstructed image
			# output_path = f"{name}/{name}_D_{quality}.bmp"
			output_path = f"folder/{name}_D_{quality}.bmp"
			os.makedirs(os.path.dirname(output_path), exist_ok=True)
			write_bmp(output_path, R_reconstructed, G_reconstructed, B_reconstructed, width, height)
			
			# Save original if needed
			if original and quality == max_quality:
				# output_path = f"{name}/{name}_Orig.bmp"
				output_path = f"folder/{name}_Orig.bmp"
				write_bmp(output_path, R, G, B, width, height)
		
		print("\n\nDone!!!")
	
	except Exception as e:
		print(f"Error: {e}")

if __name__ == "__main__":
	main()