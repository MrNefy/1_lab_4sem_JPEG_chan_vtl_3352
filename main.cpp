//#include "Header.h"

#include <windows.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>

#include <cstdint>
#include <cstddef>

#include <map>
#include <queue>
#include <bitset>

#include <cassert>

using namespace std;

// 1. Преобразование RGB в YCbCr
void rgb_to_ycbcr(int height, int width, const vector<vector<uint8_t>>& R, const vector<vector<uint8_t>>& G, const vector<vector<uint8_t>>& B, vector<vector<uint8_t>>& Y, vector<vector<uint8_t>>& cb, vector<vector<uint8_t>>& cr) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double r = R[y][x];
			double g = G[y][x];
			double b = B[y][x];

			Y[y][x] = 0.299 * r + 0.587 * g + 0.114 * b;
			cb[y][x] = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
			cr[y][x] = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
		}
	}
}

// 2. Даунсэмплинг 4:2:0 (применяется к Cb и Cr)
vector<vector<uint8_t>> downsample(int& height, int& width, const vector<vector<uint8_t>>& c) {
	int h = 0;
	int w = 0;

	vector<vector<uint8_t>> downsampled(height, vector<uint8_t>(width, 0));

	//булево значение, определяет нечётное ли число или нет
	int odd_h = (height % 2 == 0) ? 0 : 1;//0 - нет, 1 - да, нечётное
	int odd_w = (width % 2 == 0) ? 0 : 1;

	for (size_t y = 0; h < height - odd_h; y++, h += 2)
	{
		for (size_t x = 0; w < width - odd_h; x++, w += 2)
		{// среднее арифметическое
			int arithmetic_mean = (c[h    ][w] + c[h    ][w + 1]
								 + c[h + 1][w] + c[h + 1][w + 1]) / 4;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_w)
	{// правый край
		int h = 0;
		//const
		int w = width + odd_w;
		int x = width / 2 + odd_w;
		for (size_t y = 0; h < height - odd_h; y++, h += 2)
		{
			int arithmetic_mean
				= (c[h][w] + 0
				+ c[h + 1][w] + 0) / 2;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_h)
	{// нижний край
		int w = 0;
		//const
		int h = height + odd_h;
		int y = height / 2 + odd_h;
		for (size_t x = 0; w < width - odd_w; x++, w += 2)
		{
			int arithmetic_mean
				= (c[h][w] + c[h][w + 1]
				+ 0 + 0) / 2;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_h + odd_w == 2)
	{
		int y = height / 2 + odd_h;
		int x = width / 2 + odd_w;
		downsampled[y][x] = c[height][width];
	}

	height = height / 2 + odd_h;
	width = width / 2 + odd_w;

	return downsampled;
}

// 3. Разбиение изображения на блоки NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#include <stdexcept> // для invalid_argument
#include <array>

using ui8_Block8x8 = array<array<uint8_t, 8>, 8>;

vector<ui8_Block8x8> splitInto8x8Blocks(int height, int width, const vector<vector<uint8_t>>& Ycbcr) {
	if (Ycbcr.empty() || Ycbcr[0].empty()) {
		throw invalid_argument("Input vector is empty");
	}

	vector<ui8_Block8x8> BLOCKS;// двумерный массив блоков
	
	// Проходим по изображению с шагом 8х8
	//i - строки, j - столбцы
	for (size_t i = 0; i < height; i += 8) {
		int N = 8;
		if (height - i < 8) N = height - i;

		for (size_t j = 0; j < width; j += 8) {
			ui8_Block8x8 block{};
			int M = 8;
			if (width - j < 8) M = width - j;

			// Копируем данные в блок 8х8
			//bi - строки, bj - столбцы
			for (size_t bi = 0; bi < N; bi++) {
				for (size_t bj = 0; bj < M; bj++) {
					block[bi][bj] = Ycbcr[i + bi][j + bj];
				}
			}

			BLOCKS.push_back(block);
		}
	}

	return BLOCKS;
}

// 4.1 Прямое DCT-II преобразование для блока NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#include <array>

// Предварительно вычисленные константы для DCT-II 8x8
constexpr int N = 8;
constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880; // sqrt(2)

// Коэффициенты масштабирования для DCT-II
constexpr double C0 = 1.0 / SQRT2;
constexpr double C1 = 1.0;

using d_Block8x8 = array<array<double, N>, N>;

// Таблица косинусов для ускорения вычислений
d_Block8x8 precompute_cos_table() {
	d_Block8x8 cos_table{};
	for (int u = 0; u < N; u++) {
		for (int x = 0; x < N; x++) {
			cos_table[u][x] = cos((2 * x + 1) * u * PI / (2 * N));
		}
	}
	return cos_table;
}

const auto COS_TABLE = precompute_cos_table();

// 1D DCT-II для строки/столбца
void dct_1d(const array<double, N>& input, array<double, N>& output) {
	for (int u = 0; u < N; u++) {
		double sum = 0.0;
		double cu = (u == 0) ? C0 : C1;
		for (int x = 0; x < N; x++) {
			sum += input[x] * COS_TABLE[u][x];
		}
		output[u] = sum * cu * 0.5; // Нормализация
	}
}

// 2D DCT-II для блока 8x8
d_Block8x8 dct_2d_8x8(const ui8_Block8x8& block) {
	d_Block8x8 temp{};
	d_Block8x8 coeffs{};

	// Применяем 1D DCT к каждой строке (горизонтальное преобразование)
	for (int y = 0; y < N; y++) {
		array<double, N> row{};
		for (int x = 0; x < N; x++) {
			row[x] = block[y][x];
		}
		array<double, N> dct_row{};
		dct_1d(row, dct_row);
		for (int u = 0; u < N; u++) {
			temp[y][u] = dct_row[u];
		}
	}

	// Применяем 1D DCT к каждому столбцу (вертикальное преобразование)
	for (int u = 0; u < N; u++) {
		array<double, N> column{};
		for (int y = 0; y < N; y++) {
			column[y] = temp[y][u];
		}
		array<double, N> dct_column{};
		dct_1d(column, dct_column);
		for (int v = 0; v < N; v++) {
			coeffs[v][u] = dct_column[v];
		}
	}

	return coeffs;
}

// 4.2 Обратное DCT-II преобразование для блока NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
// 1D DCT-III обратное преобразование для строки/столбца
void idct_1d(const array<double, N>& input, array<double, N>& output) {
	for (int x = 0; x < N; x++) {
		double sum = 0.0;
		for (int u = 0; u < N; ++u) {
			double cu = (u == 0) ? C0 : C1;
			sum += cu * input[u] * COS_TABLE[u][x];
		}
		output[x] = sum * 0.5; // Нормализация
	}
}

// 2D IDCT для блока 8x8
d_Block8x8 idct_2d_8x8(const d_Block8x8& coeffs) {
	d_Block8x8 temp{};
	d_Block8x8 block{};

	// Применяем 1D IDCT к каждому столбцу (вертикальное преобразование)
	for (int u = 0; u < N; u++) {
		array<double, N> column{};
		for (int v = 0; v < N; v++) {
			column[v] = coeffs[v][u];
		}
		array<double, N> idct_column{};
		idct_1d(column, idct_column);
		for (int y = 0; y < N; y++) {
			temp[y][u] = idct_column[y];
		}
	}

	// Применяем 1D IDCT к каждой строке (горизонтальное преобразование)
	for (int y = 0; y < N; y++) {
		array<double, N> row{};
		for (int u = 0; u < N; u++) {
			row[u] = temp[y][u];
		}
		array<double, N> idct_row{};
		idct_1d(row, idct_row);
		for (int x = 0; x < N; x++) {
			block[y][x] = idct_row[x];
		}
	}

	return block;
}

// 5. Генерация матрицы квантования для заданного уровня качества
using i16_Block8x8 = array<array<int16_t, N>, N>;

// Annex K стандарт JPEG (ISO/IEC 10918-1) : 1993(E)
// Стандартная матрица квантования Y для качества 50
constexpr int Luminance_quantization_table[8][8] = {
	{16, 11, 10, 16, 24, 40, 51, 61},
	{12, 12, 14, 19, 26, 58, 60, 55},
	{14, 13, 16, 24, 40, 57, 69, 56},
	{14, 17, 22, 29, 51, 87, 80, 62},
	{18, 22, 37, 56, 68, 109, 103, 77},
	{24, 35, 55, 64, 81, 104, 113, 92},
	{49, 64, 78, 87, 103, 121, 120, 101},
	{72, 92, 95, 98, 112, 100, 103, 99}
};
// Стандартная матрица квантования Cb и Cr для качества 50
constexpr int Chrominance_quantization_table[8][8] = {
	{17, 18, 24, 47, 99, 99, 99, 99},
	{18, 21, 26, 66, 99, 99, 99, 99},
	{24, 26, 56, 99, 99, 99, 99, 99},
	{47, 66, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
	{99, 99, 99, 99, 99, 99, 99, 99},
};

void generate_quantization_matrix(int quality, i16_Block8x8 q_matrix, const int (&Quantization_table)[8][8]) {
	// Корректируем качество (1-100)
	quality = max(1, min(100, quality));

	// Вычисляем scale_factor
	double scale_factor;
	if (quality < 50) {
		scale_factor = 50.0 / quality;  // Для quality=1 -> scale_factor=50
	}
	else {
		scale_factor = 2.0 - (quality / 50.0);  // Для quality=100 -> scale_factor=0
	}

	// Масштабируем стандартную матрицу Cb/Cr
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			double q = Quantization_table[y][x] * scale_factor;
			q_matrix[y][x] = max(1, static_cast<int>(round(q)));
		}
	}
}

// 6.1 Квантование DCT коэффициентов
void quantize(const d_Block8x8& dct_coeffs, const i16_Block8x8& q_matrix, i16_Block8x8& quantized) {
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			quantized[y][x] = static_cast<int>(round(dct_coeffs[y][x] / q_matrix[y][x]));
		}
	}
}

// 6.2 Обратное квантование (восстановление DCT-коэффициентов)
void dequantize(const i16_Block8x8& quantized, const i16_Block8x8& q_matrix, d_Block8x8& dct_coeffs) {
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			dct_coeffs[y][x] = static_cast<double>(quantized[y][x]) * q_matrix[y][x];
		}
	}
}

// 7.1 Зигзаг-сканирование блока
constexpr int zigzag_order[64] = {
	0,
	1, 8,
	16, 9, 2,
	3, 10, 17, 24,
	32, 25, 18, 11, 4,
	5, 12, 19, 26, 33, 40,
	48, 41, 34, 27, 20, 13, 6,
	7, 14, 21, 28, 35, 42, 49, 56,
	57, 50, 43, 36, 29, 22, 15,
	23, 30, 37, 44, 51, 58,
	59, 52, 45, 38, 31,
	39, 46, 53, 60,
	61, 54, 47,
	55, 62,
	63
};

// 0  1  2  3  4  5  6  7
// 8  9 10 11 12 13 14 15
//16 17 18 19 20 21 22 23
//24 25 26 27 28 29 30 31
//32 33 34 35 36 37 38 39
//40 41 42 43 44 45 46 47
//48 49 50 51 52 53 54 55
//56 57 58 59 60 61 62 63

vector<int16_t> zigzag_scan_fast(const i16_Block8x8& quantized) {
	vector<int16_t> zigzag(64);
	for (int i = 0; i < N * N; i++) {
		int idx = zigzag_order[i];
		zigzag[i] = quantized[idx / N][idx % N];
	}
	return zigzag;
}

// 7.2 Обратное зигзаг-сканирование блока
i16_Block8x8 inverse_zigzag_scan(const vector<int16_t>& zigzag_coeffs) {
	i16_Block8x8 block{};

	for (int i = 0; i < N * N; i++) {
		int idx = zigzag_order[i];              // Получаем позицию в зигзаг-порядке
		block[idx / N][idx % N] = zigzag_coeffs[i]; // Заполняем блок
	}

	return block;
}

// после зиг-заг обхода обрабатываем полученный массив с DC и AC коэффициентами.
// 8.1 Разностное кодирование DC коэффициентов
void dc_difference(vector<int16_t>& str) {
	if (str.empty()) return;
	size_t size = str.size();
	
	for (size_t i = 64; i < size; i += 64) {
		str[i] -= str[i - 64];
	}
}

// 8.2 Обратное разностное кодирование DC коэффициентов
// ТРЕБУЕТ ПЕРЕДЕЛКИ
void reverse_difference_dc(const vector<int16_t>& delta_encoded) {
	if (delta_encoded.empty()) return;

	vector<int32_t> dc_coeffs(delta_encoded.size());
	dc_coeffs[0] = delta_encoded[0];

	for (size_t i = 1, j = 64; i < delta_encoded.size(); i++, j += 64) {
		dc_coeffs[i] = dc_coeffs[i - 1] + delta_encoded[i];
	}
}

// 9. Переменного кодирования разностей DC и AC коэффициентов.
vector<int16_t> intToBinaryVector(int16_t num, int positive1_or_negative0 = 1/*влияет на биты, будут ли биты инвертивными*/) {
	vector<int16_t> bits;

	if (num == 0) {
		bits.push_back(0);
		return bits;
	}

	if (positive1_or_negative0 == 0) num *= -1;

	// Разложение числа на биты
	while (num > 0) {
		bits.push_back(num % 2 == positive1_or_negative0 ? 1 : 0); // Младший бит
		num /= 2;
	}

	// Чтобы биты шли в привычном порядке (старший бит первым), развернём вектор
	reverse(bits.begin(), bits.end());

	return bits;
}

bool rle_encode_ac(int16_t num, vector<int16_t>& rle_str, int& zero_count, bool& EOB);

// типо продолжение разностного кодирования DC, а также мы кодируем AC
void preparing_for_coding_dc_and_ac(vector<int16_t>& str) {
	vector<int16_t> output;
	output.reserve(str.size());

	dc_difference(str);
	//вся запись будет не побитовой
	bool EOB;
	size_t size = str.size();
	for (size_t i = 0; i < size; i += 64)// блок
	{
		vector<int16_t> temp;
		// {DC coeffs}
		if (str[i] >= 0)
		{//запись категорий DC и сам DC ввиде бинарного кода в вектор, но запись не бинарная.
			temp = intToBinaryVector(str[i], 1);
		}
		else
		{//если число отрицательно, то мы инвертируем его биты
			temp = intToBinaryVector(str[i], 0);
		}
		output.push_back(static_cast<int16_t>(temp.size()));
		copy(temp.begin(), temp.end(), back_inserter(output));

		// {AC coeffs}
		int zero_count = 0;
		for (size_t j = i + 1; j < 64; j++)// блок
		{// подсчёт нулей и его запись, запись категорий AC с самим коэффициентом и так заного пока не 0,0 (EOB).
			EOB = false;
			if (rle_encode_ac(str[j], output, zero_count, EOB)) continue;

			if (str[j] >= 0)
			{//запись категорий DC и сам DC ввиде бинарного кода в вектор, но запись не бинарная.
				temp = intToBinaryVector(str[j], 1);
			}
			else
			{//если число отрицательно, то мы инвертируем его биты
				temp = intToBinaryVector(str[j], 0);
			}

			output.push_back(static_cast<int16_t>(temp.size()));
			copy(temp.begin(), temp.end(), back_inserter(output));
		}

		// Добавляем маркер конца блока (0,0)
//ЕСЛИ ЧИСЛО НУЛЕЙ УКАЗАНО 0 И СЛЕДУЮЩЕЕ ЗНАЧЕНИЕ ОЖИДАЕТСЯ ЧИСЛО НЕ 0, НО ОКАЗАЛОСЬ ИМ, ТО ЭТО МАРКЕР КОНЦА, ОБОЗНАЧАЮЩИЙ, ЧТО ДАЛЬШЕ ОСТАЮТСЯ ТОЛЬКО НУЛИ.
		if (zero_count != 0)
		{
			temp.push_back(0);
			temp.push_back(0);
		}
		if (EOB)
		{//случай когда попалось ровно 16 нулей до конца блока
			temp[temp.size() - 2] = 0;
		}
	}

	str = output;
}

// 10.1
// RLE кодирование AC коэффициентов
bool rle_encode_ac(int16_t num, vector<int16_t>& rle_str, int& zero_count, bool& EOB) {
	if (num == 0) {
		zero_count++;
		if (zero_count > 15)
		{
			rle_str.push_back(15);
			rle_str.push_back(0);
			zero_count = 0;
			EOB = true;
		}
		return 1;
	}
	else {
		rle_str.push_back(zero_count);
		return 0;
	}
}

// 10.2 Обратное RLE кодирование AC коэффициентов
// ТРЕБУЕТ ПЕРЕДЕЛКИ
vector<int> rle_decode_ac(const vector<pair<int, int>>& rle_encoded, int total_ac) {
	vector<int> ac_coeffs;

	for (const auto& run_value : rle_encoded) {
		if (run_value.first == 0 && run_value.second == 0) {
			// Маркер конца блока - заполняем оставшиеся нулями
			while (ac_coeffs.size() < total_ac) {
				ac_coeffs.push_back(0);
			}
			break;
		}

		// Добавляем нули
		for (int i = 0; i < run_value.first; ++i) {
			ac_coeffs.push_back(0);
		}

		// Добавляем ненулевое значение
		ac_coeffs.push_back(run_value.second);
	}

	return ac_coeffs;
}

// 11. Кодирования разностей  DC коэффициентов и последовательностей  Run/Size  по таблице кодов Хаффмана и упаковки результата в байтовую строку.
#include <string_view>//C++17 и новее
// Annex K стандарт JPEG (ISO/IEC 10918-1) : 1993(E)
constexpr string_view Luminance_DC_differences[12] = {
	"00",//0
	"010",//1
	"011",//2
	"100",//3
	"101",//4
	"110",//5
	"1110",//6
	"11110",//7
	"111110",//8
	"1111110",//9
	"11111110",//10
	"111111110",//11
};
constexpr string_view Luminance_AC[16][11] = {
			{
/*0 / 0 (EOB)14*/ "1010",
/*0 / 1 12*/ "00",
/*0 / 2 12*/ "01",
/*0 / 3 13*/ "100",
/*0 / 4 14*/ "1011",
/*0 / 5 15*/ "11010",
/*0 / 6 17*/ "1111000",
/*0 / 7 18*/ "11111000",
/*0 / 8 10*/ "1111110110",
/*0 / 9 16*/ "1111111110000010",
/*0 / A 16*/ "1111111110000011"
			},
			{
/*empty*/    "",
/*1 / 1  4*/ "1100",
/*1 / 2  5*/ "11011",
/*1 / 3  7*/ "1111001",
/*1 / 4  9*/ "111110110",
/*1 / 5 11*/ "11111110110",
/*1 / 6 16*/ "1111111110000100",
/*1 / 7 16*/ "1111111110000101",
/*1 / 8 16*/ "1111111110000110",
/*1 / 9 16*/ "1111111110000111",
/*1 / A 16*/ "1111111110001000"
			},
			{
/*empty*/    "",
/*2 / 1  5*/ "11100",
/*2 / 2  8*/ "11111001",
/*2 / 3 10*/ "1111110111",
/*2 / 4 12*/ "111111110100",
/*2 / 5 16*/ "1111111110001001",
/*2 / 6 16*/ "1111111110001010",
/*2 / 7 16*/ "1111111110001011",
/*2 / 8 16*/ "1111111110001100",
/*2 / 9 16*/ "1111111110001101",
/*2 / A 16*/ "1111111110001110"
			},
			{
/*empty*/    "",
/*3 / 1  6*/ "111010",
/*3 / 2  9*/ "111110111",
/*3 / 3 12*/ "111111110101",
/*3 / 4 16*/ "1111111110001111",
/*3 / 5 16*/ "1111111110010000",
/*3 / 6 16*/ "1111111110010001",
/*3 / 7 16*/ "1111111110010010",
/*3 / 8 16*/ "1111111110010011",
/*3 / 9 16*/ "1111111110010100",
/*3 / A 16*/ "1111111110010101"
			},
			{
/*empty*/    "",
/*4 / 1  6*/ "111011",
/*4 / 2 10*/ "1111111000",
/*4 / 3 16*/ "1111111110010110",
/*4 / 4 16*/ "1111111110010111",
/*4 / 5 16*/ "1111111110011000",
/*4 / 6 16*/ "1111111110011001",
/*4 / 7 16*/ "1111111110011010",
/*4 / 8 16*/ "1111111110011011",
/*4 / 9 16*/ "1111111110011100",
/*4 / A 16*/ "1111111110011101"
			},
			{
/*empty*/    "",
/*5 / 1  7*/ "1111010",
/*5 / 2 11*/ "11111110111",
/*5 / 3 16*/ "1111111110011110",
/*5 / 4 16*/ "1111111110011111",
/*5 / 5 16*/ "1111111110100000",
/*5 / 6 16*/ "1111111110100001",
/*5 / 7 16*/ "1111111110100010",
/*5 / 8 16*/ "1111111110100011",
/*5 / 9 16*/ "1111111110100100",
/*5 / A 16*/ "1111111110100101"
			},
			{
/*empty*/    "",
/*6 / 1  7*/ "1111011",
/*6 / 2 12*/ "111111110110",
/*6 / 3 16*/ "1111111110100110",
/*6 / 4 16*/ "1111111110100111",
/*6 / 5 16*/ "1111111110101000",
/*6 / 6 16*/ "1111111110101001",
/*6 / 7 16*/ "1111111110101010",
/*6 / 8 16*/ "1111111110101011",
/*6 / 9 16*/ "1111111110101100",
/*6 / A 16*/ "1111111110101101"
			},
			{
/*empty*/    "",
/*7 / 1  8*/ "11111010",
/*7 / 2 12*/ "111111110111",
/*7 / 3 16*/ "1111111110101110",
/*7 / 4 16*/ "1111111110101111",
/*7 / 5 16*/ "1111111110110000",
/*7 / 6 16*/ "1111111110110001",
/*7 / 7 16*/ "1111111110110010",
/*7 / 8 16*/ "1111111110110011",
/*7 / 9 16*/ "1111111110110100",
/*7 / A 16*/ "1111111110110101"
			},
			{
/*empty*/    "",
/*8 / 1  9*/ "111111000",
/*8 / 2 15*/ "111111111000000",
/*8 / 3 16*/ "1111111110110110",
/*8 / 4 16*/ "1111111110110111",
/*8 / 5 16*/ "1111111110111000",
/*8 / 6 16*/ "1111111110111001",
/*8 / 7 16*/ "1111111110111010",
/*8 / 8 16*/ "1111111110111011",
/*8 / 9 16*/ "1111111110111100",
/*8 / A 16*/ "1111111110111101"
			},
			{
/*empty*/    "",
/*9 / 1  9*/ "111111001",
/*9 / 2 16*/ "1111111110111110",
/*9 / 3 16*/ "1111111110111111",
/*9 / 4 16*/ "1111111111000000",
/*9 / 5 16*/ "1111111111000001",
/*9 / 6 16*/ "1111111111000010",
/*9 / 7 16*/ "1111111111000011",
/*9 / 8 16*/ "1111111111000100",
/*9 / 9 16*/ "1111111111000101",
/*9 / A 16*/ "1111111111000110"
			},
			{
/*empty*/    "",
/*A / 1  9*/ "111111010",
/*A / 2 16*/ "1111111111000111",
/*A / 3 16*/ "1111111111001000",
/*A / 4 16*/ "1111111111001001",
/*A / 5 16*/ "1111111111001010",
/*A / 6 16*/ "1111111111001011",
/*A / 7 16*/ "1111111111001100",
/*A / 8 16*/ "1111111111001101",
/*A / 9 16*/ "1111111111001110",
/*A / A 16*/ "1111111111001111"
			},
			{
/*empty*/    "",
/*B / 1 10*/ "1111111001",
/*B / 2 16*/ "1111111111010000",
/*B / 3 16*/ "1111111111010001",
/*B / 4 16*/ "1111111111010010",
/*B / 5 16*/ "1111111111010011",
/*B / 6 16*/ "1111111111010100",
/*B / 7 16*/ "1111111111010101",
/*B / 8 16*/ "1111111111010110",
/*B / 9 16*/ "1111111111010111",
/*B / A 16*/ "1111111111011000"
			},
			{
/*empty*/    "",
/*C / 1 10*/ "1111111010",
/*C / 2 16*/ "1111111111011001",
/*C / 3 16*/ "1111111111011010",
/*C / 4 16*/ "1111111111011011",
/*C / 5 16*/ "1111111111011100",
/*C / 6 16*/ "1111111111011101",
/*C / 7 16*/ "1111111111011110",
/*C / 8 16*/ "1111111111011111",
/*C / 9 16*/ "1111111111100000",
/*C / A 16*/ "1111111111100001"
			},
			{
/*empty*/    "",
/*D / 1 11*/ "11111111000",
/*D / 2 16*/ "1111111111100010",
/*D / 3 16*/ "1111111111100011",
/*D / 4 16*/ "1111111111100100",
/*D / 5 16*/ "1111111111100101",
/*D / 6 16*/ "1111111111100110",
/*D / 7 16*/ "1111111111100111",
/*D / 8 16*/ "1111111111101000",
/*D / 9 16*/ "1111111111101001",
/*D / A 16*/ "1111111111101010"
			},
			{
/*empty*/    "",
/*E / 1 16*/ "1111111111101011",
/*E / 2 16*/ "1111111111101100",
/*E / 3 16*/ "1111111111101101",
/*E / 4 16*/ "1111111111101110",
/*E / 5 16*/ "1111111111101111",
/*E / 6 16*/ "1111111111110000",
/*E / 7 16*/ "1111111111110001",
/*E / 8 16*/ "1111111111110010",
/*E / 9 16*/ "1111111111110011",
/*E / A 16*/ "1111111111110100"
			},
			{
/*F / 0 (ZRL)11*/ "11111111001",
/*F / 1 16*/ "1111111111110101",
/*F / 2 16*/ "1111111111110110",
/*F / 3 16*/ "1111111111110111",
/*F / 4 16*/ "1111111111111000",
/*F / 5 16*/ "1111111111111001",
/*F / 6 16*/ "1111111111111010",
/*F / 7 16*/ "1111111111111011",
/*F / 8 16*/ "1111111111111100",
/*F / 9 16*/ "1111111111111101",
/*F / A 16*/ "1111111111111110"
			}
};
constexpr string_view Chrominance_DC_differences[12] = {
	"00",//0
	"01",//1
	"10",//2
	"110",//3
	"1110",//4
	"11110",//5
	"111110",//6
	"1111110",//7
	"11111110",//8
	"111111110",//9
	"1111111110",//10
	"11111111110",//11
};
constexpr string_view Chrominance_AC[16][11] = {
			{
/*0 / 0 (EOB)2*/ "00",
/*0 / 1  2*/ "01",
/*0 / 2  3*/ "100",
/*0 / 3  4*/ "1010",
/*0 / 4  5*/ "11000",
/*0 / 5  5*/ "11001",
/*0 / 6  6*/ "111000",
/*0 / 7  7*/ "1111000",
/*0 / 8  9*/ "111110100",
/*0 / 9 10*/ "1111110110",
/*0 / A 12*/ "111111110100"
			},
			{
/*empty*/    "",
/*1 / 1  4*/ "1011",
/*1 / 2  6*/ "111001",
/*1 / 3  8*/ "11110110",
/*1 / 4  9*/ "111110101",
/*1 / 5 11*/ "11111110110",
/*1 / 6 12*/ "111111110101",
/*1 / 7 16*/ "1111111110001000",
/*1 / 8 16*/ "1111111110001001",
/*1 / 9 16*/ "1111111110001010",
/*1 / A 16*/ "1111111110001011"
			},
			{
/*empty*/    "",
/*2 / 1  5*/ "11010",
/*2 / 2  8*/ "11110111",
/*2 / 3 10*/ "1111110111",
/*2 / 4 12*/ "111111110110",
/*2 / 5 15*/ "111111111000010",
/*2 / 6 16*/ "1111111110001100",
/*2 / 7 16*/ "1111111110001101",
/*2 / 8 16*/ "1111111110001110",
/*2 / 9 16*/ "1111111110001111",
/*2 / A 16*/ "1111111110010000"
			},
			{
/*empty*/    "",
/*3 / 1 15*/ "11011",
/*3 / 2 18*/ "11111000",
/*3 / 3 10*/ "1111111000",
/*3 / 4 12*/ "111111110111",
/*3 / 5 16*/ "1111111110010001",
/*3 / 6 16*/ "1111111110010010",
/*3 / 7 16*/ "1111111110010011",
/*3 / 8 16*/ "1111111110010100",
/*3 / 9 16*/ "1111111110010101",
/*3 / A 16*/ "1111111110010110"
			},
			{
/*empty*/    "",
/*4 / 1  6*/ "111010",
/*4 / 2  9*/ "111110110",
/*4 / 3 16*/ "1111111110010111",
/*4 / 4 16*/ "1111111110011000",
/*4 / 5 16*/ "1111111110011001",
/*4 / 6 16*/ "1111111110011010",
/*4 / 7 16*/ "1111111110011011",
/*4 / 8 16*/ "1111111110011100",
/*4 / 9 16*/ "1111111110011101",
/*4 / A 16*/ "1111111110011110"
			},
			{
/*empty*/    "",
/*5 / 1  6*/ "111011",
/*5 / 2 10*/ "1111111001",
/*5 / 3 16*/ "1111111110011111",
/*5 / 4 16*/ "1111111110100000",
/*5 / 5 16*/ "1111111110100001",
/*5 / 6 16*/ "1111111110100010",
/*5 / 7 16*/ "1111111110100011",
/*5 / 8 16*/ "1111111110100100",
/*5 / 9 16*/ "1111111110100101",
/*5 / A 16*/ "1111111110100110"
			},
			{
/*empty*/    "",
/*6 / 1  7*/ "1111001",
/*6 / 2 11*/ "11111110111",
/*6 / 3 16*/ "1111111110100111",
/*6 / 4 16*/ "1111111110101000",
/*6 / 5 16*/ "1111111110101001",
/*6 / 6 16*/ "1111111110101010",
/*6 / 7 16*/ "1111111110101011",
/*6 / 8 16*/ "1111111110101100",
/*6 / 9 16*/ "1111111110101101",
/*6 / A 16*/ "1111111110101110"
			},
			{
/*empty*/    "",
/*7 / 1  7*/ "1111010",
/*7 / 2 11*/ "11111111000",
/*7 / 3 16*/ "1111111110101111",
/*7 / 4 16*/ "1111111110110000",
/*7 / 5 16*/ "1111111110110001",
/*7 / 6 16*/ "1111111110110010",
/*7 / 7 16*/ "1111111110110011",
/*7 / 8 16*/ "1111111110110100",
/*7 / 9 16*/ "1111111110110101",
/*7 / A 16*/ "1111111110110110"
			},
			{
/*empty*/    "",
/*8 / 1  8*/ "11111001",
/*8 / 2 16*/ "1111111110110111",
/*8 / 3 16*/ "1111111110111000",
/*8 / 4 16*/ "1111111110111001",
/*8 / 5 16*/ "1111111110111010",
/*8 / 6 16*/ "1111111110111011",
/*8 / 7 16*/ "1111111110111100",
/*8 / 8 16*/ "1111111110111101",
/*8 / 9 16*/ "1111111110111110",
/*8 / A 16*/ "1111111110111111"
			},
			{
/*empty*/    "",
/*9 / 1  9*/ "111110111",
/*9 / 2 16*/ "1111111111000000",
/*9 / 3 16*/ "1111111111000001",
/*9 / 4 16*/ "1111111111000010",
/*9 / 5 16*/ "1111111111000011",
/*9 / 6 16*/ "1111111111000100",
/*9 / 7 16*/ "1111111111000101",
/*9 / 8 16*/ "1111111111000110",
/*9 / 9 16*/ "1111111111000111",
/*9 / A 16*/ "1111111111001000"
			},
			{
/*empty*/    "",
/*A / 1  9*/ "111111000",
/*A / 2 16*/ "1111111111001001",
/*A / 3 16*/ "1111111111001010",
/*A / 4 16*/ "1111111111001011",
/*A / 5 16*/ "1111111111001100",
/*A / 6 16*/ "1111111111001101",
/*A / 7 16*/ "1111111111001110",
/*A / 8 16*/ "1111111111001111",
/*A / 9 16*/ "1111111111010000",
/*A / A 16*/ "1111111111010001"
			},
			{
/*empty*/    "",
/*B / 1  9*/ "111111001",
/*B / 2 16*/ "1111111111010010",
/*B / 3 16*/ "1111111111010011",
/*B / 4 16*/ "1111111111010100",
/*B / 5 16*/ "1111111111010101",
/*B / 6 16*/ "1111111111010110",
/*B / 7 16*/ "1111111111010111",
/*B / 8 16*/ "1111111111011000",
/*B / 9 16*/ "1111111111011001",
/*B / A 16*/ "1111111111011010"
			},
			{
/*empty*/    "",
/*C / 1  9*/ "111111010",
/*C / 2 16*/ "1111111111011011",
/*C / 3 16*/ "1111111111011100",
/*C / 4 16*/ "1111111111011101",
/*C / 5 16*/ "1111111111011110",
/*C / 6 16*/ "1111111111011111",
/*C / 7 16*/ "1111111111100000",
/*C / 8 16*/ "1111111111100001",
/*C / 9 16*/ "1111111111100010",
/*C / A 16*/ "1111111111100011"
			},
			{
/*empty*/    "",
/*D / 1 11*/ "11111111001",
/*D / 2 16*/ "1111111111100100",
/*D / 3 16*/ "1111111111100101",
/*D / 4 16*/ "1111111111100110",
/*D / 5 16*/ "1111111111100111",
/*D / 6 16*/ "1111111111101000",
/*D / 7 16*/ "1111111111101001",
/*D / 8 16*/ "1111111111101010",
/*D / 9 16*/ "1111111111101011",
/*D / A 16*/ "1111111111101100"
			},
			{
/*empty*/    "",
/*E / 1 14*/ "11111111100000",
/*E / 2 16*/ "1111111111101101",
/*E / 3 16*/ "1111111111101110",
/*E / 4 16*/ "1111111111101111",
/*E / 5 16*/ "1111111111110000",
/*E / 6 16*/ "1111111111110001",
/*E / 7 16*/ "1111111111110010",
/*E / 8 16*/ "1111111111110011",
/*E / 9 16*/ "1111111111110100",
/*E / A 16*/ "1111111111110101"
			},
			{
/*F / 0 (ZRL)10*/ "1111111010",
/*F / 1 15*/ "111111111000011",
/*F / 2 16*/ "1111111111110110",
/*F / 3 16*/ "1111111111110111",
/*F / 4 16*/ "1111111111111000",
/*F / 5 16*/ "1111111111111001",
/*F / 6 16*/ "1111111111111010",
/*F / 7 16*/ "1111111111111011",
/*F / 8 16*/ "1111111111111100",
/*F / 9 16*/ "1111111111111101",
/*F / A 16*/ "1111111111111110"
			}
};

// Все таблицы найдёшь, поставив курсор на имя функцим и нажав f12
string HA_encode(const vector<int16_t>& data, const string_view (&DC_differences)[12], const string_view (&AC)[16][11]) {
	string encoded;

	size_t size = data.size();
	for (size_t i = 0; i < size; i++)
	{
		// DC
		encoded += DC_differences[data[i]];
		int k_size = data[i];
		for (int k = 0; k < k_size; k++)
		{// записть кода 0/1
			encoded += data[++i];
		}

		// AC
		int count = 1;// мы уже обработали DC, поэтому 1.
		while (count >= 64)
		{//0-15 кол. нулей
			i++;
			if (data[i] == 0 && data[i + 1] == 0)
			{
				encoded += AC[0][0];
				break;
			}

			if (data[i] == 15 && data[i + 1] == 0)
			{
				encoded += AC[15][0];
				count += 16;
				continue;
			}

			count += data[i] + 1;
			encoded += AC[data[i]][data[i + 1]];

			int k_size = data[i + 1];
			for (int k = 0; k < k_size; k++)
			{// записть кода 0/1
				encoded += data[++i];
			}
		}
	}

	return encoded;
}

// Упаковка битовой строки в байты
vector<uint8_t> pack_bits_to_bytes(const string& bit_string) {
	vector<uint8_t> output;
	size_t len = bit_string.length();
	uint8_t zero_bits = 0;
	output.push_back(zero_bits);

	for (size_t i = 0; i < len; i += 8) {
		string byte_str = bit_string.substr(i, 8);
		if (byte_str.length() < 8) {
			zero_bits = 8 - byte_str.length();
			byte_str.append(zero_bits, '0'); // Дополняем нулями
		}

		bitset<8> bits(byte_str);
		output.push_back(static_cast<uint8_t>(bits.to_ulong()));
	}

	output[0] = zero_bits;
	return output;
}

int main() {
	ifstream ifT;
	//ofstream ofT;
	//ofstream console;

	setlocale(LC_ALL, "ru");
	SetConsoleOutputCP(CP_UTF8);

	string filename = "grey.raw"; //800 x 600 (width x height)
	int width = 800;//512
	int height = 600;

	ifT.open(filename, ios::binary);
	if (!ifT.is_open()) {
		cerr << "Error opening file: " << filename << endl;
		return 0;
	}

	ifT.seekg(0, std::ios::end);  // Перемещаем указатель в конец файла
	auto size = ifT.tellg();       // Получаем позицию (размер файла)
	ifT.seekg(0, std::ios::beg);   // Возвращаем указатель в начало
	cout << "check size: " << size << '\n';
	if (size != 3 * height * width)
	{
		cerr << "Error the resolution does not match the size: " << size << endl;
		return 0;
	}
	cout << "Параметры изображения:\n1) Размер: " << size << " байт\n";
	cout << "2) Количество: " << size/3 << " pixels\n";
	cout << "3) Разрешение: " << height << "x" << width << " байт\n";

	// encode
	
	//array<array<uint8_t, width>, height> r;
	vector<vector<uint8_t>> r(height, vector<uint8_t>(width));
	vector<vector<uint8_t>> g(height, vector<uint8_t>(width));
	vector<vector<uint8_t>> b(height, vector<uint8_t>(width));

	uint8_t color[3];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (!ifT.read(reinterpret_cast<char*>(color), 3)) {
				throw runtime_error("Ошибка чтения файла");
			}
			r[y][x] = color[0];
			g[y][x] = color[1];
			b[y][x] = color[2];
		}
	}

	vector<vector<uint8_t>> Y(height, vector<uint8_t>(width));
	vector<vector<uint8_t>> cb(height, vector<uint8_t>(width));
	vector<vector<uint8_t>> cr(height, vector<uint8_t>(width));

	rgb_to_ycbcr(height, width, r, g, b, Y, cb, cr);

	//4:2:0 - берём по 1 цвету в двух блоках по 4 элемента
	//цвет усреднённый среди 4 элементов
	if (height % 8 != 0 || width % 8 != 0) {
		cout << "разрешение не делится на 8";
		return 0;
	}
	int height_c = height;
	int width_c = width;
	// переписать даунсэплинг чтобы сохранялась размерность массивов
	cb = downsample(height_c, width_c, cb);
	cr = downsample(height_c, width_c, cr);

	// Разбиение на блоки и запись блоков в одномерном векторе
	// i8_Block8x8
	auto Y_blocks = splitInto8x8Blocks(height, width, Y);
	auto cb_blocks = splitInto8x8Blocks(height_c, width_c, cb);
	auto cr_blocks = splitInto8x8Blocks(height_c, width_c, cr);

	size_t Y_blSize = Y_blocks.size();
	size_t cb_blSize = cb_blocks.size();// совпадает с cr_blocks.size() в давнной реализации

	vector<i16_Block8x8> Y_done(Y_blSize);
	vector<i16_Block8x8> cb_done(cb_blSize);
	vector<i16_Block8x8> cr_done(cb_blSize);

	int quality = 100;
	i16_Block8x8 q0_matrix{};
	i16_Block8x8 q1_matrix{};
	generate_quantization_matrix(quality, q0_matrix, Luminance_quantization_table);
	generate_quantization_matrix(quality, q1_matrix, Chrominance_quantization_table);

	for (size_t x = 0; x < Y_blSize; x++)
	{// d_Block8x8
		auto Y_dct = dct_2d_8x8(Y_blocks[x]);
		quantize(Y_dct, q0_matrix, Y_done[x]);// запись в Y_done
	}
	for (size_t x = 0; x < cb_blSize; x++)
	{// d_Block8x8
		auto cb_dct = dct_2d_8x8(cb_blocks[x]);
		auto cr_dct = dct_2d_8x8(cr_blocks[x]);

		quantize(cb_dct, q1_matrix, cb_done[x]);// запись в cb_done
		quantize(cr_dct, q1_matrix, cr_done[x]);// запись в cr_done
	}

	// Зиг-заг обход для каждого блока
	vector<int16_t> str1;
	vector<int16_t> str2;
	vector<int16_t> str3;
	str1.reserve(Y_blSize * 64 + 2 * cb_blSize * 64);
	str2.reserve(cb_blSize * 64);
	str3.reserve(cb_blSize * 64);

	for (size_t x = 0; x < Y_blSize; x++)
	{// Y
		auto str = zigzag_scan_fast(Y_done[x]);
		copy(str.begin(), str.end(), back_inserter(str1));
	}
	for (size_t x = 0; x < cb_blSize; x++)
	{// Cb
		auto str = zigzag_scan_fast(cb_done[x]);
		copy(str.begin(), str.end(), back_inserter(str2));
	}
	for (size_t x = 0; x < cb_blSize; x++)
	{// Cr
		auto str = zigzag_scan_fast(cr_done[x]);
		copy(str.begin(), str.end(), back_inserter(str3));
	}
	
	preparing_for_coding_dc_and_ac(str1);
	preparing_for_coding_dc_and_ac(str2);
	preparing_for_coding_dc_and_ac(str3);

	string code;
	code  = HA_encode(str1, Luminance_DC_differences, Luminance_AC);
	code += HA_encode(str2, Chrominance_DC_differences, Chrominance_AC);
	code += HA_encode(str3, Chrominance_DC_differences, Chrominance_AC);

	vector<uint8_t> output = pack_bits_to_bytes(code);

	for (char c : output)
	{
		cout << c;
	}

	// decode

	ifT.close();
	//ofT.close();
	//console.close();
	return 0;
}