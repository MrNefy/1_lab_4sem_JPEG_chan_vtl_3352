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

#include <array>

using namespace std;

constexpr double PR[3][3] = {
	{1, 0, 1.402},
	{1, -0.344136, -0.714136},
	{1, 1.772, 0}
};

int divUp(int x, int y)
{// идея была такая (x + y - 1) / y
	//если мы добавим делитель к знамнателю и поделим на делитель, то по сравнению с прошлым результатом тут добавится единица
	// то есть, если добавить числителю делитель на 1 меньше, то результату добавится число всегда меньшее единице, но достаточное, чтобы округлить вверх
	return (x - 1) / y + 1;// упрощённая форма
}

// 1. Преобразование RGB в YCbCr
using inputArray = vector<vector<int16_t>>;

void rgb_to_ycbcr(int height, int width
	, const inputArray& R, const inputArray& G, const inputArray& B
	, inputArray& Y, inputArray& cb, inputArray& cr) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double r = R[y][x];
			double g = G[y][x];
			double b = B[y][x];

			Y[y][x] = static_cast<int16_t>(0.299 * r + 0.587 * g + 0.114 * b);
			cb[y][x] = static_cast<int16_t>(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
			cr[y][x] = static_cast<int16_t>(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);
		}
	}
}

void ycbcr_to_rgb(int height, int width
	, inputArray& R, inputArray& G, inputArray& B
	, const inputArray& Y, const inputArray& cb, const inputArray& cr
	, const double RG[3][3]) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double y_val = Y[y][x];
			double cb_val = cb[y][x] - 128;
			double cr_val = cr[y][x] - 128;

			double r = RG[0][0] * y_val + RG[0][1] * cb_val + RG[0][2] * cr_val;
			double g = RG[1][0] * y_val + RG[1][1] * cb_val + RG[1][2] * cr_val;
			double b = RG[2][0] * y_val + RG[2][1] * cb_val + RG[2][2] * cr_val;

			R[y][x] = static_cast<int16_t>(max(0.0, min(255.0, r)));
			G[y][x] = static_cast<int16_t>(max(0.0, min(255.0, g)));
			B[y][x] = static_cast<int16_t>(max(0.0, min(255.0, b)));
		}
	}
}

// 2. Даунсэмплинг 4:2:0 (применяется к Cb и Cr)
inputArray downsample(int height, int width, const inputArray& c);

// 3. Разбиение изображения на блоки NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#include <array>

constexpr int N = 8;
using i16_Block8x8 = array<array<int16_t, N>, N>;

vector<i16_Block8x8> splitInto8x8Blocks(int height, int width, const inputArray& Y_Cb_Cr);

// 4.1 Прямое DCT-II преобразование для блока NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
// Предварительно вычисленные константы для DCT-II 8x8
constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880; // sqrt(2)

// Коэффициенты масштабирования для DCT-II
constexpr double C0 = 1.0 / SQRT2;
constexpr double C1 = 1.0;

using d_Block8x8 = array<array<double, N>, N>;

// Таблица косинусов для ускорения вычислений
d_Block8x8 precompute_cos_table() {
	d_Block8x8 cos_table;
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
d_Block8x8 dct_2d_8x8(const i16_Block8x8& block) {
	d_Block8x8 temp;
	d_Block8x8 coeffs;

	// Применяем 1D DCT к каждой строке (горизонтальное преобразование)
	for (int y = 0; y < N; y++) {
		array<double, N> row;
		for (int x = 0; x < N; x++) {
			row[x] = block[y][x] - 128;
		}
		array<double, N> dct_row{};
		dct_1d(row, dct_row);
		temp[y] = dct_row;
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
i16_Block8x8 idct_2d_8x8(const d_Block8x8& coeffs) {
	d_Block8x8 temp;
	i16_Block8x8 block;

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
			block[y][x] = static_cast<int16_t>(round(idct_row[x])) + 128;
		}
	}

	return block;
}

// 5. Генерация матрицы квантования для заданного уровня качества
// Annex K стандарт JPEG (ISO/IEC 10918-1) : 1993(E)
// Стандартная матрица квантования Y
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
// Стандартная матрица квантования Cb и Cr
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

void generate_quantization_matrix(int quality, d_Block8x8& q_matrix, const int (&Quantization_table)[8][8]) {
	// Корректируем качество (1-100)
	quality = max(1, min(100, quality));

	// Вычисляем scale_factor
	double scaleFactor;
	if (quality < 50) {
		scaleFactor = 200.0 / quality;  // Для Q < 50
	}
	else {
		scaleFactor = 8 * (1.0 - 0.01 * quality);  // Для Q >= 50
	}

	// Масштабируем стандартную матрицу Cb/Cr
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			double q = Quantization_table[y][x] * scaleFactor;
			q_matrix[y][x] = max(1.0, min(255.0, q));
		}
	}
}

// 6.1 Квантование DCT коэффициентов
void quantize(const d_Block8x8& dct_coeffs, const d_Block8x8& q_matrix, i16_Block8x8& quantized) {
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			quantized[y][x] = static_cast<int16_t>(round(dct_coeffs[y][x] / q_matrix[y][x]));
		}
	}
}

// 6.2 Обратное квантование (восстановление DCT-коэффициентов)
d_Block8x8 dequantize(const i16_Block8x8& quantized, const d_Block8x8& q_matrix) {
	d_Block8x8 dct_coeffs;
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			dct_coeffs[y][x] = static_cast<double>(quantized[y][x]) * q_matrix[y][x];
		}
	}

	return dct_coeffs;
}

// 7.1 Зигзаг-сканирование блока
constexpr int zigzag_sequence[64] = {
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

vector<int16_t> zigzag_scan(const i16_Block8x8& block) {
	vector<int16_t> str(64);

	for (int i = 0; i < N * N; i++) {
		int idx = zigzag_sequence[i];
		str[i] = block[idx / N][idx % N];
	}

	return str;
}

// 7.2 Обратное зигзаг-сканирование блока
i16_Block8x8 inverse_zigzag_scan(const array<int16_t, 64>& str) {
	i16_Block8x8 block{};

	for (int i = 0; i < 64; i++) {
		int idx = zigzag_sequence[i];
		block[idx / 8][idx % 8] = str[i];
	}

	return block;
}

// после зиг-заг обхода обрабатываем полученный массив с DC и AC коэффициентами.
// 8.1 Разностное кодирование DC коэффициентов
void dc_difference(vector<int16_t>& data) {
	size_t size = data.size();
	vector<int16_t> temp(size / 64);

	for (size_t i = 0, t = 0; i < size; i += 64, t++) {
		temp[t] = data[i];
	}

	for (size_t i = 64, t = 0; i < size; i += 64, t++) {
		data[i] -= temp[t];
	}
}

// 8.2 Обратное разностное кодирование DC коэффициентов
// ТРЕБУЕТ ПЕРЕДЕЛКИ
void reverse_dc_difference(vector<array<int16_t, 64>>& data) {
	size_t size = data.size();

	for (size_t i = 1; i < size; i++) {
		data[i][0] += data[i - 1][0];
	}
}

// 9. Переменного кодирования разностей DC и AC коэффициентов.
vector<int16_t> intToBinaryVector(int16_t num, int positive1_or_negative0 = 1/*влияет на биты, будут ли биты инвертивными*/) {
	vector<int16_t> bits;

	if (positive1_or_negative0 == 0) num *= -1;

	// Разложение числа на биты
	while (num > 0) {
		bits.push_back(num % 2 == positive1_or_negative0); // Младший бит
		num /= 2;
	}

	// Чтобы биты шли в привычном порядке (старший бит первым), развернём вектор
	reverse(bits.begin(), bits.end());

	return bits;
}

bool rle_encode_ac(int16_t cur, vector<int16_t>& out_rle, int& zero_count, bool& EOB, bool& ZRL, vector<int16_t>& tempZRL);

// типо продолжение разностного кодирования DC, а также мы кодируем AC
void preparing_for_coding_dc_and_ac(vector<int16_t>& data) {
	vector<int16_t> output;
	output.reserve(data.size());

	dc_difference(data);

	//вся запись будет не побитовой
	size_t size = data.size();

	for (size_t i = 0; i < size; i += 64)// блок
	{
		// запись DC
		vector<int16_t> temp;
		// {DC coeffs}
		if (data[i] == 0)
		{
			output.push_back(0);// запись КАТЕГОРИИ = 0 без записи кода
		}
		else
		{
			if (data[i] > 0)
			{//запись категорий DC и сам DC ввиде бинарного кода в вектор, но запись не на самом деле не бинарная.
				temp = intToBinaryVector(data[i], 1);
			}
			else
			{//если число отрицательно, то мы инвертируем его биты
				temp = intToBinaryVector(data[i], 0);
			}
			output.push_back(static_cast<int16_t>(temp.size()));// запись КАТЕГОРИИ
			copy(temp.begin(), temp.end(), back_inserter(output));// запись КОДА
		}




		// {AC coeffs}
		int zero_count = 0;
		bool EOB = false;
		bool ZRL = false;
		vector<int16_t> tempZRL;
		for (size_t j = 1; j < 64; j++)// оставшиеся 63 AC коэффициенты блока
		{// подсчёт нулей и его запись, запись категорий AC с самим коэффициентом и так заного пока не 0,0 (EOB).
			if (rle_encode_ac(data[j + i], output, zero_count, EOB, ZRL, tempZRL)) continue;
			// запись НУЛЕЙ AC происхлдит в функции rle_encode_ac

			// запись AC
			if (data[j + i] >= 0)
			{//запись категорий AC и сам AC ввиде бинарного кода в вектор, но запись на самом деле не бинарная.
				temp = intToBinaryVector(data[j + i], 1);
			}
			else
			{//если число отрицательно, то мы инвертируем его биты
				temp = intToBinaryVector(data[j + i], 0);
			}

			output.push_back(static_cast<int16_t>(temp.size()));// запись КАТЕГОРИИ
			copy(temp.begin(), temp.end(), back_inserter(output));// запись КОДА
		}

		// когда до конца блока все нули
		if (EOB)
		{
			output.push_back(0);
			output.push_back(0);
		}
	}

	data = output;
}

// 10. RLE кодирование AC коэффициентов
bool rle_encode_ac(int16_t cur, vector<int16_t>& out_rle, int& zero_count, bool& EOB, bool& ZRL, vector<int16_t>& tempZRL) {
	if (cur == 0)
	{// попался 0
		zero_count++;
		EOB = true;
		if (zero_count == 15)
		{//ZRL or EOB. Если ZRL то мы не записываем код, т.к. там 0
			tempZRL.push_back(15);// запись НУЛЕЙ
			tempZRL.push_back(0);// запись КАТЕГОРИИ
			zero_count = 0;
			ZRL = true;
		}
		return true;
	}
	else
	{
		// tempZRL
		if (ZRL)
		{// если был ZRL, то он не пустой
			// запись ZRL
			copy(tempZRL.begin(), tempZRL.end(), back_inserter(out_rle));

			tempZRL.clear();
			ZRL = false;
		}

		out_rle.push_back(zero_count);// запись НУЛЕЙ AC
		zero_count = 0;
		EOB = false;
		return false;
	}
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
/*0 / 1  2*/ "00",
/*0 / 2  2*/ "01",
/*0 / 3  3*/ "100",
/*0 / 4  4*/ "1011",
/*0 / 5  5*/ "11010",
/*0 / 6  7*/ "1111000",
/*0 / 7  8*/ "11111000",
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
/*3 / 1  5*/ "11011",
/*3 / 2  8*/ "11111000",
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

string HA_encode(const vector<int16_t>& data, const string_view(&DC_differences)[12], const string_view(&AC)[16][11]) {
	string encoded;
	size_t size = data.size();
	for (size_t i = 0; i < size; i++)
	{
		// DC
		encoded += DC_differences[data[i]];// код КАТЕГОРИИ
		int k_size = data[i];// длина битовой строки

		for (int k = 0; k < k_size; k++)
		{// записть кода
			i++;
			encoded += to_string(data[i]);
		}

		// AC
		int count = 1;// мы уже обработали DC, поэтому 1.
		while (count < 64)// блок до 64 (последний индекс = 63)
		{
			i++;

			if (data[i] == 0 && data[i + 1] == 0)
			{// EOB
				encoded += AC[0][0];
				i++;
				break;
			}

			if (data[i] == 15 && data[i + 1] == 0)
			{// ZRL
				encoded += AC[15][0];
				count += 15;
				i++;
				continue;
			}

			count += 1 + data[i];// добавили число + кол нулей в счётчик блока
			encoded += AC[data[i]][data[i + 1]];// запись кода таблицы "кол-во нулей/категория"

			i++;
			int k_size = data[i];
			for (int k = 0; k < k_size; k++)
			{// записть кода 0/1
				i++;
				encoded += to_string(data[i]);
			}
			int up = 0;

		}
	}

	return encoded;
}

// Упаковка битовой строки в байты
vector<uint8_t> pack_bits_to_bytes(const string& bit_str) {
	vector<uint8_t> output;
	size_t len = bit_str.length();

	for (size_t i = 0; i < len; i += 8) {
		string byte_str = bit_str.substr(i, 8);

		if (byte_str.length() < 8) {
			uint8_t zero_bits = 8 - byte_str.length();
			byte_str.append(zero_bits, '0');// Дополняем нулями
		}

		bitset<8> bits(byte_str);
		int8_t num = static_cast<uint8_t>(bits.to_ulong());
		output.push_back(num);
	}

	return output;
}

bool writeBMP(const string& filename, const inputArray& r, const inputArray& g, const inputArray& b, int width, int height) {
	ofstream file(filename, ios::binary);
	if (!file) {
		cerr << "Не удалось открыть файл для записи: " << filename << '\n';
		return false;
	}

	// Размер файла (54 байта заголовок + 3 * width * height)
	const int fileSize = 54 + 3 * width * height;

	// Заголовок BMP (14 байт)
	const uint8_t bmpHeader[14] = {
		'B', 'M',                                   // Сигнатура
		static_cast<uint8_t>(fileSize),              // Размер файла (младший байт)
		static_cast<uint8_t>(fileSize >> 8),         // ...
		static_cast<uint8_t>(fileSize >> 16),        // ...
		static_cast<uint8_t>(fileSize >> 24),        // Старший байт
		0, 0, 0, 0,                                 // Зарезервировано
		54, 0, 0, 0                                 // Смещение до данных пикселей (54 байта)
	};

	// Заголовок DIB (40 байт)
	const uint8_t dibHeader[40] = {
		40, 0, 0, 0,                                // Размер DIB-заголовка
		static_cast<uint8_t>(width),                 // Ширина (младший байт)
		static_cast<uint8_t>(width >> 8),           // ...
		static_cast<uint8_t>(width >> 16),          // ...
		static_cast<uint8_t>(width >> 24),          // Старший байт
		static_cast<uint8_t>(height),                // Высота
		static_cast<uint8_t>(height >> 8),          // ...
		static_cast<uint8_t>(height >> 16),         // ...
		static_cast<uint8_t>(height >> 24),         // ...
		1, 0,                                       // Количество плоскостей (1)
		24, 0,                                      // Бит на пиксель (24 = RGB)
		0, 0, 0, 0,                                 // Сжатие (нет)
		0, 0, 0, 0,                                 // Размер изображения (можно 0)
		0, 0, 0, 0,                                 // Горизонтальное разрешение
		0, 0, 0, 0,                                 // Вертикальное разрешение
		0, 0, 0, 0,                                 // Палитра (не используется)
		0, 0, 0, 0                                  // Важные цвета (все)
	};

	// Записываем заголовки
	file.write(reinterpret_cast<const char*>(bmpHeader), 14);
	file.write(reinterpret_cast<const char*>(dibHeader), 40);

	// Выравнивание строк (BMP требует, чтобы каждая строка была кратна 4 байтам)
	const int padding = (4 - (width * 3) % 4) % 4;
	const uint8_t padBytes[3] = { 0, 0, 0 };

	// Записываем пиксели (снизу вверх, BGR-порядок)
	for (int y = height - 1; y >= 0; --y) {
		for (int x = 0; x < width; ++x) {
			// Ограничиваем значения 0-255 и конвертируем в uint8_t
			uint8_t blue = static_cast<int16_t>(max(0.0, min(255.0, b[y][x])));
			uint8_t green = static_cast<int16_t>(max(0.0, min(255.0, g[y][x])));
			uint8_t red = static_cast<int16_t>(max(0.0, min(255.0, r[y][x])));

			// Пиксель в формате BGR (не RGB!)
			file.put(blue);
			file.put(green);
			file.put(red);
		}
		// Записываем выравнивание, если нужно
		if (padding > 0) {
			file.write(reinterpret_cast<const char*>(padBytes), padding);
		}
	}

	file.close();
	return true;
}

inputArray downsample(int height, int width, const inputArray& c) {
	//булево значение, определяет нечётное ли число или нет
	bool odd_h = height % 2 != 0;//0 - нет, 1 - да, нечётное
	bool odd_w = width % 2 != 0;

	inputArray downsampled(divUp(height, 2), vector<int16_t>(divUp(width, 2), 0));

	for (size_t y = 0, h = 0; h < height - odd_h; y++, h += 2)
	{
		for (size_t x = 0, w = 0; w < width - odd_h; x++, w += 2)
		{// среднее арифметическое
			int arithmetic_mean = (c[h][w] + c[h][w + 1]
				+ c[h + 1][w] + c[h + 1][w + 1]) / 4;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_w)
	{// правый край
		// индексы правых краёв
		int w = width - 1;// const
		int x = width / 2;// width нечётное
		for (size_t y = 0, h = 0; h < height - odd_h; y++, h += 2)
		{
			int arithmetic_mean
				= (c[h][w] + 0
					+ c[h + 1][w] + 0) / 2;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_h)
	{// нижний край
		// индексы нижних краёв
		int h = height - 1;// const
		int y = height / 2;// height нечётное
		for (size_t x = 0, w = 0; w < width - odd_w; x++, w += 2)
		{
			int arithmetic_mean
				= (c[h][w] + c[h][w + 1]
					+ 0 + 0) / 2;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_h + odd_w == 2)
	{// уголок
		int w = width - 1;// const
		int h = height - 1;// const

		int y = height / 2;// height нечётное (разрешение/size - 1)
		int x = width / 2;// width нечётное (разрешение/size - 1)
		downsampled[y][x] = c[h][w];
	}

	return downsampled;
}

inputArray upScale(int height, int width, const inputArray& c) {
	inputArray up(height, vector<int16_t>(width));


	//булево значение, определяет нечётное ли число или нет
	bool odd_h = height % 2 != 0;//0 - нет, 1 - да, нечётное
	bool odd_w = width % 2 != 0;

	for (size_t y = 0, h = 0; y < height / 2; y++, h += 2)
	{
		for (size_t x = 0, w = 0; x < width / 2; x++, w += 2)
		{
			int Ycbcr_pixel = c[y][x];
			up[h  ][w  ] = Ycbcr_pixel; up[h  ][w+1] = Ycbcr_pixel;
			up[h+1][w  ] = Ycbcr_pixel; up[h+1][w+1] = Ycbcr_pixel;
		}
	}

	if (odd_w)
	{// правый край
		// индексы правых краёв
		int w = width - 1;// const
		int x = width / 2;// width нечётное
		for (size_t y = 0, h = 0; y < height / 2; y++, h += 2)
		{
			int Ycbcr_pixel = c[y][x];
			up[h  ][w  ] = Ycbcr_pixel;
			up[h+1][w  ] = Ycbcr_pixel;
		}
	}

	if (odd_h)
	{// нижний край
		// индексы нижних краёв
		int h = height - 1;// const
		int y = height / 2;// height нечётное
		for (size_t x = 0, w = 0; x < width / 2; x++, w += 2)
		{
			int Ycbcr_pixel = c[y][x];
			up[h  ][w  ] = Ycbcr_pixel; up[h  ][w+1] = Ycbcr_pixel;
		}
	}

	if (odd_h + odd_w == 2)
	{// уголок
		int w = width - 1;// const
		int h = height - 1;// const

		int y = height / 2;// height нечётное (разрешение/size - 1)
		int x = width / 2;// width нечётное (разрешение/size - 1)
		up[h][w] = c[y][x];
	}

	return up;
}

vector<i16_Block8x8> splitInto8x8Blocks(int height, int width, const inputArray& Y_Cb_Cr) {
	vector<i16_Block8x8> BLOCKS;// двумерный массив блоков

	// Проходим по изображению с шагом 8х8
	//i - строки, j - столбцы
	for (size_t i = 0; i < height; i += 8) {
		int Nt = 8;
		if (height - i < 8) Nt = height - i;

		for (size_t j = 0; j < width; j += 8) {
			int M = 8;
			if (width - j < 8) M = width - j;

			// Копируем данные в блок 8х8
			//bi - строки, bj - столбцы
			i16_Block8x8 block{};
			for (size_t bi = 0; bi < Nt; bi++) {
				for (size_t bj = 0; bj < M; bj++) {
					block[bi][bj] = Y_Cb_Cr[i + bi][j + bj];
				}
			}

			BLOCKS.push_back(block);
		}
	}

	return BLOCKS;
}

inputArray marge8x8Blocks(int height, int width, const vector<i16_Block8x8>& blocks/*Y/Cb/Cr*/, int lever) {
	if (lever)
	{
		height = divUp(height, 2);
		width = divUp(width, 2);
	}

	inputArray out(height, vector<int16_t>(width));
	int h_blocks = divUp(height, 8);
	int w_blocks = divUp(width, 8);
	int position = 0;

	for (size_t i = 0; i < h_blocks; i++)
	{
		for (size_t u = 0; u < w_blocks; u++)
		{
			for (size_t bi = 0; bi < 8; bi++)
			{
				if (i * 8 + bi >= height) continue;

				for (size_t bu = 0; bu < 8; bu++)
				{
					if (u * 8 + bu >= width) continue;
					
					out[i * 8 + bi][u * 8 + bu] = blocks[position][bi][bu];
				}
			}

			position++;
		}
	}

	return out;
}

void JPEG_decode_HA_RLE(vector<array<int16_t, 64>>& out, string str, int size_Bl8x8, const string_view(&DC)[12], const string_view(&AC)[16][11], int& tmp) {
	ofstream outp("console.txt");
	
	for (size_t num_block = 0; num_block < size_Bl8x8; num_block++)
	{
		// DC coeffs
		if (str.substr(tmp, 2) == "00")
		{
			tmp += 2;// сдвигаем курсор
			outp << tmp << '\n';
		}
		else
		{
			bool search = true;
			int length = 2;// длина кода

			while (search)
			{
				if (length > 11)
				{
					cout << "DC Difference length ERROR: not found. tmp = " << tmp << '\n';
				}

				string_view code(str.data() + tmp, length);

				// поиск по таблице
				for (size_t d = 1; d < 12; d++)
				{
					if (length == DC[d].length())
					{
						if (code == DC[d])
						{
							search = false;
							tmp += length;// сдвигаем курсор на кол. битов отведённых на код категории
							outp << tmp << '\n';

							// перевод числа 2->10 систему счисления
							string bits = str.substr(tmp, d);
							int minus = 1;

							if (bits[0] == '0')
							{
								for (char& c : bits)
								{// char& — ссылка, меняет исходные данные
									c ^= 1;// Инвертируем биты
								}
								minus = -1;
							}

							tmp += d;// сдвигаем курсор на кол. битов отведённых двоичное число
							outp << tmp << '\n';
							out[num_block][0] = minus * stoi(bits, nullptr, 2);// nullptr нужен просто, чтобы функция не сохраняла лишний раз pos, т.к. не нужен
							break;
						}
					}
					else if (length < DC[d].length())
					{
						break;
					}
				}
				length++;
			}
		}

		// AC coeffs
		int EOB = false;
		int count = 0;// count должен стоять на последнем записанном числе и никак иначе

		// 63 это индекс последнего коэффициента массива
		while (count != 63)// count никогда не будет больше > 63
		{
			bool search = true;
			int length = 2;// длина кода

			while (search)
			{

				string_view code(str.data() + tmp, length);

				if (length > 16)
				{
					cout << "AC length ERROR: not found. tmp = " << tmp << '\n';
				}

				if (code == AC[0][0])
				{
					EOB = true;
					tmp += AC[0][0].length();
					outp << tmp << '\n';
					break;
				}

				// поиск по таблице
				for (size_t a = 0; a < 16; a++)
				{
					for (size_t c = 0; c < 11; c++)
					{
						if (length == AC[a][c].length())
						{
							if (code == AC[a][c])
							{
								search = false;
								tmp += length;// сдвигаем курсор на кол. битов отведённых на код (кол. нулей/категория)
								outp << tmp << '\n';
								count += a;// прибавил количество нулей

								if (a == 15 && c == 0)
								{
									break;
								}

								string bits = str.substr(tmp, c);
								int minus = 1;

								if (bits[0] == '0')
								{
									for (char& c : bits)
									{// char& — ссылка, меняет исходные данные
										c ^= 1;// Инвертируем биты
									}
									minus = -1;
								}

								// перевод числа 2->10 систему счисления
								tmp += c;// сдвигаем курсор на кол. битов отведённых двоичное число
								outp << tmp << '\n';
								count++;
								out[num_block][count] = minus * stoi(bits, nullptr, 2);// nullptr нужен просто, чтобы функция не сохраняла лишний раз pos, т.к. не нужен
								break;
							}
						}
						else if (length < AC[a][c].length())
						{
							if (!(a == 0 && c == 0))
							{
								break;
							}
						}
					}

					if (!search) break;
				}

				length++;
			}

			if (EOB) break;
}}
	
	outp.close();
}

// Запись в файл
void write_quant_table(const int(&quant_table)[8][8], ofstream& out) {
	for (size_t i = 0; i < 8; i++)
	{
		for (size_t u = 0; u < 8; u++)
		{
			char sym = quant_table[i][u];// Байт в ASCII или просто обычный символ char
			out.write(&sym, 1);// Записываем 1 байт (sizeof(char) = 1 byte)
		}
	}
}

void write_DC_coeff(const string_view(&DC)[12], string& text) {
	for (size_t i = 0; i < 12; i++)
	{
		int8_t length = DC[i].length() - 1;// записываем длину - 1
		bitset<4> bits(length);
		text += bits.to_string();
		text += DC[i];
	}
}

void write_AC_coeff(const string_view(&AC)[16][11], string& text) {
	// EOB
	int8_t length = AC[0][0].length() - 1;// записываем длину - 1
	bitset<4> bits(length);
	text += bits.to_string();
	text += AC[0][0];

	// ZRL
	length = AC[15][0].length() - 1;// записываем длину - 1
	bits = bitset<4>(length);
	text += bits.to_string();
	text += AC[15][0];

	// остальные коды
	for (size_t i = 0; i < 16; i++)
	{
		for (size_t u = 1; u < 11; u++)// пропускаем первый элемент, т.к. он пустой (u = 1)
		{
			int8_t length = AC[i][u].length() - 1;// записываем длину - 1
			bits = bitset<4>(length);
			text += bits.to_string();
			text += AC[i][u];
		}
	}
}

void writing_the_compressed_file(string link, vector<uint8_t> out_bytes, int width, int height, int quality) {
	ofstream out(link, ios::binary);

	// 1. Запись таблиц квантования
	write_quant_table(Luminance_quantization_table, out);
	write_quant_table(Chrominance_quantization_table, out);

	// 2. Разрешение
	out.write(reinterpret_cast<char*>(&width), sizeof(int));
	out.write(reinterpret_cast<char*>(&height), sizeof(int));

	// 3. Качество
	out.write(reinterpret_cast<char*>(&quality), sizeof(int));

	// 4. Запись таблиц Хаффмана
	string text;

	write_DC_coeff(Luminance_DC_differences, text);
	write_DC_coeff(Chrominance_DC_differences, text);

	write_AC_coeff(Luminance_AC, text);
	write_AC_coeff(Chrominance_AC, text);

	vector<uint8_t> coeff = pack_bits_to_bytes(text);
	text = "";

	int size_coeff = coeff.size();
	for (size_t i = 0; i < size_coeff; i++)
	{
		char sym = coeff[i];// Байт в ASCII или просто обычный символ char
		out.write(&sym, 1);// Записываем 1 байт и sizeof = 8 (const)
	}

	// 5. Запись формулы для RGB to YCbCr
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t u = 0; u < 3; u++)
		{
			out.write(reinterpret_cast<const char*>(&PR[i][u]), sizeof(double));
		}
	}

	// 6. Запись сжатых данных
	size_t size_output = out_bytes.size();
	for (size_t i = 0; i < size_output; i++)
	{
		char sym = out_bytes[i];
		out.write(&sym, 1);// Записываем 1 байт (sizeof(char) = 1 byte)
	}

	out.close();
}

// Бинарное чтение метаданных и сжатых данных
void read_quant_table(ifstream& in, int quant_table[8][8]) {
	for (size_t i = 0; i < 8; i++)
	{
		for (size_t u = 0; u < 8; u++)
		{
			char sym;
			in.read(&sym, 1);
			quant_table[i][u] = sym;
}}}

void check(int& cursor, ifstream& in, bitset<8>& bits) {
	if (cursor < 0)
	{
		cursor = 7;
		char sym;
		in.read(&sym, 1);
		bits = bitset<8>(sym);
}}

string code(int& cursor, ifstream& in, bitset<8>& bits) {
	int num = 0;
	bits.to_string();

	for (int i = 3; i >= 0; i--)
	{
		cursor--;
		check(cursor, in, bits);
		num += bits[cursor] << i;
	}

	int length = num + 1;// длина кода
	string str;
	for (size_t i = 0; i < length; i++)
	{
		cursor--;
		check(cursor, in, bits);
		str += bits[cursor] == 1 ? '1' : '0';
	}

	return str;
}

void read_DC_coeff(ifstream& in, int& cursor, bitset<8>& bits, string DC[12]) {
	for (int count = 0; count < 12; count++)
	{
		DC[count] = code(cursor, in, bits);
	}
}

void read_AC_coeff(ifstream& in, int& cursor, bitset<8>& bits, string AC[16][11]) {
	int f = 0;

	for (size_t t = 0; t < 2; t++)
	{//EOB ZRL
		AC[f][0] = code(cursor, in, bits);
		f = 15;
	}
	for (int count = 1; count <= 14; count++)
	{
		AC[count][0] = "";
	}

	for (int count = 0; count < 16; count++)
	{
		for (size_t cnt = 1; cnt < 11; cnt++)
		{
			AC[count][cnt] = code(cursor, in, bits);
		}
	}
}

void read_coeff_ycbcr_to_rgb(ifstream& in, double RG[3][3]) {
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t u = 0; u < 3; u++)
		{
			in.read(reinterpret_cast<char*>(&RG[i][u]), 8);
		}
	}
}

string read_compressed_data(ifstream& in) {
	string str = "";

	char byte;
	while (in.read(&byte, 1))
	{
		bitset<8> bits(byte);
		for (int8_t i = 7; i > -1; i--)
		{
			str += to_string(bits[i]);
		}
	}

	return str;
}

string reading_the_compressed_file(string link
	, int& width, int& height, int& quality
	, int Lumin_QT[8][8], int Chrom_QT[8][8]
	, string_view L_DC[12], string_view C_DC[12]
	, string_view L_AC[16][11], string_view C_AC[16][11],
	double RG[3][3]) {
	ifstream in(link, ios::binary);

	read_quant_table(in, Lumin_QT);
	read_quant_table(in, Chrom_QT);

	in.read(reinterpret_cast<char*>(&width), sizeof(int));
	in.read(reinterpret_cast<char*>(&height), sizeof(int));
	in.read(reinterpret_cast<char*>(&quality), sizeof(int));

	int cursor = 0;
	bitset<8> bits;

	static string Lumin_DC[12];
	static string Chrom_DC[12];
	read_DC_coeff(in, cursor, bits, Lumin_DC);
	read_DC_coeff(in, cursor, bits, Chrom_DC);
	for (size_t i = 0; i < 12; i++){
		L_DC[i] = Lumin_DC[i];
		C_DC[i] = Chrom_DC[i];
	}

	static string Lumin_AC[16][11];
	static string Chrom_AC[16][11];
	read_AC_coeff(in, cursor, bits, Lumin_AC);
	read_AC_coeff(in, cursor, bits, Chrom_AC);
	for (size_t i = 0; i < 16; i++){
		for (size_t u = 0; u < 11; u++){
			L_AC[i][u] = Lumin_AC[i][u];
			C_AC[i][u] = Chrom_AC[i][u];
	}}

	read_coeff_ycbcr_to_rgb(in, RG);

	string str = read_compressed_data(in);

	in.close();
	return str;
}

//
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
// 
//

int main() {
	ifstream ifT;

	setlocale(LC_ALL, "ru");
	SetConsoleOutputCP(CP_UTF8);

	// images
	// grey 800 x 600 (width x height)
	// test8 8 x 8
	// test16 16 x 16
	// test32 32 x 31
	// Lenna - 512 x 512
	// forza - 2048 x 2048
	//

	string file = "forza.raw";
	bool original = 0;//1 - записываем оригинал фото в .bmp на диск, 0 - не записываем

	int width = 2048;//800 512 2048
	int height = 2048;//600 512 2048

	int min_quality = 0;
	int max_quality = 100;
	int step = 20;

	string link;
	size_t dot_pos = file.rfind('.');
	string name = file.substr(0, dot_pos);

	ifT.open(file, ios::binary);
	if (!ifT.is_open()) {
		cerr << "Error opening file: " << file << endl;
		return 0;
	}

	ifT.seekg(0, ios::end);  // Перемещаем указатель в конец файла
	auto size = ifT.tellg();       // Получаем позицию (размер файла)
	ifT.seekg(0, ios::beg);   // Возвращаем указатель в начало
	cout << "check size: " << size << '\n';
	if (size != 3 * height * width)
	{
		cerr << "Error the resolution does not match the size: " << size << "\nResoult count: " << 3 * height * width << "\n";
		return 0;
	}
	cout << "Параметры изображения:\n1) Размер: " << size << " байт\n";
	cout << "2) Количество: " << size/3 << " pixels\n";
	cout << "3) Разрешение: " << width << "x" << height << " байт\n";

	inputArray r(height, vector<int16_t>(width));
	inputArray g(height, vector<int16_t>(width));
	inputArray b(height, vector<int16_t>(width));

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
	
	for (int quality = max_quality; quality >= min_quality; quality -= step)
	{
		// encode
		cout << "Quality: " << quality << '\n';

		inputArray Y(height, vector<int16_t>(width));
		inputArray cb(height, vector<int16_t>(width));
		inputArray cr(height, vector<int16_t>(width));

		rgb_to_ycbcr(height, width, r, g, b, Y, cb, cr);

		//4:2:0 - берём по 1 цвету в двух блоках по 4 элемента
		//цвет усреднённый среди 4 элементов
		// переписать даунсэплинг чтобы сохранялась размерность массивов
		cb = downsample(height, width, cb);
		cr = downsample(height, width, cr);

		// Разбиение на блоки и запись блоков в одномерном векторе
		// i8_Block8x8
		auto Y_blocks = splitInto8x8Blocks(height, width, Y);
		auto cb_blocks = splitInto8x8Blocks(divUp(height, 2), divUp(width, 2), cb);
		auto cr_blocks = splitInto8x8Blocks(divUp(height, 2), divUp(width, 2), cr);


		Y = marge8x8Blocks(height, width, Y_blocks, 0);
		cb = marge8x8Blocks(height, width, cb_blocks, 1);
		cr = marge8x8Blocks(height, width, cr_blocks, 1);


		Y_blocks = splitInto8x8Blocks(height, width, Y);
		cb_blocks = splitInto8x8Blocks(divUp(height, 2), divUp(width, 2), cb);
		cr_blocks = splitInto8x8Blocks(divUp(height, 2), divUp(width, 2), cr);

		size_t sizeY_Bl8x8 = Y_blocks.size();
		size_t sizeC_Bl8x8 = cb_blocks.size();// совпадает с cr_blocks.size()

		d_Block8x8 q0_matrix{};
		d_Block8x8 q1_matrix{};
		generate_quantization_matrix(quality, q0_matrix, Luminance_quantization_table);
		generate_quantization_matrix(quality, q1_matrix, Chrominance_quantization_table);

		// квантование
		for (size_t x = 0; x < sizeY_Bl8x8; x++)
		{// d_Block8x8
			auto Y_dct = dct_2d_8x8(Y_blocks[x]);// запись обработанного через dct блок
			quantize(Y_dct, q0_matrix, Y_blocks[x]);// запись в Y_blocks
		}
		for (size_t x = 0; x < sizeC_Bl8x8; x++)
		{// d_Block8x8
			auto cb_dct = dct_2d_8x8(cb_blocks[x]);
			auto cr_dct = dct_2d_8x8(cr_blocks[x]);
			quantize(cb_dct, q1_matrix, cb_blocks[x]);// запись в cb_done
			quantize(cr_dct, q1_matrix, cr_blocks[x]);// запись в cr_done
		}

		// Зиг-заг обход для каждого блока
		vector<int16_t> str1;
		vector<int16_t> str2;
		vector<int16_t> str3;
		str1.reserve(sizeY_Bl8x8 * 64);
		str2.reserve(sizeC_Bl8x8 * 64);
		str3.reserve(sizeC_Bl8x8 * 64);

		for (size_t x = 0; x < sizeY_Bl8x8; x++)
		{// Y
			auto str = zigzag_scan(Y_blocks[x]);
			copy(str.begin(), str.end(), back_inserter(str1));
		}
		for (size_t x = 0; x < sizeC_Bl8x8; x++)
		{// Cb
			auto str = zigzag_scan(cb_blocks[x]);
			copy(str.begin(), str.end(), back_inserter(str2));
		}
		for (size_t x = 0; x < sizeC_Bl8x8; x++)
		{// Cr
			auto str = zigzag_scan(cr_blocks[x]);
			copy(str.begin(), str.end(), back_inserter(str3));
		}

		preparing_for_coding_dc_and_ac(str1);
		preparing_for_coding_dc_and_ac(str2);
		preparing_for_coding_dc_and_ac(str3);

		string coder = "";
		coder += HA_encode(str1, Luminance_DC_differences, Luminance_AC);
		coder += HA_encode(str2, Chrominance_DC_differences, Chrominance_AC);
		coder += HA_encode(str3, Chrominance_DC_differences, Chrominance_AC);

		vector<uint8_t> output = pack_bits_to_bytes(coder);// байтовая строка сжатых данных
		coder = "";
		// end encode





		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		link = "C:/Users/lin/Desktop/4 семестр/АиСД 4сем/Лаба 2 картинки/" + name + '/' + name + "_E_" + to_string(quality) + ".bin";
		// Запись в файл
		writing_the_compressed_file(link, output, width, height, quality);
		width = height = quality = 0;
		ifstream file(link, ios::binary | ios::ate);
		streampos fileSize = file.tellg();
		cout << "\n\nРазмер сжатых данных без метаданных:\n" << fileSize << " байт\nили\n" << fileSize / 1024 << "Кб\n\n";
		file.close();

		// Считывание из файла метаданных
		int Lumin_QT[8][8];// Luminance_quantization_table
		int Chrom_QT[8][8];// Chrominance_quantization_table
		string_view Lumin_DC[12];
		string_view Chrom_DC[12];
		string_view Lumin_AC[16][11];
		string_view Chrom_AC[16][11];
		double RG[3][3];
		string str = reading_the_compressed_file(link
			, width, height, quality
			, Lumin_QT, Chrom_QT
			, Lumin_DC, Chrom_DC
			, Lumin_AC, Chrom_AC
			, RG);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





		// decode

		sizeY_Bl8x8 = divUp(width, 8) * divUp(height, 8);

		vector<i16_Block8x8> vecY_Bl8x8(sizeY_Bl8x8);

		sizeC_Bl8x8 = divUp(divUp(width, 2), 8) * divUp(divUp(height, 2), 8);

		vector<i16_Block8x8> vecCb_Bl8x8(sizeC_Bl8x8);
		vector<i16_Block8x8> vecCr_Bl8x8(sizeC_Bl8x8);

		// векторы блоков
		vector<array<int16_t, 64>> strY(sizeY_Bl8x8);
		vector<array<int16_t, 64>> strCb(sizeC_Bl8x8);
		vector<array<int16_t, 64>> strCr(sizeC_Bl8x8);

		// Из output создадим битовую строку
		size = output.size();
		str = "";
		for (uint8_t byte : output)
		{
			bitset<8> bits(byte);
			for (int8_t i = 7; i > -1; i--)
			{
				str += to_string(bits[i]);
			}
		}

		int temp = 0;
		JPEG_decode_HA_RLE(strY, str, sizeY_Bl8x8, Lumin_DC, Lumin_AC, temp);
		JPEG_decode_HA_RLE(strCb, str, sizeC_Bl8x8, Chrom_DC, Chrom_AC, temp);
		JPEG_decode_HA_RLE(strCr, str, sizeC_Bl8x8, Chrom_DC, Chrom_AC, temp);

		reverse_dc_difference(strY);
		reverse_dc_difference(strCb);
		reverse_dc_difference(strCr);

		// Обратный зиг-заг обход для всех блоков
		for (size_t i = 0; i < sizeY_Bl8x8; i++)
		{// Y
			vecY_Bl8x8[i] = inverse_zigzag_scan(strY[i]);
		}
		for (size_t i = 0; i < sizeC_Bl8x8; i++)
		{// Cb
			vecCb_Bl8x8[i] = inverse_zigzag_scan(strCb[i]);
		}
		for (size_t i = 0; i < sizeC_Bl8x8; i++)
		{// Cr
			vecCr_Bl8x8[i] = inverse_zigzag_scan(strCr[i]);
		}

		vector<i16_Block8x8> aY(sizeY_Bl8x8);
		vector<i16_Block8x8> aCb(sizeC_Bl8x8);
		vector<i16_Block8x8> aCr(sizeC_Bl8x8);

		d_Block8x8 q0_2matrix{};
		d_Block8x8 q1_2matrix{};
		generate_quantization_matrix(quality, q0_2matrix, Lumin_QT);
		generate_quantization_matrix(quality, q1_2matrix, Chrom_QT);

		for (size_t x = 0; x < sizeY_Bl8x8; x++)
		{
			d_Block8x8 doub_Y = dequantize(vecY_Bl8x8[x], q0_2matrix);
			aY[x] = idct_2d_8x8(doub_Y);
		}
		for (size_t x = 0; x < sizeC_Bl8x8; x++)
		{
			d_Block8x8 doub_Cb = dequantize(vecCb_Bl8x8[x], q1_2matrix);
			d_Block8x8 doub_Cr = dequantize(vecCr_Bl8x8[x], q1_2matrix);
			aCb[x] = idct_2d_8x8(doub_Cb);
			aCr[x] = idct_2d_8x8(doub_Cr);
		}

		inputArray Y2 = marge8x8Blocks(height, width, aY, 0);
		inputArray cb2 = marge8x8Blocks(height, width, aCb, 1);
		inputArray cr2 = marge8x8Blocks(height, width, aCr, 1);

		cb2 = upScale(height, width, cb2);
		cr2 = upScale(height, width, cr2);

		inputArray r2(height, vector<int16_t>(width));
		inputArray g2(height, vector<int16_t>(width));
		inputArray b2(height, vector<int16_t>(width));

		ycbcr_to_rgb(height, width
			, r2, g2, b2
			, Y2, cb2, cr2
			, RG);

		link = "C:/Users/lin/Desktop/4 семестр/АиСД 4сем/Лаба 2 картинки/" + name + '/' + name + "_D_" + to_string(quality) + ".bmp";
		writeBMP(link, r2, g2, b2, width, height);

		if (original)
		{
			link = "C:/Users/lin/Desktop/4 семестр/АиСД 4сем/Лаба 2 картинки/" + name + '/' + name + "_Orig.bmp";
			writeBMP(link, r, g, b, width, height);
		}
	}

	cout << "Done!!!";
	ifT.close();
	return 0;
}