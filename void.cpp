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

int divUp(int x, int y)
{// èäåÿ áûëà òàêàÿ (x + y - 1) / y
	//åñëè ìû äîáàâèì äåëèòåëü ê çíàìíàòåëþ è ïîäåëèì íà äåëèòåëü, òî ïî ñðàâíåíèþ ñ ïðîøëûì ðåçóëüòàòîì òóò äîáàâèòñÿ åäèíèöà
	// òî åñòü, åñëè äîáàâèòü ÷èñëèòåëþ äåëèòåëü íà 1 ìåíüøå, òî ðåçóëüòàòó äîáàâèòñÿ ÷èñëî âñåãäà ìåíüøåå åäèíèöå, íî äîñòàòî÷íîå, ÷òîáû îêðóãëèòü ââåðõ
	return (x - 1) / y + 1;// óïðîù¸ííàÿ ôîðìà
}

// 1. Ïðåîáðàçîâàíèå RGB â YCbCr
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
	, const inputArray& Y, const inputArray& cb, const inputArray& cr) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double y_val = Y[y][x];
			double cb_val = cb[y][x] - 128;
			double cr_val = cr[y][x] - 128;

			double r = y_val + 1.402 * cr_val;
			double g = y_val - 0.344136 * cb_val - 0.714136 * cr_val;
			double b = y_val + 1.772 * cb_val;

			R[y][x] = static_cast<int16_t>(max(0.0, min(255.0, r)));
			G[y][x] = static_cast<int16_t>(max(0.0, min(255.0, g)));
			B[y][x] = static_cast<int16_t>(max(0.0, min(255.0, b)));
		}
	}
}

// 2. Äàóíñýìïëèíã 4:2:0 (ïðèìåíÿåòñÿ ê Cb è Cr)
inputArray downsample(int height, int width, const inputArray& c);

// 3. Ðàçáèåíèå èçîáðàæåíèÿ íà áëîêè NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
#include <array>

constexpr int N = 8;
using i16_Block8x8 = array<array<int16_t, N>, N>;

vector<i16_Block8x8> splitInto8x8Blocks(int height, int width, const inputArray& Y_Cb_Cr);

// 4.1 Ïðÿìîå DCT-II ïðåîáðàçîâàíèå äëÿ áëîêà NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
// Ïðåäâàðèòåëüíî âû÷èñëåííûå êîíñòàíòû äëÿ DCT-II 8x8
constexpr double PI = 3.14159265358979323846;
constexpr double SQRT2 = 1.41421356237309504880; // sqrt(2)

// Êîýôôèöèåíòû ìàñøòàáèðîâàíèÿ äëÿ DCT-II
constexpr double C0 = 1.0 / SQRT2;
constexpr double C1 = 1.0;

using d_Block8x8 = array<array<double, N>, N>;

// Òàáëèöà êîñèíóñîâ äëÿ óñêîðåíèÿ âû÷èñëåíèé
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

// 1D DCT-II äëÿ ñòðîêè/ñòîëáöà
void dct_1d(const array<double, N>& input, array<double, N>& output) {
	for (int u = 0; u < N; u++) {
		double sum = 0.0;
		double cu = (u == 0) ? C0 : C1;
		for (int x = 0; x < N; x++) {
			sum += input[x] * COS_TABLE[u][x];
		}
		output[u] = sum * cu * 0.5; // Íîðìàëèçàöèÿ
	}
}

// 2D DCT-II äëÿ áëîêà 8x8
d_Block8x8 dct_2d_8x8(const i16_Block8x8& block) {
	d_Block8x8 temp;
	d_Block8x8 coeffs;

	// Ïðèìåíÿåì 1D DCT ê êàæäîé ñòðîêå (ãîðèçîíòàëüíîå ïðåîáðàçîâàíèå)
	for (int y = 0; y < N; y++) {
		array<double, N> row;
		for (int x = 0; x < N; x++) {
			row[x] = block[y][x];
		}
		array<double, N> dct_row{};
		dct_1d(row, dct_row);
		temp[y] = dct_row;
	}

	// Ïðèìåíÿåì 1D DCT ê êàæäîìó ñòîëáöó (âåðòèêàëüíîå ïðåîáðàçîâàíèå)
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

// 4.2 Îáðàòíîå DCT-II ïðåîáðàçîâàíèå äëÿ áëîêà NxN
// VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
// 1D DCT-III îáðàòíîå ïðåîáðàçîâàíèå äëÿ ñòðîêè/ñòîëáöà
void idct_1d(const array<double, N>& input, array<double, N>& output) {
	for (int x = 0; x < N; x++) {
		double sum = 0.0;
		for (int u = 0; u < N; ++u) {
			double cu = (u == 0) ? C0 : C1;
			sum += cu * input[u] * COS_TABLE[u][x];
		}
		output[x] = sum * 0.5; // Íîðìàëèçàöèÿ
	}
}

// 2D IDCT äëÿ áëîêà 8x8
i16_Block8x8 idct_2d_8x8(const d_Block8x8& coeffs) {
	d_Block8x8 temp;
	i16_Block8x8 block;

	// Ïðèìåíÿåì 1D IDCT ê êàæäîìó ñòîëáöó (âåðòèêàëüíîå ïðåîáðàçîâàíèå)
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

	// Ïðèìåíÿåì 1D IDCT ê êàæäîé ñòðîêå (ãîðèçîíòàëüíîå ïðåîáðàçîâàíèå)
	for (int y = 0; y < N; y++) {
		array<double, N> row{};
		for (int u = 0; u < N; u++) {
			row[u] = temp[y][u];
		}
		array<double, N> idct_row{};
		idct_1d(row, idct_row);
		for (int x = 0; x < N; x++) {
			block[y][x] = static_cast<int16_t>(round(idct_row[x]));
		}
	}

	return block;
}

// 5. Ãåíåðàöèÿ ìàòðèöû êâàíòîâàíèÿ äëÿ çàäàííîãî óðîâíÿ êà÷åñòâà
// Annex K ñòàíäàðò JPEG (ISO/IEC 10918-1) : 1993(E)
// Ñòàíäàðòíàÿ ìàòðèöà êâàíòîâàíèÿ Y äëÿ êà÷åñòâà 50
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
// Ñòàíäàðòíàÿ ìàòðèöà êâàíòîâàíèÿ Cb è Cr äëÿ êà÷åñòâà 50
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
	// Êîððåêòèðóåì êà÷åñòâî (1-100)
	quality = max(1, min(100, quality));

	// Âû÷èñëÿåì scale_factor
	double scaleFactor;
	if (quality < 50) {
		scaleFactor = 200.0 / quality;  // Äëÿ Q < 50
	}
	else {
		scaleFactor = 8 * (1.0 - 0.01 * quality);  // Äëÿ Q >= 50
	}

	// Ìàñøòàáèðóåì ñòàíäàðòíóþ ìàòðèöó Cb/Cr
	for (int y = 0; y < 8; y++) {
		for (int x = 0; x < 8; x++) {
			double q = Quantization_table[y][x] * scaleFactor;
			q_matrix[y][x] = max(1.0, min(255.0, q));
		}
	}
}

// 6.1 Êâàíòîâàíèå DCT êîýôôèöèåíòîâ
void quantize(const d_Block8x8& dct_coeffs, const d_Block8x8& q_matrix, i16_Block8x8& quantized) {
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			quantized[y][x] = static_cast<int16_t>(round(dct_coeffs[y][x] / q_matrix[y][x]));
		}
	}
}

// 6.2 Îáðàòíîå êâàíòîâàíèå (âîññòàíîâëåíèå DCT-êîýôôèöèåíòîâ)
d_Block8x8 dequantize(const i16_Block8x8& quantized, const d_Block8x8& q_matrix) {
	d_Block8x8 dct_coeffs;
	for (int y = 0; y < N; y++) {
		for (int x = 0; x < N; x++) {
			dct_coeffs[y][x] = static_cast<double>(quantized[y][x]) * q_matrix[y][x];
		}
	}

	return dct_coeffs;
}

// 7.1 Çèãçàã-ñêàíèðîâàíèå áëîêà
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

// 7.2 Îáðàòíîå çèãçàã-ñêàíèðîâàíèå áëîêà
i16_Block8x8 inverse_zigzag_scan(const array<int16_t, 64>& str) {
	i16_Block8x8 block{};

	for (int i = 0; i < 64; i++) {
		int idx = zigzag_sequence[i];
		block[idx / 8][idx % 8] = str[i];
	}

	return block;
}

// ïîñëå çèã-çàã îáõîäà îáðàáàòûâàåì ïîëó÷åííûé ìàññèâ ñ DC è AC êîýôôèöèåíòàìè.
// 8.1 Ðàçíîñòíîå êîäèðîâàíèå DC êîýôôèöèåíòîâ
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

// 8.2 Îáðàòíîå ðàçíîñòíîå êîäèðîâàíèå DC êîýôôèöèåíòîâ
// ÒÐÅÁÓÅÒ ÏÅÐÅÄÅËÊÈ
void reverse_dc_difference(vector<array<int16_t, 64>>& data) {
	size_t size = data.size();

	for (size_t i = 1; i < size; i++) {
		data[i][0] += data[i - 1][0];
	}
}

// 9. Ïåðåìåííîãî êîäèðîâàíèÿ ðàçíîñòåé DC è AC êîýôôèöèåíòîâ.
vector<int16_t> intToBinaryVector(int16_t num, int positive1_or_negative0 = 1/*âëèÿåò íà áèòû, áóäóò ëè áèòû èíâåðòèâíûìè*/) {
	vector<int16_t> bits;

	if (num == 0) {// ýòî ìîæíî óäàëèòü ïî èäåè
		bits.push_back(0);
		return bits;
	}

	if (positive1_or_negative0 == 0) num *= -1;

	// Ðàçëîæåíèå ÷èñëà íà áèòû
	while (num > 0) {
		bits.push_back(num % 2 == positive1_or_negative0); // Ìëàäøèé áèò
		num /= 2;
	}

	// ×òîáû áèòû øëè â ïðèâû÷íîì ïîðÿäêå (ñòàðøèé áèò ïåðâûì), ðàçâåðí¸ì âåêòîð
	reverse(bits.begin(), bits.end());

	return bits;
}

bool rle_encode_ac(int16_t cur, vector<int16_t>& out_rle, int& zero_count, bool& EOB, bool& ZRL, vector<int16_t>& tempZRL);

// òèïî ïðîäîëæåíèå ðàçíîñòíîãî êîäèðîâàíèÿ DC, à òàêæå ìû êîäèðóåì AC
void preparing_for_coding_dc_and_ac(vector<int16_t>& data) {
	vector<int16_t> output;
	output.reserve(data.size());

	dc_difference(data);

	//âñÿ çàïèñü áóäåò íå ïîáèòîâîé
	size_t size = data.size();

	for (size_t i = 0; i < size; i += 64)// áëîê
	{
		// çàïèñü DC
		vector<int16_t> temp;
		// {DC coeffs}
		if (data[i] == 0)
		{
			output.push_back(0);// çàïèñü ÊÀÒÅÃÎÐÈÈ = 0 áåç çàïèñè êîäà
		}
		else
		{
			if (data[i] > 0)
			{//çàïèñü êàòåãîðèé DC è ñàì DC ââèäå áèíàðíîãî êîäà â âåêòîð, íî çàïèñü íå íà ñàìîì äåëå íå áèíàðíàÿ.
				temp = intToBinaryVector(data[i], 1);
			}
			else
			{//åñëè ÷èñëî îòðèöàòåëüíî, òî ìû èíâåðòèðóåì åãî áèòû
				temp = intToBinaryVector(data[i], 0);
			}
			output.push_back(static_cast<int16_t>(temp.size()));// çàïèñü ÊÀÒÅÃÎÐÈÈ
			copy(temp.begin(), temp.end(), back_inserter(output));// çàïèñü ÊÎÄÀ
		}




		// {AC coeffs}
		int zero_count = 0;
		bool EOB = false;
		bool ZRL = false;
		vector<int16_t> tempZRL;
		for (size_t j = 1; j < 64; j++)// îñòàâøèåñÿ 63 AC êîýôôèöèåíòû áëîêà
		{// ïîäñ÷¸ò íóëåé è åãî çàïèñü, çàïèñü êàòåãîðèé AC ñ ñàìèì êîýôôèöèåíòîì è òàê çàíîãî ïîêà íå 0,0 (EOB).
			if (rle_encode_ac(data[j + i], output, zero_count, EOB, ZRL, tempZRL)) continue;
			// çàïèñü ÍÓËÅÉ AC ïðîèñõëäèò â ôóíêöèè rle_encode_ac

			// çàïèñü AC
			if (data[j + i] >= 0)
			{//çàïèñü êàòåãîðèé AC è ñàì AC ââèäå áèíàðíîãî êîäà â âåêòîð, íî çàïèñü íà ñàìîì äåëå íå áèíàðíàÿ.
				temp = intToBinaryVector(data[j + i], 1);
			}
			else
			{//åñëè ÷èñëî îòðèöàòåëüíî, òî ìû èíâåðòèðóåì åãî áèòû
				temp = intToBinaryVector(data[j + i], 0);
			}

			output.push_back(static_cast<int16_t>(temp.size()));// çàïèñü ÊÀÒÅÃÎÐÈÈ
			copy(temp.begin(), temp.end(), back_inserter(output));// çàïèñü ÊÎÄÀ
		}

		// êîãäà äî êîíöà áëîêà âñå íóëè
		if (EOB)
		{
			output.push_back(0);
			output.push_back(0);
		}
	}

	data = output;
}

// 10. RLE êîäèðîâàíèå AC êîýôôèöèåíòîâ
bool rle_encode_ac(int16_t cur, vector<int16_t>& out_rle, int& zero_count, bool& EOB, bool& ZRL, vector<int16_t>& tempZRL) {
	if (cur == 0)
	{// ïîïàëñÿ 0
		zero_count++;
		EOB = true;
		if (zero_count == 15)
		{//ZRL or EOB. Åñëè ZRL òî ìû íå çàïèñûâàåì êîä, ò.ê. òàì 0
			tempZRL.push_back(15);// çàïèñü ÍÓËÅÉ
			tempZRL.push_back(0);// çàïèñü ÊÀÒÅÃÎÐÈÈ
			zero_count = 0;
			ZRL = true;
		}
		return true;
	}
	else
	{
		// tempZRL
		if (ZRL)
		{// åñëè áûë ZRL, òî îí íå ïóñòîé
			// çàïèñü ZRL
			copy(tempZRL.begin(), tempZRL.end(), back_inserter(out_rle));

			tempZRL.clear();
			ZRL = false;
		}

		out_rle.push_back(zero_count);// çàïèñü ÍÓËÅÉ AC
		zero_count = 0;
		EOB = false;
		return false;
	}
}

// 11. Êîäèðîâàíèÿ ðàçíîñòåé  DC êîýôôèöèåíòîâ è ïîñëåäîâàòåëüíîñòåé  Run/Size  ïî òàáëèöå êîäîâ Õàôôìàíà è óïàêîâêè ðåçóëüòàòà â áàéòîâóþ ñòðîêó.
#include <string_view>//C++17 è íîâåå
// Annex K ñòàíäàðò JPEG (ISO/IEC 10918-1) : 1993(E)
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
	int bug = false;

	string encoded;
	size_t size = data.size();
	for (size_t i = 0; i < size; i++)
	{
		// DC
		encoded += DC_differences[data[i]];// êîä ÊÀÒÅÃÎÐÈÈ
		int k_size = data[i];// äëèíà áèòîâîé ñòðîêè

		for (int k = 0; k < k_size; k++)
		{// çàïèñòü êîäà
			i++;
			encoded += to_string(data[i]);
		}

		// AC
		int count = 1;// ìû óæå îáðàáîòàëè DC, ïîýòîìó 1.
		while (count < 64)// áëîê äî 64 (ïîñëåäíèé èíäåêñ = 63)
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

			count += 1 + data[i];// äîáàâèëè ÷èñëî + êîë íóëåé â ñ÷¸ò÷èê áëîêà
			if (count > 64)
			{
				bug = true;
			}
			encoded += AC[data[i]][data[i + 1]];// çàïèñü êîäà òàáëèöû "êîë-âî íóëåé/êàòåãîðèÿ"

			i++;
			int k_size = data[i];
			for (int k = 0; k < k_size; k++)
			{// çàïèñòü êîäà 0/1
				i++;
				encoded += to_string(data[i]);
			}
			int up = 0;

		}
	}

	if (bug)
	{
		cout << "bug (count): have\n";
	}
	else
	{
		cout << "bug (count): none\n";
	}

	return encoded;
}

// Óïàêîâêà áèòîâîé ñòðîêè â áàéòû
vector<uint8_t> pack_bits_to_bytes(const string& bit_str) {
	vector<uint8_t> output;
	size_t len = bit_str.length();

	for (size_t i = 0; i < len; i += 8) {
		string byte_str = bit_str.substr(i, 8);

		if (byte_str.length() < 8) {
			uint8_t zero_bits = 8 - byte_str.length();
			byte_str.append(zero_bits, '0');// Äîïîëíÿåì íóëÿìè
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
		cerr << "Íå óäàëîñü îòêðûòü ôàéë äëÿ çàïèñè: " << filename << '\n';
		return false;
	}

	// Ðàçìåð ôàéëà (54 áàéòà çàãîëîâîê + 3 * width * height)
	const int fileSize = 54 + 3 * width * height;

	// Çàãîëîâîê BMP (14 áàéò)
	const uint8_t bmpHeader[14] = {
		'B', 'M',                                   // Ñèãíàòóðà
		static_cast<uint8_t>(fileSize),              // Ðàçìåð ôàéëà (ìëàäøèé áàéò)
		static_cast<uint8_t>(fileSize >> 8),         // ...
		static_cast<uint8_t>(fileSize >> 16),        // ...
		static_cast<uint8_t>(fileSize >> 24),        // Ñòàðøèé áàéò
		0, 0, 0, 0,                                 // Çàðåçåðâèðîâàíî
		54, 0, 0, 0                                 // Ñìåùåíèå äî äàííûõ ïèêñåëåé (54 áàéòà)
	};

	// Çàãîëîâîê DIB (40 áàéò)
	const uint8_t dibHeader[40] = {
		40, 0, 0, 0,                                // Ðàçìåð DIB-çàãîëîâêà
		static_cast<uint8_t>(width),                 // Øèðèíà (ìëàäøèé áàéò)
		static_cast<uint8_t>(width >> 8),           // ...
		static_cast<uint8_t>(width >> 16),          // ...
		static_cast<uint8_t>(width >> 24),          // Ñòàðøèé áàéò
		static_cast<uint8_t>(height),                // Âûñîòà
		static_cast<uint8_t>(height >> 8),          // ...
		static_cast<uint8_t>(height >> 16),         // ...
		static_cast<uint8_t>(height >> 24),         // ...
		1, 0,                                       // Êîëè÷åñòâî ïëîñêîñòåé (1)
		24, 0,                                      // Áèò íà ïèêñåëü (24 = RGB)
		0, 0, 0, 0,                                 // Ñæàòèå (íåò)
		0, 0, 0, 0,                                 // Ðàçìåð èçîáðàæåíèÿ (ìîæíî 0)
		0, 0, 0, 0,                                 // Ãîðèçîíòàëüíîå ðàçðåøåíèå
		0, 0, 0, 0,                                 // Âåðòèêàëüíîå ðàçðåøåíèå
		0, 0, 0, 0,                                 // Ïàëèòðà (íå èñïîëüçóåòñÿ)
		0, 0, 0, 0                                  // Âàæíûå öâåòà (âñå)
	};

	// Çàïèñûâàåì çàãîëîâêè
	file.write(reinterpret_cast<const char*>(bmpHeader), 14);
	file.write(reinterpret_cast<const char*>(dibHeader), 40);

	// Âûðàâíèâàíèå ñòðîê (BMP òðåáóåò, ÷òîáû êàæäàÿ ñòðîêà áûëà êðàòíà 4 áàéòàì)
	const int padding = (4 - (width * 3) % 4) % 4;
	const uint8_t padBytes[3] = { 0, 0, 0 };

	// Çàïèñûâàåì ïèêñåëè (ñíèçó ââåðõ, BGR-ïîðÿäîê)
	for (int y = height - 1; y >= 0; --y) {
		for (int x = 0; x < width; ++x) {
			// Îãðàíè÷èâàåì çíà÷åíèÿ 0-255 è êîíâåðòèðóåì â uint8_t
			uint8_t blue = static_cast<int16_t>(max(0.0, min(255.0, b[y][x])));
			uint8_t green = static_cast<int16_t>(max(0.0, min(255.0, g[y][x])));
			uint8_t red = static_cast<int16_t>(max(0.0, min(255.0, r[y][x])));

			// Ïèêñåëü â ôîðìàòå BGR (íå RGB!)
			file.put(blue);
			file.put(green);
			file.put(red);
		}
		// Çàïèñûâàåì âûðàâíèâàíèå, åñëè íóæíî
		if (padding > 0) {
			file.write(reinterpret_cast<const char*>(padBytes), padding);
		}
	}

	file.close();
	return true;
}

inputArray downsample(int height, int width, const inputArray& c) {
	//áóëåâî çíà÷åíèå, îïðåäåëÿåò íå÷¸òíîå ëè ÷èñëî èëè íåò
	bool odd_h = height % 2 != 0;//0 - íåò, 1 - äà, íå÷¸òíîå
	bool odd_w = width % 2 != 0;

	inputArray downsampled(divUp(height, 2), vector<int16_t>(divUp(width, 2), 0));

	for (size_t y = 0, h = 0; h < height - odd_h; y++, h += 2)
	{
		for (size_t x = 0, w = 0; w < width - odd_h; x++, w += 2)
		{// ñðåäíåå àðèôìåòè÷åñêîå
			int arithmetic_mean = (c[h][w] + c[h][w + 1]
				+ c[h + 1][w] + c[h + 1][w + 1]) / 4;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_w)
	{// ïðàâûé êðàé
		// èíäåêñû ïðàâûõ êðà¸â
		int w = width - 1;// const
		int x = width / 2;// width íå÷¸òíîå
		for (size_t y = 0, h = 0; h < height - odd_h; y++, h += 2)
		{
			int arithmetic_mean
				= (c[h][w] + 0
					+ c[h + 1][w] + 0) / 2;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_h)
	{// íèæíèé êðàé
		// èíäåêñû íèæíèõ êðà¸â
		int h = height - 1;// const
		int y = height / 2;// height íå÷¸òíîå
		for (size_t x = 0, w = 0; w < width - odd_w; x++, w += 2)
		{
			int arithmetic_mean
				= (c[h][w] + c[h][w + 1]
					+ 0 + 0) / 2;
			downsampled[y][x] = arithmetic_mean;
		}
	}

	if (odd_h + odd_w == 2)
	{// óãîëîê
		int w = width - 1;// const
		int h = height - 1;// const

		int y = height / 2;// height íå÷¸òíîå (ðàçðåøåíèå/size - 1)
		int x = width / 2;// width íå÷¸òíîå (ðàçðåøåíèå/size - 1)
		downsampled[y][x] = c[h][w];
	}

	return downsampled;
}

inputArray upScale(int height, int width, const inputArray& c) {
	inputArray up(height, vector<int16_t>(width));


	//áóëåâî çíà÷åíèå, îïðåäåëÿåò íå÷¸òíîå ëè ÷èñëî èëè íåò
	bool odd_h = height % 2 != 0;//0 - íåò, 1 - äà, íå÷¸òíîå
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
	{// ïðàâûé êðàé
		// èíäåêñû ïðàâûõ êðà¸â
		int w = width - 1;// const
		int x = width / 2;// width íå÷¸òíîå
		for (size_t y = 0, h = 0; y < height / 2; y++, h += 2)
		{
			int Ycbcr_pixel = c[y][x];
			up[h  ][w  ] = Ycbcr_pixel;
			up[h+1][w  ] = Ycbcr_pixel;
		}
	}

	if (odd_h)
	{// íèæíèé êðàé
		// èíäåêñû íèæíèõ êðà¸â
		int h = height - 1;// const
		int y = height / 2;// height íå÷¸òíîå
		for (size_t x = 0, w = 0; x < width / 2; x++, w += 2)
		{
			int Ycbcr_pixel = c[y][x];
			up[h  ][w  ] = Ycbcr_pixel; up[h  ][w+1] = Ycbcr_pixel;
		}
	}

	if (odd_h + odd_w == 2)
	{// óãîëîê
		int w = width - 1;// const
		int h = height - 1;// const

		int y = height / 2;// height íå÷¸òíîå (ðàçðåøåíèå/size - 1)
		int x = width / 2;// width íå÷¸òíîå (ðàçðåøåíèå/size - 1)
		up[h][w] = c[y][x];
	}

	return up;
}

vector<i16_Block8x8> splitInto8x8Blocks(int height, int width, const inputArray& Y_Cb_Cr) {
	vector<i16_Block8x8> BLOCKS;// äâóìåðíûé ìàññèâ áëîêîâ

	// Ïðîõîäèì ïî èçîáðàæåíèþ ñ øàãîì 8õ8
	//i - ñòðîêè, j - ñòîëáöû
	for (size_t i = 0; i < height; i += 8) {
		int Nt = 8;
		if (height - i < 8) Nt = height - i;

		for (size_t j = 0; j < width; j += 8) {
			int M = 8;
			if (width - j < 8) M = width - j;

			// Êîïèðóåì äàííûå â áëîê 8õ8
			//bi - ñòðîêè, bj - ñòîëáöû
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
			tmp += 2;// ñäâèãàåì êóðñîð
			outp << tmp << '\n';
		}
		else
		{
			bool search = true;
			int length = 2;// äëèíà êîäà

			while (search)
			{
				if (length > 11)
				{
					cout << "DC Difference length ERROR: not found. tmp = " << tmp << '\n';
				}

				string_view code(str.data() + tmp, length);

				// ïîèñê ïî òàáëèöå
				for (size_t d = 1; d < 12; d++)
				{
					if (length == DC[d].length())
					{
						if (code == DC[d])
						{
							search = false;
							tmp += length;// ñäâèãàåì êóðñîð íà êîë. áèòîâ îòâåä¸ííûõ íà êîä êàòåãîðèè
							outp << tmp << '\n';

							// ïåðåâîä ÷èñëà 2->10 ñèñòåìó ñ÷èñëåíèÿ
							string bits = str.substr(tmp, d);
							int minus = 1;

							if (bits[0] == '0')
							{
								for (char& c : bits)
								{// char&  ññûëêà, ìåíÿåò èñõîäíûå äàííûå
									c ^= 1;// Èíâåðòèðóåì áèòû
								}
								minus = -1;
							}

							tmp += d;// ñäâèãàåì êóðñîð íà êîë. áèòîâ îòâåä¸ííûõ äâîè÷íîå ÷èñëî
							outp << tmp << '\n';
							out[num_block][0] = minus * stoi(bits, nullptr, 2);// nullptr íóæåí ïðîñòî, ÷òîáû ôóíêöèÿ íå ñîõðàíÿëà ëèøíèé ðàç pos, ò.ê. íå íóæåí
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
		int count = 0;// count äîëæåí ñòîÿòü íà ïîñëåäíåì çàïèñàííîì ÷èñëå è íèêàê èíà÷å

		// 63 ýòî èíäåêñ ïîñëåäíåãî êîýôôèöèåíòà ìàññèâà
		while (count != 63)// count íèêîãäà íå áóäåò áîëüøå > 63
		{
			bool search = true;
			int length = 2;// äëèíà êîäà

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

				// ïîèñê ïî òàáëèöå
				for (size_t a = 0; a < 16; a++)
				{
					for (size_t c = 0; c < 11; c++)
					{
						if (length == AC[a][c].length())
						{
							if (code == AC[a][c])
							{
								search = false;
								tmp += length;// ñäâèãàåì êóðñîð íà êîë. áèòîâ îòâåä¸ííûõ íà êîä (êîë. íóëåé/êàòåãîðèÿ)
								outp << tmp << '\n';
								count += a;// ïðèáàâèë êîëè÷åñòâî íóëåé

								if (a == 15 && c == 0)
								{
									break;
								}

								string bits = str.substr(tmp, c);
								int minus = 1;

								if (bits[0] == '0')
								{
									for (char& c : bits)
									{// char&  ññûëêà, ìåíÿåò èñõîäíûå äàííûå
										c ^= 1;// Èíâåðòèðóåì áèòû
									}
									minus = -1;
								}

								// ïåðåâîä ÷èñëà 2->10 ñèñòåìó ñ÷èñëåíèÿ
								tmp += c;// ñäâèãàåì êóðñîð íà êîë. áèòîâ îòâåä¸ííûõ äâîè÷íîå ÷èñëî
								outp << tmp << '\n';
								count++;
								out[num_block][count] = minus * stoi(bits, nullptr, 2);// nullptr íóæåí ïðîñòî, ÷òîáû ôóíêöèÿ íå ñîõðàíÿëà ëèøíèé ðàç pos, ò.ê. íå íóæåí
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
		}

		if (num_block - size_Bl8x8 == -1) cout << "num block: " << tmp;
	}
	
	outp.close();
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

	ifstream p("C:/Users/lin/Desktop/word_fix.txt");
	ifstream l("C:/Users/lin/source/repos/compression/compression/console.txt");

	/*char s1;
	char s2;
	int t = 0;
	while (1)
	{
		t++;
		if (!p.read(&s1, 1)) break;
		if (!l.read(&s2, 1)) break;
		if (s1 != s2)
		{
			cout << t;
			break;
		}
	}

	p.close();
	l.close();*/

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

	string file = "Lenna.raw";
	bool compress = 0;
	bool original = 0;

	constexpr int width = 512;//800 512 2048
	constexpr int height = 512;//600 512 2048

	int min_quality = 0;
	int max_quality = 100;
	int step = 100;

	string link;
	size_t dot_pos = file.rfind('.');
	string name = file.substr(0, dot_pos);

	ifT.open(file, ios::binary);
	if (!ifT.is_open()) {
		cerr << "Error opening file: " << file << endl;
		return 0;
	}

	ifT.seekg(0, ios::end);  // Ïåðåìåùàåì óêàçàòåëü â êîíåö ôàéëà
	auto size = ifT.tellg();       // Ïîëó÷àåì ïîçèöèþ (ðàçìåð ôàéëà)
	ifT.seekg(0, ios::beg);   // Âîçâðàùàåì óêàçàòåëü â íà÷àëî
	cout << "check size: " << size << '\n';
	if (size != 3 * height * width)
	{
		cerr << "Error the resolution does not match the size: " << size << "\nResoult count: " << 3 * height * width << "\n";
		return 0;
	}
	cout << "Ïàðàìåòðû èçîáðàæåíèÿ:\n1) Ðàçìåð: " << size << " áàéò\n";
	cout << "2) Êîëè÷åñòâî: " << size/3 << " pixels\n";
	cout << "3) Ðàçðåøåíèå: " << width << "x" << height << " áàéò\n";

	inputArray r(height, vector<int16_t>(width));
	inputArray g(height, vector<int16_t>(width));
	inputArray b(height, vector<int16_t>(width));

	uint8_t color[3];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (!ifT.read(reinterpret_cast<char*>(color), 3)) {
				throw runtime_error("Îøèáêà ÷òåíèÿ ôàéëà");
			}
			r[y][x] = color[0];
			g[y][x] = color[1];
			b[y][x] = color[2];
		}
	}
	
	for (int quality = max_quality; quality > min_quality; quality -= step)
	{
		// encode
		cout << "Quality: " << quality << '\n';

		inputArray Y(height, vector<int16_t>(width));
		inputArray cb(height, vector<int16_t>(width));
		inputArray cr(height, vector<int16_t>(width));

		rgb_to_ycbcr(height, width, r, g, b, Y, cb, cr);

		//4:2:0 - áåð¸ì ïî 1 öâåòó â äâóõ áëîêàõ ïî 4 ýëåìåíòà
		//öâåò óñðåäí¸ííûé ñðåäè 4 ýëåìåíòîâ
		// ïåðåïèñàòü äàóíñýïëèíã ÷òîáû ñîõðàíÿëàñü ðàçìåðíîñòü ìàññèâîâ
		cb = downsample(height, width, cb);
		cr = downsample(height, width, cr);

		// Ðàçáèåíèå íà áëîêè è çàïèñü áëîêîâ â îäíîìåðíîì âåêòîðå
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
		size_t sizeC_Bl8x8 = cb_blocks.size();// ñîâïàäàåò ñ cr_blocks.size()

		d_Block8x8 q0_matrix{};
		d_Block8x8 q1_matrix{};
		generate_quantization_matrix(quality, q0_matrix, Luminance_quantization_table);
		generate_quantization_matrix(quality, q1_matrix, Chrominance_quantization_table);

		// êâàíòîâàíèå
		for (size_t x = 0; x < sizeY_Bl8x8; x++)
		{// d_Block8x8
			auto Y_dct = dct_2d_8x8(Y_blocks[x]);// çàïèñü îáðàáîòàííîãî ÷åðåç dct áëîê
			quantize(Y_dct, q0_matrix, Y_blocks[x]);// çàïèñü â Y_blocks
		}
		for (size_t x = 0; x < sizeC_Bl8x8; x++)
		{// d_Block8x8
			auto cb_dct = dct_2d_8x8(cb_blocks[x]);
			auto cr_dct = dct_2d_8x8(cr_blocks[x]);
			quantize(cb_dct, q1_matrix, cb_blocks[x]);// çàïèñü â cb_done
			quantize(cr_dct, q1_matrix, cr_blocks[x]);// çàïèñü â cr_done
		}

		// Çèã-çàã îáõîä äëÿ êàæäîãî áëîêà
		vector<int16_t> str1;
		vector<int16_t> str2;
		vector<int16_t> str3;
		str1.reserve(sizeY_Bl8x8 * 64);
		str2.reserve(sizeC_Bl8x8 * 64);
		str3.reserve(sizeC_Bl8x8 * 64);

		cout << '\n';
		cout << "êîë. áëîêîâ Y: " << sizeY_Bl8x8 << '\n';
		cout << "êîë. áëîêîâ Cb: " << sizeC_Bl8x8 << '\n';
		cout << "êîë. áëîêîâ Cr: " << sizeC_Bl8x8 << '\n';

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


		/*ofstream ou("console.txt");
		ou << "[" << str1[0];
		for (size_t i = 1; i < str1.size(); i++) ou << ", " << str1[i];
		ou << "]";
		ou.close();*/

		auto arr1 = str1;
		auto arr2 = str2;
		auto arr3 = str3;

		preparing_for_coding_dc_and_ac(str1);
		preparing_for_coding_dc_and_ac(str2);
		preparing_for_coding_dc_and_ac(str3);

		string coder = "";
		coder += HA_encode(str1, Luminance_DC_differences, Luminance_AC);
		coder += HA_encode(str2, Chrominance_DC_differences, Chrominance_AC);
		coder += HA_encode(str3, Chrominance_DC_differences, Chrominance_AC);

		vector<uint8_t> output = pack_bits_to_bytes(coder);

		cout << "\n\nÐàçìåð ñæàòûõ äàííûõ áåç ìåòàäàííûõ:\n" << output.size() << " áàéò\nèëè\n" << output.size() / 1024 << "Êá\n\n";

		// Çàïèñü â ôàéë

		// çàïèñü òàáëèö

		// çàïèñü ñæàòûõ äàííûõ
		//ofstream ou("console.txt", ios::binary);
		for (size_t i = 0; i < output.size(); i++)
		{
			char space = output[i];  // Áàéò â ASCII èëè ïðîñòî îáû÷íûé ñèìâîë char
			//ou.write(&space, sizeof(space));  // Çàïèñûâàåì 1 áàéò è sizeof = 8 (const)
		}
		//ou.close();

		/*
		ofstream ou("console.txt", ios::binary);
		ifstream in("console.txt", ios::binary);
		for (size_t i = 0; i < out2.size(); i++)
		{
			in.get(reinterpret_cast<char&>(out2[i]));
			//ou << out2[i];
		}
		//ou.close();
		in.close();
		for (size_t i = 0; i < out2.size(); i++)
		{
			if (out2[i] != output[i])
			{
				cout << "\n" << i;
			}
		}
		if (out2 == output)
		{
			cout << "\n\n\n" << 1;
		}
		else cout << 0;
		cout << "\ndone";
		cout << "done";*/

		// decode

		//
		// ?*+$&^-ÒÎÅÐÈß-^&$+*?
		// 
		// çíàÿ ðàçðåøåíèå óçíàåì äëèíó âåêòîðà áëîêîâ 8õ8 ó Y, Cb, Cr
		// 
		// ïðèìåð 1.
		// ðàçðåøåíèå 800 x 600
		// 
		// ÿðêîñòü
		// ðàçáèåíèå íà áëîêè 8õ8
		// 800 / 8 = 100
		// 600 / 8 = 75
		// âñå áåç îñòàòêà
		// êîë. áëîêîâ:
		// 100 * 75 = 7500
		// 
		// öâåò
		// äàóíñýìïëèíã
		// 800 / 2 = 400
		// 600 / 2 = 300
		// âñå áåç îñòàòêà
		// ðàçáèåíèå íà áëîêè 8õ8
		// 400 / 8 = 50
		// 300 / 8 = 37,5
		// îêðóãëèì 37,5 -> 38
		// êîë. áëîêîâ:
		// 50 * 38 = 1900
		// 
		// ïðèåìð 2.
		// ðàçðåøåíèå 803 õ 607
		// 
		// ÿðêîñòü
		// ðàçáèåíèå íà áëîêè 8õ8
		// divUp(803, 8) = 101
		// divUp(607, 8) = 76
		// âñå èìåþò îñòàòêè, ïîýòîìó îêðóãëèë
		// êîë. áëîêîâ:
		// 101 * 76 = 7676
		// 
		// öâåò
		// äàóíñýìïëèíã
		// divUp(803, 2) = 402
		// divUp(607, 2) = 304
		// ðàçáèåíèå íà áëîêè 8õ8
		// divUp(402, 8) = 51
		// divUp(304, 8) = 38 - áåç îñòàòêà
		// êîë. áëîêîâ:
		// 51 * 38 = 1938
		//

		sizeY_Bl8x8 = divUp(width, 8) * divUp(height, 8);

		vector<i16_Block8x8> vecY_Bl8x8(sizeY_Bl8x8);

		sizeC_Bl8x8 = divUp(divUp(width, 2), 8) * divUp(divUp(height, 2), 8);

		vector<i16_Block8x8> vecCb_Bl8x8(sizeC_Bl8x8);
		vector<i16_Block8x8> vecCr_Bl8x8(sizeC_Bl8x8);

		//
		// ïðèìåð òîãî êàê âûãëÿäèò ãîòîâûé HA, íî íóëè è êàòåãîðèè íå çàêîäèðîâàíû
		// DC 3 010 AC:
		// 1) 1 1 0
		// 3) 0 2 10
		// 4) 1 2 10
		// 6) 0 1 1
		// 7) 1 1 1
		// 9) 1 2 01
		// 11) 0 2 10
		// 12) 0 2 01
		// 13) 0 1 0
		// 14) 5 1 0
		// 20) 0 1 0
		// 21) 0 1 1
		// 22) 4 1 0
		// 27) 0 1 1
		// 28) 1 1 0
		// 30) 1 1 1
		// 32) 1 1 1
		// 34) 1 1 1
		// 36) 3 1 0
		// 40) 0 1 1
		// 41) 2 1 0
		// 44) 0 1 0
		// 45) 0 1 0
		// 46) 3 1 0
		// 50) 7 1 1
		// 58) 2 1 1
		// 61) 2 1 0
		// âñåãî 64 ÷èñëà
		// 
		// äîëæíû ïîëó÷èòü ýòî
		// -5 0 - 1 2 0 2 1 0 1 0 - 2 2 - 2 - 1 0 0 0 0 0 - 1 - 1 1 0 0 0 0 - 1 1 0 - 1 0 1 0 1 0 1 0 0 0 - 1 1 0 0 - 1 - 1 - 1 0 0 0 - 1 0 0 0 0 0 0 0 1 0 0 1 0 0 - 1
		// 
		// îáðàòíûé zig-zag
		// âîçâðàùàòü áóäåì ïî áëîêó
		//

		// âåêòîðû áëîêîâ
		vector<array<int16_t, 64>> strY(sizeY_Bl8x8);
		vector<array<int16_t, 64>> strCb(sizeC_Bl8x8);
		vector<array<int16_t, 64>> strCr(sizeC_Bl8x8);

		cout << '\n';
		cout << "êîë. áëîêîâ Y: " << sizeY_Bl8x8 << '\n';
		cout << "êîë. áëîêîâ Cb: " << sizeC_Bl8x8 << '\n';
		cout << "êîë. áëîêîâ Cr: " << sizeC_Bl8x8 << '\n';

		//sizeC_Bl8x8
		// Ðàñêîäèðóåì êàòåãîðèè DC è ïîëó÷èì êîëè÷åñòâî áèò ÷èñëà
		// Ïåðåâåä¸ì äâè÷íîå ÷èñëî â äåñÿòè÷íîå (îáîçíà÷åíèå: ïåðåâîä ÷èñëà 2->10)
		// Ðàñêîäèðóåì êîë. íóëåé/êàòåãîðèþ AC è ïîëó÷èì êîëè÷åñòâî íóëåé è áèò ÷èñëà
		// Ïåðåâîä ÷èñëà 2->10
		// Count äëÿ ïðîâåðêè, ÷òî ðîâíî 64 öèôðû çàïèñàëè
		// Ïåðåõîä ê ñëåäóþùåìó áëîêó
		//

		// Áèíàðíîå ÷òåíèå èç ôàéëà
		vector<uint8_t> out2(output.size());
		ifstream in("console.txt", ios::binary);
		for (size_t i = 0; i < out2.size(); i++)
		{
			in.get(reinterpret_cast<char&>(out2[i]));
		}
		in.close();

		// Èç out2 ñîçäàäèì áèòîâóþ ñòðîêó
		size = output.size();
		string str = "";
		for (uint8_t byte : output)
		{
			bitset<8> bits(byte);
			for (int8_t i = 7; i > -1; i--)
			{
				str += to_string(bits[i]);
			}
		}

		int temp = 0;
		JPEG_decode_HA_RLE(strY, str, sizeY_Bl8x8, Luminance_DC_differences, Luminance_AC, temp);
		JPEG_decode_HA_RLE(strCb, str, sizeC_Bl8x8, Chrominance_DC_differences, Chrominance_AC, temp);
		JPEG_decode_HA_RLE(strCr, str, sizeC_Bl8x8, Chrominance_DC_differences, Chrominance_AC, temp);

		reverse_dc_difference(strY);
		reverse_dc_difference(strCb);
		reverse_dc_difference(strCr);

		bool ok = true;// check JPEG decoder

		// Îáðàòíûé çèã-çàã îáõîä äëÿ âñåõ áëîêîâ
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

		for (size_t x = 0; x < sizeY_Bl8x8; x++)
		{
			d_Block8x8 doub_Y = dequantize(vecY_Bl8x8[x], q0_matrix);
			aY[x] = idct_2d_8x8(doub_Y);
		}
		for (size_t x = 0; x < sizeC_Bl8x8; x++)
		{
			d_Block8x8 doub_Cb = dequantize(vecCb_Bl8x8[x], q1_matrix);
			d_Block8x8 doub_Cr = dequantize(vecCr_Bl8x8[x], q1_matrix);
			aCb[x] = idct_2d_8x8(doub_Cb);
			aCr[x] = idct_2d_8x8(doub_Cr);
		}

		cout << '\n';
		inputArray Y2 = marge8x8Blocks(height, width, aY, 0);
		cout << "\nlever marge: " << 0;
		inputArray cb2 = marge8x8Blocks(height, width, aCb, 1);
		cout << "\nlever marge: " << 1;
		inputArray cr2 = marge8x8Blocks(height, width, aCr, 1);
		cout << "\nlever marge: " << 1;

		cb2 = upScale(height, width, cb2);
		cr2 = upScale(height, width, cr2);

		inputArray r2(height, vector<int16_t>(width));
		inputArray g2(height, vector<int16_t>(width));
		inputArray b2(height, vector<int16_t>(width));

		ycbcr_to_rgb(height, width, r2, g2, b2, Y2, cb2, cr2);

		link = "C:/Users/lin/Desktop/4 ñåìåñòð/ÀèÑÄ 4ñåì/Ëàáà 2 êàðòèíêè/" + name + '/' + name + "_D_" + to_string(quality) + ".bmp";
		writeBMP(link, r2, g2, b2, width, height);

		if (original)
		{
			link = "C:/Users/lin/Desktop/4 ñåìåñòð/ÀèÑÄ 4ñåì/Ëàáà 2 êàðòèíêè/" + name + '/' + name + "_Orig.bmp";
			writeBMP(link, r, g, b, width, height);
		}
	}

	cout << "\n\nDone!!!";
	ifT.close();
	return 0;
}
