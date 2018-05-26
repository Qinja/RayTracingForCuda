#pragma once
#include <iostream>
void WriteBmp(int w, int h,const unsigned char *data,int length, const char * fileName)
{
#pragma pack(2)//必须得写，否则sizeof得不到正确的结果  
	typedef unsigned char  BYTE;
	typedef unsigned short WORD;
	typedef unsigned long  DWORD;
	typedef long    LONG;
	typedef struct {
		WORD    bfType;
		DWORD   bfSize;
		WORD    bfReserved1;
		WORD    bfReserved2;
		DWORD   bfOffBits;
	} BITMAPFILEHEADER;

	typedef struct {
		DWORD      biSize;
		LONG       biWidth;
		LONG       biHeight;
		WORD       biPlanes;
		WORD       biBitCount;
		DWORD      biCompression;
		DWORD      biSizeImage;
		LONG       biXPelsPerMeter;
		LONG       biYPelsPerMeter;
		DWORD      biClrUsed;
		DWORD      biClrImportant;
	} BITMAPINFOHEADER;
	// Define BMP Size  
	long size = h * w * 3;
	int x, y;
	long index;

	// Part.1 Create Bitmap File Header  
	BITMAPFILEHEADER fileHeader;

	fileHeader.bfType = 0x4D42;
	fileHeader.bfReserved1 = 0;
	fileHeader.bfReserved2 = 0;
	fileHeader.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + size;
	fileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	// Part.2 Create Bitmap Info Header  
	BITMAPINFOHEADER bitmapHeader = { 0 };

	bitmapHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapHeader.biHeight = h;
	bitmapHeader.biWidth = w;
	bitmapHeader.biPlanes = 3;
	bitmapHeader.biBitCount = 24;
	bitmapHeader.biSizeImage = size;
	bitmapHeader.biCompression = 0; //BI_RGB  

									// Part.3 Create Data  
	BYTE *bits = (BYTE *)malloc(size);

	for (int i = 0, x = 0, index = 0; x < w; x++)
	{
		for (y = 0; y < h; y++, index += 3, i+= length)
		{
			bits[index + 0] = data[i+2]; // Blue  
			bits[index + 1] = data[i+1];   // Green  
			bits[index + 2] = data[i+0];    // Red  
		}
	}
	// Write to file  
	FILE *output = fopen(fileName, "wb");

	if (output == NULL)
	{
		printf("Cannot open file!\n");
	}
	else
	{
		fwrite(&fileHeader, sizeof(BITMAPFILEHEADER), 1, output);
		fwrite(&bitmapHeader, sizeof(BITMAPINFOHEADER), 1, output);
		fwrite(bits, size, 1, output);
		fclose(output);
	}
}
