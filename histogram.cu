#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define TILE_WIDTH 16

typedef struct {
	unsigned char red, green, blue;
} PPMPixel;

typedef struct {
	int x, y;
	PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

//funcao que le o imagem ppm
static PPMImage *readPPM(const char *filename);

__global__ void device_histogram(PPMPixel *image ,float *h, int *l, int *c){
	
	int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
	int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
	
	int n = *l * *c;
	
	int j = threadIdx.x*blockDim.x + threadIdx.y;
	
	int x, y, z;
	
	__shared__ float h_private[64];
	
	//inicializando o contador do bin na copia privada de h
	if(j < 64){
		h_private[j] = 0;
	}
	__syncthreads();

	if(row < *l && col < *c){
		for (x = 0; x <= 3; x++) {
			for (y = 0; y <= 3; y++) {
				for (z = 0; z <= 3; z++) {
					if (image[*c*row + col].red == x && image[*c*row + col].green == y 
						&& image[*c*row + col].blue == z) {
						atomicAdd(&(h_private[x*16+y*4+z]), 1.0f);
					}
				}
			}
		}
		
	}

	__syncthreads();

	//adicionando em h com a normalizacao
	if(j < 64){
		atomicAdd(&(h[j]), h_private[j]/n);
	}
	
}

void Histogram(PPMImage *image, float *h) {
	
	int rows, cols, i;
	int *d_r, *d_c;
	float *d_h;
	PPMPixel *d_image;	

	float n = image->y * image->x;

	cols = image->x;
	rows = image->y;

	
	size_t bytes = sizeof(float)*64;

	for (i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}


	cudaMalloc((void**)&d_r, sizeof(int));
	cudaMalloc((void**)&d_c, sizeof(int));
	cudaMalloc((void**)&d_h, bytes);
	cudaMalloc((void**)&d_image, sizeof(PPMPixel)*cols*rows);

	dim3 dimGrid(ceil((float) cols/TILE_WIDTH), ceil((float) rows/TILE_WIDTH ), 1);//numero de blocos de threads
   	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1); //numero de threads por bloco
    
	cudaMemcpy(d_h, h, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_image, image->data, sizeof(PPMPixel)*cols*rows, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, &rows, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, &cols, sizeof(int), cudaMemcpyHostToDevice);
	
	device_histogram<<<dimGrid , dimBlock>>>(d_image, d_h, d_r, d_c);
	
	cudaMemcpy(h, d_h, bytes, cudaMemcpyDeviceToHost);
   
    cudaFree(d_c);
    cudaFree(d_r); 
    cudaFree(d_h);
    cudaFree(d_image);
}


int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < 64; i++) h[i] = 0.0;

	t_start = rtclock();
	Histogram(image, h);
	t_end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", h[i]);
	}
	printf("\n");
	fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
	free(h);

	return 0;
}

static PPMImage *readPPM(const char *filename) {
	char buff[16];
	PPMImage *img;
	FILE *fp;
	int c, rgb_comp_color;
	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file '%s'\n", filename);
		exit(1);
	}

	if (!fgets(buff, sizeof(buff), fp)) {
		perror(filename);
		exit(1);
	}

	if (buff[0] != 'P' || buff[1] != '6') {
		fprintf(stderr, "Invalid image format (must be 'P6')\n");
		exit(1);
	}

	img = (PPMImage *) malloc(sizeof(PPMImage));
	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	c = getc(fp);
	while (c == '#') {
		while (getc(fp) != '\n')
			;
		c = getc(fp);
	}

	ungetc(c, fp);
	if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
		fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
		exit(1);
	}

	if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
		fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
				filename);
		exit(1);
	}

	if (rgb_comp_color != RGB_COMPONENT_COLOR) {
		fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
		exit(1);
	}

	while (fgetc(fp) != '\n');
	img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

	if (!img) {
		fprintf(stderr, "Unable to allocate memory\n");
		exit(1);
	}

	if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
		fprintf(stderr, "Error loading image '%s'\n", filename);
		exit(1);
	}

	fclose(fp);
	return img;
}
/*
entrada, tempo_serial, tempo_GPU_criar_buffer, tempo_GPU_offload_enviar, tempo_kernel, tempo_GPU_offload_receber, GPU_total, speedup 
arq1.ppm, 0.218407, 0.112063, 0.008347, 0.000029, 0.003583, 0.121043, 1.804375305
arq2.ppm, 0.410912, 0.146193, 0.018426, 0.000035, 0.013859, 0.139981, 2.935484101 
arq3.ppm, 1.532259, 0.110749, 0.072426, 0.000042, 0.055907, 0.239100, 6.408444166


*/