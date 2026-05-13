//
//  imagesProcessor.cu
// this code is incomplete but will be updated eventually
//
//  Created by Nolan Jones on 2/5/26.
//

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstdint>

void dataTransfer(uint8_t *to, uint8_t *from, size_t length, cudaMemcpyKind direction){
    cudaError_t error;
    error = cudaMemcpy(to, from, length * sizeof(uint8_t), (cudaMemcpyKind)direction);
    if(error != cudaSuccess){
            printf("dataTransfer error: %s\n", cudaGetErrorString(error));
            exit(2);
    }
}
void gpuMalloc(uint8_t **d_g, size_t length){
    cudaError_t error;
    error = cudaMalloc ((void **) d_g, length * sizeof(uint8_t));
    if(error != cudaSuccess){
            printf("gpuMalloc error: %s\n", cudaGetErrorString(error));
            exit(1);
    }
}
void gpuFree(uint8_t *d_g){
    cudaError_t error;
    error = cudaFree(d_g);
    if(error != cudaSuccess){
            printf("gpuFree error: %s\n", cudaGetErrorString(error));
            exit(3);
    }

}
void synchronizeKernel(){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
            printf("Kernel Launch Failed\n");
            exit(4);
    }
}

__global__ void rgb_to_gray_GPU(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *gr, unsigned int width, unsigned int height){
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i = row * width + col;
    
    if(row < height && col < width){
        gr[i] = r[i] * 3/10 + g[i]*6/10 + b[i]/10;
    }
}

__global__ void gray_threshold_GPU(uint8_t *g, unsigned int width, unsigned int height, unsigned int threshold){
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i = row * width + col;
    
    if(row < height && col < width){
        if(gr[i] <= threshold){
            gr[i] = 0;
        }else{
            gr[i] = 255;
        }
    }
}

__global__ void blur_GPU(uint8_t *origional, uint8_t *blurred, unsigned int width, unsigned int height, unsigned int blursize){
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width){
        int sum = 0;
        int count = 0;
        for(int i = row - blursize; i < row + blursize + 1; i++){
            for(int j = col - blursize; j < col + blursize + 1; j++){
                if(i >= 0 && i < height && j >= 0 && j < width){
                    sum += (int)origional[i * width + j];
                    count++;
                }
            }
        }
        blurred[row * width + col] = (uint8_t)(sum/count);
   }
    
}
__global__ void blur_shared_GPU(uint8_t *origional, uint8_t *blurred, unsigned int width, unsigned int height, int blursize){
    // Compute global pixel coordinates
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory tile dimensions (block + halo)
    unsigned int shared_width = blockDim.x + 2 * blursize;
    unsigned int shared_height = blockDim.y + 2 * blursize;

    // Dynamic shared memory
    extern __shared__ uint8_t tile[];

    // Each thread loads multiple pixels in the shared memory tile
    for(int i = threadIdx.y; i < (int)shared_height; i += blockDim.y){
        for(int j = threadIdx.x; j < (int)shared_width; j += blockDim.x){
            int global_x = blockIdx.x * blockDim.x + j - blursize;
            int global_y = blockIdx.y * blockDim.y + i - blursize;

            if(global_x >= 0 && global_x < (int)width && global_y >= 0 && global_y < (int)height){
                tile[i * shared_width + j] = origional[global_y * width + global_x];
            }else{
                tile[i * shared_width + j] = 0;
            }
        }
    }

    __syncthreads();

    // Compute blur via shared memory
    if (col < width && row < height) {
        int sum = 0;
        int count = 0;

        unsigned int shared_x = threadIdx.x + blursize;
        unsigned int shared_y = threadIdx.y + blursize;

        for (int i = -blursize; i <= blursize; i++) {
            for (int j = -blursize; j <= blursize; j++) {
                sum += tile[(shared_y + i) * shared_width + (shared_x + j)];
                count++;
            }
        }

        blurred[row * width + col] = (uint8_t)(sum / count);
    }
}

void rgb_to_gray_GPU(uint8_t *r, uint8_t *g, uint8_t *b, uint8_t *gr, unsigned int width, unsigned int height){
    
    uint8_t *r_g, *g_g, *b_g, *gr_g;
    
    size_t pixels = width * height;
    size_t y = ceil(width/4.0);
    size_t x = ceil(height/4.0);
    unsigned int threads = ceil(sqrt(pixels/(y * x)));
    
    dim3 dimGrid(x, y, 1);
    dim3 dimBlock(threads, threads, 1);
    
    gpuMalloc(&r_g, pixels);
    gpuMalloc(&g_g, pixels);
    gpuMalloc(&b_g, pixels);
    gpuMalloc(&gr_g, pixels);
    
    dataTransfer(r_g, r, pixels, cudaMemcpyHostToDevice);
    dataTransfer(g_g, g, pixels, cudaMemcpyHostToDevice);
    dataTransfer(b_g, b, pixels, cudaMemcpyHostToDevice);
    
    
    rgb_to_gray_GPU <<<dimGrid, dimBlock>>> (r_g, g_g, b_g, gr_g, width, height);
    cudaDeviceSynchronize();
    synchronizeKernel();
    
    dataTransfer(gr, gr_g, pixels, cudaMemcpyDeviceToHost);
    
    cudaFree(r_g);
    cudaFree(g_g);
    cudaFree(b_g);
    cudaFree(gr_g);
}

void blur_gray(uint8_t *g, unsigned int width, unsigned int height, unsigned int blursize, bool shared){
    uint8_t *g_g;
    uint8_t *g_o;
    
    size_t pixels = width * height;
    size_t y = ceil(width/4.0);
    size_t x = ceil(height/4.0);
    unsigned int threads = ceil(sqrt(pixels/(y * x)));
    
    dim3 dimGrid(x, y, 1);
    dim3 dimBlock(threads, threads, 1);
    
    gpuMalloc(&g_g, pixels);
    gpuMalloc(&g_o, pixels);
    
    dataTransfer(g_g, g, pixels, cudaMemcpyHostToDevice);
    
    if(shared == true){
        unsigned int sharedMemSize = (block.x + 2*blursize) * (block.y + 2*blursize) * sizeof(uint8_t);
        gaussian_blur_shared_kernel <<<grid, block, sharedMemSize>>> (g_g, g_o, width, height, blursize);
    }else{
        gaussian_blur_kernel <<<grid, block>>> (g_g, g_o, width, height, blursize);
    }
    cudaDeviceSynchronize();
    synchronizeKernel();
    
    dataTransfer(g, g_o, pixels, cudaMemcpyDeviceToHost);
    
    cudaFree(g_g);
    cudaFree(g_o);
}

void blur_rgb(uint8_t *r, uint8_t *g, uint8_t *b, unsigned int width, unsigned int height, unsigned int blursize, bool shared){
    //will overwrite cpu code conversion ^^^
    uint8_t *r_g, *g_g, *b_g;
    uint8_t *r_o, *g_o, *b_o;
    
    size_t pixels = width * height;
    size_t y = ceil(width/4.0);
    size_t x = ceil(height/4.0);
    unsigned int threads = ceil(sqrt(pixels/(y * x)));
    
    dim3 dimGrid(x, y, 1);
    dim3 dimBlock(threads, threads, 1);
    
    gpuMalloc(&r_g, pixels);
    gpuMalloc(&g_g, pixels);
    gpuMalloc(&b_g, pixels);
    gpuMalloc(&r_o, pixels);
    gpuMalloc(&g_o, pixels);
    gpuMalloc(&b_o, pixels);
    
    dataTransfer(r_g, r, pixels, cudaMemcpyHostToDevice);
    dataTransfer(g_g, g, pixels, cudaMemcpyHostToDevice);
    dataTransfer(b_g, b, pixels, cudaMemcpyHostToDevice);
    
    if(shared == true){
        unsigned int sharedMemSize = (block.x + 2*blursize) * (block.y + 2*blursize) * sizeof(uint8_t);
        gaussian_blur_shared_GPU <<<grid, block, sharedMemSize>>> (r_g, r_o, width, height, blursize);
        gaussian_blur_shared_GPU <<<grid, block, sharedMemSize>>> (g_g, g_o, width, height, blursize);
        gaussian_blur_shared_GPU <<<grid, block, sharedMemSize>>> (b_g, b_o, width, height, blursize);
    }else{
        gaussian_blur_kernel <<<grid, block>>> (r_g, r_o, width, height, blursize);
        gaussian_blur_kernel <<<grid, block>>> (g_g, g_o, width, height, blursize);
        gaussian_blur_kernel <<<grid, block>>> (b_g, b_o, width, height, blursize);
    }
    
    cudaDeviceSynchronize();
    synchronizeKernel();
    
    dataTransfer(r, r_o, pixels, cudaMemcpyDeviceToHost);
    dataTransfer(g, g_o, pixels, cudaMemcpyDeviceToHost);
    dataTransfer(b, b_o, pixels, cudaMemcpyDeviceToHost);
    
    cudaFree(r_g);
    cudaFree(g_g);
    cudaFree(b_g);
    cudaFree(r_o);
    cudaFree(g_o);
    cudaFree(b_o);
    
}

int main(int c, char **argv){
    if(c < 2){
        if(c != 1){
            printf("Missing first file\n");
        }
        if(c != 2){
            printf("Missing second file\n");
        }
        exit(1);
    }
    bool quit = false;
    int choice = 0;
    
    do{
        switch(case){
            case 1:
                
            case 2:
                
            case 3:
                
            case 0:
                quit = true;
                break
            default:
                
                break;
        }
    }while(quit == false);
    
    unsigned int width = 900;
    unsigned int height = 675;
    const size_t pixels = width * height;
    uint8_t red[pixels], green[pixels], blue[pixels], gray[pixels];
    
    FILE *inFS = fopen(argv[1], "r");
    for(size_t i = 0; i < pixels; i++){
        fscanf(inFS, "%hhu %hhu %hhu", &red[i], &green[i], &blue[i]);
    }
    fclose(inFS);
    
    
    rgb_to_gray(red, green, blue, gray, width, height);
    
    FILE outFS = fopen(argv[2], "w");
    for(size_t i = 0; i < pixels; i++){
        fprintf(outFS, "%hhu\n", gray[i]);
    }
    fclose(outFS);
    
    
    return 0;
}
