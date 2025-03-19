//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <cuda_runtime.h>

#include "optixRaycastingKernels.h"

#include <sutil/vec_math.h>
#include <cstdio>

//define a macro for the max range


inline int idivCeil( int x, int y )
{
    return ( x + y - 1 ) / y;
}

float3 rotate_vec3_xz(float3 input, float theta){
    /*Performs a rotation in the xz axis*/
    float3 output;
    output.x = input.x * cos(theta) - input.z * sin(theta);
    output.y = input.y;
    output.z = input.x * sin(theta) + input.z * cos(theta);
    return output;
}

__device__ float3 rotate_vec3_xz_device(float3 input, float theta){
    /*Performs a rotation in the xz axis*/
    float3 output;
    output.x = input.x * cos(theta) - input.z * sin(theta);
    output.y = input.y;
    output.z = input.x * sin(theta) + input.z * cos(theta);
    return output;
}

__global__ void createRaysSideScanKernel_RATL( Ray* rays, int width, int height, float3 pose0, float3 dpose, 
                                                float phi0, float dphi, float theta, float max_range)
{
    const int rayx = threadIdx.x + blockIdx.x * blockDim.x;
    const int rayy = threadIdx.y + blockIdx.y * blockDim.y;
    if( rayx >= width || rayy >= height )
        return;

    float3 ray_origin = pose0 + dpose * rayy;
    float altitude = ray_origin.y;
    // float slant_sample = altitude + EPS + dslant * rayx;
    
    // float phi = acos(altitude/slant_sample) + PI;
    float phi = phi0 + dphi*rayx;
    
    // printf("slant_sample: %f, altitude: %f, theta: %f\n", slant_sample, altitude, theta);
    float3 ray_dir = make_float3(cos(phi), sin(phi), 0.0f);
    float3 rotated_ray_dir = rotate_vec3_xz_device(ray_dir, theta);
    // output.x = ray_dir.x * cos(theta) - ray_dir.z * sin(theta);
    // output.y = ray_dir.y;
    // output.z = ray_dir.x * sin(theta) + ray_dir.z * cos(theta);

    const int idx    = rayx + rayy * width;
    rays[idx].origin = ray_origin;
    rays[idx].tmin   = MIN_RANGE;
    rays[idx].dir    = rotated_ray_dir; //rays pointing down
    rays[idx].tmax   = max_range;
}



// Note: uses left handed coordinate system
void createRaysSideScanOnDevice_RATL( Ray* rays_device, int width, int height, float3 start_pose, float3 end_pose, float theta, float max_range) 
{
    // const float3 bbspan = bbmax - bbmin;
    // float        dx     = bbspan.x * ( 1 + 2 * padding ) / width;
    // float        dz     = bbspan.z * ( 1 + 2 * padding ) / height;
    // float        x0     = bbmin.x - bbspan.x * padding + dx / 2;
    // float        z0     = bbmin.z - bbspan.z * padding + dz / 2;
    // float        y      = 20;
    
    float3 dpose = (end_pose - start_pose)/height;
    float dslant = max_range / width; //for optimal binning

    float phi0 = -PI/2.0f;
    float dphi = -PI/2.0f/width;

    float3 pose0 = start_pose;

    //print everything 
    printf("dx: %f, dy: %f, dz: %f, x0: %f, y0: %f, z0: %f, dslant: %f\n", dpose.x, dpose.y, dpose.z, pose0.x, pose0.y, pose0.z, dslant);
    dim3 blockSize( 32, 16 );
    dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ) );
    createRaysSideScanKernel_RATL<<<gridSize, blockSize>>>( rays_device, width, height, pose0, dpose, phi0, dphi, theta, max_range);
}

__global__ void createRaysSideScanKernel( Ray* rays, int width, int height, float3 pose0, float3 dpose, float dslant, float max_range)
{
    const int rayx = threadIdx.x + blockIdx.x * blockDim.x;
    const int rayy = threadIdx.y + blockIdx.y * blockDim.y;
    if( rayx >= width || rayy >= height )
        return;

    float3 ray_origin = pose0 + dpose * rayy;
    float altitude = ray_origin.y;
    float slant_sample = altitude + EPS + dslant * rayx;
    
    float theta = acos(altitude/slant_sample) - PI/2.0f;
    // if(rayy == 1){
    //     printf("%f\n", theta);
    // }
    // printf("slant_sample: %f, altitude: %f, theta: %f\n", slant_sample, altitude, theta);
    float3 ray_dir = make_float3(cos(theta), sin(theta), 0.0f);
    const int idx    = rayx + rayy * width;
    rays[idx].origin = ray_origin;
    rays[idx].tmin   = MIN_RANGE;
    rays[idx].dir    = ray_dir; //rays pointing down
    rays[idx].tmax   = max_range;
}


// Note: uses left handed coordinate system
void createRaysSideScanOnDevice( Ray* rays_device, int width, int height, float3 start_pose, float3 end_pose, float vertical_angle, float max_range) 
{
    // const float3 bbspan = bbmax - bbmin;
    // float        dx     = bbspan.x * ( 1 + 2 * padding ) / width;
    // float        dz     = bbspan.z * ( 1 + 2 * padding ) / height;
    // float        x0     = bbmin.x - bbspan.x * padding + dx / 2;
    // float        z0     = bbmin.z - bbspan.z * padding + dz / 2;
    // float        y      = 20;
    
    float3 dpose = (end_pose - start_pose)/height;
    float dslant =  max_range / width; //for optimal binning

    float3 pose0 = start_pose;

    //print everything 
    printf("dx: %f, dy: %f, dz: %f, x0: %f, y0: %f, z0: %f, dslant: %f\n", dpose.x, dpose.y, dpose.z, pose0.x, pose0.y, pose0.z, dslant);
    dim3 blockSize( 32, 16 );
    dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ) );
    createRaysSideScanKernel<<<gridSize, blockSize>>>( rays_device, width, height, pose0, dpose, dslant, max_range);
}

/*
Basic Debugging using orthonormal downward facing rays into the scene. 
*/
__global__ void createRaysDownwardKernel( Ray* rays, int width, int height, float x0, float z0, float y, float dx, float dz )
{
    const int rayx = threadIdx.x + blockIdx.x * blockDim.x;
    const int rayy = threadIdx.y + blockIdx.y * blockDim.y;
    if( rayx >= width || rayy >= height )
        return;

    const int idx    = rayx + rayy * width;
    rays[idx].origin = make_float3( x0 + rayx * dx, y, z0 + rayy * dz);
    rays[idx].tmin   = 0.0f;
    rays[idx].dir    = make_float3( 0, -1, 0 ); //rays pointing down
    rays[idx].tmax   = 150.0f;
}


// Note: uses left handed coordinate system
void createRaysDownwardOnDevice( Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, float padding )
{
    const float3 bbspan = bbmax - bbmin;
    float        dx     = bbspan.x * ( 1 + 2 * padding ) / width;
    float        dz     = bbspan.z * ( 1 + 2 * padding ) / height;
    float        x0     = bbmin.x - bbspan.x * padding + dx / 2;
    float        z0     = bbmin.z - bbspan.z * padding + dz / 2;
    float        y      = 20;

    //print everythign 
    printf("dx: %f, dz: %f, x0: %f, z0: %f, y: %f\n", dx, dz, x0, z0, y);
    dim3 blockSize( 32, 16 );
    dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ) );
    createRaysDownwardKernel<<<gridSize, blockSize>>>( rays_device, width, height, x0, z0, y, dx, dz );
}


__global__ void createRaysOrthoKernel( Ray* rays, int width, int height, float x0, float y0, float z, float dx, float dy )
{
    const int rayx = threadIdx.x + blockIdx.x * blockDim.x;
    const int rayy = threadIdx.y + blockIdx.y * blockDim.y;
    if( rayx >= width || rayy >= height )
        return;

    const int idx    = rayx + rayy * width;
    rays[idx].origin = make_float3( x0 + rayx * dx, y0 + rayy * dy, z );
    rays[idx].tmin   = 0.0f;
    rays[idx].dir    = make_float3( 0, 0, 1 ); 
    rays[idx].tmax   = 30.0f;
}


// Note: uses left handed coordinate system
void createRaysOrthoOnDevice( Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, float padding )
{
    const float3 bbspan = bbmax - bbmin;
    float        dx     = bbspan.x * ( 1 + 2 * padding ) / width;
    float        dy     = bbspan.y * ( 1 + 2 * padding ) / height;
    float        x0     = bbmin.x - bbspan.x * padding + dx / 2;
    float        y0     = bbmin.y - bbspan.y * padding + dy / 2;
    float        z      = bbmin.z - fmaxf( bbspan.z, 1.0f ) * .001f;

    //print everything
    printf("dx: %f, dy: %f, x0: %f, y0: %f, z: %f\n", dx, dy, x0, y0, z);

    dim3 blockSize( 32, 16 );
    dim3 gridSize( idivCeil( width, blockSize.x ), idivCeil( height, blockSize.y ) );
    createRaysOrthoKernel<<<gridSize, blockSize>>>( rays_device, width, height, x0, y0, z, dx, dy );
}


__global__ void translateRaysKernel( Ray* rays, int count, float3 offset )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx >= count )
        return;

    rays[idx].origin = rays[idx].origin + offset;
}


void translateRaysOnDevice( Ray* rays_device, int count, float3 offset )
{
    const int blockSize  = 512;
    const int blockCount = idivCeil( count, blockSize );
    translateRaysKernel<<<blockCount, blockSize>>>( rays_device, count, offset );
}


__global__ void shadeHitsKernel( float3* image, int count, const Hit* hits )
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
   
    if( idx >= count )
        return;

    const float3 backgroundColor = make_float3( 0.5f, 0.0f, 0.0f );
    if( hits[idx].t < 0.0f )
    {
        image[idx] = backgroundColor;
    }
    else
    {
        // image[idx] = 0.5f * hits[idx].geom_normal + make_float3( 0.5f, 0.5f, 0.5f );
        // image[idx] = make_float3((float)hits[idx].t, (float)hits[idx].t, (float)hits[idx].t);
        image[idx] = make_float3((float)hits[idx].t, (float)hits[idx].color.x, 0.0f);
    }
}


void shadeHitsOnDevice( float3* image_device, int count, const Hit* hits_device )
{
    const int blockSize  = 512;
    const int blockCount = idivCeil( count, blockSize );
    shadeHitsKernel<<<blockCount, blockSize>>>( image_device, count, hits_device );
}

