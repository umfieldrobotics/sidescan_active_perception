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

#pragma once

//
// Kernels for processing hits and rays outside of OptiX
//

#define PI 3.14159265358979323846f
#define MIN_RANGE 0.0f
#define PPM 150.0f
#define EPS 1e-6f
struct Ray
{
    float3 origin;
    float  tmin;
    float3 dir;
    float  tmax;
};

struct Hit
{
    float  t;
    float3 geom_normal;
    float3 color;
};
void createRaysSideScanOnDevice_RATL( Ray* rays_device, int width, int height, float3 start_pose, float3 end_pose, float theta, float max_range); 
void createRaysOrthoOnDevice( Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, float padding );
void createRaysDownwardOnDevice( Ray* rays_device, int width, int height, float3 bbmin, float3 bbmax, float padding );
void createRaysSideScanOnDevice( Ray* rays_device, int width, int height, float3 start_pose, float3 end_pose, float vertical_angle, float max_range);
void translateRaysOnDevice( Ray* rays_device, int count, float3 offset );
float3 rotate_vec3_xz(float3 input, float theta);
void shadeHitsOnDevice( float3* image_device, int count, const Hit* hits_device );

