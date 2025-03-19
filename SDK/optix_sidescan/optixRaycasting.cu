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

#include <optix.h>

#include "optixRaycasting.h"
#include "optixRaycastingKernels.h"

#include "cuda/LocalGeometry.h"
#include "cuda/whitted.h"

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}


extern "C" __global__ void __raygen__from_buffer()
{
    const uint3        idx        = optixGetLaunchIndex();
    const uint3        dim        = optixGetLaunchDimensions();
    const unsigned int linear_idx = idx.z * dim.y * dim.x + idx.y * dim.x + idx.x;

    unsigned int t, cx, cy, cz;
    Ray          ray = params.rays[linear_idx];
    optixTrace( params.handle, ray.origin, ray.dir, ray.tmin, ray.tmax, 0.0f, OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, t, cx, cy, cz );

    Hit hit;
    
    
    hit.t             = int_as_float( t );
    hit.color.x       = int_as_float( cx );
    hit.color.y       = int_as_float( cy );
    hit.color.z       = int_as_float( cz );
    // printf("Hit Color: %f, %f, %f\n", hit.color.x, hit.color.y, hit.color.z);
    params.hits[linear_idx] = hit;
}


extern "C" __global__ void __miss__buffer_miss()
{
    optixSetPayload_0( float_as_int( -1.0f ) );
    optixSetPayload_1( float_as_int( 1.0f ) );
    optixSetPayload_2( float_as_int( 0.0f ) );
    optixSetPayload_3( float_as_int( 0.0f ) );
}


extern "C" __global__ void __closesthit__buffer_hit()
{
    const float t = optixGetRayTmax(); 

    whitted::HitGroupData* rt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    LocalGeometry          geom    = getLocalGeometry( rt_data->geometry_data );

    // if(rt_data->material_data.pbr.base_color_tex){
    //     // printf("Base Color: %f, %f, %f\n", rt_data->material_data.pbr.base_color.x, rt_data->material_data.pbr.base_color.y, rt_data->material_data.pbr.base_color.z);
    //     // printf("UV Coords: %f, %f\n", geom.UV.x, geom.UV.y);
    // }
    
    // float4 mask = tex2D<float4>( rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y );
    // float4 base_color = tex2D<float4>( rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y );
    float4 base_color = (rt_data->material_data.pbr.base_color_tex) ? tex2D<float4>( rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y ) : rt_data->material_data.pbr.base_color;
    // tex2D<float4>( rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y )
    // if(rt_data->material_data.pbr.base_color_tex){ //check if the base_color_tex field is populated
    //     base_color = tex2D<float4>( rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y );
    // }
    // else{
    //     base_color = rt_data->material_data.pbr.base_color;
    // }
    //print hte base_color 
    // if(rt_data->material_data.pbr.base_color_tex){
    //     printf("Base Color: %f, %f, %f\n", base_color.x, base_color.y, base_color.z);
    // }
    
    // float4  base_color = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    // float4 base_color = rt_data->material_data.pbr.base_color;
    float3 normal = normalize( optixTransformNormalFromObjectToWorldSpace( geom.N ) );
    float3 grazing = normalize( optixGetWorldRayDirection() );

    float lambertian = fmaxf( 0.0f, dot( normal, -grazing ) );
    // printf("Lambertian: %f\n, Normal: <%f, %f, %f>, Grazing: <%f, %f, %f>", lambertian, normal.x, normal.y, normal.z, grazing.x, grazing.y, grazing.z);
    // // //print the color 
    // if(base_color.x > 0.0f || base_color.y > 0.0f || base_color.z > 0.0f){
    //     printf("Base Color: %f, %f, %f\n", base_color.x, base_color.y, base_color.z);
    // }
    

    // Set the hit data
    optixSetPayload_0( float_as_int( t ) );
    optixSetPayload_1( float_as_int( lambertian*base_color.x) );
    optixSetPayload_2( float_as_int( lambertian*base_color.y) );
    optixSetPayload_3( float_as_int( lambertian*base_color.z) );
}


extern "C" __global__ void __anyhit__texture_mask()
{
    whitted::HitGroupData* rt_data = (whitted::HitGroupData*)optixGetSbtDataPointer();
    LocalGeometry          geom    = getLocalGeometry( rt_data->geometry_data );

    float4 mask = tex2D<float4>( rt_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y );
    if( mask.x < 0.5f && mask.y < 0.5f )
    {
        optixIgnoreIntersection();
    }
}

