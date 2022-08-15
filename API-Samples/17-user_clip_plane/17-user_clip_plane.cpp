/*
 * Vulkan Samples
 *
 * Copyright (C) 2015-2020 Valve Corporation
 * Copyright (C) 2015-2020 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
VULKAN_SAMPLE_SHORT_DESCRIPTION
Draw Cube
*/

/* This is part of the draw cube progression */

#include <util_init.hpp>
#include <assert.h>
#include <string.h>
#include <cstdlib>
#include "cube_data.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500
#define BINDING_COUNT 3
#define UNIFORM_DESC_ENABLE 1
#define UNIFORM_MVP_IDX 0
#define UNIFORM_PLANE_IDX 1
#define SCALE_RATE (0.6f)

/* We've setup cmake to process 15-draw_cube.vert and 15-draw_cube.frag                   */
/* files containing the glsl shader code for this sample.  The generate-spirv script uses */
/* glslangValidator to compile the glsl into spir-v and places the spir-v into a struct   */
/* into a generated header file                                                           */

const std::string MODEL_PATH = "../../API-Samples/data/bigunfinal.obj";

std::vector<VertexUV> vertices;
std::vector<uint32_t> indices;

void loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    std::cout << "Load model begin!\n";

    tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str());
    std::cout << "Load model succeed!\n";

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            VertexUV vertex{};

            vertex.posX = attrib.vertices[3 * index.vertex_index + 0] * SCALE_RATE;
            vertex.posY = attrib.vertices[3 * index.vertex_index + 1] * SCALE_RATE;
            vertex.posZ = attrib.vertices[3 * index.vertex_index + 2] * SCALE_RATE;
            vertex.posW = 1.f;

            vertex.u = attrib.texcoords[2 * index.texcoord_index + 0];
            vertex.v = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1];

            vertices.push_back(vertex);
            indices.push_back(indices.size());
        }
    }

    std::cout << "Load model complete!\n";
}

#if UNIFORM_DESC_ENABLE
void init_plane_uniform_buffer(struct sample_info &info, VkDescriptorBufferInfo *bufferInfo)
{
    VkResult U_ASSERT_ONLY res;
    bool U_ASSERT_ONLY pass;
    VkBuffer buf;
    VkDeviceMemory mem;

    glm::vec4 usePlane = glm::vec4(0.0f, -1.0f, 0.0f, 6.0f);
    
    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.pNext = NULL;
    buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buf_info.size = sizeof(usePlane);
    buf_info.queueFamilyIndexCount = 0;
    buf_info.pQueueFamilyIndices = NULL;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    buf_info.flags = 0;
    res = vkCreateBuffer(info.device, &buf_info, NULL, &buf);
    assert(res == VK_SUCCESS);

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(info.device, buf, &mem_reqs);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = NULL;
    alloc_info.memoryTypeIndex = 0;

    alloc_info.allocationSize = mem_reqs.size;
    pass = memory_type_from_properties(info, mem_reqs.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       &alloc_info.memoryTypeIndex);
    assert(pass && "No mappable, coherent memory");

    res = vkAllocateMemory(info.device, &alloc_info, NULL, &mem);
    assert(res == VK_SUCCESS);

    uint8_t *pData;
    res = vkMapMemory(info.device, mem, 0, mem_reqs.size, 0, (void **)&pData);
    assert(res == VK_SUCCESS);

    memcpy(pData, &usePlane, sizeof(usePlane));

    vkUnmapMemory(info.device, mem);

    res = vkBindBufferMemory(info.device, buf, mem, 0);
    assert(res == VK_SUCCESS);

    bufferInfo->buffer = buf;
    bufferInfo->offset = 0;
    bufferInfo->range = sizeof(usePlane);
}

static VkResult init_uniform_descriptor(struct sample_info &info)
{
    VkResult res;
    int i;

    /* 1. Create descriptor layout and pipeline layout */

    // Create two layout to contain two uniform buffer data.
    VkDescriptorSetLayoutBinding uniform_binding[BINDING_COUNT] = {};
    for (i = 0; i < 2; i++) {
        uniform_binding[i].binding = i;
        uniform_binding[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniform_binding[i].descriptorCount = 1;
        uniform_binding[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        uniform_binding[i].pImmutableSamplers = NULL;
    }

    uniform_binding[2].binding = 2;
    uniform_binding[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    uniform_binding[2].descriptorCount = 1;
    uniform_binding[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    uniform_binding[2].pImmutableSamplers = NULL;

    VkDescriptorSetLayoutCreateInfo uniform_layout_info = {};
    uniform_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    uniform_layout_info.pNext = NULL;
    uniform_layout_info.bindingCount = BINDING_COUNT;
    uniform_layout_info.pBindings = uniform_binding;

    // Create set, using createInfo
    VkDescriptorSetLayout descriptor_layouts[1] = {};
    res = vkCreateDescriptorSetLayout(info.device, &uniform_layout_info, NULL, descriptor_layouts);
    if (res)
        return res;

    // Create pipeline layout with descriptor set
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = NULL;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = NULL;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = descriptor_layouts;
    res = vkCreatePipelineLayout(info.device, &pipelineLayoutCreateInfo, NULL, &info.pipeline_layout);
    if (res)
        return res;

    /* 2. Create a single pool to contain data for our two descriptor sets */
    VkDescriptorPoolSize type_count[2] = {};
    type_count[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    type_count[0].descriptorCount = 2;
    type_count[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    type_count[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.pNext = NULL;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = sizeof(type_count) / sizeof(VkDescriptorPoolSize);
    pool_info.pPoolSizes = type_count;

    VkDescriptorPool descriptor_pool[1] = {};
    res = vkCreateDescriptorPool(info.device, &pool_info, NULL, descriptor_pool);
    if (res)
        return res;

    VkDescriptorSetAllocateInfo alloc_info[1] = {};
    alloc_info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info[0].pNext = NULL;
    alloc_info[0].descriptorPool = descriptor_pool[0];
    alloc_info[0].descriptorSetCount = 1;
    alloc_info[0].pSetLayouts = descriptor_layouts;

    // Populate descriptor sets
    info.desc_set.resize(1);
    res = vkAllocateDescriptorSets(info.device, alloc_info, info.desc_set.data());
    if (res)
        return res;

    // Using empty brace initializer on the next line triggers a bug in older
    // versions of gcc, so memset instead
    VkWriteDescriptorSet descriptor_writes[BINDING_COUNT];
    memset(descriptor_writes, 0, sizeof(descriptor_writes));

    // TODO: Populate with info about our uniform buffer
    descriptor_writes[0] = {};
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].pNext = NULL;
    descriptor_writes[0].dstSet = info.desc_set[0];
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_writes[0].pBufferInfo = &info.uniform_data.buffer_info; // populated by init_uniform_buffer()
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].dstBinding = 0;

    VkDescriptorBufferInfo bufferInfo;
    init_plane_uniform_buffer(info, &bufferInfo);

    descriptor_writes[1] = {};
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].pNext = NULL;
    descriptor_writes[1].dstSet = info.desc_set[0];
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_writes[1].pBufferInfo = &bufferInfo; // populated by init_plane_uniform_buffer()
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].dstBinding = 1;

    descriptor_writes[2] = {};
    descriptor_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[2].dstSet = info.desc_set[0];
    descriptor_writes[2].dstBinding = 2;
    descriptor_writes[2].descriptorCount = 1;
    descriptor_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptor_writes[2].pImageInfo = &info.texture_data.image_info;
    descriptor_writes[2].dstArrayElement = 0;

    vkUpdateDescriptorSets(info.device, BINDING_COUNT, descriptor_writes, 0, NULL);

    return res;
}
#endif

void getKeyboardInputForAngle(struct sample_info &info)
{
    MSG msg;

    GetMessage(&msg, nullptr, 0, 0);
    if (msg.message == WM_INPUT) {
        UsedRawInput(msg.lParam, info);
        if (GET_RAWINPUT_CODE_WPARAM(msg.wParam) == RIM_INPUT)
            DispatchMessage(&msg);
    }
    else {
        DispatchMessage(&msg);
    }

/*     std::cout << "angle_x_delta = " << info.angle_x_delta << std::endl;
    std::cout << "angle_y_delta = " << info.angle_y_delta << std::endl;
 */
}

void updateUniformMap(struct sample_info &info)
{
    float fov = glm::radians(45.0f);
    if (info.width > info.height) {
        fov *= static_cast<float>(info.height) / static_cast<float>(info.width);
    }
    info.Projection = glm::perspective(fov, static_cast<float>(info.width) / static_cast<float>(info.height), 0.1f, 100.0f);
    info.View = glm::lookAt(glm::vec3(20, 15, -10),  // Camera is at (20, 15, -10), in World Space
                            glm::vec3(0, 0, 0),     // and looks at the origin
                            glm::vec3(0, 1, 0)     // Head is up (set to 0,-1,0 to look upside-down)
    );
    info.View = glm::rotate(info.View, glm::radians(info.angle_x_delta), glm::vec3(1, 0, 0));
    info.View = glm::rotate(info.View, glm::radians(info.angle_y_delta), glm::vec3(0, 1, 0));
    info.View = glm::rotate(info.View, glm::radians(info.angle_z_delta), glm::vec3(0, 0, 1));

    info.Model = glm::mat4(1.0f);
    // Vulkan clip space has inverted Y and half Z.
    info.Clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f);

    info.MVP = info.Clip * info.Projection * info.View * info.Model;

    uint8_t *pData;
    int res = vkMapMemory(info.device, info.uniform_data.mem, 0, sizeof(info.MVP), 0, (void **)&pData);
    assert(res == VK_SUCCESS);

    memcpy(pData, &info.MVP, sizeof(info.MVP));

    vkUnmapMemory(info.device, info.uniform_data.mem);
}

VkResult VK_init(struct sample_info &info) {
    VkResult res = VK_SUCCESS;
    const bool depthPresent = true;
    VkShaderModuleCreateInfo vert_info = {};
    VkShaderModuleCreateInfo frag_info = {};

    init_global_layer_properties(info);
    init_instance_extension_names(info);
    init_device_extension_names(info);
    init_instance(info, "User Clip Plane");
    init_enumerate_device(info);
    init_window_size(info, WINDOW_WIDTH, WINDOW_HEIGHT);
    init_connection(info);
    init_window(info);
    init_swapchain_extension(info);
    init_device(info);

    init_command_pool(info);
    init_command_buffer(info);

    init_device_queue(info);
    init_swap_chain(info);
    init_depth_buffer(info);

    loadModel();

    init_texture(info);
    init_uniform_buffer(info); // set up info.MVP and info.uniform_data
    init_renderpass(info, depthPresent);
#include "17-user_clip_plane.vert.h"
#include "17-user_clip_plane.frag.h"
    vert_info.sType = frag_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vert_info.codeSize = sizeof(__user_clip_plane_vert);
    vert_info.pCode = __user_clip_plane_vert;
    frag_info.codeSize = sizeof(__user_clip_plane_frag);
    frag_info.pCode = __user_clip_plane_frag;
    init_shaders(info, &vert_info, &frag_info);
    init_framebuffers(info, depthPresent);

    init_vertex_buffer(info, vertices.data(), vertices.size() * sizeof(VertexUV), sizeof(vertices[0]), true);
    std::cout << vertices.size() << std::endl;
#if UNIFORM_DESC_ENABLE
    res = init_uniform_descriptor(info);
#else
    init_descriptor_and_pipeline_layouts(info, false);
    init_descriptor_pool(info, false);
    init_descriptor_set(info, false);
#endif

    init_pipeline_cache(info);
    init_pipeline(info, depthPresent);

    return res;
}

void VK_Shutdown(struct sample_info &info) {
    destroy_pipeline(info);
    destroy_pipeline_cache(info);
    destroy_descriptor_pool(info);
    destroy_vertex_buffer(info);
    destroy_framebuffers(info);
    destroy_shaders(info);
    destroy_renderpass(info);
    destroy_descriptor_and_pipeline_layouts(info);
    destroy_uniform_buffer(info);
    destroy_depth_buffer(info);
    destroy_swap_chain(info);
    destroy_command_buffer(info);
    destroy_command_pool(info);
    destroy_device(info);
    destroy_window(info);
    destroy_instance(info);
}

int sample_main(int argc, char *argv[]) {
    VkResult U_ASSERT_ONLY res;
    struct sample_info info = {};
    VkSemaphore imageAcquiredSemaphore = {};
    VkFence drawFence = {};
    bool first_fence = true;
    int frame_loop = 1;

    process_command_line_args(info, argc, argv);
    res = VK_init(info);
    assert(res == VK_SUCCESS);

    /* VULKAN_KEY_START */
    while (frame_loop) {
        VkClearValue clear_values[2];
        clear_values[0].color.float32[0] = 1.0f;
        clear_values[0].color.float32[1] = 1.0f;
        clear_values[0].color.float32[2] = 1.0f;
        clear_values[0].color.float32[3] = 1.0f;
        clear_values[1].depthStencil.depth = 1.0f;
        clear_values[1].depthStencil.stencil = 0;
        
        VkSemaphoreCreateInfo imageAcquiredSemaphoreCreateInfo;
        imageAcquiredSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        imageAcquiredSemaphoreCreateInfo.pNext = NULL;
        imageAcquiredSemaphoreCreateInfo.flags = 0;

        res = vkCreateSemaphore(info.device, &imageAcquiredSemaphoreCreateInfo, NULL, &imageAcquiredSemaphore);
        assert(res == VK_SUCCESS);

        // Get the index of the next available swapchain image:
        res = vkAcquireNextImageKHR(info.device, info.swap_chain, UINT64_MAX, imageAcquiredSemaphore, VK_NULL_HANDLE,
                                    &info.current_buffer);
        assert(res == VK_SUCCESS);

        getKeyboardInputForAngle(info);
        updateUniformMap(info);

        execute_begin_command_buffer(info); // vkBeginCommandBuffer

        VkRenderPassBeginInfo rp_begin;
        rp_begin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_begin.pNext = NULL;
        rp_begin.renderPass = info.render_pass;
        rp_begin.framebuffer = info.framebuffers[info.current_buffer];
        rp_begin.renderArea.offset.x = 0;
        rp_begin.renderArea.offset.y = 0;
        rp_begin.renderArea.extent.width = info.width;
        rp_begin.renderArea.extent.height = info.height;
        rp_begin.clearValueCount = 2;
        rp_begin.pClearValues = clear_values;

        vkCmdBeginRenderPass(info.cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline);
        vkCmdBindDescriptorSets(info.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipeline_layout, 0, NUM_DESCRIPTOR_SETS,
                                info.desc_set.data(), 0, NULL);

        const VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(info.cmd, 0, 1, &info.vertex_buffer.buf, offsets);

        init_viewports(info);
        init_scissors(info);

        vkCmdDraw(info.cmd, vertices.size(), 1, 0, 0);
        vkCmdEndRenderPass(info.cmd);
        res = vkEndCommandBuffer(info.cmd);
        const VkCommandBuffer cmd_bufs[] = {info.cmd};

        if (first_fence) {
            VkFenceCreateInfo fenceInfo;
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.pNext = NULL;
            fenceInfo.flags = 0;
            vkCreateFence(info.device, &fenceInfo, NULL, &drawFence);
            first_fence = false;
        }

        VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit_info[1] = {};
        submit_info[0].pNext = NULL;
        submit_info[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info[0].waitSemaphoreCount = 1;
        submit_info[0].pWaitSemaphores = &imageAcquiredSemaphore;
        submit_info[0].pWaitDstStageMask = &pipe_stage_flags;
        submit_info[0].commandBufferCount = 1;
        submit_info[0].pCommandBuffers = cmd_bufs;
        submit_info[0].signalSemaphoreCount = 0;
        submit_info[0].pSignalSemaphores = NULL;

        /* Queue the command buffer for execution */
        res = vkQueueSubmit(info.graphics_queue, 1, submit_info, drawFence);
        assert(res == VK_SUCCESS);

        /* Now present the image in the window */

        VkPresentInfoKHR present;
        present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        present.pNext = NULL;
        present.swapchainCount = 1;
        present.pSwapchains = &info.swap_chain;
        present.pImageIndices = &info.current_buffer;
        present.pWaitSemaphores = NULL;
        present.waitSemaphoreCount = 0;
        present.pResults = NULL;

        /* Make sure command buffer is finished before presenting */
        do {
            res = vkWaitForFences(info.device, 1, &drawFence, VK_TRUE, FENCE_TIMEOUT);
        } while (res == VK_TIMEOUT);

        vkResetFences(info.device, 1, &drawFence);

        assert(res == VK_SUCCESS);
        res = vkQueuePresentKHR(info.present_queue, &present);
        assert(res == VK_SUCCESS);
    }
    /* VULKAN_KEY_END */

    system("pause");

    vkDestroySemaphore(info.device, imageAcquiredSemaphore, NULL);
    vkDestroyFence(info.device, drawFence, NULL);

    VK_Shutdown(info);

    return 0;
}