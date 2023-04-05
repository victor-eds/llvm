// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1 client-api-for-address-space-mapping=Metal' -verify-diagnostics %s | FileCheck %s
// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1 client-api-for-address-space-mapping=Vulkan' -verify-diagnostics %s | FileCheck %s
// RUN: mlir-opt -convert-spirv-to-llvm='use-opaque-pointers=1 client-api-for-address-space-mapping=WebGPU' -verify-diagnostics %s | FileCheck %s

module {  // expected-warning-re {{address space mapping for client {{.*}} not implemented}}
  // CHECK:                llvm.func @pointerUniformConstant(!llvm.ptr)
  spirv.func @pointerUniformConstant(!spirv.ptr<i1, UniformConstant>) "None"

  // CHECK:                llvm.mlir.global external constant @varUniformConstant() {addr_space = 0 : i32} : i1
  spirv.GlobalVariable @varUniformConstant : !spirv.ptr<i1, UniformConstant>

  // CHECK:                llvm.func @pointerInput(!llvm.ptr)
  spirv.func @pointerInput(!spirv.ptr<i1, Input>) "None"

  // CHECK:                llvm.mlir.global external constant @varInput() {addr_space = 0 : i32} : i1
  spirv.GlobalVariable @varInput : !spirv.ptr<i1, Input>

  // CHECK:                llvm.func @pointerUniform(!llvm.ptr)
  spirv.func @pointerUniform(!spirv.ptr<i1, Uniform>) "None"

  // CHECK:                llvm.func @pointerOutput(!llvm.ptr)
  spirv.func @pointerOutput(!spirv.ptr<i1, Output>) "None"

  // CHECK:                llvm.mlir.global external @varOutput() {addr_space = 0 : i32} : i1
  spirv.GlobalVariable @varOutput : !spirv.ptr<i1, Output>

  // CHECK:                llvm.func @pointerWorkgroup(!llvm.ptr)
  spirv.func @pointerWorkgroup(!spirv.ptr<i1, Workgroup>) "None"

  // CHECK:                llvm.func @pointerCrossWorkgroup(!llvm.ptr)
  spirv.func @pointerCrossWorkgroup(!spirv.ptr<i1, CrossWorkgroup>) "None"

  // CHECK:                llvm.func @pointerPrivate(!llvm.ptr)
  spirv.func @pointerPrivate(!spirv.ptr<i1, Private>) "None"

  // CHECK:                llvm.mlir.global private @varPrivate() {addr_space = 0 : i32} : i1
  spirv.GlobalVariable @varPrivate : !spirv.ptr<i1, Private>

  // CHECK:                llvm.func @pointerFunction(!llvm.ptr)
  spirv.func @pointerFunction(!spirv.ptr<i1, Function>) "None"

  // CHECK:                 llvm.func @pointerGeneric(!llvm.ptr)
  spirv.func @pointerGeneric(!spirv.ptr<i1, Generic>) "None"

  // CHECK:                llvm.func @pointerPushConstant(!llvm.ptr)
  spirv.func @pointerPushConstant(!spirv.ptr<i1, PushConstant>) "None"

  // CHECK:                llvm.func @pointerAtomicCounter(!llvm.ptr)
  spirv.func @pointerAtomicCounter(!spirv.ptr<i1, AtomicCounter>) "None"

  // CHECK:                llvm.func @pointerImage(!llvm.ptr)
  spirv.func @pointerImage(!spirv.ptr<i1, Image>) "None"

  // CHECK:                llvm.func @pointerStorageBuffer(!llvm.ptr)
  spirv.func @pointerStorageBuffer(!spirv.ptr<i1, StorageBuffer>) "None"

  // CHECK:                llvm.mlir.global external @varStorageBuffer() {addr_space = 0 : i32} : i1
  spirv.GlobalVariable @varStorageBuffer : !spirv.ptr<i1, StorageBuffer>

  // CHECK:                llvm.func @pointerCallableDataKHR(!llvm.ptr)
  spirv.func @pointerCallableDataKHR(!spirv.ptr<i1, CallableDataKHR>) "None"

  // CHECK:                llvm.func @pointerIncomingCallableDataKHR(!llvm.ptr)
  spirv.func @pointerIncomingCallableDataKHR(!spirv.ptr<i1, IncomingCallableDataKHR>) "None"

  // CHECK:                llvm.func @pointerRayPayloadKHR(!llvm.ptr)
  spirv.func @pointerRayPayloadKHR(!spirv.ptr<i1, RayPayloadKHR>) "None"

  // CHECK:                llvm.func @pointerHitAttributeKHR(!llvm.ptr)
  spirv.func @pointerHitAttributeKHR(!spirv.ptr<i1, HitAttributeKHR>) "None"

  // CHECK:                llvm.func @pointerIncomingRayPayloadKHR(!llvm.ptr)
  spirv.func @pointerIncomingRayPayloadKHR(!spirv.ptr<i1, IncomingRayPayloadKHR>) "None"

  // CHECK:                llvm.func @pointerShaderRecordBufferKHR(!llvm.ptr)
  spirv.func @pointerShaderRecordBufferKHR(!spirv.ptr<i1, ShaderRecordBufferKHR>) "None"

  // CHECK:                llvm.func @pointerPhysicalStorageBuffer(!llvm.ptr)
  spirv.func @pointerPhysicalStorageBuffer(!spirv.ptr<i1, PhysicalStorageBuffer>) "None"

  // CHECK:                llvm.func @pointerCodeSectionINTEL(!llvm.ptr)
  spirv.func @pointerCodeSectionINTEL(!spirv.ptr<i1, CodeSectionINTEL>) "None"

  // CHECK:                llvm.func @pointerDeviceOnlyINTEL(!llvm.ptr)
  spirv.func @pointerDeviceOnlyINTEL(!spirv.ptr<i1, DeviceOnlyINTEL>) "None"

  // CHECK:                llvm.func @pointerHostOnlyINTEL(!llvm.ptr)
  spirv.func @pointerHostOnlyINTEL(!spirv.ptr<i1, HostOnlyINTEL>) "None"
}
