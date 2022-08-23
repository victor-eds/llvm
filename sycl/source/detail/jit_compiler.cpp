//==--- jit_compiler.cpp - SYCL runtime JIT compiler for kernel fusion -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <KernelFusion.h>
#include <detail/device_image_impl.hpp>
#include <detail/jit_compiler.hpp>
#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/kernel_bundle.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

jit_compiler::jit_compiler() : MJITContext{new ::jit_compiler::JITContext{}} {}

jit_compiler::~jit_compiler() = default;

static ::jit_compiler::BinaryFormat
translateBinaryImageFormat(pi::PiDeviceBinaryType Type) {
  switch (Type) {
  case PI_DEVICE_BINARY_TYPE_SPIRV:
    return ::jit_compiler::BinaryFormat::SPIRV;
  case PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE:
    return ::jit_compiler::BinaryFormat::LLVM;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Format unsupported for JIT compiler");
  }
}

static ::jit_compiler::ParameterKind
translateArgType(kernel_param_kind_t Kind) {
  using PK = ::jit_compiler::ParameterKind;
  using kind = kernel_param_kind_t;
  switch (Kind) {
  case kind::kind_accessor:
    return PK::Accessor;
  case kind::kind_std_layout:
    return PK::StdLayout;
  case kind::kind_sampler:
    return PK::Sampler;
  case kind::kind_pointer:
    return PK::Pointer;
  case kind::kind_specialization_constants_buffer:
    return PK::SpecConstBuffer;
  case kind::kind_stream:
    return PK::Stream;
  case kind::kind_invalid:
  default:
    return PK::Invalid;
  }
}

std::unique_ptr<detail::CG>
jit_compiler::fuseKernels(QueueImplPtr Queue,
                          detail::FusionList &InputKernels) {
  // Retrieve the device binary from each of the input
  // kernels to hand them over to the JIT compiler.
  std::vector<::jit_compiler::SYCLKernelInfo> InputKernelInfo;
  std::vector<std::string> InputKernelNames;
  // Collect argument information from all input kernels.
  std::vector<std::vector<char>> ArgsStorage;
  std::vector<detail::AccessorImplPtr> AccStorage;
  std::vector<Requirement *> Requirements;
  std::vector<detail::EventImplPtr> Events;
  std::vector<ArgDesc> FusedArgs;
  NDRDescT NDRDesc;
  int FusedArgIndex = 0;
  // TODO(Lukas, ONNX-399): Collect information about streams and auxiliary
  // resources (which contain reductions) and figure out how to fuse them.
  for (auto &RawCmd : InputKernels) {
    auto *KernelCmd = static_cast<ExecCGCommand *>(RawCmd.get());
    auto &CG = KernelCmd->getCG();
    assert(CG.getType() == CG::Kernel);
    auto *KernelCG = static_cast<CGExecKernel *>(&CG);

    auto KernelName = KernelCG->MKernelName;
    const RTDeviceBinaryImage *DeviceImage;
    RT::PiProgram Program = nullptr;
    if (KernelCG->getKernelBundle() != nullptr) {
      // Retrieve the device image from the kernel bundle.
      auto KernelBundle = KernelCG->getKernelBundle();
      kernel_id KernelID =
          detail::ProgramManager::getInstance().getSYCLKernelID(KernelName);

      auto SyclKernel = detail::getSyclObjImpl(
          KernelBundle->get_kernel(KernelID, KernelBundle));

      DeviceImage = SyclKernel->getDeviceImage()->get_bin_image_ref();
      Program = SyclKernel->getDeviceImage()->get_program_ref();
    } else if (KernelCG->MSyclKernel != nullptr) {
      DeviceImage =
          KernelCG->MSyclKernel->getDeviceImage()->get_bin_image_ref();
      Program = KernelCG->MSyclKernel->getDeviceImage()->get_program_ref();
    } else {
      auto ContextImpl = Queue->getContextImplPtr();
      auto Context = detail::createSyclObjFromImpl<context>(ContextImpl);
      auto DeviceImpl = Queue->getDeviceImplPtr();
      auto Device = detail::createSyclObjFromImpl<device>(DeviceImpl);
      DeviceImage = &detail::ProgramManager::getInstance().getDeviceImage(
          KernelCG->MOSModuleHandle, KernelName, Context, Device);
      Program = detail::ProgramManager::getInstance().createPIProgram(
          *DeviceImage, Context, Device);
    }
    ProgramManager::KernelArgMask EliminatedArgs;
    if (Program && (KernelCG->MSyclKernel == nullptr ||
                    !KernelCG->MSyclKernel->isCreatedFromSource())) {
      EliminatedArgs =
          detail::ProgramManager::getInstance().getEliminatedKernelArgMask(
              KernelCG->MOSModuleHandle, Program, KernelName);
    }

    // Collect information about the arguments of this kernel.

    // Might need to sort the arguments in case they are not already sorted, see
    // also the similar code in commands.cpp.
    auto Args = KernelCG->MArgs;
    std::sort(Args.begin(), Args.end(), [](const ArgDesc &A, const ArgDesc &B) {
      return A.MIndex < B.MIndex;
    });

    ::jit_compiler::SYCLArgumentDescriptor ArgDescriptor;
    size_t ArgIndex = 0;
    for (auto &Arg : Args) {
      ArgDescriptor.Kinds.push_back(translateArgType(Arg.MType));
      // DPC++ internally uses 'true' to indicate that an argument has been
      // eliminated, while the JIT compiler uses 'true' to indicate an argument
      // is used. Translate this here.
      ArgDescriptor.UsageMask.emplace_back(
          (EliminatedArgs.empty() || !EliminatedArgs[ArgIndex++]));
      // Add to the argument list of the fused kernel, but with the correct new
      // index in the fused kernel.
      FusedArgs.emplace_back(Arg.MType, Arg.MPtr, Arg.MSize, FusedArgIndex++);
    }

    // TODO(Lukas, ONNX-399): Check for the correct kernel bundle state of the
    // device image?
    auto &RawDeviceImage = DeviceImage->getRawData();
    auto DeviceImageSize = static_cast<size_t>(RawDeviceImage.BinaryEnd -
                                               RawDeviceImage.BinaryStart);
    // Set 0 as the number of address bits, because the JIT compiler can set
    // this field based on information from SPIR-V/LLVM module's data-layout.
    ::jit_compiler::SYCLKernelBinaryInfo BinInfo{
        translateBinaryImageFormat(DeviceImage->getFormat()), 0,
        RawDeviceImage.BinaryStart, DeviceImageSize};

    InputKernelInfo.emplace_back(KernelName, ArgDescriptor, BinInfo);
    InputKernelNames.push_back(KernelName);

    // Collect information for the fused kernel

    // TODO(Lukas, ONNX-399): Currently assuming the NDRDesc is identical for
    // all input kernels. Actually verify this here or in the graph_builder.
    NDRDesc = KernelCG->MNDRDesc;
    // We need to copy the storages here. The input CGs might be eliminated
    // before the fused kernel gets executed, so we need to copy the storages
    // here to make sure the arguments don't die on us before executing the
    // fused kernel.
    ArgsStorage.insert(ArgsStorage.end(), KernelCG->getArgsStorage().begin(),
                       KernelCG->getArgsStorage().end());
    AccStorage.insert(AccStorage.end(), KernelCG->getAccStorage().begin(),
                      KernelCG->getAccStorage().end());
    // TODO(Lukas, ONNX-399): Does the MSharedPtrStorage contain any information
    // about actual shared pointers beside the kernel bundle and handler impl?
    // If yes, we might need to copy it here.
    Requirements.insert(Requirements.end(), KernelCG->MRequirements.begin(),
                        KernelCG->MRequirements.end());
    Events.insert(Events.end(), KernelCG->MEvents.begin(),
                  KernelCG->MEvents.end());
  }

  // TODO(Lukas, ONNX-399): Fill the following with useful information about the
  // kernels.
  ::jit_compiler::ParamIdentList ParamIdentities;
  std::vector<::jit_compiler::ParameterInternalization> Internalization;
  std::vector<::jit_compiler::JITConstant> JITConstants;

  auto FusedKernelInfo = ::jit_compiler::KernelFusion::fuseKernels(
      *MJITContext, InputKernelInfo, InputKernelNames, "fused", ParamIdentities,
      /* TODO(Lukas, ONNX-399) Use value from property */ -1, Internalization,
      JITConstants);

  auto PIDeviceBinaries = createPIDeviceBinary(FusedKernelInfo);
  detail::ProgramManager::getInstance().addImages(PIDeviceBinaries);

  // Create a kernel bundle for the fused kernel.
  // Kernel bundles are stored in the CG as one of the "extended" members.
  auto FusedKernelId = detail::ProgramManager::getInstance().getSYCLKernelID(
      FusedKernelInfo.Name);
  std::vector<std::shared_ptr<const void>> RawExtendedMembers;

  std::shared_ptr<detail::kernel_bundle_impl> KernelBundleImplPtr =
      detail::getSyclObjImpl(get_kernel_bundle<bundle_state::executable>(
          Queue->get_context(), {Queue->get_device()}, {FusedKernelId}));

  auto CGType = static_cast<CG::CGTYPE>(getVersionedCGType(
      CG::CGTYPE::Kernel, static_cast<int>(CG::CG_VERSION::V1)));
  std::unique_ptr<detail::CG> FusedCG;
  FusedCG.reset(new detail::CGExecKernel(
      NDRDesc, nullptr, nullptr, std::move(KernelBundleImplPtr), std::move(ArgsStorage), std::move(AccStorage),
      std::move(RawExtendedMembers), std::move(Requirements), std::move(Events),
      std::move(FusedArgs), FusedKernelInfo.Name, OSUtil::DummyModuleHandle, {},
      {}, CGType));
  return FusedCG;
}

pi_device_binaries jit_compiler::createPIDeviceBinary(
    ::jit_compiler::SYCLKernelInfo &FusedKernelInfo) {

  DeviceBinaryContainer Binary;

  // Create an offload entry for the fused kernel.
  // It seems to be OK to set zero for most of the information here, at least
  // that is the case for compiled SPIR-V binaries.
  OffloadEntryContainer Entry{FusedKernelInfo.Name, nullptr, 0, 0, 0};
  Binary.addOffloadEntry(std::move(Entry));

  // Create a property entry for the argument usage mask for the fused kernel.
  auto ArgMask = encodeArgUsageMask(FusedKernelInfo.Args.UsageMask);
  PropertyContainer ArgMaskProp{FusedKernelInfo.Name, ArgMask.data(),
                                ArgMask.size(),
                                pi_property_type::PI_PROPERTY_TYPE_BYTE_ARRAY};

  // Create a property set for the argument usage masks of all kernels
  // (currently only one).
  PropertySetContainer ArgMaskPropSet{
      __SYCL_PI_PROPERTY_SET_KERNEL_PARAM_OPT_INFO};

  ArgMaskPropSet.addProperty(std::move(ArgMaskProp));

  Binary.addProperty(std::move(ArgMaskPropSet));

  DeviceBinariesCollection Collection;
  Collection.addDeviceBinary(std::move(Binary),
                             FusedKernelInfo.BinaryInfo.BinaryStart,
                             FusedKernelInfo.BinaryInfo.BinarySize,
                             FusedKernelInfo.BinaryInfo.AddressBits);

  JITDeviceBinaries.push_back(std::move(Collection));
  return JITDeviceBinaries.back().getPIDeviceStruct();
}

std::vector<uint8_t> jit_compiler::encodeArgUsageMask(
    const ::jit_compiler::ArgUsageMask &Mask) const {
  // This must match the decoding logic in program_manager.cpp.
  constexpr uint64_t NBytesForSize = 8;
  constexpr uint64_t NBitsInElement = 8;
  uint64_t Size = static_cast<uint64_t>(Mask.size());
  // Round the size to the next multiple of 8
  uint64_t RoundedSize =
      ((Size + (NBitsInElement - 1)) & (~(NBitsInElement - 1)));
  std::vector<uint8_t> Encoded((RoundedSize / NBitsInElement) + NBytesForSize,
                               0u);
  // First encode the size of the actual mask
  for (size_t i = 0; i < NBytesForSize; ++i) {
    uint8_t Byte =
        static_cast<uint8_t>((RoundedSize >> i * NBitsInElement) & 0xFF);
    Encoded[i] = Byte;
  }
  // Encode the actual mask bit-wise
  for (size_t i = 0; i < Size; ++i) {
    // DPC++ internally uses 'true' to indicate that an argument has been
    // eliminated, while the JIT compiler uses 'true' to indicate an argument
    // is used. Translate this here.
    if (!Mask[i]) {
      uint8_t &Byte = Encoded[NBytesForSize + (i / NBitsInElement)];
      Byte |= static_cast<uint8_t>((1 << (i % NBitsInElement)));
    }
  }
  return Encoded;
}

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
