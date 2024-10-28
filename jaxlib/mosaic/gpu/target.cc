/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "jaxlib/mosaic/gpu/target.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/include/llvm/MC/MCSubtargetInfo.h"
#include "llvm/include/llvm/MC/TargetRegistry.h"

namespace mosaic::gpu {

absl::StatusOr<std::pair<std::string, std::string>> GetSmAndPtxIsaVersion(
    int major, int minor) {
  // "base" compute capability as reported by the driver.
  // For example for a Hopper H200 GPU this would return sm_90, and never
  // sm_90a.
  std::string sm_base = absl::StrCat("sm_", major, minor);

  const std::string triple = "nvptx64-nvidia-cuda";
  std::string error;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(triple, error);
  if (target == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Failed to lookup LLVM target based on triple %s: %s", triple, error));
  }

  // Check if there's a variant of the current SM that ends in "a"
  // (has architecture-specific capabilities)
  const char* sm_arch_specific = nullptr;
  {
    // generic subtarget
    std::unique_ptr<const llvm::MCSubtargetInfo> subtarget_info{
        target->createMCSubtargetInfo(triple, "", "")};
    if (subtarget_info == nullptr) {
      return absl::InternalError(absl::StrFormat(
          "Failed to get generic LLVM subtarget info for triple %s", triple));
    }
    for (const llvm::SubtargetSubTypeKV& subtype :
         subtarget_info->getAllProcessorDescriptions()) {
      if (absl::StartsWith(subtype.Key, sm_base) &&
          absl::EndsWith(subtype.Key, "a")) {
        sm_arch_specific = subtype.Key;
        break;
      }
    }
  }

  const std::string sm = sm_arch_specific ? sm_arch_specific : sm_base;

  std::unique_ptr<const llvm::MCSubtargetInfo> subtarget_info{
      target->createMCSubtargetInfo(triple, sm, "")};
  if (subtarget_info == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Failed to get LLVM subtarget info for sm %s", sm));
  }

  for (const llvm::SubtargetFeatureKV& feature :
       subtarget_info->getEnabledProcessorFeatures()) {
    if (absl::StartsWith(feature.Key, "ptx")) {
      std::string ptx_isa = feature.Key;
      return std::make_pair(sm, ptx_isa);
    }
  }
  return absl::InternalError(absl::StrFormat(
      "Failed to find a PTX ISA LLVM subtarget feature for %s", sm));
}

}  // namespace mosaic::gpu