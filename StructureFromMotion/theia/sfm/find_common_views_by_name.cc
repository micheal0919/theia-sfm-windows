// Copyright (C) 2015 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)

#include "theia/sfm/find_common_views_by_name.h"

#include <string>
#include <vector>

#include "theia/sfm/reconstruction.h"
#include "theia/sfm/types.h"

namespace theia {

// Finds all views that are common to both reconstructions. Views are considered
// to be the same if they have the same view name. The names of all common views
// are returned.
std::vector<std::string> FindCommonViewsByName(
    const Reconstruction& reconstruction1,
    const Reconstruction& reconstruction2) {
  std::vector<std::string> common_view_names;
  common_view_names.reserve(reconstruction1.NumViews());

  const auto& view_ids1 = reconstruction1.ViewIds();
  for (const ViewId view_id1 : view_ids1) {
    const std::string name = reconstruction1.View(view_id1)->Name();
    const ViewId view_id2 = reconstruction2.ViewIdFromName(name);
    if (view_id2 != kInvalidViewId) {
      common_view_names.emplace_back(name);
    }
  }
  return common_view_names;
}

}  // namespace theia
